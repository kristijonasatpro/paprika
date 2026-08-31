#!/usr/bin/env python3
"""Kmynas (Parakeet TDT) transcription with speaker labels and timestamps.

The Whisper-based sibling of this script is transcribe_file.py. The difference
that matters is the punctuation stage: Kmynas emits punctuation and casing
itself, so there is no separate tagger to run and no word-preservation check
that can refuse a chunk. Diarization, turn building and the output writers are
shared — imported from transcribe_file rather than duplicated.

Runs in the NeMo venv (.venv-nemo), which is kept separate from the Whisper one
because NeMo pins versions that conflict with it.

    python transcribe_kmynas.py recording.mp3 --speakers 2
"""
from __future__ import annotations

import argparse
import gc
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from chunk_longform import find_cut_points, speech_regions  # noqa: E402
from transcribe_file import (build_turns, diarize, load_audio,  # noqa: E402
                             quiet_transformers, smooth_speakers, speaker_at,
                             write_outputs)

RATE = 16000
MODEL = os.environ.get("KMYNAS_MODEL", str(HERE / "kmynas-parakeet-lt-v3.nemo"))


def pick_device():
    """(device_string, human_label) — CUDA, then Apple MPS, then CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda", "NVIDIA GPU"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", "Apple GPU"
    return "cpu", "CPU (no GPU found)"


VOWELS = set("aeiouyąęėįųū")
# Vowel-less tokens that are nonetheless real: acronyms are handled by the
# all-caps test, these are the spoken fillers that survive it.
FILLER_RE = re.compile(r"^[mhn]{2,}$", re.IGNORECASE)


def guard_invalid_ids(model) -> bool:
    """Drop out-of-vocabulary ids before detokenization.

    In half precision the decoder can leak an id from the blank/duration slots
    (>= vocab_size) into a hypothesis' token list. It is not logit saturation —
    the peak is nowhere near the fp16 range — but a backend defect, and it
    crashes detokenization outright rather than degrading the output. v2 died
    this way deterministically on one recording. v3 has not reproduced it here
    across 39 minutes of audio, but the failure is cheap to make impossible: an
    id the vocabulary cannot express carries no text to lose.
    """
    try:
        dec, vocab = model.decoding, model.tokenizer.vocab_size
    except AttributeError:
        return False

    def guarded(orig):
        def inner(ids, *a, **kw):
            return orig([i for i in (int(x) for x in ids) if i < vocab], *a, **kw)
        return inner

    n = 0
    for name in ("decode_ids_to_tokens", "decode_ids_to_str"):
        orig = getattr(dec, name, None)
        if orig is not None:
            setattr(dec, name, guarded(orig))
            n += 1
    return n > 0


def enable_confidence(model) -> bool:
    """Turn on per-token confidence. False if this model cannot provide it.

    Word-level aggregation is requested as token-level and folded up by hand:
    NeMo's own word aggregation raises "Something went wrong with word-level
    confidence aggregation" on this tokenizer, while token confidences line up
    exactly with the word timestamps once grouped on the sentencepiece word
    marker (verified: 200 groups, 200 words, identical strings).
    """
    from omegaconf import open_dict
    try:
        cfg = model.cfg.decoding
        with open_dict(cfg):
            cfg.confidence_cfg = {
                "preserve_frame_confidence": True,
                "preserve_token_confidence": True,
                "preserve_word_confidence": False,
                "exclude_blank": True,
                "aggregation": "min",
                # Tsallis entropy at alpha=1/3 with min aggregation is NVIDIA's
                # recommended setting for separating hallucinations.
                "method_cfg": {"name": "entropy", "entropy_type": "tsallis",
                               "alpha": 0.33, "entropy_norm": "lin"},
            }
        model.change_decoding_strategy(cfg)
        return True
    except Exception:
        return False


def word_confidence(hyp, tokenizer) -> list[float] | None:
    """Per-word confidence, aligned 1:1 with hyp.timestamp["word"]."""
    tc = getattr(hyp, "token_confidence", None)
    ids = getattr(hyp, "y_sequence", None)
    ts = getattr(hyp, "timestamp", None)
    if tc is None or ids is None or not isinstance(ts, dict) or not ts.get("word"):
        return None
    words = ts["word"]
    ids = [int(i) for i in (ids.tolist() if hasattr(ids, "tolist") else ids)]
    if len(ids) != len(tc):
        return None
    try:
        pieces = tokenizer.ids_to_tokens(ids)
    except Exception:
        return None
    groups: list[list] = []
    for piece, c in zip(pieces, tc):
        if piece.startswith("\u2581") or not groups:
            groups.append([piece.replace("\u2581", ""), float(c)])
        else:
            groups[-1][0] += piece
            groups[-1][1] = min(groups[-1][1], float(c))

    # Consume groups until their text matches each timestamped word. A plain
    # 1:1 zip desynchronises because NeMo does not split before an opening
    # quote — it emits `aš „Registration“` as a single word while the
    # sentencepiece marker starts a new group at the quote.
    out: list[float] = []
    gi = 0
    for w in words:
        target = "".join((w.get("word") or "").split())
        acc, conf = "", 1.0
        while gi < len(groups) and acc != target:
            acc += groups[gi][0]
            conf = min(conf, groups[gi][1])
            gi += 1
        if acc != target:
            return None                  # lost sync; score nothing
        out.append(conf)
    return out


def implausible(word: str) -> bool:
    """True if `word` cannot be Lithuanian: it contains no vowel.

    Acronyms (PVM, LT, SMK) and spoken fillers (mhm) are vowel-less but real,
    so both are excluded. What is left is the hallucination family the model
    emits on non-speech — lch, chl, sxchn, chnz, Zl.
    """
    w = "".join(c for c in word if c.isalpha())
    if len(w) < 1 or set(w.lower()) & VOWELS:
        return False
    if w.isupper() and len(w) >= 2:
        return False                      # acronym
    return not FILLER_RE.match(w)


def drop_hallucinations(words: list[dict], min_conf: float
                        ) -> tuple[list[dict], list[dict]]:
    """Remove words that are BOTH low-confidence AND orthographically impossible.

    Either test alone is too blunt. Confidence alone costs ~1.9% of correct
    words at the threshold that catches most garbage, because quietly spoken
    real words score low too. The vowel test alone would delete nothing wrong
    but also cannot tell a hallucinated "chl" from the acronym "PVM". Together
    they are precise: measured on a 35-min conference call, ch-garbage sits at median confidence
    0.92 against 0.998 for normal tokens.
    """
    keep, dropped = [], []
    for w in words:
        c = w.get("conf")
        if c is not None and c < min_conf and implausible(w["w"]):
            dropped.append(w)
        else:
            keep.append(w)
    return keep, dropped


def split_glued_quotes(words: list[dict]) -> tuple[list[dict], int]:
    """Separate an opening quote the model glued to its neighbour.

    A documented quirk of this checkpoint: it writes `zodis„kitas` with no space,
    and sometimes carries a whole quoted phrase inside one timestamped word
    (`as „Registration“`). Everything downstream treats a word as atomic — the
    seam dedup, the lexicon, speaker attachment and the subtitle wrapper all
    key on whole words — so the pair stays fused in the transcript and counts
    as one word. Split before the quote and share the span by character
    length; the halves are close enough in time that speaker attachment and
    cue breaks land where they should.
    """
    out: list[dict] = []
    n = 0
    for w in words:
        t = w["w"]
        lead = len(t) - len(t.lstrip("„\"'("))
        parts = [p for p in re.split(r"\s+|(?=„)", t[lead:]) if p]
        if len(parts) < 2:
            out.append(w)
            continue
        parts[0] = t[:lead] + parts[0]
        span = float(w["end"]) - float(w["start"])
        total = sum(len(p) for p in parts) or 1
        cur = float(w["start"])
        for part in parts:
            step = span * len(part) / total
            out.append({**w, "w": part, "start": cur, "end": cur + step})
            cur += step
        n += len(parts) - 1
    return out, n


def vad_blocks(pcm: np.ndarray, target: float = 120.0, max_len: float = 300.0,
               pad: float = 0.1, progress: bool = True) -> list[tuple[int, int]]:
    """Speech-tight blocks. The silence between them is never decoded.

    Every block begins at a speech onset and ends at a speech offset, and the
    non-speech between two blocks is dropped rather than split. That is what
    WhisperX, Silero and NVIDIA's own ASR+VAD pipeline all do, and the reason
    is specific to this model: its preprocessor uses `normalize: per_feature`,
    which z-scores each mel bin over every frame of the block. Padding beyond
    the length is excluded from that statistic, but silence INSIDE the block is
    not — so silence at a block edge shifts the features of every speech frame
    in it. Measured on this audio, half a second of room tone at each edge of a
    25 s block moves speech features by ~0.13 SD; NeMo issue #15757 has
    parakeet-tdt-0.6b-v3 returning the empty string for 2.2 s of speech once
    400 ms of trailing silence is appended.

    Cutting at the CENTRE of a pause — which is what an earlier version of this
    code did — therefore puts silence on both sides of the seam and damages
    both neighbours. Cutting at its EDGES and discarding it damages neither.

    Silero is used rather than an energy gate because on this material room
    tone and quiet speech overlap in energy: Silero-speech frames reach down to
    RMS 0.0008 while Silero-nonspeech frames reach up to 0.0083.
    """
    from silero_vad import load_silero_vad, get_speech_timestamps
    import torch
    model = load_silero_vad()
    speech = get_speech_timestamps(
        torch.from_numpy(pcm), model, sampling_rate=RATE,
        threshold=0.5, min_silence_duration_ms=100, speech_pad_ms=int(pad * 1000))
    if not speech:
        return [(0, len(pcm))]

    blocks: list[tuple[int, int]] = []
    start = speech[0]["start"]
    end = speech[0]["end"]
    for seg in speech[1:]:
        if (seg["end"] - start) / RATE > target and end > start:
            blocks.append((start, end))
            start = seg["start"]
        elif (seg["end"] - start) / RATE > max_len:
            blocks.append((start, end))
            start = seg["start"]
        end = seg["end"]
    blocks.append((start, end))

    if progress:
        speech_s = sum(e - s for s, e in blocks) / RATE
        dropped = len(pcm) / RATE - speech_s
        print(f"  VAD: {len(speech)} speech segments -> {len(blocks)} blocks, "
              f"{speech_s/60:.1f} min speech ({dropped/60:.1f} min of non-speech "
              f"dropped, never decoded)", flush=True)
    return blocks


def _key(w: dict) -> str:
    return "".join(c for c in w["w"].lower() if c.isalnum())


def dedup_seams(words: list[dict], overlap: float,
                span: int = 4) -> tuple[list[dict], int]:
    """Drop words the block on each side of a seam both transcribed.

    Ownership by midpoint is not enough on its own: the two decodes of the same
    overlapped audio align it differently, so one copy can land either side of
    the cut and survive twice (the same word emitted twice). This matches
    the tail of one block against the head of the next and removes the repeated
    run.

    Only word pairs that straddle a block boundary AND sit inside the overlap
    window are considered, because genuine disfluent repetition is everywhere
    in spontaneous speech and must survive.
    """
    if not words or overlap <= 0:
        # Contiguous blocks share no audio, so nothing can be transcribed
        # twice and every match here would be genuine repetition.
        for w in words:
            w.pop("blk", None)
        return sorted(words, key=lambda w: w["start"]), 0
    out: list[dict] = [w for w in words if w["blk"] == words[0]["blk"]]
    dropped = 0
    for blk in sorted({w["blk"] for w in words})[1:]:
        head = [w for w in words if w["blk"] == blk]
        if out and head:
            seam = min(w["start"] for w in head)
            for k in range(min(span, len(out), len(head)), 0, -1):
                tail = out[-k:]
                keys = [_key(w) for w in tail]
                if not all(keys) or keys != [_key(w) for w in head[:k]]:
                    continue    # punctuation-only tokens must not match blindly
                if tail[0]["start"] < seam - overlap - 0.5:
                    continue
                if head[k - 1]["end"] > seam + overlap + 0.5:
                    continue
                head = head[k:]
                dropped += k
                break
        out.extend(head)
    out.sort(key=lambda w: w["start"])
    for w in out:
        w.pop("blk", None)
    return out, dropped


def transcribe_words(pcm: np.ndarray, model, device: str, batch_size: int = 1,
                     target: float = 30.0, min_len: float = 15.0,
                     max_len: float = 35.0, overlap: float = 1.3,
                     tokenizer=None, blocks: list[tuple[int, int]] | None = None,
                     progress: bool = True) -> list[dict]:
    """Pause-aligned blocks -> words with absolute timestamps.

    ~30 s blocks, NOT the model's 0.5-15 s training utterance length.

    Shortening blocks to match the training distribution sounds right and is
    wrong. Measured on a 7-min hall recording with v2-final, holding precision fixed:

        block  words  correct probes  errors
         30 s    761       3/5           1/5
          8 s    750       2/5           3/5

    At 30 s the model resolves multi-word phrases that it garbles at 8 s, where
    the first word of each is mis-recognised and drags the rest with it. Longer
    blocks give the decoder more context to resolve exactly the words that are
    ambiguous in isolation. The training-length figure describes the training
    distribution, not an inference constraint.

    batch_size defaults to 1. Batching is what makes this model's GPU memory
    scale: at batch 4 there are four 35 s blocks of activations live at once.

    Each block is decoded with `overlap` seconds of neighbouring audio spliced
    onto both ends, and words are kept only by the block whose core span holds
    their midpoint. A word straddling a cut is therefore decoded whole by both
    neighbours and emitted once. Contiguous blocks split it instead, and the
    orphaned tail gets capitalised as a new sentence: on a 35-min call with v3,
    31 of 60 seams did exactly that, splitting a word across the cut so the
    tail became a spurious capitalised token. Set overlap=0 for the old
    behaviour.
    """
    if blocks is None:
        regions = speech_regions(pcm)
        blocks = [(r0 + a, r0 + b) for r0, r1 in regions
                  for a, b in find_cut_points(pcm[r0:r1], target=target,
                                              min_len=min_len, max_len=max_len)]
    blocks = [(x, y) for x, y in blocks if y - x >= int(0.2 * RATE)]
    if not blocks:
        return []
    speech_s = sum(y - x for x, y in blocks) / RATE
    if progress:
        skipped = len(pcm) / RATE - speech_s
        print(f"  {len(blocks)} blocks, {speech_s/60:.1f} min of speech"
              + (f" ({skipped/60:.1f} min skipped as silence)"
                 if skipped > 5 else ""), flush=True)

    import soundfile as sf
    tmp = tempfile.mkdtemp(prefix="kmynas_")
    pad = int(max(0.0, overlap) * RATE)
    paths, spans, cores = [], [], []
    for i, (x, y) in enumerate(blocks):
        lo, hi = max(0, x - pad), min(len(pcm), y + pad)
        p = f"{tmp}/b{i:05d}.wav"
        sf.write(p, pcm[lo:hi], RATE)
        paths.append(p)
        spans.append((lo / RATE, hi / RATE))
        # Outer edges reach past the audio so nothing decoded at the very
        # start or end of the file is discarded as somebody else's overlap.
        cores.append((-1e9 if i == 0 else x / RATE,
                      1e9 if i == len(blocks) - 1 else y / RATE))

    t0 = time.monotonic()
    out = model.transcribe(paths, batch_size=batch_size, timestamps=True,
                           verbose=progress)
    dt = time.monotonic() - t0
    if progress:
        print(f"  decode {dt:.0f}s for {speech_s:.0f}s (RTF {dt/speech_s:.3f})",
              flush=True)

    words: list[dict] = []
    merged = 0
    for i, ((lo_s, hi_s), (c0, c1), h) in enumerate(zip(spans, cores, out)):
        ts = getattr(h, "timestamp", None)
        if isinstance(ts, dict) and ts.get("word"):
            wc = word_confidence(h, tokenizer) if tokenizer else None
            if wc is not None and len(wc) != len(ts["word"]):
                wc = None            # alignment broke; score nothing rather than mis-score
            pairs = [((w.get("word") or w.get("char") or "").strip(),
                      lo_s + float(w["start"]), lo_s + float(w["end"]),
                      wc[k] if wc else None)
                     for k, w in enumerate(ts["word"])]
        else:
            # No per-word timings for this block: keep the text rather than
            # drop it, spread evenly so it still lands in the right place.
            txt = (h.text if hasattr(h, "text") else str(h)).strip()
            toks = txt.split()
            step = (hi_s - lo_s) / len(toks) if toks else 0.0
            pairs = [(tk, lo_s + step * j, lo_s + step * (j + 1), None)
                     for j, tk in enumerate(toks)]
        for txt, a, b, conf in pairs:
            if not txt:
                continue
            if not c0 <= (a + b) / 2 < c1:
                merged += 1          # same word, kept by the neighbouring block
                continue
            words.append({"w": txt, "start": a, "end": b, "blk": i,
                          "conf": conf})
    words.sort(key=lambda w: (w["blk"], w["start"]))
    words, redup = dedup_seams(words, overlap)
    merged += redup
    if progress and pad:
        print(f"  {overlap:.1f}s overlap: {merged} duplicate words dropped "
              f"at {len(blocks)-1} seams ({redup} by n-gram match)", flush=True)
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass
    return words


def load_lexicon(path: str) -> list[tuple[str, str, tuple[str, ...]]]:
    """(variant_stem, canonical_stem, blocked_prefixes), longest variant first.

    A stem match is blunt: `meil` -> `email` also rewrites `meilės` (love), and
    that word occurs 610 times in 10M tokens of Lithuanian while the borrowing
    is confined to business speech. So each row may carry a third
    tab-separated field listing prefixes that must never be rewritten.
    """
    pairs: list[tuple[str, str, tuple[str, ...]]] = []
    for line in open(path, encoding="utf-8"):
        line = line.split("#", 1)[0].rstrip()
        if not line.strip() or "\t" not in line:
            continue
        parts = line.split("\t")
        canon, variants = parts[0].strip(), parts[1]
        blocked = tuple(b.strip().lower() for b in parts[2].split(",")
                        if b.strip()) if len(parts) > 2 else ()
        for v in variants.split(","):
            v = v.strip()
            if v:
                pairs.append((v, canon, blocked))
    pairs.sort(key=lambda p: -len(p[0]))
    return pairs


def normalise_lexicon(words: list[dict],
                      pairs: list[tuple[str, str, tuple[str, ...]]]
                      ) -> tuple[list[dict], list[tuple[str, str]]]:
    """Rewrite loanword stems onto one spelling, keeping the Lithuanian ending.

    The model spells the same borrowed word many ways in one recording — email
    appeared 180 times across 19 spellings, half of them dropping the initial
    e (`meilo`, `meilą`), and cancel across three stems. Every variant costs a
    word, and no downstream check can catch them because each one is a
    perfectly ordinary-looking Lithuanian string.

    Only the stem is substituted, so inflection survives: meilo -> emailo.
    Case of the first letter is preserved. This deliberately does NOT impose an
    orthography — each canonical form is the one this model already emits most
    often — because the loanword convention is still an open question.
    """
    subs: list[tuple[str, str]] = []
    for idx, w in enumerate(words):
        t = w["w"]
        lead = len(t) - len(t.lstrip("„\"'("))
        core = t[lead:]
        for variant, canon, blocked in pairs:
            low = core.lower()
            if blocked and low.startswith(blocked):
                continue
            if low.startswith(variant):
                rest = core[len(variant):]
                new = canon + rest
                if core[:1].isupper():
                    new = new[:1].upper() + new[1:]
                new = t[:lead] + new
                if new != t:
                    subs.append((t, new))
                    w["w"] = new
                    # The model sometimes splits the loanword in two ("e" +
                    # "meilų"). Rewriting only the tail would leave the orphan
                    # "e" sitting in front of the repaired word.
                    if idx and canon.startswith(words[idx - 1]["w"].strip(",.").lower()) \
                            and len(words[idx - 1]["w"].strip(",.")) <= 2:
                        words[idx - 1]["w"] = ""
                break
    kept = [w for w in words if w["w"]]
    return kept, subs


def enable_boosting(model, path: str, alpha: float) -> int:
    """Bias decoding toward a domain word list. Returns how many terms loaded.

    Product and vendor names are the one error class neither chunking nor the
    confidence filter can touch: the model renders them phonetically and
    inconsistently, and the phonetic form is a perfectly ordinary Lithuanian
    word so nothing downstream can flag it. Measured across our transcripts the
    same word lands several ways — one borrowed word appeared 55 times one way and 53 another,
    and another borrowed verb appeared across three different stems.

    Fusion is applied to the non-blank vocabulary only, so this can re-rank one
    word against another but can never suppress an emission.

    MEASURED 2026-08-30 ON v3: it does not work for this failure. With four
    distinctive domain terms at alpha 1.0,
    exactly one word in a 35-minute recording changed, and NONE of the intended
    targets moved: each stayed in its phonetic spelling. The
    phonetic rendering is not a near-competitor in the lattice; the model
    commits to it acoustically and a decode-time bonus cannot reach it.

    Two further traps if you retry. Short terms leak: a two-character term's
    initial token splits unrelated words, turning `emailai` into `e meilai`.
    And boosting a form the model already gets right makes it worse
    — a list containing `notificationas` overrode the model's own, better
    `notification'as`. Kept as an opt-in flag, off by default.
    """
    from omegaconf import open_dict
    terms = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    if not terms:
        return 0
    cfg = model.cfg.decoding
    with open_dict(cfg):
        cfg.greedy = cfg.get("greedy", {})
        cfg.greedy.boosting_tree = {
            "key_phrases_list": terms,
            "context_score": 1.0,
            "depth_scaling": 2.0,   # NeMo's documented value for CTC/RNN-T/TDT
            "use_triton": False,    # Triton path is CUDA-only; MPS needs the
                                    # pure-PyTorch fallback
        }
        cfg.greedy.boosting_tree_alpha = alpha
    model.change_decoding_strategy(cfg)
    return len(terms)


def block_length(model, override: float | None = None
                 ) -> tuple[float, float, float]:
    """(target, min, max) block seconds. 35 s unless `override` says otherwise.

    Matching the model's embedded training cap sounds obviously right and
    measured worse. v3 carries max_duration 20, so decoding in 10-20 s blocks
    keeps it inside its training distribution — but on a 35-min call that scored
    20.34% against 19.56% for 18-35 s blocks, and reading the two confirmed the
    trade rather than contradicting it: 20 s recovers two words and drops or
    corrupts three others. Longer
    blocks give the decoder more context to resolve ambiguous words, and that
    keeps outweighing the distribution mismatch. Left as a flag, not a default,
    because the balance may flip on cleaner or more monologue-like audio.
    """
    if override is not None:
        return override * 0.85, override * 0.5, override
    return 30.0, 15.0, 35.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--speakers", type=int, default=-1,
                    help="known speaker count; -1 lets the threshold decide")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--diar-min-off", type=float, default=0.5,
                    help="seconds a voice must go quiet before a speaker "
                         "change is believed. Lower (~0.15) for overlapping "
                         "conversation, where short turns are otherwise lost")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--no-vad", action="store_true",
                    help="disable VAD block placement and fall back to the "
                         "older pause-scored fixed-target chunker")
    ap.add_argument("--vad-target", type=float, default=60.0,
                    help="target block length in seconds. 60 measured best on "
                         "16 GB: 120 is no better and peaks 3.6 GB of swap")
    ap.add_argument("--lexicon", default=None,
                    help="TSV of loanword spelling variants to collapse onto "
                         "one form (see lexicon.tsv)")
    ap.add_argument("--boost-file", default=None,
                    help="file of domain terms (one per line) to bias decoding "
                         "toward; fixes phonetic renderings of product names")
    ap.add_argument("--boost-alpha", type=float, default=1.0,
                    help="strength of --boost-file bias")
    ap.add_argument("--block-secs", type=float, default=None,
                    help="longest block to decode (default 35). Setting this "
                         "to the model's training max_duration measured worse, "
                         "not better — see block_length()")
    ap.add_argument("--min-confidence", type=float, default=0.98,
                    help="drop words below this confidence that are also "
                         "orthographically impossible in Lithuanian; 0 disables")
    ap.add_argument("--overlap", type=float, default=1.3,
                    help="seconds of context spliced onto each block end; 0 "
                         "restores contiguous blocks and their seam artefacts")
    ap.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16",
                    help="fp16 halves device memory and decodes ~15%% faster. "
                         "Measured against fp32 on 39 minutes of Lithuanian "
                         "speech: 0.39%% of words differ and neither is "
                         "consistently better. v2 checkpoints need fp32 — they "
                         "crash in fp16")
    ap.add_argument("--no-diar", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    quiet_transformers()
    import torch
    import nemo.collections.asr as nemo_asr

    pcm = load_audio(args.input)
    dur = len(pcm) / RATE
    dev, where = pick_device()
    print(f"audio: {dur/60:.1f} min   device: {dev} — {where}", flush=True)

    t0 = time.monotonic()
    half = dev != "cpu" and args.dtype != "fp32"
    # Restore to CPU, cast there, and only then move to the device. Restoring
    # straight onto the device puts a full fp32 copy of the weights in its
    # allocator, and that copy sets the high-water mark for the whole run even
    # though it is dead a moment later — so `--dtype fp16` bought nothing.
    # Measured on MPS, 7-minute recording, peak device allocation:
    #
    #     restore onto device      fp32 6.12 GiB    fp16 6.13 GiB
    #     restore onto CPU first   fp32 4.06 GiB    fp16 2.22 GiB
    #
    # Transcripts are byte-identical either way, word timings included. The
    # fp32 copy still exists, it just lives in ordinary RAM where it is cheap
    # and immediately reclaimable.
    model = nemo_asr.models.ASRModel.restore_from(
        args.model, map_location=torch.device("cpu"))
    model = model.eval()
    if half:
        # Half precision on CPU is emulated and slower, same as the Whisper
        # path. On GPU it decodes faster.
        model = model.half() if args.dtype == "fp16" \
            else model.to(torch.bfloat16)
    model = model.to(dev)
    if half:
        guard_invalid_ids(model)
    print(f"  model loaded in {time.monotonic()-t0:.0f}s ({args.dtype})",
          flush=True)

    if args.boost_file:
        try:
            n = enable_boosting(model, args.boost_file, args.boost_alpha)
            print(f"  word boosting: {n} terms (alpha {args.boost_alpha})", flush=True)
        except Exception as exc:
            print(f"  word boosting unavailable ({type(exc).__name__}: {exc}); "
                  "continuing without it", flush=True)

    scoring = args.min_confidence > 0 and enable_confidence(model)
    if args.min_confidence > 0 and not scoring:
        print("  note: this model cannot report confidence; filter disabled",
              flush=True)

    blocks = None
    overlap = args.overlap
    if not args.no_vad:
        try:
            blocks = vad_blocks(pcm, target=args.vad_target,
                                max_len=args.vad_target * 2.5,
                                progress=not args.quiet)
        except ImportError:
            print("  note: silero-vad not installed; falling back to the "
                  "pause-scored chunker (pip install silero-vad)", flush=True)
        # Boundaries already sit in non-speech, so neighbours share no words to
        # merge — and splicing overlap back on would re-import the very silence
        # the VAD just removed.
    if blocks is not None:
        overlap = 0.0
    target, min_len, max_len = block_length(model, args.block_secs)
    if blocks is None:
        print(f"  blocks: {min_len:.0f}-{max_len:.0f}s (target {target:.0f}s)",
              flush=True)

    words = transcribe_words(pcm, model, dev, batch_size=args.batch_size,
                             target=target, min_len=min_len, max_len=max_len,
                             overlap=overlap, blocks=blocks,
                             tokenizer=model.tokenizer if scoring else None,
                             progress=not args.quiet)
    if not words:
        raise SystemExit("no speech found")
    print(f"decode: {len(words)} words", flush=True)

    if scoring:
        words, dropped = drop_hallucinations(words, args.min_confidence)
        if dropped:
            shown = ", ".join(repr(d["w"]) for d in dropped[:8])
            print(f"  dropped {len(dropped)} low-confidence non-words: {shown}"
                  + (" ..." if len(dropped) > 8 else ""), flush=True)

    words, glued = split_glued_quotes(words)
    if glued:
        print(f"  split {glued} words fused to an opening quote", flush=True)

    # Decoding is done; drop the acoustic model before diarization. It is
    # ~5 GB in fp32 and nothing below touches it, but it stayed resident on the
    # device for the whole diarization pass — which is itself memory-hungry and
    # runs on CPU — roughly doubling peak footprint on a 16 GB machine for no
    # reason.
    del model
    gc.collect()
    if dev == "mps":
        torch.mps.empty_cache()
    elif dev == "cuda":
        torch.cuda.empty_cache()

    if not args.no_diar:
        t0 = time.monotonic()
        segs = diarize(pcm, num_speakers=args.speakers,
                       threshold=args.threshold, min_off=args.diar_min_off)
        n_spk = len({s for _, _, s in segs})
        print(f"diarize: {len(segs)} segments, {n_spk} speakers, "
              f"{time.monotonic()-t0:.0f}s", flush=True)
        for w in words:
            w["spk"] = speaker_at(segs, w["start"], w["end"])
        moved = smooth_speakers(words)
        if moved:
            print(f"  smoothed {moved} words out of one-off speaker runs",
                  flush=True)

    if args.lexicon:
        words, subs = normalise_lexicon(words, load_lexicon(args.lexicon))
        if subs:
            import collections as _c
            top = _c.Counter(f"{a} -> {b}" for a, b in subs).most_common(8)
            print(f"  lexicon: {len(subs)} loanword spellings normalised; "
                  + ", ".join(f"{k} (x{v})" for k, v in top), flush=True)

    turns = build_turns(words)
    src = pathlib.Path(args.input)
    if args.out_dir:
        out_dir = pathlib.Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / src.stem
    else:
        base = src.parent / src.stem
    write_outputs(base, turns, words)
    print(f"\n{len(turns)} turns, {len(words)} words, {dur:.0f}s total",
          flush=True)
    print(f"wrote {base}.txt / .srt / .vtt / .json", flush=True)


if __name__ == "__main__":
    main()
