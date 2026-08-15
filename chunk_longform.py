#!/usr/bin/env python3
"""Long audio -> transcript with BOUNDED memory, without losing sentences.

The problem this solves. Two options existed and both were bad:

  * transformers' chunked pipeline (chunk_length_s=30) cuts at a fixed stride,
    transcribes each window independently, then MERGES by matching text in the
    overlaps. When the two sides disagree it discards the span it cannot align.
    Measured 2026-08-15: it silently dropped 30 words from one recording and 52
    from another, both times at a seam, on clean audio. The output reads
    perfectly — there is no marker that anything is missing.
  * Whisper's native long-form decoding is correct (it carries decoder state
    across windows and needs no merge) but holds the whole feature sequence:
    ~18 GB for 22 minutes with word timestamps, scaling with duration. A
    two-hour recording will not fit on a 24 GB machine.

The fix is to cut only where nobody is speaking. If every boundary falls in
silence then no word is split, the segments are independent by construction,
and concatenation needs no merge step at all — the failure mode is designed
out rather than patched. Memory is then bounded by the segment length, not the
file length.

Context is carried forward as an initial prompt so a segment beginning after a
pause still knows what preceded it, which is what native long-form does
internally with previous tokens.
"""
from __future__ import annotations

import numpy as np

RATE = 16000


def find_cut_points(pcm: np.ndarray, sr: int = RATE, target: float = 55.0,
                    min_len: float = 25.0, max_len: float = 70.0,
                    frame_ms: int = 50) -> list[tuple[int, int]]:
    """Segment boundaries placed in a genuine PAUSE near `target`.

    max_len may exceed 30 s ONLY because each block is decoded with Whisper's
    native long-form algorithm, which slides its own 30 s encoder window and
    carries state across. Handing a >30 s clip to a single forward pass would
    silently truncate to the first 30 s.

    Two earlier versions were both wrong, each caught by a positive control on
    synthetic speech with known pause positions:

      1. argmin over the whole window returned the EARLIEST minimum on ties, so
         the median segment came out at exactly min_len — fixed-interval
         cutting wearing a pause-aligned costume.
      2. "frames below the window's 25th percentile, nearest to target" assumes
         pauses make up a quarter of the window. When they make up less — 20%
         in the control, and continuous speech gives less still — the
         percentile lands INSIDE the speech distribution and the chosen frame
         is merely quiet-ish speech. The control cut at rms 0.048 where true
         silence (0.000) sat three seconds away.

    Now each frame is scored on how quiet it is RELATIVE to the window's own
    range, plus a small penalty for straying from target. Where real silence
    exists the quiet term is 0 for every silent frame and the distance term
    picks the one nearest target; where none exists the score degrades to the
    best available compromise instead of pretending.
    """
    win = int(frame_ms / 1000 * sr)
    n_fr = len(pcm) // win
    if n_fr == 0:
        return [(0, len(pcm))]
    rms = np.sqrt((pcm[:n_fr * win].reshape(n_fr, win).astype(np.float32) ** 2)
                  .mean(axis=1) + 1e-12)
    segs, start = [], 0
    while start < len(pcm):
        if len(pcm) - start <= int(max_len * sr):
            segs.append((start, len(pcm)))
            break
        lo, hi = start + int(min_len * sr), min(len(pcm), start + int(max_len * sr))
        f_lo, f_hi = lo // win, min(hi // win, n_fr)
        if f_hi <= f_lo:
            cut = hi
        else:
            w = rms[f_lo:f_hi]
            f_tgt = (start + int(target * sr)) // win - f_lo
            floor = float(w.min())
            span = max(float(w.max()) - floor, 1e-9)
            quiet = (w - floor) / span                     # 0 = quietest here
            dist = np.abs(np.arange(len(w)) - f_tgt) / max(1, len(w))
            # 0.35 weights length regularity against silence. Raising it drifts
            # back toward fixed-interval cutting; lowering it lets a block run
            # to max_len chasing a marginally quieter frame.
            best = int(np.argmin(quiet + 0.35 * dist))
            cut = (f_lo + best) * win
        segs.append((start, cut))
        start = cut
    return segs


def speech_regions(pcm: np.ndarray, sr: int = RATE, rms_thr: float = 1e-4,
                   frame_ms: int = 100, pad: float = 0.3,
                   min_gap: float = 1.5) -> list[tuple[int, int]]:
    """Sample ranges that actually contain sound, gaps >= min_gap removed.

    Whisper narrates silence. One press-conference recording opens with 6.2 minutes of
    DIGITAL silence (rms exactly 0.0) and the offline transcript opens with
    "vyriausybe prarytu ar jusu skelbt visok jusu jusu jus galvoju" — confident
    Lithuanian describing nothing. Decoding silence cannot produce a right
    answer, so the cheapest correct move is to not decode it. An RMS gate costs
    microseconds and removes the failure mode instead of filtering it later.

    Regions are padded by `pad` so a consonant onset just under threshold is
    not clipped, and gaps shorter than min_gap are kept as real speech pauses.
    """
    win = int(frame_ms / 1000 * sr)
    n_fr = len(pcm) // win
    if n_fr == 0:
        return [(0, len(pcm))]
    rms = np.sqrt((pcm[:n_fr * win].reshape(n_fr, win).astype(np.float32) ** 2)
                  .mean(axis=1) + 1e-12)
    voiced = rms > rms_thr
    if not voiced.any():
        return []
    regs: list[list[int]] = []
    gap_fr = int(min_gap * 1000 / frame_ms)
    f = 0
    while f < n_fr:
        if voiced[f]:
            g = f
            while g < n_fr:
                if voiced[g]:
                    g += 1
                    continue
                nxt = g
                while nxt < n_fr and not voiced[nxt]:
                    nxt += 1
                if nxt - g >= gap_fr or nxt >= n_fr:
                    break
                g = nxt
            regs.append([f * win, min(len(pcm), g * win)])
            f = g
        f += 1
    p = int(pad * sr)
    return [(max(0, a - p), min(len(pcm), b + p)) for a, b in regs]


def transcribe_words(pcm: np.ndarray, model, processor, device: str = "cpu",
                     language: str = "lithuanian", sr: int = RATE,
                     target: float = 25.0, min_len: float = 12.0,
                     max_len: float = 29.0, gate_silence: bool = True,
                     verbose: bool = False, progress: bool = False) -> list[dict]:
    """Same pause-aligned scheme as transcribe(), but returns WORDS with
    absolute timestamps so punctuation and diarization have something to
    attach to.

    Blocks are capped at 29 s here rather than the 70 s used by transcribe().
    That is not a style choice: past 30 s generate() switches to its long-form
    path, where token_timestamps come back per internal segment and no longer
    index the block cleanly. Staying under 30 s keeps one forward pass per
    block, so a token's timestamp is simply its offset from the block start.
    Cuts still land in pauses, so nothing is split across a boundary.
    """
    import sys
    import time as _time

    import torch
    regions = (speech_regions(pcm, sr) if gate_silence
               else [(0, len(pcm))])
    # Enumerate every block up front so progress can show a real denominator
    # and an ETA. Without this the run printed nothing between "audio: 7.6 min"
    # and completion, so a slow run was indistinguishable from a hung one —
    # which is how the first Colab attempt was killed by hand.
    blocks = [(r0 + a, r0 + b) for r0, r1 in regions
              for a, b in find_cut_points(pcm[r0:r1], sr, target=target,
                                          min_len=min_len, max_len=max_len)]
    blocks = [(x, y) for x, y in blocks if y - x >= int(0.2 * sr)]
    speech_s = sum(y - x for x, y in blocks) / sr
    if progress:
        skipped = len(pcm) / sr - speech_s
        print(f"  {len(blocks)} blocks, {speech_s/60:.1f} min of speech"
              + (f" ({skipped/60:.1f} min silence skipped)" if skipped > 5 else ""),
              flush=True)
    t_start = _time.monotonic()
    words: list[dict] = []
    for i, (x, y) in enumerate(blocks):
            clip, t0 = pcm[x:y], x / sr
            a, b = x, y
            feats = processor(clip, sampling_rate=sr, return_tensors="pt",
                              return_attention_mask=True)
            with torch.no_grad():
                out = model.generate(
                    feats.input_features.to(device, model.dtype),
                    attention_mask=feats.attention_mask.to(device),
                    language=language, task="transcribe",
                    return_timestamps=True, return_token_timestamps=True,
                    return_dict_in_generate=True,
                    condition_on_prev_tokens=False,
                    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                    logprob_threshold=-1.0, compression_ratio_threshold=1.35,
                    no_speech_threshold=0.6, repetition_penalty=1.05)
            toks = out["sequences"][0]
            tts = out.get("token_timestamps")
            tts = tts[0].tolist() if tts is not None else None
            got = words_from_tokens(toks, tts, processor, t0, (b - a) / sr)
            _widen(got, limit=b / sr)
            words.extend(got)
            if verbose:
                print(f"  {t0:7.1f}s +{(b-a)/sr:5.1f}s  {len(got):4}w", flush=True)
            elif progress:
                el = _time.monotonic() - t_start
                eta = el / (i + 1) * (len(blocks) - i - 1)
                sys.stdout.write(
                    f"\r  block {i+1}/{len(blocks)}  {len(words)} words  "
                    f"{el:.0f}s elapsed, ~{eta:.0f}s left   ")
                sys.stdout.flush()
    if progress and not verbose and blocks:
        sys.stdout.write("\r" + " " * 70 + "\r")
        sys.stdout.flush()
    return words


def _widen(words: list[dict], limit: float, min_dur: float = 0.08) -> None:
    """Give zero-width words a real end time, in place.

    Whisper routinely emits end == start for short words (19% of them on the
    a press-conference recording). A subtitle cue of zero length never renders, and a
    zero-width interval overlaps nothing, which breaks any downstream
    alignment. Extend to the next word's start where there is room, so no word
    is stretched over its neighbour.
    """
    for i, w in enumerate(words):
        if w["end"] - w["start"] >= min_dur:
            continue
        nxt = words[i + 1]["start"] if i + 1 < len(words) else limit
        w["end"] = max(w["end"], min(w["start"] + min_dur, max(nxt, w["start"])))


def words_from_tokens(toks, tts, processor, t0: float,
                      dur: float = 0.0) -> list[dict]:
    """Group Whisper's BPE tokens into words with absolute timestamps.

    Tokens are grouped FIRST and decoded as a group. Decoding them one at a
    time corrupts text: Whisper uses byte-level BPE, so a Lithuanian character
    can span two tokens — 'patirti' came back as 'patirt' + U+FFFD + U+FFFD
    because the final letter's two bytes were decoded separately. A word
    boundary is a space, and no character spans a space, so grouping on the
    leading-space marker and decoding each group whole is always safe.
    """
    ids = toks.tolist()
    names = processor.tokenizer.convert_ids_to_tokens(ids)
    groups: list[list[int]] = []
    for i, (tid, nm) in enumerate(zip(ids, names)):
        if nm.startswith("<|") and nm.endswith("|>"):
            continue                      # special / timestamp token
        if nm.startswith("Ġ") or not groups:
            groups.append([i])
        else:
            groups[-1].append(i)
    out: list[dict] = []
    for g in groups:
        txt = processor.decode([ids[i] for i in g],
                               skip_special_tokens=True).strip()
        if not txt:
            continue
        if tts is not None:
            a = tts[g[0]] if g[0] < len(tts) else 0.0
            b = tts[g[-1]] if g[-1] < len(tts) else a
        else:
            a = b = 0.0
        out.append({"w": txt, "start": t0 + a, "end": t0 + b})
    if tts is None and out and dur:  # no alignment heads: spread evenly
        n = len(out)
        for i, w in enumerate(out):
            w["start"], w["end"] = t0 + dur * i / n, t0 + dur * (i + 1) / n
    return out


def transcribe(pcm: np.ndarray, model, processor, device: str = "cpu",
               language: str = "lithuanian", sr: int = RATE,
               context_words: int = 0, verbose: bool = False) -> str:
    """Pause-aligned blocks, each decoded with NATIVE long-form.

    context_words defaults to 0. Carrying the previous words as an initial
    prompt SOUNDS right — it is what native long-form does internally — but
    measured 2026-08-15 it cost 71 words on a 10-minute file (354 vs 425)
    because stripping the prompt back out of the output is unreliable: token
    slicing removes real content when generate does not echo the prompt at the
    expected offset, and string matching fails when the model paraphrases its
    opening. Blocks already begin after a pause, which is where context matters
    least. Set >0 only with a verified strip.
    """
    import torch
    segs = find_cut_points(pcm, sr)
    out: list[str] = []
    for i, (a, b) in enumerate(segs):
        clip = pcm[a:b]
        feats = processor(clip, sampling_rate=sr, return_tensors="pt",
                          truncation=False, padding="longest",
                          return_attention_mask=True)
        kw = dict(language=language, task="transcribe", return_timestamps=True,
                  condition_on_prev_tokens=False,
                  temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                  logprob_threshold=-1.0, compression_ratio_threshold=1.35,
                  no_speech_threshold=0.6,
                  # NO no_repeat_ngram_size. Tried n=6 to kill the
                  # galim/isivaizdziuksta loops: it worked, and cost 63 words
                  # (354 vs native 417) because meeting speech repeats phrases
                  # legitimately — "ar galime pritarti", "aciu, pritarta" recur
                  # by design in a committee session. compression_ratio_threshold
                  # already catches DEGENERATE output and falls back through the
                  # temperature ladder, which targets loops without punishing
                  # real repetition.
                  repetition_penalty=1.05)
        prev = " ".join(" ".join(out).split()[-context_words:]) if out else ""
        n_prompt = 0
        if prev.strip() and context_words:
            pids = processor.get_prompt_ids(prev, return_tensors="pt")
            kw["prompt_ids"] = pids.to(device)
            n_prompt = int(pids.shape[-1])
        with torch.no_grad():
            ids = model.generate(feats.input_features.to(device, torch.float32),
                                 attention_mask=feats.attention_mask.to(device),
                                 **kw)
        # TOKEN-level prompt strip. The string-prefix heuristic used first left
        # the prompt in the output when the model paraphrased its opening,
        # duplicating a phrase at one seam.
        seq = ids[0]
        if n_prompt:
            seq = seq[n_prompt:] if seq.shape[-1] > n_prompt else seq
        txt = processor.decode(seq, skip_special_tokens=True).strip()
        if prev and txt.lower().startswith(prev.lower()[:24]):
            txt = txt[len(prev):].strip()
        if verbose:
            print(f"  seg {i:3} {a/sr:7.1f}-{b/sr:7.1f}s ({(b-a)/sr:5.1f}s) "
                  f"{len(txt.split()):4}w", flush=True)
        if txt:
            out.append(txt)
    return " ".join(out)
