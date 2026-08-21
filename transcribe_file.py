#!/usr/bin/env python3
"""Long Lithuanian audio -> punctuated, speaker-labelled transcript. Local.

Pipeline, and why each stage is where it is:

  1. decode   pause-aligned blocks, each a single forward pass, silence gated
              out before it reaches the model (chunk_longform.transcribe_words)
  2. punctuate  ONNX tagger with a word-preservation contract (punct_restore)
  3. diarize  pyannote segmentation + speaker embeddings + clustering, run
              through sherpa-onnx so it is ONNX-only and needs no gated model
  4. attach   speakers land on WORDS, not on text, then turns are grown from
              the word sequence

Punctuation runs BEFORE speaker attachment on purpose. The tagger needs long
runs of text to place a sentence boundary; punctuating each speaker turn in
isolation would strip exactly the cross-sentence context it uses. Because the
tagger's contract guarantees the word sequence is unchanged (case and
punctuation only), the punctuated tokens zip back onto the timed words 1:1 —
so the ordering costs nothing.

Diarization is NOT native to Whisper. Whisper has no notion of who is speaking;
anything claiming otherwise is a second model bolted alongside. Here that is
pyannote-segmentation-3.0 (powerset overlap-aware segmentation) plus a speaker
embedding model, both ONNX, clustered offline. No audio leaves the machine.

Usage:
    python transcribe_file.py INPUT.{m4a,mov,wav,mp3} [--speakers N]
                              [--model DIR] [--no-diar] [--out-dir DIR]
Writes .txt (speaker-labelled), .srt, .vtt and .json beside the input.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RATE = 16000

# Model resolution, in order: $LT_ASR_MODEL, then the local training output if
# this is a checkout of the training repo, then the published weights. The
# fallback is what makes the file runnable outside this machine unchanged.
HF_MODEL = "kristijonas/paprika-whisper-lt-v3"
_LOCAL = HERE / "acceptance-e/model/final"
DEFAULT_MODEL = (os.environ.get("LT_ASR_MODEL")
                 or (str(_LOCAL) if _LOCAL.exists() else HF_MODEL))

DIAR_DIR = pathlib.Path(os.environ.get("LT_ASR_DIAR_DIR", HERE / "diar-models"))
SEG_MODEL = DIAR_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
# Chosen by measurement on a 10-minute press-conference recording, scored against its
# known question/answer structure (3 journalist questions alternating with 3
# answers, 417 words). At a FIXED two speakers — the only honest comparison,
# since more clusters give the scorer more label permutations to exploit:
#     3dspeaker campplus zh_en   98.8% words, 99.4% of answer words
#     wespeaker en_voxceleb      86.1% words, 98.7% of answer words
# wespeaker merged the first question into the first answer. 3dspeaker also
# returned the SAME labelling at every clustering setting tried (threshold
# .55/.70/.85 and n=2/3/4), which is what finding real structure looks like.
EMB_MODEL = DIAR_DIR / "3dspeaker_campplus_zh_en.onnx"


def pick_device(torch) -> tuple[str, object]:
    """(device, dtype) — CUDA, then Apple MPS, then CPU.

    CUDA MUST be tested first. An earlier version checked only MPS and fell
    through to CPU everywhere else, so every Colab and every Linux GPU box ran
    on CPU while the notebook cheerfully printed the name of the unused T4.
    A 7.6-minute file went from about a minute to not finishing, with no output
    saying why.

    fp16 on GPU, fp32 on CPU: half precision on CPU is emulated and slower.
    """
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def quiet_transformers() -> None:
    """Silence benign, unavoidable transformers chatter.

    Every one of these fires on a correct run: the logits-processor notices are
    caused by our own generate kwargs, return_segments is set by transformers
    itself, and the BPE cleanup notice describes a default we already want.
    Printed once per call they bury the actual progress output, and a user
    watching a wall of warnings cannot tell a working run from a broken one.
    """
    import logging
    import warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
    # onnxruntime announces that CUDAExecutionProvider is unavailable. The
    # punctuation and diarization models are meant to run on CPU, so this is
    # working as intended — but on a GPU box it reads like the GPU was missed.
    warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")
    try:
        from transformers.utils import logging as tlog
        tlog.set_verbosity_error()
    except Exception:
        pass


# --- audio ---------------------------------------------------------------
def _available_ram_gb():
    """Best-effort free RAM. Returns None where it cannot be determined."""
    try:
        import os as _os
        return (_os.sysconf("SC_AVPHYS_PAGES") * _os.sysconf("SC_PAGE_SIZE")) / 1e9
    except (ValueError, OSError, AttributeError):
        return None


def load_audio(path: str, gain: bool = True) -> np.ndarray:
    """Decode anything ffmpeg understands to 16 kHz mono float32."""
    import soundfile as sf
    p = pathlib.Path(path)
    if p.suffix.lower() == ".wav":
        a, sr = sf.read(p, dtype="float32")
        if a.ndim > 1:
            a = a[:, 0]
        if sr != RATE:
            a, sr = _resample(a, sr), RATE
    else:
        raw = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(p),
             "-f", "f32le", "-ac", "1", "-ar", str(RATE), "-"],
            check=True, capture_output=True).stdout
        a = np.frombuffer(raw, dtype=np.float32).copy()
    rms = float(np.sqrt((a ** 2).mean() + 1e-12))
    if rms < 1e-5:
        raise SystemExit(f"{p.name}: audio is silent (rms={rms:.2e}) — "
                         "nothing to transcribe. Check the source file.")
    if gain and rms < 0.02:
        # Quiet phone recordings decode measurably worse. Normalise to a peak
        # of 0.95 rather than to an RMS target, so no sample ever clips.
        peak = float(np.abs(a).max())
        if peak > 0:
            a = a * min(0.95 / peak, 20.0)
    return a


def _resample(a: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    return librosa.resample(a, orig_sr=sr, target_sr=RATE)


# Diarization models, fetched on first use from the sherpa-onnx releases.
# These are ONNX exports, so no Hugging Face gate to accept — pyannote's own
# weights require agreeing to conditions on the website, which cannot be done
# from a script and would block anyone cloning this.
DIAR_URLS = {
    "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2":
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    "3dspeaker_campplus_zh_en.onnx":
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-recongition-models/"
        "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
}


def fetch_diar_models(quiet: bool = False) -> bool:
    """Download the two diarization models if absent. Returns True if ready."""
    if SEG_MODEL.exists() and EMB_MODEL.exists():
        return True
    DIAR_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in DIAR_URLS.items():
        dest = DIAR_DIR / name
        final = DIAR_DIR / name.replace(".tar.bz2", "")
        if final.exists() or (final / "model.onnx").exists():
            continue
        if not quiet:
            print(f"  fetching {name} …", flush=True)
        try:
            subprocess.run(["curl", "-sSLf", "-o", str(dest), url], check=True)
            if name.endswith(".tar.bz2"):
                subprocess.run(["tar", "xjf", str(dest), "-C", str(DIAR_DIR)],
                               check=True)
                dest.unlink(missing_ok=True)
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"  could not fetch {name}: {e}", flush=True)
            return False
    return SEG_MODEL.exists() and EMB_MODEL.exists()


# --- diarization ---------------------------------------------------------
def diarize(pcm: np.ndarray, num_speakers: int = -1,
            threshold: float = 0.55) -> list[tuple[float, float, int]]:
    """[(start_s, end_s, speaker_id)] — empty list if models are missing."""
    try:
        import sherpa_onnx
    except ImportError:
        print("  diarization skipped: sherpa-onnx not installed", flush=True)
        return []
    if not fetch_diar_models():
        print(f"  diarization skipped: models unavailable under {DIAR_DIR}",
              flush=True)
        return []
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(SEG_MODEL))),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(EMB_MODEL)),
        # num_clusters=-1 lets the threshold decide the speaker count. Pass
        # --speakers N when the count is known: it is strictly more reliable
        # than a threshold, which has to guess where one voice ends.
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers, threshold=threshold),
        min_duration_on=0.3, min_duration_off=0.5)
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    res = sd.process(pcm).sort_by_start_time()
    return [(s.start, s.end, s.speaker) for s in res]


def speaker_at(turns: list[tuple[float, float, int]], t0: float,
               t1: float) -> int | None:
    """Speaker whose segment overlaps [t0,t1] most.

    The zero-width case is the whole reason this function is not two lines.
    Whisper gives 19% of words (80 of 417 on a press-conference recording) an end
    timestamp equal to their start. A zero-width interval has zero overlap
    with every segment, so `ov > best_ov` never fired, and those words fell
    through to a fallback that ranked segments by distance to the nearest
    BOUNDARY — which, for a word sitting mid-way through a 57 s answer, is
    some other segment entirely. That misrouted a fifth of all words and was
    the real source of the speaker flipping mid-sentence, not the clustering.
    """
    if not turns:
        return None
    if t1 <= t0:
        t1 = t0 + 0.05
    best, best_ov = None, 0.0
    for a, b, spk in turns:
        ov = min(t1, b) - max(t0, a)
        if ov > best_ov:
            best, best_ov = spk, ov
    if best is not None:
        return best
    mid = 0.5 * (t0 + t1)
    for a, b, spk in turns:          # containment before proximity
        if a <= mid <= b:
            return spk
    return min(turns, key=lambda s: min(abs(mid - s[0]), abs(mid - s[1])))[2]


def smooth_speakers(words: list[dict], min_run: int = 4,
                    min_secs: float = 1.0) -> int:
    """Absorb implausibly short speaker runs into their surroundings.

    Assigning each word to whichever diarization segment overlaps it most is
    correct per word and wrong per sentence: at a segment boundary the labels
    alternate, and the transcript reads "Kalbetojas 10: Manau, / Kalbetojas 8:
    kad programa / Kalbetojas 10: yra". Nobody speaks one word and stops. A run
    that is both short in words and short in seconds, with the SAME speaker on
    both sides, is a boundary artefact rather than a turn, so it is absorbed.

    Both conditions are required. A genuine one-word turn ("Aciu.") is short in
    words but usually stands alone between different speakers, and an
    interjection long enough to last a second is left alone.
    Returns the number of words relabelled.
    """
    runs: list[list[int]] = []
    for i, w in enumerate(words):
        if runs and words[runs[-1][0]].get("spk") == w.get("spk"):
            runs[-1].append(i)
        else:
            runs.append([i])
    changed = 0
    for r in range(1, len(runs) - 1):
        idx = runs[r]
        prev_s = words[runs[r - 1][0]].get("spk")
        next_s = words[runs[r + 1][0]].get("spk")
        if prev_s is None or prev_s != next_s:
            continue
        span = words[idx[-1]]["end"] - words[idx[0]]["start"]
        if len(idx) < min_run and span < min_secs:
            for i in idx:
                words[i]["spk"] = prev_s
            changed += len(idx)
    return changed


# --- assembly ------------------------------------------------------------
def build_turns(words: list[dict], max_gap: float = 1.2,
                max_words: int = 110) -> list[dict]:
    """Group consecutive same-speaker words into turns, split on long pauses
    so a monologue does not become one unreadable block."""
    turns: list[dict] = []
    for w in words:
        spk = w.get("spk")
        if (turns and turns[-1]["spk"] == spk
                and w["start"] - turns[-1]["end"] <= max_gap
                and len(turns[-1]["words"]) < max_words):
            turns[-1]["words"].append(w["w"])
            turns[-1]["end"] = w["end"]
        else:
            turns.append({"spk": spk, "start": w["start"], "end": w["end"],
                          "words": [w["w"]]})
    for t in turns:
        t["text"] = " ".join(t["words"])
    return turns


def build_cues(words: list[dict], max_secs: float = 6.0, max_chars: int = 84,
               line_chars: int = 42, max_gap: float = 1.5) -> list[dict]:
    """Screen-ready subtitle cues — NOT the same thing as speaker turns.

    Turns are the right unit for a transcript someone reads; they are the
    wrong unit for subtitles. Grouping by turn produced a single 28-second cue
    holding a whole paragraph, which no player can display. Cues are therefore
    split independently: at most max_secs, at most max_chars, never across a
    speaker change or a long pause, and preferentially at a sentence end so a
    cue does not break mid-clause.
    """
    cues: list[dict] = []
    cur: list[dict] = []

    def flush():
        if not cur:
            return
        cues.append({"spk": cur[0].get("spk"), "start": cur[0]["start"],
                     "end": cur[-1]["end"],
                     "text": " ".join(w["w"] for w in cur)})
        cur.clear()

    for w in words:
        if cur:
            same = w.get("spk") == cur[0].get("spk")
            length = len(" ".join(x["w"] for x in cur)) + 1 + len(w["w"])
            if (not same or w["start"] - cur[-1]["end"] > max_gap
                    or w["end"] - cur[0]["start"] > max_secs
                    or length > max_chars):
                flush()
        cur.append(w)
        # A sentence has ended and the cue is already substantial: break here
        # rather than carrying two half-sentences into one cue.
        if (w["w"].endswith((".", "!", "?", "…"))
                and len(" ".join(x["w"] for x in cur)) > max_chars * 0.45):
            flush()
    flush()
    for c in cues:
        c["lines"] = _wrap(c["text"], line_chars)
    return cues


def _wrap(text: str, width: int, max_lines: int = 2) -> list[str]:
    """Balance across at most max_lines; a subtitle that overflows is worse
    than one that runs slightly wide, so the last line absorbs the excess."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width and len(lines) < max_lines - 1:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def _ts(t: float, comma: bool = False) -> str:
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", sep)


def _out(base: pathlib.Path, ext: str) -> pathlib.Path:
    """base + '.ext', APPENDING rather than replacing.

    pathlib's with_suffix() replaces the last dotted component, so an input
    named 'rec.16k.wav' produced 'rec.txt' — it treated '.16k' as the
    extension. Dotted filenames are ordinary ('interview.2026-08-15.m4a'), and
    the failure is silent: the file is written, just not where the log says,
    and it can overwrite an unrelated neighbour.
    """
    return base.parent / f"{base.name}.{ext}"


def write_outputs(base: pathlib.Path, turns: list[dict], words: list[dict],
                  names: dict[int, str] | None = None) -> None:
    names = names or {}

    def label(spk):
        return names.get(spk, f"Kalbėtojas {spk + 1}") if spk is not None else ""

    lines = []
    for t in turns:
        who = label(t["spk"])
        stamp = f"[{int(t['start'])//60:02d}:{int(t['start'])%60:02d}]"
        lines.append(f"{stamp} {who}: {t['text']}" if who
                     else f"{stamp} {t['text']}")
    _out(base, "txt").write_text("\n\n".join(lines), encoding="utf-8")

    cues = build_cues(words)
    srt, vtt = [], ["WEBVTT", ""]
    prev_spk = object()
    for i, c in enumerate(cues, 1):
        # Name the speaker only when it CHANGES. Repeating it on every cue
        # eats a third of the screen line for no information.
        who = label(c["spk"]) if c["spk"] != prev_spk else ""
        prev_spk = c["spk"]
        body = "\n".join(c["lines"])
        if who:
            body = f"{who}:\n{body}" if len(c["lines"]) < 2 else f"{who}: {body}"
        srt += [str(i), f"{_ts(c['start'], True)} --> {_ts(c['end'], True)}",
                body, ""]
        vtt += [f"{_ts(c['start'])} --> {_ts(c['end'])}", body, ""]
    _out(base, "srt").write_text("\n".join(srt), encoding="utf-8")
    _out(base, "vtt").write_text("\n".join(vtt), encoding="utf-8")
    _out(base, "json").write_text(json.dumps(
        {"turns": [{k: v for k, v in t.items() if k != "words"} for t in turns],
         "words": words}, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-word-ts", action="store_true",
                    help="skip per-word timestamps. Whisper's word alignment "
                         "materialises the full cross-attention stack and costs "
                         "~6.4 GB of RAM per block regardless of block length, "
                         "which is what kills a free Colab runtime. Without it "
                         "word times are spread evenly inside each block: "
                         "coarser, but the transcript and speaker turns survive")
    ap.add_argument("--speakers", type=int, default=-1,
                    help="known speaker count; -1 = decide by threshold")
    ap.add_argument("--diar-threshold", type=float, default=0.55)
    ap.add_argument("--no-diar", action="store_true")
    ap.add_argument("--no-punct", action="store_true")
    ap.add_argument("--no-gate", action="store_true",
                    help="do not skip silent regions (slower, hallucinates)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--target", type=float, default=25.0)
    ap.add_argument("--verbose", action="store_true",
                    help="per-block timings and word counts")
    ap.add_argument("--quiet", action="store_true",
                    help="no progress line")
    ap.add_argument("--loud", action="store_true",
                    help="keep transformers' warnings")
    args = ap.parse_args()

    if not args.loud:
        quiet_transformers()
    import torch
    from transformers import (WhisperForConditionalGeneration,
                              WhisperProcessor)
    from chunk_longform import transcribe_words

    t_all = time.monotonic()
    pcm = load_audio(args.input)
    dur = len(pcm) / RATE

    dev, dt = pick_device(torch)
    where = (torch.cuda.get_device_name(0) if dev == "cuda"
             else "Apple GPU" if dev == "mps" else "CPU (no GPU found)")
    # Say where this is running BEFORE the slow part. A silent CPU fallback is
    # indistinguishable from a hang, which is exactly how this failed on Colab.
    print(f"audio: {dur/60:.1f} min   device: {dev} — {where}", flush=True)
    if not args.no_word_ts:
        avail = _available_ram_gb()
        if avail is not None and avail < 13.0:
            print(f"  WARNING: {avail:.1f} GB RAM free. Per-word timestamps need "
                  f"about 10 GB peak and WILL be killed here (a free Colab "
                  f"runtime dies around block 4). Rerun with --no-word-ts.",
                  flush=True)
    if dev == "cpu":
        print("  WARNING: CPU is roughly 20x slower than a GPU here; expect "
              f"~{dur/60*3:.0f}-{dur/60*4:.0f} min for this file.", flush=True)
    proc = WhisperProcessor.from_pretrained(args.model, language="lithuanian",
                                            task="transcribe")
    mdl = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=dt).to(dev).eval()

    t0 = time.monotonic()
    words = transcribe_words(pcm, mdl, proc, device=dev, word_ts=not args.no_word_ts,
                             target=args.target,
                             gate_silence=not args.no_gate,
                             verbose=args.verbose, progress=not args.quiet)
    print(f"decode: {len(words)} words in {time.monotonic()-t0:.0f}s "
          f"(RTF {(time.monotonic()-t0)/dur:.2f})", flush=True)
    if not words:
        raise SystemExit("no speech found")

    if not args.no_punct:
        from punct_restore import Punctuator
        print("punctuating …", flush=True)
        t0 = time.monotonic()
        raw = " ".join(w["w"] for w in words)
        txt, st = Punctuator().restore(raw)
        toks = txt.split()
        if len(toks) == len(words):
            for w, tk in zip(words, toks):
                w["w"] = tk
        else:  # contract says this cannot happen; if it does, keep raw words
            print(f"  punctuation SKIPPED: {len(toks)} tokens vs "
                  f"{len(words)} words", flush=True)
        print(f"punctuate: {st['chunks']} chunks, {st['refused']} refused, "
              f"{time.monotonic()-t0:.0f}s", flush=True)

    turns_d: list[tuple[float, float, int]] = []
    if not args.no_diar:
        print("diarizing … (first run downloads two small models)", flush=True)
        t0 = time.monotonic()
        turns_d = diarize(pcm, args.speakers, args.diar_threshold)
        n_spk = len({s for _, _, s in turns_d})
        print(f"diarize: {len(turns_d)} segments, {n_spk} speakers, "
              f"{time.monotonic()-t0:.0f}s", flush=True)
        for w in words:
            w["spk"] = speaker_at(turns_d, w["start"], w["end"])
        moved = smooth_speakers(words)
        if moved:
            print(f"  smoothed {moved} words out of one-off speaker runs",
                  flush=True)
        # Automatic speaker counting is the weakest link and fails in BOTH
        # directions depending on the room: on one press conference it found
        # 9-11 speakers for two people, on a meeting recording it merged a
        # floor question into the chair's answer. Segmentation proposes the
        # boundaries correctly (longest segment 19.8 s on comparable audio);
        # it is clustering that cannot always tell two similar voices on one
        # distant microphone apart. Nothing here can fix that silently, so the
        # least it can do is say when the result looks implausible.
        if args.speakers < 0 and n_spk == 1 and dur > 60:
            print("  NOTE: only one speaker found in "
                  f"{dur/60:.0f} min. If more than one person speaks, rerun "
                  "with --speakers N — a known count beats the threshold.",
                  flush=True)

    turns = build_turns(words)
    # A turn far longer than anyone speaks uninterrupted usually means a
    # speaker change was missed, not that somebody talked for a minute
    # straight. Worth surfacing: the transcript still READS fine, so there is
    # nothing in the output itself to suggest the labels are wrong.
    if not args.no_diar and turns:
        longest = max(t["end"] - t["start"] for t in turns)
        if longest > 60:
            print(f"  NOTE: longest turn is {longest:.0f}s. If a question or "
                  "interjection is buried inside it, the speaker change was "
                  "missed — rerun with --speakers N, or --diar-threshold 0.4 "
                  "to split more eagerly.", flush=True)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else pathlib.Path(args.input).parent
    base = out_dir / pathlib.Path(args.input).stem
    write_outputs(base, turns, words)
    print(f"\n{len(turns)} turns, {len(words)} words, "
          f"{time.monotonic()-t_all:.0f}s total", flush=True)
    print(f"wrote {base}.txt / .srt / .vtt / .json", flush=True)


if __name__ == "__main__":
    main()
