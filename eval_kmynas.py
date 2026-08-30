#!/usr/bin/env python3
"""Compare two Kmynas checkpoints on the same audio, on the axes that matter.

Built to evaluate a mid-training v2 against released v1. Runs both models over
identical pause-aligned blocks and reports, per file:

  * unk rate      — whether the quotation-mark vocab gap is fixed. This is the
                    headline check: `⁇` is the <unk> token, and in v1 100% of
                    them sat at a correct quote position, so the model knew
                    where quotes went but could not encode them.
  * word count    — coverage. A drop can mean the model is dropping speech.
  * word-level diff between checkpoints, so changes can be read rather than
    guessed at.
  * speed / RTF.

    python eval_kmynas.py --a v1.nemo --b v2.nemo audio/x.m4a audio/y.mp3
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unicodedata

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chunk_longform import find_cut_points, speech_regions  # noqa: E402

RATE = 16000
UNK = "⁇"


def load_pcm(path: str) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path,
         "-f", "f32le", "-ac", "1", "-ar", str(RATE), "-"],
        check=True, capture_output=True).stdout
    a = np.frombuffer(raw, dtype=np.float32).copy()
    rms = float(np.sqrt((a ** 2).mean() + 1e-12))
    if rms < 0.02:
        peak = float(np.abs(a).max())
        if peak > 0:
            a = a * min(0.95 / peak, 20.0)
    return a


def blocks_for(pcm: np.ndarray) -> list[tuple[int, int]]:
    regions = speech_regions(pcm)
    b = [(r0 + x, r0 + y) for r0, r1 in regions
         for x, y in find_cut_points(pcm[r0:r1], target=30.0,
                                     min_len=15.0, max_len=35.0)]
    return [(x, y) for x, y in b if y - x >= int(0.2 * RATE)]


def run_model(ckpt: str, files: list[str], device: str, dtype: str) -> dict:
    """{file: {"text":…, "words":[…], "decode_s":…, "speech_s":…}}"""
    import torch
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.restore_from(
        ckpt, map_location=torch.device(device))
    model = model.to(device).eval()
    if device != "cpu" and dtype == "fp16":
        model = model.half()

    out: dict = {}
    for f in files:
        pcm = load_pcm(f)
        blks = blocks_for(pcm)
        speech_s = sum(y - x for x, y in blks) / RATE
        tmp = tempfile.mkdtemp(prefix="ev_")
        paths, starts = [], []
        for i, (x, y) in enumerate(blks):
            import soundfile as sf
            p = f"{tmp}/b{i:04d}.wav"
            sf.write(p, pcm[x:y], RATE)
            paths.append(p)
            starts.append(x / RATE)
        t0 = time.monotonic()
        hyps = model.transcribe(paths, batch_size=1, timestamps=True)
        dt = time.monotonic() - t0
        words = []
        texts = []
        for st, h in zip(starts, hyps):
            texts.append((h.text if hasattr(h, "text") else str(h)).strip())
            ts = getattr(h, "timestamp", None)
            if isinstance(ts, dict) and ts.get("word"):
                for w in ts["word"]:
                    t = (w.get("word") or w.get("char") or "").strip()
                    if t:
                        words.append({"w": t, "start": st + float(w["start"]),
                                      "end": st + float(w["end"])})
        out[f] = {"text": " ".join(texts), "words": words,
                  "decode_s": dt, "speech_s": speech_s}
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return out


def norm(s: str) -> str:
    s = s.lower()
    return "".join(c for c in s if not unicodedata.category(c).startswith("P"))


def report(a: dict, b: dict, files: list[str], name_a: str, name_b: str) -> None:
    for f in files:
        ra, rb = a[f], b[f]
        ta, tb = ra["text"], rb["text"]
        wa, wb = ta.split(), tb.split()
        ua, ub = ta.count(UNK), tb.count(UNK)
        print(f"\n{'='*72}\n{pathlib.Path(f).name}  ({ra['speech_s']/60:.1f} min speech)\n{'='*72}")
        print(f"{'metric':<28}{name_a:>18}{name_b:>18}")
        print(f"{'words':<28}{len(wa):>18}{len(wb):>18}")
        print(f"{'unk (⁇) tokens':<28}{ua:>18}{ub:>18}")
        print(f"{'unk per 1000 words':<28}{1000*ua/max(1,len(wa)):>18.1f}"
              f"{1000*ub/max(1,len(wb)):>18.1f}")
        print(f"{'decode s':<28}{ra['decode_s']:>18.1f}{rb['decode_s']:>18.1f}")
        print(f"{'RTF':<28}{ra['decode_s']/ra['speech_s']:>18.3f}"
              f"{rb['decode_s']/rb['speech_s']:>18.3f}")
        na, nb = [norm(x) for x in wa], [norm(x) for x in wb]
        sm = difflib.SequenceMatcher(None, na, nb, autojunk=False)
        print(f"{'word similarity A vs B':<28}{100*sm.ratio():>17.1f}%")
        subs = [(" ".join(wa[i1:i2]), " ".join(wb[j1:j2]))
                for t, i1, i2, j1, j2 in sm.get_opcodes() if t == "replace"]
        print(f"\nsample changes ({len(subs)} substitution regions), {name_a} -> {name_b}:")
        for x, y in subs[:20]:
            if x and y and len(x) < 45 and len(y) < 45:
                print(f"   {x!r} -> {y!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--a", required=True, help="baseline .nemo")
    ap.add_argument("--b", required=True, help="candidate .nemo")
    ap.add_argument("--name-a", default="v1")
    ap.add_argument("--name-b", default="v2")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from transcribe_file import quiet_transformers
    quiet_transformers()
    import torch
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {dev}  dtype: {args.dtype}", flush=True)

    print(f"\n--- running {args.name_a} ---", flush=True)
    ra = run_model(args.a, args.files, dev, args.dtype)
    print(f"--- running {args.name_b} ---", flush=True)
    rb = run_model(args.b, args.files, dev, args.dtype)

    report(ra, rb, args.files, args.name_a, args.name_b)
    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(
            {args.name_a: {k: {kk: vv for kk, vv in v.items() if kk != "words"}
                           for k, v in ra.items()},
             args.name_b: {k: {kk: vv for kk, vv in v.items() if kk != "words"}
                           for k, v in rb.items()}},
            ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
