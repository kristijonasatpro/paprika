#!/usr/bin/env python3
"""Same job as transcribe_file.py, but with the Kmynas NeMo checkpoint.

Long Lithuanian audio -> punctuated, speaker-labelled transcript. Local, no
API key, nothing uploaded. Writes .txt / .srt / .vtt / .json beside the input.

Why a second script rather than a flag: the two models need different
pipelines, not different parameters.

  paprika (transcribe_file.py)   Whisper fine-tune, ~800M. Punctuation comes
                                 from a SEPARATE ONNX tagger run after decoding.
                                 Per-word timestamps mean cross-attention
                                 alignment, ~6.4 GB per block.
  kmynas  (this script)          Parakeet TDT fine-tune, 600M. Punctuation and
                                 casing come from the model itself, so there is
                                 no tagger stage and no chance of the tagger
                                 disagreeing with the words. Word timestamps
                                 are near-free. Roughly 7x faster on the same
                                 machine (RTF ~0.04 vs ~0.30).

Everything AFTER decoding is shared: diarization, speaker smoothing, turn and
cue building and the writers are imported from transcribe_file, so the two
pipelines cannot drift apart in how they lay a transcript out.

Which to use: paprika still reads better on clean, prepared speech. kmynas is
much faster, a third smaller, and punctuates natively. Try both on your own
audio -- they fail differently, and which failure you can live with is the
whole decision.

Usage:
    python transcribe_kmynas.py INPUT.{m4a,mov,wav,mp3} [--speakers N]
                                [--model PATH.nemo] [--no-diar] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import time

# Shared with the paprika pipeline on purpose -- see module docstring.
from transcribe_file import (RATE, build_turns, diarize, load_audio,
                             smooth_speakers, speaker_at, write_outputs)

HF_REPO = "kristijonas/kmynas-parakeet-lt-v3"
DEFAULT_MODEL = os.environ.get("KMYNAS_MODEL", "kmynas-v3-final.nemo")


def quiet_nemo() -> None:
    """NeMo prints its entire config on restore; keep the console readable."""
    import logging
    for name in ("nemo_logger", "nemo_logging", "nemo"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        from nemo.utils import logging as nl
        nl.setLevel(logging.ERROR)
    except Exception:
        pass


def load_model(spec: str):
    """Load a local .nemo, or fetch the published checkpoint once.

    NOTE the published repo is currently PRIVATE, so the download path needs
    `huggingface-cli login` with access to it. Passing a local file is the
    normal route meanwhile.
    """
    import nemo.collections.asr as nemo_asr
    p = pathlib.Path(spec).expanduser()
    if not p.exists():
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise SystemExit(f"model not found at {p}; pass --model PATH.nemo")
        print(f"fetching {HF_REPO} (once, ~2.5 GB) …", flush=True)
        p = pathlib.Path(hf_hub_download(HF_REPO, "kmynas-parakeet-lt-v3.nemo"))
    m = nemo_asr.models.ASRModel.restore_from(str(p), map_location="cpu").cpu().eval()

    # fp16 on Apple's MPS backend leaked blank/duration-slot ids past the
    # vocabulary and crashed SentencePiece. This runs fp32 where that cannot
    # happen; the guard costs nothing and turns any future leak into a skipped
    # token instead of a crash.
    dec, orig, vocab = m.decoding, m.decoding.decode_ids_to_tokens, m.tokenizer.vocab_size
    dec.decode_ids_to_tokens = lambda ids: orig([i for i in ids if i < vocab])

    # Word timestamps need BOTH the decoding config and return_hypotheses on
    # transcribe(); the config alone silently yields None. Confidence is
    # deliberately not requested: it is near-zero for every word inside a
    # padded region whether that word is right or wrong, and NeMo's aggregator
    # crashes on this model's quote-glue output.
    try:
        from copy import deepcopy
        from omegaconf import open_dict
        cfg = deepcopy(m.cfg.decoding)
        with open_dict(cfg):
            cfg.strategy = "greedy_batch"
            cfg.rnnt_timestamp_type = "word"
        m.change_decoding_strategy(cfg, verbose=False)
    except Exception as e:
        print(f"  word timestamps unavailable ({type(e).__name__}); times will "
              f"be spread evenly inside each block", flush=True)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="path to a .nemo checkpoint (or $KMYNAS_MODEL)")
    ap.add_argument("--speakers", type=int, default=-1,
                    help="known speaker count; -1 = decide by threshold. Pass "
                         "it when you know: unaided the clusterer over-splits")
    ap.add_argument("--diar-threshold", type=float, default=0.55)
    ap.add_argument("--no-diar", action="store_true")
    ap.add_argument("--threads", type=int,
                    default=max((os.cpu_count() or 4) // 2, 2))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    quiet_nemo()
    import torch
    torch.set_num_threads(args.threads)
    from kmynas_decode import transcribe_words

    t_all = time.monotonic()
    pcm = load_audio(args.input)
    dur = len(pcm) / RATE
    print(f"audio: {dur/60:.1f} min   {args.threads} threads   CPU", flush=True)

    t0 = time.monotonic()
    model = load_model(args.model)
    print(f"model loaded in {time.monotonic()-t0:.0f}s", flush=True)

    t0 = time.monotonic()
    words = transcribe_words(pcm, model, batch_size=args.batch_size,
                             progress=not args.quiet, verbose=args.verbose)
    el = time.monotonic() - t0
    print(f"decode: {len(words)} words in {el:.0f}s (RTF {el/dur:.2f})", flush=True)
    if not words:
        raise SystemExit("no speech found")

    if not args.no_diar:
        print("diarizing … (first run downloads two small models)", flush=True)
        t0 = time.monotonic()
        turns_d = diarize(pcm, args.speakers, args.diar_threshold)
        print(f"diarize: {len(turns_d)} segments, "
              f"{len({s for _, _, s in turns_d})} speakers, "
              f"{time.monotonic()-t0:.0f}s", flush=True)
        for w in words:
            w["spk"] = speaker_at(turns_d, w["start"], w["end"])
        smooth_speakers(words)

    out_dir = pathlib.Path(args.out_dir or pathlib.Path(args.input).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / pathlib.Path(args.input).stem
    write_outputs(base, build_turns(words), words)
    print(f"\ndone in {time.monotonic()-t_all:.0f}s -> "
          f"{base}.txt/.srt/.vtt/.json", flush=True)


if __name__ == "__main__":
    main()
