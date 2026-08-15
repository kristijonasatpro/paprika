#!/usr/bin/env python3
"""Phase D Track 2: punctuation + casing restoration for LT ASR output.

Runs entirely on-prem (ONNX tagger, cached locally) — the whole point of the
product is that audio and transcript never leave the machine, so an LLM API
is disqualified on architecture before accuracy even enters.

Chosen by measurement, not preference (T356 head-to-head on the 11 gold
speeches): this tagger scored comma 84.7 / period 88.5 / casing 91.7 F1 with
0/11 word mismatches; the codex-LLM arm scored lower on every class AND
corrupted words in 5 of 11 files.

WORD PRESERVATION IS A HARD CONTRACT. A punctuator that edits words is not a
punctuator, it is an unreviewed rewrite of evidence. If the restored word
sequence differs from the input in anything but case, we return the input
untouched and count the refusal.

Usage:
    from punct_restore import Punctuator
    p = Punctuator()
    text, stats = p.restore("jis sakė kad viskas gerai")
"""
from __future__ import annotations

import os
import re
import unicodedata as ud

MODEL_ID = "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
PUNCT = ",.;:!?…"
STRIP = PUNCT + "\"'()–—-«»„“"
# Long transcripts must be split (xlm-roberta caps at 512 subword tokens), but
# a naive split is not free: cutting mid-sentence cost period F1 88.6 -> 82.2
# on the gold set, because the boundary word loses its right context. So each
# chunk carries OVERLAP words of lead-in whose output is discarded — they exist
# only to give the first kept word real context. Measured 2026-08-10.
CHUNK_WORDS = int(os.environ.get("CHUNK_WORDS", "200"))
OVERLAP = int(os.environ.get("PUNCT_OVERLAP", "40"))


def _words(text: str) -> list[str]:
    """Case- and punctuation-free word sequence — the identity that must
    survive restoration."""
    out = []
    for tok in text.split():
        core = tok.strip(STRIP)
        if core:
            out.append(ud.normalize("NFC", core.lower()))
    return out


class Punctuator:
    def __init__(self, model_id: str = MODEL_ID):
        from punctuators.models import PunctCapSegModelONNX
        self.m = PunctCapSegModelONNX.from_pretrained(model_id)

    def _infer(self, text: str) -> str:
        segs = self.m.infer([text], apply_sbd=True)[0]
        return " ".join(s.strip() for s in segs if s.strip())

    def restore(self, text: str) -> tuple[str, dict]:
        """Return (restored_text, stats). stats.refused is the count of chunks
        where the guard rejected the tagger output and kept the input."""
        src = " ".join(text.split())
        if not src:
            return "", {"chunks": 0, "refused": 0, "words": 0}
        toks = src.split()
        step = max(1, CHUNK_WORDS - 2 * OVERLAP)
        done, refused, n_chunks, i, first = [], 0, 0, 0, True
        while i < len(toks):
            chunk = toks[i:i + CHUNK_WORDS]
            is_last = i + CHUNK_WORDS >= len(toks)
            n_chunks += 1
            # Keep only the MIDDLE of each chunk. The lead-in was emitted by
            # the previous chunk; the trailing margin is dropped because the
            # tagger always terminates its last word with a period — keeping
            # chunk edges cost period F1 82.2 -> 79.8 before this.
            lo = 0 if first else min(OVERLAP, len(chunk))
            hi = len(chunk) if is_last else max(lo, len(chunk) - OVERLAP)
            joined = " ".join(chunk)
            try:
                got = self._infer(joined)
            except Exception:
                got = None
            # guard: the tagger may change case and add punctuation, nothing
            # else. Any word-level edit and we keep the input verbatim.
            if got is not None and _words(got) == _words(joined):
                done.extend(got.split()[lo:hi])
            else:
                done.extend(chunk[lo:hi])
                refused += 1
            if is_last:
                break
            i += step
            first = False
        out = " ".join(done)
        return out, {"chunks": n_chunks, "refused": refused,
                     "words": len(_words(src))}


def sentence_case_fallback(text: str) -> str:
    """Last-resort formatting when the tagger refuses everything: capitalize
    after terminal punctuation only. Never invents punctuation."""
    out = re.sub(r"(^|[.!?]\s+)([a-ząčęėįšųūž])",
                 lambda m: m.group(1) + m.group(2).upper(), text)
    return out


if __name__ == "__main__":
    import sys
    p = Punctuator()
    raw = sys.stdin.read() if not sys.argv[1:] else " ".join(sys.argv[1:])
    txt, st = p.restore(raw)
    print(txt)
    print(f"[chunks={st['chunks']} refused={st['refused']} words={st['words']}]",
          file=sys.stderr)
