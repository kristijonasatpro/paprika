#!/usr/bin/env python3
"""Long audio -> timed words, with a Kmynas NeMo checkpoint. CPU-friendly.

The model is trained on 0.5-15 s utterances, so long audio has to be cut. Three
things here were each paid for with a measured failure; none is decoration.

1. CUTS LAND IN PAUSES. A fixed grid slices words in half. Cuts are placed at
   the quietest point inside a search window instead.

2. SILENCE IS NEVER SENT TO THE MODEL. Given digital silence the decoder emits
   a phantom word: measured 2026-08-29 over 67 min of Lithuanian, 69 of 169
   chunks decoded to exactly "Mums", every one at -240 dBFS (all-zero samples),
   while the quietest REAL chunk sat at -38.1 dBFS. The gate also makes
   silence-heavy recordings ~2.5x faster, because nothing is decoded there.

3. CHUNKS OVERLAP, AND THE SEAM IS DEDUPLICATED. Each chunk is padded on both
   sides so the boundary word has context, which means a 2*OVERLAP_S window is
   transcribed twice and one copy must go. The deletion is CAPPED at what was
   actually heard twice: without that cap a single stray anchor once erased 12
   real words -- an entire meeting agenda item -- with nothing visible to the
   reader. A visible duplicate is recoverable; a silent deletion is not.

Word timestamps come from NeMo directly and cost almost nothing here, which is
why per-word timing is always on. (Getting them out requires
`return_hypotheses=True, timestamps=True` on transcribe() -- setting the
decoding config alone silently returns None.)
"""
from __future__ import annotations

import difflib
import os
import re
import tempfile
import pathlib

RATE = 16_000
MIN_S = float(os.environ.get("MIN_S", "10.0"))
MAX_S = float(os.environ.get("MAX_S", "20.0"))
OVERLAP_S = float(os.environ.get("OVERLAP_S", "1.3"))
# Every measured phantom was exactly digital zero, so this sits far below the
# quietest real speech seen (-38.1 dBFS) rather than hugging that boundary.
# It is an ENERGY gate, not a voice detector: steady noise above it survives.
SILENCE_DBFS = float(os.environ.get("SILENCE_DBFS", "-80"))

_NONWORD = re.compile(r"[^\w]", re.UNICODE)
_FINAL_VOWEL = re.compile(r"[aąeęėiįyouųū]$")


def fold(w: str) -> str:
    """Normalise a word for seam comparison.

    Two hearings of the same audio differ mostly by Lithuanian final-vowel
    inflection ("apkalbėta"/"apkalbėtą"), so one final vowel is stemmed. The
    ORIGINAL word is always what gets emitted.
    """
    w = _NONWORD.sub("", w.lower())
    return _FINAL_VOWEL.sub("", w) if len(w) > 3 else w


def words_match(x: str, y: str) -> bool:
    """Same word heard twice? Fuzzy matching is deliberately hard to trigger.

    A false match here authorises DELETING text, so the bar sits above every
    false pair observed and below every true one. An earlier, looser version
    matched genuinely different Lithuanian words -- seimas/šeimas (parliament
    vs family), klausimas/klausimynas (question vs questionnaire).
    """
    if x == y:
        return len(x) > 1
    lo, hi = (x, y) if len(x) <= len(y) else (y, x)
    if len(lo) < 5:
        return False
    if hi.startswith(lo):                 # ilgamečio/ilgamečius
        return True
    if len(hi) - len(lo) <= 1 and len(hi) >= 6:
        return difflib.SequenceMatcher(a=x, b=y, autojunk=False).ratio() >= 0.88
    return False


def _choose_block(A, B, max_cut):
    """Longest run of matching words the seam is ALLOWED to delete.

    A seam deletes from BOTH sides: len(A)-a words off A's tail and b words off
    B's head. The cap is on the TOTAL -- bounding each side separately still
    permits twice the budget. Searched explicitly rather than with
    difflib.get_matching_blocks(), which returns one decomposition and can
    commit to a spurious pairing whose implied cut is then rejected, throwing
    away a good anchor that was present. A and B are capped at W words, so this
    is bounded work.

    A block resting entirely on fuzzy matches may not authorise a deletion.
    """
    best = None
    for a in range(len(A)):
        cut_a = len(A) - a
        if cut_a > max_cut:
            continue
        for b in range(min(len(B), max_cut - cut_a + 1)):
            n = exact = 0
            while (a + n < len(A) and b + n < len(B)
                   and words_match(A[a + n], B[b + n])):
                exact += A[a + n] == B[b + n]
                n += 1
            if not n or not exact:
                continue
            if (best is None or n > best[2]
                    or (n == best[2] and (len(A) - a) + b
                        < (len(A) - best[0]) + best[1])):
                best = (a, b, n)
    return best


def merge_word_seams(chunks, overlap_s=OVERLAP_S):
    """Drop the duplicated span at each seam, on WORDS rather than on text.

    chunks: per chunk, a list of word dicts with "w", "start", "end".
    Only what was heard twice may be deleted: at the fastest sustained rate
    measured (2.86 words/s) a 2*overlap window holds about 7 words, and 4.0 w/s
    leaves room for local bursts.
    """
    max_cut = max(4, int(2 * overlap_s * 4.0))
    W = 12
    out = [list(chunks[0])] if chunks else []
    for cur in chunks[1:]:
        prev = out[-1]
        base = max(0, len(prev) - W)
        A = [fold(w["w"]) for w in prev[base:]]
        B = [fold(w["w"]) for w in cur[:W]]
        blk = _choose_block(A, B, max_cut) if A and B else None
        if blk and (blk[2] >= 2 or (blk[2] == 1 and len(A) > blk[0]
                                    and len(A[blk[0]]) >= 5)):
            out[-1] = prev[:base + blk[0]]
            out.append(list(cur[blk[1]:]))
        else:
            out.append(list(cur))
    return out


def split_on_pauses(audio, sr=RATE):
    """Cut at the quietest point in [MIN_S, MAX_S] so boundaries fall in pauses."""
    import numpy as np
    hop = int(0.02 * sr)
    frames = len(audio) // hop
    energy = np.sqrt(np.array([np.mean(audio[i * hop:(i + 1) * hop] ** 2) + 1e-12
                               for i in range(frames)]))
    cuts, start = [], 0
    while start < len(audio):
        lo = start + int(MIN_S * sr)
        hi = min(start + int(MAX_S * sr), len(audio))
        if hi >= len(audio) - int(0.5 * sr):
            cuts.append((start, len(audio)))
            break
        f_lo = lo // hop
        f_hi = max(hi // hop, f_lo + 1)
        end = (f_lo + int(np.argmin(energy[f_lo:f_hi]))) * hop
        cuts.append((start, end))
        start = end
    return cuts


def fix_spacing(text: str) -> str:
    """Restore the space the model drops before an opening quote or dash.

    The tokenizer has no word-initial `▁„` piece, so at generation the model
    must emit a standalone `▁` and sometimes does not, producing `knyga„Kitas`.
    Lithuanian typography puts a space before an opening `„` without exception,
    so this is deterministic rather than a guess.
    """
    text = re.sub(r"(?<=[^\s])„", " „", text)
    text = re.sub(r"(?<=[^\s])–(?=[^\s])", " – ", text)
    return re.sub(r"\s+", " ", text).strip()


def transcribe_words(pcm, model, batch_size=4, progress=True, verbose=False):
    """Decode long audio to timed words. Returns [{"w","start","end"}, ...]."""
    import numpy as np
    import soundfile as sf

    cuts = split_on_pauses(pcm)
    pad = int(OVERLAP_S * RATE)
    work = pathlib.Path(tempfile.mkdtemp())
    paths, spans = [], []
    for i, (a, b) in enumerate(cuts):
        a2 = max(0, a - (pad if i else 0))
        b2 = min(len(pcm), b + (pad if i < len(cuts) - 1 else 0))
        p = work / f"chunk{i:05d}.wav"
        sf.write(p, pcm[a2:b2], RATE)
        paths.append(str(p))
        spans.append(a2 / RATE)

    keep, gated = [], []
    for i, p in enumerate(paths):
        seg, _ = sf.read(p, dtype="float32")
        db = (20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)
              if len(seg) else -240.0)
        (keep if db > SILENCE_DBFS else gated).append(i)
    if gated and progress:
        secs = sum((cuts[i][1] - cuts[i][0]) / RATE for i in gated)
        print(f"  silence gate: {len(gated)} of {len(paths)} blocks "
              f"({secs / 60:.1f} min) below {SILENCE_DBFS:.0f} dBFS, skipped",
              flush=True)
    if not keep:
        return []

    hyps = model.transcribe([paths[i] for i in keep], batch_size=batch_size,
                            return_hypotheses=True, timestamps=True,
                            verbose=False)
    assert len(hyps) == len(keep), f"{len(hyps)} results for {len(keep)} blocks"

    chunks = [[] for _ in paths]
    for idx, h in zip(keep, hyps):
        ts = (h.timestamp or {}).get("word") if isinstance(
            getattr(h, "timestamp", None), dict) else None
        if not ts:                       # no word times: fall back to block text
            txt = fix_spacing(str(getattr(h, "text", h)))
            words = txt.split()
            a, b = cuts[idx][0] / RATE, cuts[idx][1] / RATE
            step = (b - a) / max(len(words), 1)
            chunks[idx] = [{"w": w, "start": a + k * step,
                            "end": a + (k + 1) * step}
                           for k, w in enumerate(words)]
            continue
        base = spans[idx]
        chunks[idx] = [{"w": fix_spacing(w["word"]),
                        "start": base + w["start"], "end": base + w["end"]}
                       for w in ts if w.get("word", "").strip()]
        if verbose:
            print(f"    block {idx}: {len(chunks[idx])} words", flush=True)

    merged = merge_word_seams(chunks) if OVERLAP_S > 0 else chunks
    return [w for ws in merged for w in ws]
