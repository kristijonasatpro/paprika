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
                    frame_ms: int = 50, pause_weight: float = 0.0
                    ) -> list[tuple[int, int]]:
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
            if pause_weight > 0:
                # Score contiguous QUIET RUNS rather than single frames, on
                # the theory that a long pause marks a sentence boundary while
                # a short one is a breath.
                #
                # MEASURED 2026-08-30 AND REFUTED. At pause_weight 0.6 the
                # spurious-capital rate at seams went UP, 12% -> 21% over 76
                # seams on three recordings, and reading the 317 changed spans
                # showed real losses: words collapsed or truncated
                # mid-token, e.g. a five-syllable word losing its ending. Two likely reasons: the longest pause near target
                # is usually a speaker hesitating MID-sentence, which is the
                # worst place to cut; and cutting at a pause centre starts each
                # block with ~0.5 s of silence, which is what makes this model
                # hallucinate (a spurious `cha` duly appeared). Default 0 = off.
                thr = floor + 0.20 * span
                quiet_fr = w <= thr
                runs, i = [], 0
                while i < len(quiet_fr):
                    if quiet_fr[i]:
                        j = i
                        while j < len(quiet_fr) and quiet_fr[j]:
                            j += 1
                        runs.append((i, j))
                        i = j
                    else:
                        i += 1
                if runs:
                    longest = max(j - i for i, j in runs)
                    best_run = min(
                        runs,
                        key=lambda r: (0.35 * abs((r[0] + r[1]) / 2 - f_tgt) / max(1, len(w))
                                       - pause_weight * (r[1] - r[0]) / longest))
                    best = (best_run[0] + best_run[1]) // 2
                else:
                    best = int(np.argmin(quiet + 0.35 * dist))
            else:
                best = int(np.argmin(quiet + 0.35 * dist))
            cut = (f_lo + best) * win
        segs.append((start, cut))
        start = cut
    return segs


def speech_regions(pcm: np.ndarray, sr: int = RATE, rms_thr: float | None = None,
                   frame_ms: int = 100, pad: float = 0.3,
                   min_gap: float = 1.5, level_frac: float = 0.0,
                   max_cut: float = 0.4) -> list[tuple[int, int]]:
    """Sample ranges that actually contain sound, gaps >= min_gap removed.

    Whisper narrates silence. One recording opens with 6.2 minutes of DIGITAL
    silence (rms exactly 0.0) and the offline transcript opens with a fluent,
    confident Lithuanian sentence describing nothing at all. Decoding silence
    cannot produce a right answer, so the cheapest correct move is to not
    decode it. An RMS gate costs microseconds and removes the failure mode
    instead of filtering it later.

    `level_frac` > 0 derives the threshold from the recording (that fraction of
    its 75th-percentile frame) instead of using the 1e-4 constant. It defaults
    to 0 — OFF — and should stay off on conversational audio.

    The constant is genuinely near-useless: measured 2026-08-30 it sits 5-8x
    BELOW the noise floor of an ordinary room, so it never fires and a 35-min
    conversational recording returns ONE region. But raising it does not work either.
    On that recording room tone runs p5=0.00055 / p10=0.00081, while real
    quiet words measure 0.00053-0.00081 —
    speech and silence OVERLAP in energy, so no threshold separates them. At
    level_frac=0.02 the gate deleted 18 spans of real speech, among them the
    a required pronoun mid-clause and a pair of spoken digits. Separating these needs spectral features, i.e. a real VAD
    (MarbleNet or Silero), not an RMS comparison. Until then the downstream
    confidence filter in the kmynas repo handles what silence produces.

    Regions are padded by `pad` so a consonant onset just under threshold is
    not clipped, and gaps shorter than min_gap are kept as real speech pauses.
    """
    win = int(frame_ms / 1000 * sr)
    n_fr = len(pcm) // win
    if n_fr == 0:
        return [(0, len(pcm))]
    rms = np.sqrt((pcm[:n_fr * win].reshape(n_fr, win).astype(np.float32) ** 2)
                  .mean(axis=1) + 1e-12)
    if rms_thr is None:
        rms_thr = 1e-4 if level_frac <= 0 else max(
            1e-4, level_frac * float(np.percentile(rms, 75)))
        if (rms > rms_thr).mean() < 1.0 - max_cut:
            rms_thr = 1e-4          # would cut too much; distrust the estimate
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
                     verbose: bool = False, progress: bool = False,
                     word_ts: bool = True) -> list[dict]:
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
              + (f" ({skipped/60:.1f} min skipped as silence)" if skipped > 5 else ""),
              flush=True)
        # Name the skipped ranges, not just the total. "6.3 min skipped" gives
        # no way to tell correct silence-dropping from a gate that ate real
        # speech — and the transcript reads perfectly either way, so nothing
        # else will reveal it. With ranges printed, a user who knows their
        # recording can spot a wrong one immediately.
        gaps, prev = [], 0
        for x, y in blocks:
            if x - prev > 10 * sr:
                gaps.append((prev / sr, x / sr))
            prev = y
        if (len(pcm) - prev) > 10 * sr:
            gaps.append((prev / sr, len(pcm) / sr))
        for g0, g1 in gaps[:6]:
            print(f"    skipped {int(g0)//60:02d}:{int(g0)%60:02d}"
                  f"–{int(g1)//60:02d}:{int(g1)%60:02d}  ({g1-g0:.0f}s)",
                  flush=True)
        if gaps:
            print("    (if speech is in there, rerun with --no-gate)",
                  flush=True)
    t_start = _time.monotonic()
    words: list[dict] = []
    empty: list[tuple[int, int]] = []
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
                    return_timestamps=True, return_token_timestamps=word_ts,
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
            if not got:
                empty.append((x, y))
            words.extend(got)
            if verbose:
                print(f"  {t0:7.1f}s +{(b-a)/sr:5.1f}s  {len(got):4}w", flush=True)
            elif progress:
                el = _time.monotonic() - t_start
                eta = el / (i + 1) * (len(blocks) - i - 1)
                text, show = _progress_line(i, len(blocks), len(words), el, eta,
                                            _interactive())
                if show:
                    sys.stdout.write(text)
                    sys.stdout.flush()
    # only a TTY needs the in-place line wiped; a notebook printed real lines
    if progress and not verbose and blocks and _interactive():
        sys.stdout.write("\r" + " " * 70 + "\r")
        sys.stdout.flush()
    if progress and empty:
        # A block the model returned nothing for is the OTHER way audio goes
        # missing silently: the gate kept it, so it never appears as skipped,
        # and the transcript simply has no words there. Usually it really is
        # non-speech (applause, music, room noise), but if the user knows
        # somebody was talking, this is where to look.
        tot = sum(y - x for x, y in empty) / sr
        print(f"  {len(empty)} block(s), {tot:.0f}s, produced no words:",
              flush=True)
        for x, y in empty[:6]:
            print(f"    {int(x/sr)//60:02d}:{int(x/sr)%60:02d}"
                  f"–{int(y/sr)//60:02d}:{int(y/sr)%60:02d}", flush=True)
    return words


def _interactive():
    """True only for a real terminal. Notebooks and pipes get whole lines."""
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


# Progress rendering. A terminal gets an in-place \r line that ticks 1..N.
# A notebook does NOT: Colab and Jupyter buffer carriage returns, so a \r line
# is drawn once and then never appears to move, which reads as a hung job. That
# cost a real user a cancelled run on a healthy 7-minute file (reported
# 2026-08-21). When stdout is not a TTY, emit whole lines instead, thinned out
# so a long recording does not scroll away the rest of the output.
def _progress_line(i, total, n_words, elapsed, eta, interactive):
    body = (f"  block {i+1}/{total}  {n_words} words  "
            f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left")
    if interactive:
        return "\r" + body + "   ", True
    # first, last, and roughly every 10% in between
    step = max(1, total // 10)
    show = (i == 0 or i + 1 == total or (i + 1) % step == 0)
    return (body + "\n", True) if show else ("", False)


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
