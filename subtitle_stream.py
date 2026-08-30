#!/usr/bin/env python3
"""Real-time Lithuanian subtitles: stable text on screen with bounded latency.

The problem. Whisper is an offline model. Feeding it a growing buffer and
re-printing whatever it says produces subtitles that FLICKER — each new decode
rewrites words the viewer already read, because a longer window changes the
model's mind about earlier words. Flicker is worse than latency: a caption that
rewrites itself is unreadable.

The fix is LocalAgreement-2 (Machacek et al.). Decode the buffer repeatedly;
a word is COMMITTED only once two consecutive decodes agree on it. Committed
text never changes, so the top line is stable. The disagreeing tail is shown
separately as tentative — the viewer sees it immediately, but knows it may
still move. That buys perceived latency near the decode time while the stable
line stays honest.

Measured on this M4 (fp16, MPS, v3 turbo-lineage fine-tune):

    window   decode      window   decode
      2 s     0.88 s      10 s     1.11 s
      5 s     1.10 s      15 s     1.50 s

Encoder cost is FIXED (~1.0 s) because Whisper pads every input to 30 s of mel
regardless of how much audio it holds; only token generation scales. So a short
window buys almost nothing, and the real budget question is just "is decode
faster than the step interval". At STEP=1.0 s it is not (0.88 s decode leaves
no headroom once punctuation and feature extraction are counted); at 1.25 s it
is, with margin.

SILENCE IS GATED BEFORE DECODE, not after. Pointed at a six-minute silent
lead-in, the model emits a confident, fluent Lithuanian sentence describing
nothing at all. An RMS gate costs microseconds and removes the entire failure
mode, and it also saves the decode, so it is strictly better than filtering
hallucinations afterwards.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

RATE = 16000

# --- tuning ---------------------------------------------------------------
# STEP: audio added between decodes. Must exceed decode time or the stream
#   falls behind without bound. 1.25 s measured safe; see docstring.
# WINDOW_MAX: hard cap on buffer length. Bounds both memory and decode time.
#   Whisper truncates past 30 s anyway, so exceeding it silently loses audio.
# VAD_RMS: below this a frame is silence. 1e-4 separates one recording's digital
#   silence (0.0) from its quietest real speech (0.05) by a wide margin; it is
#   deliberately NOT tuned tight, because a false "speech" costs one wasted
#   decode while a false "silence" drops a word.
STEP = float(os.environ.get("SUB_STEP", "1.5"))
WINDOW_MAX = float(os.environ.get("SUB_WINDOW_MAX", "22.0"))
VAD_RMS = float(os.environ.get("SUB_VAD_RMS", "1e-4"))
SILENCE_FLUSH = float(os.environ.get("SUB_SILENCE_FLUSH", "0.8"))
CAPTION_CHARS = int(os.environ.get("SUB_CAPTION_CHARS", "42"))
CAPTION_LINES = int(os.environ.get("SUB_CAPTION_LINES", "2"))
PUNCT_CTX = int(os.environ.get("SUB_PUNCT_CTX", "40"))
PUNCT_LIVE = int(os.environ.get("SUB_PUNCT_LIVE", "12"))
PUNCT_MIN_NEW = int(os.environ.get("SUB_PUNCT_MIN_NEW", "6"))


def _norm(w: str) -> str:
    return w.strip().lower().strip(".,!?;:…\"'()–—-„“«»")


@dataclass
class Word:
    text: str
    start: float          # absolute seconds from stream start
    end: float


@dataclass
class StreamState:
    committed: list[Word] = field(default_factory=list)
    tentative: list[Word] = field(default_factory=list)
    decodes: int = 0
    skipped_silent: int = 0
    decode_secs: list[float] = field(default_factory=list)
    commit_lag: list[float] = field(default_factory=list)


class SubtitleStream:
    """Feed audio with `push`; read `state.committed` (stable) and
    `state.tentative` (may still change)."""

    def __init__(self, model_dir: str, device: str | None = None,
                 dtype=None, language: str = "lithuanian",
                 punctuate: bool = True):
        import torch
        from transformers import (WhisperForConditionalGeneration,
                                  WhisperProcessor)
        self.torch = torch
        # CUDA first, then Apple MPS, then CPU. Checking only MPS silently sent
        # every Linux and Colab GPU user to the CPU path — see
        # transcribe_file.pick_device.
        # fp16 measured 25% faster than fp32 on MPS with IDENTICAL word counts
        # across all eight window sizes, so the speed is not bought with
        # accuracy. CPU keeps fp32: fp16 on CPU is emulated and slower.
        from transcribe_file import pick_device
        _dev, _dt = pick_device(torch)
        self.device = device or _dev
        self.dtype = dtype or _dt
        self.language = language
        self.proc = WhisperProcessor.from_pretrained(
            model_dir, language=language, task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_dir, dtype=self.dtype).to(self.device).eval()
        self.buf = np.zeros(0, dtype=np.float32)
        self.buf_t0 = 0.0
        self.clock = 0.0
        self.state = StreamState()
        self._pending = np.zeros(0, dtype=np.float32)
        self._silence_run = 0.0
        self._p_words: list[str] = []   # frozen, punctuated
        self._live: list[str] = []      # punctuated but still revisable
        self._flush = False             # set by finish(): punctuate the rest
        self.punct = None
        if punctuate:
            from punct_restore import Punctuator
            self.punct = Punctuator()

    # -- audio in ----------------------------------------------------------
    def push(self, pcm: np.ndarray) -> bool:
        """Add audio. Returns True if this triggered a decode."""
        self._pending = np.concatenate([self._pending, pcm.astype(np.float32)])
        if len(self._pending) < int(STEP * RATE):
            return False
        chunk, self._pending = self._pending, np.zeros(0, dtype=np.float32)
        self.clock += len(chunk) / RATE

        rms = float(np.sqrt((chunk ** 2).mean() + 1e-12))
        if rms < VAD_RMS:
            self._silence_run += len(chunk) / RATE
            # Silence ENDS an utterance: promote the tentative tail rather than
            # leaving it on screen forever waiting for an agreement that will
            # never come, because no further audio will change it.
            if self._silence_run >= SILENCE_FLUSH and self.state.tentative:
                self._commit(self.state.tentative, forced=True)
                self.state.tentative = []
            if not self.state.tentative:
                # Drop silent audio instead of buffering it. Keeps the window
                # full of speech and stops the model narrating the silence.
                self.buf = np.zeros(0, dtype=np.float32)
                self.buf_t0 = self.clock
                self.state.skipped_silent += 1
                return False
        else:
            self._silence_run = 0.0

        self.buf = np.concatenate([self.buf, chunk])
        if len(self.buf) > int(WINDOW_MAX * RATE):
            # Force-commit the tentative tail and hard-trim. Without this a
            # speaker who never pauses grows the buffer past Whisper's 30 s
            # limit, where audio is silently truncated.
            if self.state.tentative:
                self._commit(self.state.tentative, forced=True)
                self.state.tentative = []
            keep = int(WINDOW_MAX * RATE) // 2
            self.buf_t0 += (len(self.buf) - keep) / RATE
            self.buf = self.buf[-keep:]
        self._decode()
        return True

    # -- the agreement policy ---------------------------------------------
    def _decode(self) -> None:
        if len(self.buf) < int(1.0 * RATE):
            return
        torch = self.torch
        t0 = time.monotonic()
        feats = self.proc(self.buf, sampling_rate=RATE, return_tensors="pt",
                          return_attention_mask=True)
        with torch.no_grad():
            out = self.model.generate(
                feats.input_features.to(self.device, self.dtype),
                attention_mask=feats.attention_mask.to(self.device),
                language=self.language, task="transcribe",
                # return_timestamps="word" is a PIPELINE argument; generate()
                # wants the bool plus return_token_timestamps, which yields
                # per-token times via the model's 6 alignment heads.
                return_timestamps=True, return_token_timestamps=True,
                return_dict_in_generate=True,
                # Cap generation to what the buffer could plausibly contain.
                # Without it a degenerate repeat runs to Whisper's 448-token
                # limit: one decode in an early streaming run took 8.83 s against
                # a 1.28 s median, and a single stall that long is what makes a
                # live stream fall behind permanently.
                max_new_tokens=min(400, int(8 * len(self.buf) / RATE) + 16),
                condition_on_prev_tokens=False, temperature=0.0)
        if self.device == "mps":
            torch.mps.synchronize()
        self.state.decode_secs.append(time.monotonic() - t0)
        self.state.decodes += 1

        words = self._words_from(out)
        prev = self.state.tentative
        n = 0
        while (n < min(len(prev), len(words))
               and _norm(prev[n].text) == _norm(words[n].text)):
            n += 1
        if n:
            self._commit(words[:n])
        self.state.tentative = words[n:]
        # Trim to the last committed word so the next decode does not re-do
        # work already frozen. Memory is bounded by the tentative tail alone.
        if self.state.committed:
            cut = self.state.committed[-1].end
            off = int(max(0.0, cut - self.buf_t0) * RATE)
            if 0 < off < len(self.buf):
                self.buf = self.buf[off:]
                self.buf_t0 += off / RATE

    def _words_from(self, out) -> list[Word]:
        # generate() returns a plain dict here (return_timestamps=True forces
        # return_segments=True), not a ModelOutput — attribute access fails.
        from chunk_longform import words_from_tokens
        toks = out["sequences"][0]
        offs = out.get("token_timestamps")
        tts = offs[0].tolist() if offs is not None else None
        got = words_from_tokens(toks, tts, self.proc, self.buf_t0,
                                len(self.buf) / RATE)
        return [Word(w["w"], w["start"], w["end"]) for w in got]

    def _commit(self, words: list[Word], forced: bool = False) -> None:
        for w in words:
            self.state.commit_lag.append(max(0.0, self.clock - w.end))
        self.state.committed.extend(words)

    # -- what goes on screen ----------------------------------------------
    def caption(self) -> tuple[str, str]:
        """(stable, tentative). Stable text is punctuated and never rewritten.

        Punctuation is incremental and PERSISTED. The first version simply
        re-punctuated the last 60 words on every call and returned that joined
        to the raw head — so the head never received punctuation at all and the
        220 s stream came out entirely lower-case. The fix keeps a
        frozen punctuated prefix: each pass punctuates only the words added
        since the last pass, preceded by PUNCT_CTX raw words whose output is
        discarded (they exist to give the first new word real left context,
        the same trick punct_restore uses across its own chunks). The final
        PUNCT_LIVE words stay unfrozen because a following word can still
        change where a comma belongs.
        """
        tent = " ".join(w.text for w in self.state.tentative)
        com = self.state.committed
        if not self.punct:
            return " ".join(w.text for w in com), tent
        # Throttle: the tagger costs ~200 ms, which at STEP=1.5 s is 13% of the
        # whole budget and took keep-up to a no-headroom 0.99. Re-punctuating
        # after every single new word buys nothing — a comma needs a few words
        # of right context before it can be placed at all.
        pending = len(com) - len(self._p_words) - len(self._live)
        if pending >= PUNCT_MIN_NEW or (pending and self._flush):
            f = len(self._p_words)
            ctx = [w.text for w in com[max(0, f - PUNCT_CTX):f]]
            new = [w.text for w in com[f:]]
            got = None
            try:
                done, _ = self.punct.restore(" ".join(ctx + new))
                t = done.split()
                if len(t) == len(ctx) + len(new):
                    got = t[len(ctx):]
            except Exception:
                got = None
            if got is None:      # tagger refused: raw words, never lose text
                got = new
            keep = 0 if self._flush else max(0, len(got) - PUNCT_LIVE)
            self._p_words.extend(got[:keep])
            self._live = got[keep:]
        tail = [w.text for w in com[len(self._p_words) + len(self._live):]]
        return " ".join(self._p_words + self._live + tail), tent

    def screen_lines(self, stable: str | None = None) -> list[str]:
        """Last CAPTION_LINES lines of <=CAPTION_CHARS, stable text only.
        Pass `stable` from an existing caption() call to avoid punctuating
        the same words twice per frame."""
        if stable is None:
            stable, _ = self.caption()
        lines, cur = [], ""
        for w in stable.split():
            if len(cur) + len(w) + 1 > CAPTION_CHARS:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        if cur:
            lines.append(cur)
        return lines[-CAPTION_LINES:]

    def finish(self) -> str:
        if self.state.tentative:
            self._commit(self.state.tentative, forced=True)
            self.state.tentative = []
        self._flush = True          # punctuate the remainder, throttle aside
        stable, _ = self.caption()
        self._flush = False
        return stable


# --- runnable -------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
# Same resolution order as transcribe_file: env var, local training output,
# then the published weights — so this runs unchanged outside this machine.
from transcribe_file import DEFAULT_MODEL  # noqa: E402

LIVE_HELP = """\
Live microphone (no extra Python deps — ffmpeg does the capture):

    ffmpeg -f avfoundation -i :0 -ac 1 -ar 16000 -v error -f f32le - \\
      | python subtitle_stream.py --stdin

List input devices with:
    ffmpeg -f avfoundation -list_devices true -i ""
"""


def _render(stream: "SubtitleStream", final: bool = False) -> None:
    """Two-line stable caption plus the tentative tail, redrawn in place.

    The tentative tail is shown DIM and separate. It is the part that may
    still change, and marking it is what makes the stable line trustworthy —
    a viewer who has read a word in the bright line never sees it move.
    """
    stable, tent = stream.caption()
    stable_lines = stream.screen_lines(stable)
    sys.stdout.write("\033[2K\r")
    for ln in stable_lines:
        sys.stdout.write(f"\033[2K{ln}\n")
    sys.stdout.write(f"\033[2K\033[2m{tent}\033[0m")
    if not final:
        sys.stdout.write(f"\033[{len(stable_lines) + 1}A\r")
    sys.stdout.flush()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        epilog=LIVE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--stdin", action="store_true",
                     help="raw 16 kHz mono float32 PCM on stdin")
    src.add_argument("--file", help="wav/m4a/mp4/... replayed as a stream")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-punct", action="store_true")
    ap.add_argument("--realtime", action="store_true",
                    help="with --file, pace playback at 1x instead of ASAP")
    ap.add_argument("--out", help="write the final transcript here")
    args = ap.parse_args()

    s = SubtitleStream(args.model, punctuate=not args.no_punct)
    print(f"[{s.device} {str(s.dtype).split('.')[-1]}] listening…\n",
          file=sys.stderr, flush=True)
    blk = int(0.25 * RATE)
    t0 = time.monotonic()
    fed = 0.0
    try:
        if args.stdin:
            src_stream = sys.stdin.buffer
            while True:
                raw = src_stream.read(blk * 4)
                if not raw:
                    break
                pcm = np.frombuffer(raw, dtype=np.float32)
                fed += len(pcm) / RATE
                if s.push(pcm):
                    _render(s)
        else:
            from transcribe_file import load_audio
            a = load_audio(args.file, gain=True)
            for i in range(0, len(a), blk):
                chunk = a[i:i + blk]
                fed += len(chunk) / RATE
                if args.realtime:
                    ahead = fed - (time.monotonic() - t0)
                    if ahead > 0:
                        time.sleep(ahead)
                if s.push(chunk):
                    _render(s)
    except KeyboardInterrupt:
        pass
    final = s.finish()
    _render(s, final=True)
    st = s.state
    ds = sorted(st.decode_secs) or [0.0]
    lg = sorted(st.commit_lag) or [0.0]
    wall = time.monotonic() - t0
    print(f"\n\n[{len(final.split())} words | audio {fed:.0f}s compute {wall:.0f}s "
          f"ratio {wall/max(fed,1e-9):.2f} | decode med {ds[len(ds)//2]:.2f}s "
          f"| lag med {lg[len(lg)//2]:.2f}s p90 {lg[int(len(lg)*.9)]:.2f}s]",
          file=sys.stderr)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(final)


if __name__ == "__main__":
    main()
