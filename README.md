# paprika 🌶️ — Lithuanian speech-to-text

Runs entirely on your own machine. Two pipelines, both
producing punctuation:

- **`subtitle_stream.py`** — live subtitles from a microphone, stable enough to
  put on a screen.
- **`transcribe_file.py`** — long recordings to a punctuated, speaker-labelled
  transcript plus `.srt` / `.vtt` / `.json`.
- **`transcribe_kmynas.py`** — the same outputs from a Parakeet-TDT model
  instead of Whisper. Roughly 30× faster, and it emits punctuation itself.

Nothing leaves the machine. No API key, no upload.

Built around [`paprika-whisper-lt-v3`](https://huggingface.co/kristijonas/paprika-whisper-lt-v3),
a `whisper-large-v3-turbo` fine-tune on ~3,281 h of the Lithuanian LIEPA-3
corpus.

## Install

```bash
git clone https://github.com/kristijonasatpro/paprika
cd paprika
pip install -r requirements.txt        # ffmpeg must also be on PATH
```

First run downloads ~1.6 GB of model weights, plus ~57 MB of diarization models
if you use speaker labels. Both are cached afterwards.

## Transcribe a file

```bash
python transcribe_file.py recording.m4a --speakers 2
```

Writes `recording.txt` (speaker-labelled, punctuated), `recording.srt`,
`recording.vtt` and `recording.json` (word-level timestamps). The transcript
looks like this:

```
[00:12] Kalbėtojas 1: Labas rytas, pradėkime nuo pirmojo klausimo.

[00:19] Kalbėtojas 2: Ačiū. Manau, kad reikėtų pažiūrėti, kaip tai
veikia praktiškai, ir tik tada spręsti.
```

Useful flags: `--speakers N` when you know the count (more reliable than
letting a threshold guess), `--no-diar` to skip speaker labels, `--no-punct`
for raw output, `--model` for a different checkpoint.

Roughly 17× faster than real time on an M4 Mac mini: a 10-minute recording
takes about a minute, most of it diarization.

## The Parakeet pipeline (`transcribe_kmynas.py`)

`transcribe_kmynas.py` does the same job as `transcribe_file.py` with a
different model, and the trade is real in both directions.

| | paprika (`transcribe_file.py`) | kmynas (`transcribe_kmynas.py`) |
|---|---|---|
| model | Whisper large-v3-turbo fine-tune, ~800M | Parakeet TDT fine-tune, 600M |
| punctuation | separate ONNX tagger after decoding | from the model itself |
| word timestamps | cross-attention alignment, ~6.4 GB per block | near-free |
| speed | RTF ~0.30 | RTF ~0.03 (about 10x faster) |
| clean prepared speech | **better** | weaker |

Everything after decoding — diarization, speaker smoothing, turns, cues and the
writers — is shared code, imported from `transcribe_file`, so the two produce
identically laid-out transcripts and cannot drift apart.

**The kmynas checkpoint may be a private Hugging Face repo**, so unlike paprika
it will not download itself unless you are authenticated. Pass `--model` (or set
`$KMYNAS_MODEL`) to a local `.nemo` file.

### Which one to use

Read both on your own audio; they fail differently and the difference is not
captured by a single number. On clean, prepared speech paprika still reads
better — kmynas mangles rare proper nouns. Two of its earlier weaknesses are
handled now: duplicated words at block seams, and stray letter clusters where
an audience applauds (see the next section for how). Reach for kmynas when you
want speed, a smaller footprint, and punctuation without a second model; it
holds up better on spontaneous speech than the benchmarks suggest.

Install `nemo_toolkit[asr]` only if you want this path; it is a large
dependency and the paprika pipeline does not need it.

It needs NeMo, which pins versions that conflict with the Whisper stack, so keep
it in its own environment:

```bash
python -m venv .venv-nemo
.venv-nemo/bin/pip install nemo_toolkit[asr] silero-vad soundfile
.venv-nemo/bin/python transcribe_kmynas.py recording.m4a \
    --model your-model.nemo --speakers 2 --lexicon lexicon.tsv
```

### Why it is not just "cut into blocks and decode"

Long-form transducer decoding fails in specific, reproducible ways, and most of
this file is the handling for them. Each was measured, and several plausible
fixes were tried and rejected — those are documented in the code so they are not
retried blind.

**Blocks are placed by voice activity, not by a timer** (`--vad-target`,
default 60 s; `--no-vad` to disable). Every block starts at a speech onset and
ends at a speech offset, and the non-speech between two blocks is dropped rather
than split. This matters because the model normalises features *per utterance*:
silence inside a block shifts the mel statistics of every speech frame in it, so
a cut through the middle of a pause damages both neighbours. Cutting at a
pause's edges damages neither. An energy threshold cannot find those edges —
room tone and quiet speech overlap in RMS on real recordings — so Silero VAD
does it.

**Overlapping blocks are merged by word midpoint plus an n-gram seam pass**
(`--overlap`, default 1.3 s; ignored in VAD mode, where boundaries already sit
in silence). A word straddling a cut is decoded whole by both neighbours and
kept once. Without this, cuts land mid-word and the orphaned tail is capitalised
as a new sentence — 31 of 60 seams did that on one 35-minute recording.

**Hallucinated non-words are filtered** (`--min-confidence`, default 0.98, `0`
disables). On non-speech — breath, laughter, applause — the model has no way to
emit nothing, so it emits short consonant clusters instead. A word is dropped
only when it is *both* low-confidence *and* orthographically impossible in
Lithuanian (contains no vowel), with acronyms and `m`/`h`/`n` fillers
whitelisted. Either test alone is too blunt; together they removed 63 such
tokens across five recordings without touching a real word.

**Borrowed words are spelled one way** (`--lexicon lexicon.tsv`, off by
default). The model renders the same loanword several ways in one recording.
The table collapses the variants onto whichever form that model already produces
most often, so it removes inconsistency without imposing an orthography. Edit it
for your own domain.

### Flags worth knowing

| flag | default | what it does |
|---|---|---|
| `--speakers N` | `-1` | known speaker count; far better than letting the threshold guess |
| `--vad-target` | `60` | target block length in seconds |
| `--no-vad` | off | fall back to the older fixed-target chunker |
| `--min-confidence` | `0.98` | hallucination filter threshold; `0` disables |
| `--overlap` | `1.3` | block overlap in seconds (non-VAD mode) |
| `--lexicon` | none | loanword spelling table |
| `--diar-min-off` | `0.5` | seconds a voice must go quiet before a speaker change is believed; lower for fast turn-taking |
| `--dtype` | `fp16` | use `fp32` on Apple Silicon if output looks wrong |
| `--block-secs` | `35` | max block length in non-VAD mode |
| `--boost-file` | none | domain term list — measured ineffective, see the docstring |

### What it does not fix

Overlapping speech. When two people talk over each other the model hears a
mixture and emits one shorter, garbled stream — words get collapsed rather than
split, and no amount of block placement or filtering reaches it, because the
damage happens inside a block, not at its edges. Diarization degrades in the
same places for the same reason: while both people are speaking there is no
silence to detect, so a frame goes to whichever voice dominates it and short
interjections land on the wrong speaker. `--diar-min-off` helps when turns are
fast but *not* overlapping; it does nothing for genuine overlap. Handling that
properly needs source separation or an overlap-trained model.

### Block length and memory

Block length is the memory knob, and attention cost is quadratic in it. On a
16 GB machine 60 s blocks are comfortable; 120 s works but pushes several GB
into swap for no measured accuracy gain. Single-pass decoding of a whole file
(no blocks at all) is fine up to about 5 minutes and gets the process OOM-killed
well before 10. Note that on Apple Silicon this memory does **not** show up in
RSS — watch `sysctl -n vm.swapusage` instead.

## Memory (paprika word timestamps)

Per-word timestamps are the expensive part: Whisper's word alignment builds the
full cross-attention stack and peaks around 10 GB of RAM per block, and that
does not shrink if you use shorter blocks. On a 12.7 GB machine, such as a free
Colab runtime, the run is killed part-way through and you see only `^C`.

Pass `--no-word-ts` there. Word times are spread evenly inside each block, so
the transcript, punctuation, speaker turns and subtitle files are unchanged and
only per-word timing gets coarser. Peak drops to under 4 GB. With 16 GB or more,
leave it off and get exact word timings.

## Live subtitles

```bash
ffmpeg -f avfoundation -i :0 -ac 1 -ar 16000 -v error -f f32le - \
  | python subtitle_stream.py --stdin
```

(`-f avfoundation -i :0` is macOS; use `-f alsa -i default` on Linux or
`-f dshow` on Windows. List devices with
`ffmpeg -f avfoundation -list_devices true -i ""`.)

Or replay a file to see it work: `python subtitle_stream.py --file rec.wav --realtime`

Measured on an M4 Mac mini over 220 s of press-conference speech:

| | |
|---|---|
| keeps up at | 0.94× real time |
| spoken → stable text | 2.48 s median, 5.2 s p90 |
| words vs offline transcript | 413 vs 425 |

The screen shows two things: **stable** text that will never change, and a dim
**tentative** tail that may still move. That distinction is the whole design.
Naively re-printing Whisper's output makes captions rewrite words the viewer
already read, which is far worse to watch than a delay. Here a word becomes
stable only once two consecutive decodes agree on it (LocalAgreement-2).

## Three things worth knowing before you build on this

**1. Do not use `chunk_length_s`.** The chunked pipeline in `transformers`
cuts at a fixed stride, decodes each window independently, then merges by
matching text in the overlaps — and discards spans it cannot align. On clean
audio we measured it silently dropping 30 words from one recording and 52 from
another, both at a seam. The output reads perfectly; nothing marks the gap. We
lost three days to this, diagnosing "model defects" that were the decoder.

**2. Gate silence before decoding, not after.** Whisper narrates silence.
Given 60 s of digital silence, faint hiss, or room tone, the chunked pipeline
emits 24–164 characters of confident Lithuanian. An RMS check costs
microseconds and removes the failure mode entirely — and saves the decode.
Both pipelines here do it. (Native long-form decoding emits nothing on all
three, so this is a decoder property, not a model one.)

**3. Punctuation is a separate model, and so are speakers.** The ASR output is
lowercase and unpunctuated by design — the LIEPA-3 labels are, and a dedicated
tagger does the job better. Whisper has no notion of who is speaking; speaker
labels come from a segmentation model plus speaker embeddings, clustered
offline. Anything claiming Whisper "does diarization" is bolting on a second
model, as this does.

## How it works

```
audio ─► silence gate ─► pause-aligned blocks (<30 s) ─► Whisper ─► words+times
                                                                        │
                              punctuation tagger (word-preserving) ◄─────┤
                                                                        │
        speaker segmentation ─► embeddings ─► clustering ─► labels ◄─────┘
```

Blocks are cut where nobody is speaking, so no word is split across a boundary
and the blocks are independent by construction — there is no merge step to get
wrong. Memory is bounded by block length, not file length: a two-hour recording
costs the same 5.4 GB as a ten-minute one.

Punctuation runs *before* speaker attachment, because the tagger needs long runs
of text to place a sentence boundary. It preserves the word sequence exactly
(case and punctuation only — any word-level edit and it returns the input
untouched), so punctuated tokens map back onto timed words 1:1.

## Accuracy

WER on Lithuanian benchmarks, long-form decoding:

| | v1 | v2 | v3 |
|---|---|---|---|
| VoxPopuli LT, 11 speeches | 17.94 | 17.68 | **17.25** |
| LIEPA-3 held-out, 39 items | 8.44 | 7.67 | **6.42** |

Both are in-domain. There is no out-of-domain benchmark — the sealed set built
for it turned out to be digital silence and every figure from it was withdrawn.
Test on your own audio.

Speaker labelling, scored against known turn structure: **98.7%** of words on a
press conference (two speakers), **93.7%** frame accuracy on a 22-minute
two-person conversation.

## Requirements

Python 3.10+, ffmpeg, ~4 GB RAM for transcription. Apple Silicon (MPS) and CUDA
are used automatically when present; CPU works but is slow. Diarization is
ONNX-only and always runs on CPU.

## Licence

Code Apache-2.0. Model weights CC BY 4.0, inheriting the LIEPA-3 corpus
(VU / raštija.lt) and `svogunas/whisper-large-v3-turbo-lt` attribution chain.
