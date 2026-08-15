# paprika-lt-asr

Lithuanian speech-to-text that runs on your own machine. Two pipelines, both
producing punctuation:

- **`subtitle_stream.py`** — live subtitles from a microphone, stable enough to
  put on a screen.
- **`transcribe_file.py`** — long recordings to a punctuated, speaker-labelled
  transcript plus `.srt` / `.vtt` / `.json`.

Nothing leaves the machine. No API key, no upload.

Built around [`paprika-whisper-lt-v3`](https://huggingface.co/kristijonas/paprika-whisper-lt-v3),
a `whisper-large-v3-turbo` fine-tune on ~3,281 h of the Lithuanian LIEPA-3
corpus.

## Install

```bash
git clone https://github.com/kristijonasatpro/paprika-lt-asr
cd paprika-lt-asr
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
