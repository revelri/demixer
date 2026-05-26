# demixer

**Turn any audio track into stems, MIDI, a lead sheet, and an editable DAW session.**

demixer is an end-to-end music-understanding pipeline: it separates a song into
instrument stems, transcribes each to MIDI, analyzes tempo / key / chords / harmony,
engraves a score, and exports a ready-to-open project for your DAW — all from a single
input file.

> **Status:** early, active development. The core pipeline runs end-to-end; engines and
> output formats are still being hardened. Interfaces may change.

---

## What it does

```
  input.{mp3,flac,wav,m4a,ogg}
            │
            ▼
   ┌─────────────────┐
   │  1. ingest      │  decode → 44.1 kHz stereo float32 → EBU R128 loudness normalize
   ├─────────────────┤
   │  2. separate    │  Demucs v4 → drums · bass · other · vocals  (+ optional 6-stem,
   │                 │  BS-RoFormer vocals)
   ├─────────────────┤
   │  3. analyze     │  tempo + beats + downbeats (beat_this, confidence-gated)
   │                 │  key (essentia, on the drums-excluded mix)
   │                 │  chords (autochord triads · or BTC 170-class)
   │                 │  harmony + reharmonization  (opt-in)
   ├─────────────────┤
   │  4. transcribe  │  pitched stems → MIDI (basic-pitch · or MR-MT3)
   │                 │  drums → MIDI (spectral · or ADTOF)
   ├─────────────────┤
   │  5. score       │  quantize → MusicXML → Verovio SVG → MuseScore PDF/MSCZ/PNG/audio
   ├─────────────────┤
   │  6. project     │  Reaper .rpp · .dawproject · drag-in stem+MIDI bundle
   ├─────────────────┤
   │  7. bundle      │  everything packed into a single .demixer archive
   └─────────────────┘
```

## Install

Requires **Python 3.11** and [uv](https://docs.astral.sh/uv/). System tools: `ffmpeg`
(decode) and, for notation rendering, **MuseScore 4** on `PATH`.

```bash
git clone https://github.com/revelri/demixer
cd demixer
uv sync --extra dev          # add --extra gui for the PySide6 desktop shell
```

## Usage

```bash
# Full pipeline → ./out/<track>/ (+ .demixer archive)
uv run demixer process "song.flac" -o out/

# Faster iteration: skip the slow stages while developing
uv run demixer process song.flac -o out/ --skip separate --skip transcribe

# Pick engines
uv run demixer process song.flac -o out/ \
    --model htdemucs_6s --transcriber mt3 --chords btc --drums adtof

# Read-only harmony analysis + a tritone-substitution reharmonization
uv run demixer process song.flac -o out/ --harmony --reharmonize tritone
```

### Output bundle

```
out/<track>/
├── stems/        drums.wav · bass.wav · other.wav · vocals.wav
├── midi/         <stem>.mid per transcribed stem
├── analysis.json tempo · beats · downbeats · key · chords (with confidence)
├── score.*       musicxml · svg · pdf · mscz · preview mp3
├── <track>.rpp · <track>.dawproject · dragin/   DAW round-trip
├── harmony.json  (with --harmony)
└── <track>.demixer   single-file zip of the above
```

## Engines

| Stage | Default | Alternative |
|-------|---------|-------------|
| Separation | Demucs `htdemucs` | `htdemucs_6s` (adds guitar/piano), BS-RoFormer vocals |
| Tempo/beats | beat_this | librosa (automatic fallback) |
| Key | essentia KeyExtractor | — |
| Chords | autochord (triads) | BTC (170-class, 7ths/extensions) |
| Pitched transcription | basic-pitch | MR-MT3 |
| Drum transcription | spectral (librosa) | ADTOF (learned, GM classes) |

The heavier alternatives (MR-MT3, ADTOF, BTC, BS-RoFormer) have **incompatible
dependency stacks** (notably numpy 1.x vs 2.x), so each runs in its own isolated
virtualenv and is invoked as a subprocess worker (`src/demixer/core/workers.py`,
`scripts/*_worker.py`). The main environment stays clean; these engines are strictly
opt-in.

**BTC chord engine** (optional): clone the upstream model into `third_party/`:

```bash
git clone https://github.com/jayg996/BTC-ISMIR19 third_party/BTC
# place the pretrained .pt weights under third_party/BTC/test/ (see that repo)
```

## Architecture

```
src/demixer/
├── core/
│   ├── ingest.py            decode + loudness-normalize
│   ├── separation.py        Demucs wrapper (+ separation_roformer.py)
│   ├── analysis/            tempo_beats · key · chords · chords_btc · harmony
│   ├── transcription/       pitched · drums · mt3 · drums_adtof
│   ├── score/               quantize · musicxml · render (Verovio/MuseScore)
│   ├── project/             reaper · dawproject · dragin
│   ├── workers.py           isolated-venv subprocess worker protocol
│   └── bundle.py            .demixer archive writer
├── cli/                     `demixer process`
├── app/                     PySide6 desktop shell
└── eval/                    ground-truth matrix · consistency / SDR harnesses
```

## Evaluation

`src/demixer/eval/` holds accuracy harnesses: a ground-truth matrix scored against the
Isophonics beat/chord annotations, transcription self-consistency, and separation SDR.
The annotation set is fetched on demand:

```bash
bash tests/groundtruth/fetch_isophonics.sh
```

Harnesses that need real audio read tracks from `--music-root` / `--tracks` or the
`DEMIXER_MUSIC_ROOT` / `DEMIXER_EVAL_TRACKS` environment variables.

## Development

```bash
uv run pytest          # test suite
uv run ruff check .    # lint
uv run mypy            # type-check (strict)
```

## License

[AGPL-3.0-or-later](LICENSE). demixer builds on Demucs, beat_this, essentia, autochord,
basic-pitch, music21, and other open-source projects — each retains its own license.
