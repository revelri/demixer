"""Organic-audio eval on real library tracks (no ground-truth MIDI available).

Two signals, both computable without hand annotation:

  1. Self-consistency: transcribe a separated stem → render the MIDI back to
     audio → transcribe again → note-F1 between pass 1 and pass 2. High
     consistency means the transcriber is stable / its output is "re-readable".

  2. Harmonic fidelity (chroma cosine): compare the chroma of the ORIGINAL stem
     to the chroma of the audio re-synthesized from its transcription. Measures
     "did the MIDI capture the harmonic content of the stem" — the closest
     objective proxy for recreation fidelity without perceptual modelling.

Run:
    uv run python -m demixer.eval.organic_consistency [<audio> ...]
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from demixer.eval.render import render_midi_to_wav
from demixer.eval.synth_groundtruth import score_notes

# Tracks to evaluate. Supply your own via the CLI args or the DEMIXER_EVAL_TRACKS
# env var (os.pathsep-separated absolute paths) — this harness needs local audio
# it can't ship.
DEFAULT_TRACKS = [
    t for t in os.environ.get("DEMIXER_EVAL_TRACKS", "").split(os.pathsep) if t
]
CLIP_START_S = 30
CLIP_DUR_S = 20
# Which separated stems to transcribe as pitched content
PITCHED_STEMS = ("bass", "other", "vocals")


@dataclass(frozen=True)
class OrganicResult:
    track: str
    stem: str
    self_consistency_f1: float  # pass1 vs pass2 note F1
    chroma_similarity: float    # original stem vs re-synth, cosine on mean chroma
    n_notes: int


def _trim(src: Path, dest: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
         "-ss", str(CLIP_START_S), "-i", str(src), "-t", str(CLIP_DUR_S),
         "-ac", "2", "-ar", "44100", str(dest)],
        check=True,
    )


def _chroma_similarity(wav_a: Path, wav_b: Path) -> float:
    """Cosine similarity of time-averaged chroma vectors. 1.0 = identical pitch classes."""
    ya, sr = librosa.load(str(wav_a), sr=22_050, mono=True)
    yb, _ = librosa.load(str(wav_b), sr=22_050, mono=True)
    ca = librosa.feature.chroma_cqt(y=ya, sr=sr).mean(axis=1)
    cb = librosa.feature.chroma_cqt(y=yb, sr=sr).mean(axis=1)
    na, nb = np.linalg.norm(ca), np.linalg.norm(cb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(ca, cb) / (na * nb))


def run(tracks: list[str] | None = None, verbose: bool = True) -> list[OrganicResult]:
    from demixer.core.ingest import ingest
    from demixer.core.separation import separate, write_stems
    from demixer.core.transcription.pitched import transcribe_pitched, write_midi

    paths = [Path(t) for t in (tracks or DEFAULT_TRACKS) if Path(t).exists()]
    results: list[OrganicResult] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for src in paths:
            clip = tmpdir / f"{src.stem}.wav"
            try:
                _trim(src, clip)
            except subprocess.CalledProcessError:
                continue
            # _trim seeks to CLIP_START_S; if the source is shorter than that the
            # trim is empty (ffmpeg still exits 0). ingest raises ValueError on
            # empty/too-short audio — skip such clips rather than abort the run.
            try:
                audio = ingest(clip)
            except ValueError as exc:
                if verbose:
                    print(f"  {src.stem[:24]:24s} skipped ({exc})")
                continue
            sep = separate(audio, model_name="htdemucs")
            stem_paths = write_stems(sep, tmpdir / f"{src.stem}_stems")

            for stem in PITCHED_STEMS:
                wav = stem_paths.get(stem)
                if wav is None:
                    continue
                hint = "bass" if stem == "bass" else ("vocals" if stem == "vocals" else "other")
                midi1 = transcribe_pitched(wav, hint=hint)  # type: ignore[arg-type]
                write_midi(midi1, tmpdir / f"{src.stem}_{stem}_1.mid")

                # Re-synthesize the transcription, then transcribe again
                resynth = tmpdir / f"{src.stem}_{stem}_resynth.wav"
                render_midi_to_wav(midi1, resynth)
                midi2 = transcribe_pitched(resynth, hint=hint)  # type: ignore[arg-type]

                consistency = score_notes(midi1, midi2, f"{src.stem}/{stem}").f1
                chroma = _chroma_similarity(wav, resynth)
                n = sum(len(i.notes) for i in midi1.instruments)
                results.append(OrganicResult(src.stem, stem, consistency, chroma, n))
                if verbose:
                    print(f"  {src.stem[:24]:24s} {stem:7s} "
                          f"self-consistency-F1={consistency:.2f}  "
                          f"chroma-sim={chroma:.2f}  notes={n}")

    if verbose and results:
        mc = sum(r.self_consistency_f1 for r in results) / len(results)
        mh = sum(r.chroma_similarity for r in results) / len(results)
        print(f"\n  MEAN self-consistency-F1={mc:.3f}  MEAN chroma-sim={mh:.3f}")
    return results


if __name__ == "__main__":
    print("Organic eval on real library tracks (self-consistency + harmonic fidelity):\n")
    run(sys.argv[1:] or None)
