"""Synthetic ground-truth accuracy eval.

Generate MIDI we control → render via FluidSynth+SF2 → run the pipeline's
transcriber → score the estimate against the known source with mir_eval.

This is an *optimistic* bound: the transcriber sees clean synthesized audio,
not a Demucs-separated stem with separation artifacts. But it gives a real,
reproducible note-F1 number and isolates the transcriber from the separator.

Run:
    uv run python -m demixer.eval.synth_groundtruth
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi

from demixer.eval.render import render_midi_to_wav

ONSET_TOLERANCE_S = 0.05   # mir_eval default — 50 ms onset window
PITCH_TOLERANCE_CENTS = 50.0  # half a semitone


@dataclass(frozen=True)
class NoteScore:
    name: str
    precision: float
    recall: float
    f1: float
    ref_n: int
    est_n: int


def _midi_to_mireval(midi: pretty_midi.PrettyMIDI) -> tuple[np.ndarray, np.ndarray]:
    """PrettyMIDI → (intervals Nx2 seconds, pitches Hz) across all instruments."""
    intervals: list[tuple[float, float]] = []
    pitches: list[float] = []
    for inst in midi.instruments:
        for n in inst.notes:
            end = max(n.end, n.start + 1e-3)
            intervals.append((n.start, end))
            pitches.append(440.0 * 2 ** ((n.pitch - 69) / 12.0))
    if not intervals:
        return np.zeros((0, 2)), np.zeros((0,))
    return np.asarray(intervals), np.asarray(pitches)


def score_notes(ref: pretty_midi.PrettyMIDI, est: pretty_midi.PrettyMIDI, name: str) -> NoteScore:
    ref_int, ref_pitch = _midi_to_mireval(ref)
    est_int, est_pitch = _midi_to_mireval(est)
    if len(ref_int) == 0:
        return NoteScore(name, 0.0, 0.0, 0.0, 0, len(est_int))
    if len(est_int) == 0:
        return NoteScore(name, 0.0, 0.0, 0.0, len(ref_int), 0)
    # offset_ratio=None → onset+pitch only (standard "note F1, no offset")
    p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int, ref_pitch, est_int, est_pitch,
        onset_tolerance=ONSET_TOLERANCE_S,
        pitch_tolerance=PITCH_TOLERANCE_CENTS,
        offset_ratio=None,
    )
    return NoteScore(name, p, r, f1, len(ref_int), len(est_int))


# ---------- test corpus ----------

def _note(pitch: int, start: float, dur: float, vel: int = 90) -> pretty_midi.Note:
    return pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=start + dur)


def _make(program: int, notes: list[pretty_midi.Note]) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    inst.notes.extend(notes)
    midi.instruments.append(inst)
    return midi


def build_corpus() -> dict[str, tuple[pretty_midi.PrettyMIDI, str]]:
    """Return name -> (midi, transcribe_hint). Covers register/polyphony/tempo/timbre."""
    corpus: dict[str, tuple[pretty_midi.PrettyMIDI, str]] = {}

    # Monophonic C-major scale, piano (the easy baseline)
    scale = [_note(p, i * 0.5, 0.45) for i, p in enumerate([60, 62, 64, 65, 67, 69, 71, 72])]
    corpus["piano_scale_mono"] = (_make(0, scale), "piano")

    # Arpeggio, piano
    arp = [_note(p, i * 0.25, 0.22)
           for i, p in enumerate([60, 64, 67, 72, 67, 64, 60, 64, 67, 72])]
    corpus["piano_arpeggio"] = (_make(0, arp), "piano")

    # Block triads (polyphony), piano
    chords: list[pretty_midi.Note] = []
    for i, root in enumerate([60, 65, 67, 60]):
        for offset in (0, 4, 7):
            chords.append(_note(root + offset, i * 0.75, 0.7))
    corpus["piano_block_chords"] = (_make(0, chords), "piano")

    # Bass line (low register), finger bass (GM 33)
    bass = [_note(p, i * 0.5, 0.45) for i, p in enumerate([40, 40, 43, 45, 40, 38, 36, 43])]
    corpus["bass_line"] = (_make(33, bass), "bass")

    # Fast sixteenth-note run, piano (tests onset resolution)
    fast = [_note(60 + (i % 12), i * 0.125, 0.11) for i in range(24)]
    corpus["piano_fast_run"] = (_make(0, fast), "other")

    # Sustained strings chord (slow attack — hard for onset detection), GM 48
    strings: list[pretty_midi.Note] = []
    for offset in (0, 3, 7, 10):
        strings.append(_note(60 + offset, 0.0, 3.0))
    corpus["strings_sustained_chord"] = (_make(48, strings), "other")

    return corpus


def run(verbose: bool = True) -> list[NoteScore]:
    from demixer.core.transcription.pitched import transcribe_pitched

    corpus = build_corpus()
    scores: list[NoteScore] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for name, (midi, hint) in corpus.items():
            wav = tmpdir / f"{name}.wav"
            render_midi_to_wav(midi, wav)
            est = transcribe_pitched(wav, hint=hint)  # type: ignore[arg-type]
            score = score_notes(midi, est, name)
            scores.append(score)
            if verbose:
                print(f"  {name:28s} P={score.precision:.2f} R={score.recall:.2f} "
                      f"F1={score.f1:.2f}  (ref={score.ref_n} est={score.est_n})")

    if verbose and scores:
        mean_f1 = sum(s.f1 for s in scores) / len(scores)
        print(f"\n  MEAN note-F1 across {len(scores)} cases: {mean_f1:.3f}")
    return scores


if __name__ == "__main__":
    print("Synthetic ground-truth pitched-transcription eval (basic-pitch on clean synth audio):\n")
    run()
