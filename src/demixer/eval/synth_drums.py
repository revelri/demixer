"""Synthetic ground-truth drum-transcription eval.

Generate GM drum patterns → render via FluidSynth → run the drum transcriber →
score onset timing (mir_eval) and per-class (kick/snare/hat) accuracy.

Run:
    uv run python -m demixer.eval.synth_drums
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi

from demixer.eval.render import render_midi_to_wav

ONSET_TOLERANCE_S = 0.05

# GM drum notes we emit/classify
_KICK, _SNARE, _HAT_CLOSED, _HAT_OPEN = 36, 38, 42, 46
# Map every drum note to a coarse class for matching (closed/open hat → "hat")
_CLASS = {_KICK: "kick", _SNARE: "snare", _HAT_CLOSED: "hat", _HAT_OPEN: "hat"}


@dataclass(frozen=True)
class DrumScore:
    name: str
    onset_f1: float
    class_accuracy: float  # of matched onsets, fraction with correct coarse class
    ref_n: int
    est_n: int


def _drum_onsets(midi: pretty_midi.PrettyMIDI) -> tuple[np.ndarray, list[str]]:
    pairs = sorted(
        ((n.start, _CLASS.get(n.pitch, "other"))
         for inst in midi.instruments for n in inst.notes),
        key=lambda x: x[0],
    )
    times = np.asarray([t for t, _ in pairs])
    classes = [c for _, c in pairs]
    return times, classes


def score_drums(ref: pretty_midi.PrettyMIDI, est: pretty_midi.PrettyMIDI, name: str) -> DrumScore:
    ref_t, ref_c = _drum_onsets(ref)
    est_t, est_c = _drum_onsets(est)
    if len(ref_t) == 0:
        return DrumScore(name, 0.0, 0.0, 0, len(est_t))
    if len(est_t) == 0:
        return DrumScore(name, 0.0, 0.0, len(ref_t), 0)

    f1, _, _ = mir_eval.onset.f_measure(ref_t, est_t, window=ONSET_TOLERANCE_S)

    # Greedy nearest match within tolerance, then check class agreement
    matched = 0
    correct_class = 0
    used_est = set()
    for rt, rc in zip(ref_t, ref_c):
        best_j, best_d = -1, ONSET_TOLERANCE_S
        for j, et in enumerate(est_t):
            if j in used_est:
                continue
            d = abs(et - rt)
            if d <= best_d:
                best_j, best_d = j, d
        if best_j >= 0:
            used_est.add(best_j)
            matched += 1
            if est_c[best_j] == rc:
                correct_class += 1
    class_acc = correct_class / matched if matched else 0.0
    return DrumScore(name, f1, class_acc, len(ref_t), len(est_t))


def _drum_note(pitch: int, start: float, vel: int = 100) -> pretty_midi.Note:
    return pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=start + 0.1)


def _drum_midi(notes: list[pretty_midi.Note]) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.extend(notes)
    midi.instruments.append(inst)
    return midi


def build_corpus() -> dict[str, pretty_midi.PrettyMIDI]:
    corpus: dict[str, pretty_midi.PrettyMIDI] = {}

    # Four-on-the-floor: kick every beat, hat every off-beat, snare on 2 & 4. 120 BPM, 4 bars.
    beat = 0.5
    notes: list[pretty_midi.Note] = []
    for bar in range(4):
        b0 = bar * 4 * beat
        for k in range(4):
            notes.append(_drum_note(_KICK, b0 + k * beat))
        notes.append(_drum_note(_SNARE, b0 + 1 * beat))
        notes.append(_drum_note(_SNARE, b0 + 3 * beat))
        for h in range(8):
            notes.append(_drum_note(_HAT_CLOSED, b0 + h * beat / 2, vel=70))
    corpus["four_on_floor"] = _drum_midi(notes)

    # Sparse kick+snare backbeat only (no hats) — tests class separation cleanly
    notes2: list[pretty_midi.Note] = []
    for bar in range(4):
        b0 = bar * 4 * beat
        notes2.append(_drum_note(_KICK, b0 + 0 * beat))
        notes2.append(_drum_note(_KICK, b0 + 2 * beat))
        notes2.append(_drum_note(_SNARE, b0 + 1 * beat))
        notes2.append(_drum_note(_SNARE, b0 + 3 * beat))
    corpus["backbeat_sparse"] = _drum_midi(notes2)

    return corpus


def run(verbose: bool = True) -> list[DrumScore]:
    from demixer.core.transcription.drums import transcribe_drums

    corpus = build_corpus()
    scores: list[DrumScore] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for name, midi in corpus.items():
            wav = tmpdir / f"{name}.wav"
            render_midi_to_wav(midi, wav)
            est = transcribe_drums(wav)
            score = score_drums(midi, est, name)
            scores.append(score)
            if verbose:
                print(f"  {name:20s} onset-F1={score.onset_f1:.2f}  "
                      f"class-acc={score.class_accuracy:.2f}  "
                      f"(ref={score.ref_n} est={score.est_n})")

    if verbose and scores:
        mean_onset = sum(s.onset_f1 for s in scores) / len(scores)
        mean_class = sum(s.class_accuracy for s in scores) / len(scores)
        print(f"\n  MEAN onset-F1={mean_onset:.3f}  MEAN class-acc={mean_class:.3f}")
    return scores


if __name__ == "__main__":
    print("Synthetic drum-transcription eval (librosa onset + centroid classifier):\n")
    run()
