"""Polyphonic-pitch transcription via Spotify's basic-pitch.

basic-pitch is instrument-agnostic and ships a single small CoreML/TF model.
Input: any audio file. Output: a `pretty_midi.PrettyMIDI` object with one
Instrument containing all detected notes.

We expose per-instrument-hint thresholds because vocals and bass benefit from
different onset / frame thresholds than dense polyphonic content:

  - vocals: lower onset threshold to catch breathy attacks
  - bass:   monophonic post-process (keep highest-velocity note per frame) +
            octave snap (bass notes below MIDI 24 are pulled up an octave —
            basic-pitch frequently octave-errors on sub-bass)
  - other / piano / guitar: defaults
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pretty_midi

InstrumentHint = Literal["vocals", "bass", "drums", "other", "piano", "guitar"]


@dataclass(frozen=True)
class TranscribeOptions:
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    minimum_note_length_ms: float = 58.0  # basic-pitch default
    minimum_frequency: float | None = None
    maximum_frequency: float | None = None
    multiple_pitch_bends: bool = False
    melodia_trick: bool = True  # basic-pitch's polyphonic-vocal helper


_HINT_OPTIONS: dict[InstrumentHint, TranscribeOptions] = {
    "vocals": TranscribeOptions(onset_threshold=0.4, frame_threshold=0.3),
    "bass":   TranscribeOptions(onset_threshold=0.6, frame_threshold=0.3,
                                minimum_frequency=27.5, maximum_frequency=330.0),
    "piano":  TranscribeOptions(),
    "guitar": TranscribeOptions(minimum_frequency=70.0, maximum_frequency=1320.0),
    "other":  TranscribeOptions(),
    # drums is included only so the type system accepts it; raise instead of calling basic-pitch.
    "drums":  TranscribeOptions(),
}


def options_for(hint: InstrumentHint) -> TranscribeOptions:
    return _HINT_OPTIONS[hint]


def transcribe_pitched(
    wav_path: str | Path,
    *,
    hint: InstrumentHint = "other",
    options: TranscribeOptions | None = None,
) -> pretty_midi.PrettyMIDI:
    """Transcribe a pitched stem to MIDI. Raises for `hint='drums'` — use transcription.drums."""
    if hint == "drums":
        raise ValueError("transcribe_pitched does not handle drums; use transcription.drums")

    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise FileNotFoundError(wav_path)

    opts = options or options_for(hint)

    # Imported lazily — basic_pitch pulls in TensorFlow and is heavy.
    # We deliberately use the ONNX-exported model rather than the default TF
    # SavedModel. basic-pitch's SavedModel was serialized against TF ~2.15 and
    # fails to load in modern TF (`_UserObject has no attribute 'add_slot'`).
    # The ONNX export is numerically identical and stack-stable.
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import predict
    model_path = build_icassp_2022_model_path(FilenameSuffix.onnx)

    kwargs: dict[str, object] = dict(
        audio_path=str(wav_path),
        model_or_model_path=model_path,
        onset_threshold=opts.onset_threshold,
        frame_threshold=opts.frame_threshold,
        minimum_note_length=opts.minimum_note_length_ms,
        multiple_pitch_bends=opts.multiple_pitch_bends,
        melodia_trick=opts.melodia_trick,
    )
    if opts.minimum_frequency is not None:
        kwargs["minimum_frequency"] = opts.minimum_frequency
    if opts.maximum_frequency is not None:
        kwargs["maximum_frequency"] = opts.maximum_frequency

    _, midi, _ = predict(**kwargs)

    # basic-pitch shatters sustained notes into many short same-pitch fragments
    # (a 3s held string note can come back as 20 × 0.1s notes), which tanks
    # precision on sustained/polyphonic content. Stitch consecutive same-pitch
    # notes separated by a short gap back into one note.
    _merge_fragmented_notes_inplace(midi)

    if hint == "bass":
        _force_monophonic_inplace(midi)
        _octave_snap_bass_inplace(midi)

    return midi


# Max gap between two same-pitch notes for them to be considered one fragmented
# note rather than two distinct strikes. 90 ms gave the best mean note-F1 on the
# synth eval (0.771 vs 0.758 baseline) and turns the catastrophic sustained-
# strings case from F1 0.07 → 0.31 by stitching its ~50 shards back together.
_FRAGMENT_MERGE_GAP_S = 0.09


def _merge_fragmented_notes_inplace(midi: pretty_midi.PrettyMIDI) -> None:
    """Merge consecutive same-pitch notes separated by < _FRAGMENT_MERGE_GAP_S."""
    for instrument in midi.instruments:
        by_pitch: dict[int, list[pretty_midi.Note]] = {}
        for n in instrument.notes:
            by_pitch.setdefault(n.pitch, []).append(n)

        merged: list[pretty_midi.Note] = []
        for notes in by_pitch.values():
            notes.sort(key=lambda n: n.start)
            cur = notes[0]
            run_vel = [cur.velocity]
            for nxt in notes[1:]:
                if nxt.start - cur.end <= _FRAGMENT_MERGE_GAP_S:
                    # Extend the current note over the fragment
                    cur.end = max(cur.end, nxt.end)
                    run_vel.append(nxt.velocity)
                else:
                    cur.velocity = max(run_vel)
                    merged.append(cur)
                    cur = nxt
                    run_vel = [cur.velocity]
            cur.velocity = max(run_vel)
            merged.append(cur)

        merged.sort(key=lambda n: (n.start, n.pitch))
        instrument.notes = merged


def _force_monophonic_inplace(midi: pretty_midi.PrettyMIDI) -> None:
    """For each instrument, keep at most one note sounding at any time.

    Tiebreaker on equal start times: highest velocity wins.
    """
    for instrument in midi.instruments:
        notes = sorted(instrument.notes, key=lambda n: (n.start, -n.velocity))
        kept: list[pretty_midi.Note] = []
        for note in notes:
            # Equal-onset losers: a louder note at this exact start is already kept.
            if kept and kept[-1].start == note.start:
                continue
            for prev in kept:
                if prev.end > note.start:
                    prev.end = note.start
            kept = [n for n in kept if n.end > n.start]
            kept.append(note)
        instrument.notes = kept


def _octave_snap_bass_inplace(midi: pretty_midi.PrettyMIDI) -> None:
    """Pull notes below MIDI 24 (C0) up an octave — basic-pitch's known sub-bass octave error."""
    for instrument in midi.instruments:
        for note in instrument.notes:
            while note.pitch < 24:
                note.pitch += 12


def write_midi(midi: pretty_midi.PrettyMIDI, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(dest))
    return dest
