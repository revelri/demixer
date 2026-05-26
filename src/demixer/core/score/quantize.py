"""Snap transcribed MIDI to a clean musical grid for engraving.

basic-pitch returns notes with continuous start/duration in seconds, off-grid by
arbitrary amounts. An engraved score needs notes on exact subdivisions or music21
emits ugly tuplets — and on dense multi-part material those tuplets produce
fractional measure offsets that crash music21's own MusicXML export.

We quantize in two steps:

  1. Map each note's time to a fractional BEAT position by interpolating against
     the detected beat grid (so tempo drift in the source is absorbed by the
     mapping, not baked into the output).
  2. Round that beat position — and the note's duration in beats — to the nearest
     1/`subdivisions_per_beat` (default 16th notes), then emit the note in a
     normalized 120 BPM reference where one beat == one quarter note.

Emitting at a fixed 120 BPM (0.5 s/beat) means every note start/end lands on an
exact quarter-note fraction, so music21 reads clean note values and tiles 4/4
measures without tuplets. The *displayed* tempo is applied separately by
build_score's MetronomeMark — note-value cleanliness is decoupled from playback
speed, which is exactly what engraving wants.

Only the engraving path uses this; DAW-exported MIDI keeps the raw transcription.
"""

from __future__ import annotations

import numpy as np
import pretty_midi

_REF_SECONDS_PER_BEAT = 0.5  # 120 BPM reference: 1 beat == 1 quarter note


def quantize_midi(
    midi: pretty_midi.PrettyMIDI,
    beat_times_s: np.ndarray,
    *,
    subdivisions_per_beat: int = 4,
) -> pretty_midi.PrettyMIDI:
    """Return a NEW PrettyMIDI with notes snapped to a clean 1/subdiv beat grid.

    Output is in a 120 BPM reference (beat == quarter note) so downstream music21
    engraving sees exact note values. Notes that collapse to the same start+pitch
    after snapping are de-duplicated.
    """
    if subdivisions_per_beat < 1:
        raise ValueError("subdivisions_per_beat must be >= 1")
    if len(beat_times_s) < 2:
        return midi  # nothing to align against

    beat_times_s = np.asarray(beat_times_s, dtype=np.float64)
    beat_index = np.arange(len(beat_times_s), dtype=np.float64)
    step = 1.0 / subdivisions_per_beat

    def to_quantized_beats(t: float) -> float:
        # Fractional beat position (extrapolates linearly past the ends via np.interp clamp).
        beat = float(np.interp(t, beat_times_s, beat_index))
        return round(beat / step) * step

    out = pretty_midi.PrettyMIDI()
    for src in midi.instruments:
        dst = pretty_midi.Instrument(program=src.program, is_drum=src.is_drum, name=src.name)
        seen: set[tuple[float, int]] = set()
        for n in src.notes:
            start_beat = to_quantized_beats(n.start)
            end_beat = to_quantized_beats(n.end)
            dur_beats = max(step, end_beat - start_beat)  # at least one subdivision

            key = (round(start_beat, 6), n.pitch)
            if key in seen:
                continue
            seen.add(key)

            start_s = start_beat * _REF_SECONDS_PER_BEAT
            end_s = (start_beat + dur_beats) * _REF_SECONDS_PER_BEAT
            dst.notes.append(pretty_midi.Note(
                velocity=n.velocity, pitch=n.pitch, start=start_s, end=end_s,
            ))
        dst.notes.sort(key=lambda x: (x.start, x.pitch))
        out.instruments.append(dst)
    return out
