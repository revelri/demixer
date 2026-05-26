"""Build a MusicXML score from per-stem MIDI + tempo + key.

We use music21 because it handles MusicXML serialization, key signatures, time
signatures, tempo marks, and instrument metadata correctly. Each stem becomes
one Part in the Score, named after the stem.

Drum stems (`is_drum=True` in the PrettyMIDI) get the "Drumset" percussion
clef so engravers know to render on a single-line drum staff.

Tempo / key are stamped on the first measure of each part.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import music21 as m21
import pretty_midi

from demixer.core.analysis.key import KeyEstimate
from demixer.core.analysis.tempo_beats import TempoBeats


def build_score(
    stems: dict[str, pretty_midi.PrettyMIDI],
    *,
    tempo: TempoBeats,
    key: KeyEstimate,
    project_name: str = "demixer",
    include_drums: bool = False,
) -> m21.stream.Score:
    """Return a music21 Score with one Part per stem (preserves stem-dict order).

    `include_drums` defaults False because music21's drum-staff serialization
    produces MusicXML that crashes MuseScore (unpitched-note voice issues).
    Drum MIDI still rides in the bundle's midi/drums.mid — it's just not on
    the engraved score. Set include_drums=True if your downstream engraver
    handles percussion clefs better.
    """
    score = m21.stream.Score()
    score.metadata = m21.metadata.Metadata()
    score.metadata.title = project_name
    score.metadata.composer = "demixer"

    for stem_name, midi in stems.items():
        midi_has_drums = any(inst.is_drum for inst in midi.instruments)
        if midi_has_drums and not include_drums:
            continue
        part = _midi_to_part(midi, name=stem_name)
        # Stamp tempo + key + time sig on the first measure of every part
        m21_key = m21.key.KeySignature(key.sharps)
        m21_ts = m21.meter.TimeSignature(f"{tempo.beats_per_bar}/4")
        m21_tempo = m21.tempo.MetronomeMark(number=tempo.tempo_bpm)
        part.insert(0, m21_key)
        part.insert(0, m21_ts)
        part.insert(0, m21_tempo)
        score.append(part)

    _normalize_part_lengths(score)
    return score


def _normalize_part_lengths(score: m21.stream.Score) -> None:
    """Pad every (flat) part to the same length so they share a measure count.

    Parts are flat (notes at absolute offsets, no Measures yet — see
    `_midi_to_part`); music21's exporter makes measures once. Padding the shorter
    parts to the longest part's end with a trailing rest keeps the measure grids
    aligned across parts. We deliberately do NOT call makeNotation here — letting
    the exporter bar everything once avoids the double-measure StreamException
    that dense multi-part scores hit.
    """
    parts = list(score.parts)
    if len(parts) < 2:
        return
    target = max((p.highestTime for p in parts), default=0.0)
    for p in parts:
        gap = target - p.highestTime
        if gap > 1e-6:
            p.insert(p.highestTime, m21.note.Rest(quarterLength=gap))


def _midi_to_part(midi: pretty_midi.PrettyMIDI, *, name: str) -> m21.stream.Part:
    """Round-trip a PrettyMIDI through a temp .mid into a music21 Part."""
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        midi.write(f.name)
        try:
            stream = m21.converter.parse(f.name)
        finally:
            Path(f.name).unlink(missing_ok=True)

    # MIDI import auto-creates Measures. Re-bar them later — if a part keeps its
    # own Measures while other parts have theirs, music21's MusicXML exporter
    # double-inserts measures (makeRests/makeTies "object already found"
    # StreamException) on dense multi-part scores. So return a FLAT part holding
    # only notes at absolute offsets; the exporter makes measures exactly once.
    src_notes = stream.flatten().notes
    part = m21.stream.Part()
    for el in src_notes:
        part.insert(el.offset, el)

    part.partName = name
    part.partAbbreviation = name[:4]
    if any(inst.is_drum for inst in midi.instruments):
        part.insert(0, m21.clef.PercussionClef())
    return part


def to_musicxml(score: m21.stream.Score) -> str:
    """Serialize a music21 Score to a MusicXML string."""
    exporter = m21.musicxml.m21ToXml.GeneralObjectExporter(score)
    raw = exporter.parse()
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return raw


def write_musicxml(score: m21.stream.Score, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(to_musicxml(score))
    return dest
