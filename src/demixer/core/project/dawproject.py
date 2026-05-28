"""DAWproject (`.dawproject`) writer — Bitwig's open project-exchange format.

Covers Bitwig Studio, Studio One, Cubase, and (recent) Reaper natively. Spec:
https://github.com/bitwig/dawproject

A `.dawproject` is a zip containing:
    project.xml       — Project root: Transport, Structure, Arrangement
    metadata.xml      — title, artist, comment
    audio/<n>.wav     — audio files referenced by relative path

We emit:
  - One audio Track per stem with an Audio clip referencing audio/<name>.wav
  - One instrument Track per transcribed stem with a Notes clip containing the
    notes from the .mid (parsed via pretty_midi → inline XML; DAWproject does
    not reference external .mid files).
  - Transport: tempo + time signature from analysis
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pretty_midi

from demixer.core.analysis.key import KeyEstimate
from demixer.core.analysis.tempo_beats import TempoBeats

DAWPROJECT_VERSION = "1.0"


@dataclass(frozen=True)
class StemTrack:
    name: str
    wav_path: Path
    midi_path: Path | None  # None for stems not transcribed (e.g. drums in v1)


def _track_id_audio(name: str) -> str:
    return f"track-audio-{name}"


def _track_id_notes(name: str) -> str:
    return f"track-notes-{name}"


def _channel_id(name: str, suffix: str) -> str:
    return f"channel-{suffix}-{name}"


def _build_project_xml(
    tracks: list[StemTrack],
    tempo: TempoBeats,
    duration_s: float,
    audio_paths_in_zip: dict[str, str],
) -> ET.Element:
    root = ET.Element("Project", attrib={"version": DAWPROJECT_VERSION})

    # Application — required
    ET.SubElement(root, "Application", attrib={"name": "demixer", "version": "0.0.1"})

    # Transport — tempo + time signature
    transport = ET.SubElement(root, "Transport")
    ET.SubElement(transport, "Tempo", attrib={"value": f"{tempo.tempo_bpm:.6f}", "unit": "bpm"})
    ET.SubElement(transport, "TimeSignature",
                  attrib={"numerator": str(tempo.beats_per_bar), "denominator": "4"})

    # Structure — tracks
    structure = ET.SubElement(root, "Structure")
    for t in tracks:
        audio_track = ET.SubElement(structure, "Track", attrib={
            "id": _track_id_audio(t.name),
            "name": t.name,
            "contentType": "audio",
            "loaded": "true",
        })
        ET.SubElement(audio_track, "Channel", attrib={
            "id": _channel_id(t.name, "audio"),
            "audioChannels": "2",
            "role": "regular",
            "solo": "false",
        })
    for t in tracks:
        if t.midi_path is None:
            continue
        inst_track = ET.SubElement(structure, "Track", attrib={
            "id": _track_id_notes(t.name),
            "name": t.name + " MIDI",
            "contentType": "notes",
            "loaded": "true",
        })
        ET.SubElement(inst_track, "Channel", attrib={
            "id": _channel_id(t.name, "notes"),
            "audioChannels": "2",
            "role": "regular",
            "solo": "false",
        })

    # Arrangement — Lanes containing per-track Clip lanes
    arrangement = ET.SubElement(root, "Arrangement", attrib={"id": "arrangement-0"})
    lanes = ET.SubElement(arrangement, "Lanes",
                          attrib={"timeUnit": "seconds", "id": "lanes-arrangement"})

    for t in tracks:
        track_lane = ET.SubElement(lanes, "Lanes", attrib={
            "track": _track_id_audio(t.name),
            "id": f"lanes-audio-{t.name}",
        })
        clips = ET.SubElement(track_lane, "Clips", attrib={"id": f"clips-audio-{t.name}"})
        clip = ET.SubElement(clips, "Clip", attrib={
            "name": t.name,
            "time": "0",
            "duration": f"{duration_s:.6f}",
        })
        audio = ET.SubElement(clip, "Audio", attrib={
            "algorithm": "stretch",
            "channels": "2",
            "duration": f"{duration_s:.6f}",
            "sampleRate": "44100",
        })
        ET.SubElement(audio, "File", attrib={"path": audio_paths_in_zip[t.name]})

    for t in tracks:
        if t.midi_path is None:
            continue
        track_lane = ET.SubElement(lanes, "Lanes", attrib={
            "track": _track_id_notes(t.name),
            "id": f"lanes-notes-{t.name}",
        })
        clips = ET.SubElement(track_lane, "Clips", attrib={"id": f"clips-notes-{t.name}"})
        clip = ET.SubElement(clips, "Clip", attrib={
            "name": t.name + " MIDI",
            "time": "0",
            "duration": f"{duration_s:.6f}",
            "contentTimeUnit": "seconds",
        })
        notes_el = ET.SubElement(clip, "Notes")
        midi = pretty_midi.PrettyMIDI(str(t.midi_path))
        for instrument in midi.instruments:
            for n in instrument.notes:
                ET.SubElement(notes_el, "Note", attrib={
                    "time": f"{n.start:.6f}",
                    "duration": f"{max(n.end - n.start, 0.001):.6f}",
                    "channel": "0",
                    "key": str(int(n.pitch)),
                    "vel": f"{n.velocity / 127.0:.4f}",
                    "rel": "0.5",
                })

    return root


def _build_metadata_xml(project_name: str) -> ET.Element:
    root = ET.Element("MetaData")
    ET.SubElement(root, "Title").text = project_name
    ET.SubElement(root, "Comment").text = "Generated by demixer"
    return root


def _pretty(elem: ET.Element) -> bytes:
    # Hand-roll a stable indent (avoid ET.indent for Py<3.9 compat history; here it's fine but
    # explicit control gives us deterministic output for diffing).
    ET.indent(elem, space="  ")
    return b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(elem, encoding="utf-8")


def write_dawproject(
    out_path: str | Path,
    *,
    tracks: list[StemTrack],
    tempo: TempoBeats,
    key: KeyEstimate,
    duration_s: float,
    project_name: str = "demixer",
) -> Path:
    out_path = Path(out_path)
    if out_path.suffix != ".dawproject":
        out_path = out_path.with_suffix(".dawproject")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Map stem name → path inside the zip. Preserve the source extension so
    # FLAC stems round-trip as FLAC instead of being relabeled .wav.
    audio_paths_in_zip = {
        t.name: f"audio/{t.name}{Path(t.wav_path).suffix or '.wav'}"
        for t in tracks
    }

    project_xml = _build_project_xml(tracks, tempo, duration_s, audio_paths_in_zip)
    metadata_xml = _build_metadata_xml(project_name)

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("project.xml", _pretty(project_xml))
        z.writestr("metadata.xml", _pretty(metadata_xml))
        for t in tracks:
            # FLAC payload is already compressed — re-DEFLATEing wastes CPU for
            # zero gain. Store FLAC; deflate PCM.
            comp = (zipfile.ZIP_STORED if Path(t.wav_path).suffix.lower() == ".flac"
                    else zipfile.ZIP_DEFLATED)
            z.write(t.wav_path, arcname=audio_paths_in_zip[t.name], compress_type=comp)

    return out_path
