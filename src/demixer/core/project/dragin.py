"""Drag-in bundle — universal fallback for DAWs without project-format export.

Targets FL Studio, Ableton Live, Logic Pro, and anything else: the user opens
their DAW, drags the contents in, then assigns instruments per the suggestions.

Layout:
    out_dir/
      stems/<name>.wav          # audio
      midi/<name>.mid           # MIDI (omitted for stems with no transcription)
      instruments.json          # per-stem GM program + suggested patch + channel
      project.json              # tempo, time sig, key, duration
      README.txt                # short import instructions per DAW
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from demixer.core.analysis.key import KeyEstimate
from demixer.core.analysis.tempo_beats import TempoBeats


@dataclass(frozen=True)
class StemTrack:
    name: str
    wav_path: Path
    midi_path: Path | None  # None for stems with no transcription (e.g. drums in v1)


# General MIDI program numbers (0-indexed) chosen as default re-synthesis voices
# for each stem family. User overrides in their DAW; these are only suggestions.
_GM_PROGRAMS: dict[str, dict[str, Any]] = {
    "vocals": {"gm_program": 53, "patch_name": "Voice Oohs",
               "suggested_patch": "any vocal sampler / SFZ vocal library"},
    "bass":   {"gm_program": 38, "patch_name": "Synth Bass 1",
               "suggested_patch": "electric bass or sub-bass synth"},
    "piano":  {"gm_program": 0,  "patch_name": "Acoustic Grand Piano",
               "suggested_patch": "your favorite piano (Pianoteq / Salamander SFZ / Kontakt)"},
    "guitar": {"gm_program": 27, "patch_name": "Electric Guitar (clean)",
               "suggested_patch": "clean electric guitar sample lib"},
    "other":  {"gm_program": 0,  "patch_name": "Acoustic Grand Piano",
               "suggested_patch": "piano placeholder — re-assign to whatever the stem sounds like"},
}

_DRUM_INFO: dict[str, Any] = {
    "gm_program": 0,
    "patch_name": "Standard Drum Kit",
    "channel": 10,
    "is_drum_channel": True,
    "suggested_patch": "your DAW's GM drum kit (FL: FPC, Ableton: Drum Rack, Logic: Drum Machine Designer)",
}


_README_TEMPLATE = """\
demixer drag-in bundle
======================

Contents:
  {m}stems/ — separated audio (drop on audio tracks)
  {m}midi/  — transcribed MIDI per pitched stem (drop on instrument tracks)
  project.json    — tempo, time signature, key, duration
  instruments.json — suggested GM program / patch per stem (paths relative to here)

Project info:
  tempo:           {tempo:.2f} BPM ({beats_per_bar}/4)
  key:             {key_root} {key_scale}
  duration:        {duration:.2f}s
  stems:           {stem_names}

FL Studio:
  1. Set tempo to {tempo:.2f} in the transport.
  2. Drag {m}stems/*.wav into the Playlist (each becomes an Audio Clip).
  3. For each {m}midi/*.mid: drag into the Playlist; FL prompts to create a new
     Channel Rack instrument. Pick the suggested patch from instruments.json,
     or any sampler that matches the stem.

Ableton Live:
  1. Set tempo to {tempo:.2f}.
  2. Drag {m}stems/*.wav onto audio tracks.
  3. Drag {m}midi/*.mid onto MIDI tracks; insert Instrument Rack or your
     preferred sampler (Sampler / Simpler) on each.

Logic Pro:
  1. Set tempo to {tempo:.2f}.
  2. File ▸ Import audio for each {m}stems/*.wav.
  3. New Software Instrument track per {m}midi/*.mid, drag the .mid in,
     pick a patch from instruments.json suggestions.
"""


def write_dragin(
    out_dir: str | Path,
    *,
    tracks: list[StemTrack],
    tempo: TempoBeats,
    key: KeyEstimate,
    duration_s: float,
    copy_media: bool = True,
) -> Path:
    """Materialize a drag-in directory bundle.

    `copy_media=True` (default) copies each stem WAV + MIDI into the dragin
    folder so it's a self-contained, movable directory — right for standalone
    library use. `copy_media=False` skips the copies and points instruments.json
    at the stems/MIDI relative to the dragin dir (`../stems`, `../midi`); the
    pipeline uses this so the .demixer archive doesn't duplicate ~hundreds of MB
    of audio that already live at the bundle top level.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    media_prefix = "" if copy_media else "../"
    stems_dir = out_dir / "stems"
    midi_dir = out_dir / "midi"
    if copy_media:
        stems_dir.mkdir(parents=True, exist_ok=True)
        midi_dir.mkdir(parents=True, exist_ok=True)

    instruments: dict[str, Any] = {}
    for t in tracks:
        if copy_media:
            dst_wav = stems_dir / f"{t.name}.wav"
            if t.wav_path.resolve() != dst_wav.resolve():
                dst_wav.write_bytes(t.wav_path.read_bytes())
            if t.midi_path is not None:
                dst_midi = midi_dir / f"{t.name}.mid"
                if t.midi_path.resolve() != dst_midi.resolve():
                    dst_midi.write_bytes(t.midi_path.read_bytes())

        stem_wav = f"{media_prefix}stems/{t.name}.wav"
        midi_rel = f"{media_prefix}midi/{t.name}.mid" if t.midi_path is not None else None
        if t.name == "drums":
            instruments[t.name] = dict(_DRUM_INFO, stem_wav=stem_wav, midi=None)
        else:
            base = _GM_PROGRAMS.get(t.name, _GM_PROGRAMS["other"])
            instruments[t.name] = dict(
                base,
                channel=1,
                is_drum_channel=False,
                stem_wav=stem_wav,
                midi=midi_rel,
            )

    (out_dir / "instruments.json").write_text(json.dumps(instruments, indent=2))
    (out_dir / "project.json").write_text(json.dumps({
        "tempo_bpm": tempo.tempo_bpm,
        "beats_per_bar": tempo.beats_per_bar,
        "tempo_method": tempo.method,
        "key": {"root": key.root, "scale": key.scale, "sharps": key.sharps, "strength": key.strength},
        "duration_s": duration_s,
    }, indent=2))
    (out_dir / "README.txt").write_text(_README_TEMPLATE.format(
        m=media_prefix,
        tempo=tempo.tempo_bpm,
        beats_per_bar=tempo.beats_per_bar,
        key_root=key.root,
        key_scale=key.scale,
        duration=duration_s,
        stem_names=", ".join(t.name for t in tracks),
    ))

    return out_dir
