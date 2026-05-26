"""Reaper project (`.rpp`) writer — primary DAW export.

RPP is a plain-text node tree. Each node opens with `<NAME attrs...`, contains
indented attribute lines and/or child nodes, and closes with `>`. Reaper's
documentation: https://wiki.cockos.com/wiki/index.php/REAPER_Project_File

We emit one audio track per stem (referencing the WAV) and one MIDI track per
transcribed stem (referencing the .mid). The master gets tempo + time signature
+ a markdown note with the analysis's key.

File paths in the RPP are recorded **relative** to the .rpp file, so the project
remains portable as long as the audio/MIDI files travel alongside it.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from demixer.core.analysis.key import KeyEstimate
from demixer.core.analysis.tempo_beats import TempoBeats

REAPER_VERSION = "7.0/linux-x86_64"  # any modern Reaper accepts our subset


@dataclass(frozen=True)
class StemTrack:
    name: str
    wav_path: Path
    midi_path: Path | None  # None for stems we couldn't / didn't transcribe (e.g. drums)


def _guid() -> str:
    return "{" + str(uuid.uuid4()).upper() + "}"


def _quote(s: str) -> str:
    """Quote an RPP string value, picking a delimiter not present in the input."""
    for q in ('"', "'", "`"):
        if q not in s:
            return f"{q}{s}{q}"
    # Last resort — strip the offending chars
    return '"' + s.replace('"', "") + '"'


class _Writer:
    """Indentation-aware writer for the RPP node tree."""

    def __init__(self, fp: TextIO) -> None:
        self.fp = fp
        self._depth = 0

    def line(self, s: str) -> None:
        self.fp.write("  " * self._depth + s + "\n")

    def open(self, header: str) -> None:
        self.line("<" + header)
        self._depth += 1

    def close(self) -> None:
        self._depth -= 1
        self.line(">")


def _write_audio_track(
    w: _Writer,
    name: str,
    wav_relpath: str,
    duration_s: float,
    is_first: bool,
) -> None:
    w.open("TRACK " + _guid())
    w.line(f"NAME {_quote(name)}")
    w.line("VOLPAN 1 0 -1 -1 1")
    w.line("MUTESOLO 0 0 0")
    w.line("IPHASE 0")
    w.line(f"ISBUS {1 if is_first else 0} 0")
    w.line("BUSCOMP 0 0 0 0 0")
    w.line("SHOWINMIX 1 0.6667 0.5 1 0.5 0 0 0")

    w.open("ITEM")
    w.line("POSITION 0")
    w.line(f"LENGTH {duration_s:.6f}")
    w.line(f"NAME {_quote(name)}")
    w.line("IGUID " + _guid())
    w.line("GUID " + _guid())
    w.open("SOURCE WAVE")
    w.line(f"FILE {_quote(wav_relpath)}")
    w.close()  # SOURCE
    w.close()  # ITEM
    w.close()  # TRACK


def _write_midi_track(
    w: _Writer,
    name: str,
    midi_relpath: str,
    duration_s: float,
) -> None:
    w.open("TRACK " + _guid())
    w.line(f"NAME {_quote(name + ' MIDI')}")
    w.line("VOLPAN 1 0 -1 -1 1")
    w.line("MUTESOLO 0 0 0")
    w.line("IPHASE 0")
    w.line("ISBUS 0 0")
    w.line("BUSCOMP 0 0 0 0 0")
    w.line("SHOWINMIX 1 0.6667 0.5 1 0.5 0 0 0")

    # Reaper loads .mid as an in-project MIDI item via a FILE source node
    w.open("ITEM")
    w.line("POSITION 0")
    w.line(f"LENGTH {duration_s:.6f}")
    w.line(f"NAME {_quote(name + ' MIDI')}")
    w.line("IGUID " + _guid())
    w.line("GUID " + _guid())
    w.open("SOURCE MIDI")
    w.line(f"FILE {_quote(midi_relpath)}")
    w.close()  # SOURCE
    w.close()  # ITEM
    w.close()  # TRACK


def write_rpp(
    rpp_path: str | Path,
    *,
    tracks: list[StemTrack],
    tempo: TempoBeats,
    key: KeyEstimate,
    duration_s: float,
    project_name: str = "demixer",
) -> Path:
    """Write a Reaper project file referencing the given stems and MIDI."""
    rpp_path = Path(rpp_path).resolve()
    rpp_path.parent.mkdir(parents=True, exist_ok=True)

    with rpp_path.open("w", encoding="utf-8") as fp:
        w = _Writer(fp)
        w.open(f"REAPER_PROJECT 0.1 {_quote(REAPER_VERSION)} {int(time.time())}")
        w.line("RIPPLE 0")
        w.line("GROUPOVERRIDE 0 0 0")
        w.line("AUTOXFADE 1")
        w.line("ENVATTACH 1")
        w.line("POOLEDENVATTACH 0")
        w.line("MIXERUIFLAGS 11 48")
        w.line("PEAKGAIN 1")
        w.line("FEEDBACK 0")
        w.line("PANLAW 1")
        w.line("PROJOFFS 0 0 0")
        w.line("MAXPROJLEN 0 600")
        w.line("GRID 3199 8 1 8 1 0 0 0")
        w.line("TIMEMODE 1 5 -1 30 0 0 -1")
        w.line("VIDEO_CONFIG 0 0 256")
        w.line("PANMODE 3")
        w.line("CURSOR 0")
        w.line("ZOOM 100 0 0")
        w.line("VZOOMEX 6 0")
        w.line("USE_REC_CFG 0")
        w.line("RECMODE 1")
        w.line("SMPTESYNC 0 30 100 40 1000 300 0 0 1 0 0")
        w.line("LOOP 0")
        w.line("LOOPGRAN 0 4")
        w.line(f"RECORD_PATH {_quote('')} {_quote('')}")
        w.line("RECORD_CFG")
        w.line("APPLYFX_CFG")
        w.line(f"RENDER_FILE {_quote('')}")
        w.line(f"RENDER_PATTERN {_quote(project_name + '-render')}")
        w.line("RENDER_FMT 0 2 0")
        w.line("RENDER_1X 0")
        w.line("RENDER_RANGE 1 0 0 18 1000")
        w.line("RENDER_RESAMPLE 3 0 1")
        w.line("RENDER_ADDTOPROJ 0")
        w.line("RENDER_STEMS 0")
        w.line("RENDER_DITHER 0")
        w.line("TIMELOCKMODE 1")
        w.line("TEMPOENVLOCKMODE 1")
        w.line("ITEMMIX 0")
        w.line("DEFPITCHMODE 589824 0")
        w.line("TAKELANE 1")
        w.line("SAMPLERATE 44100 0 0")

        # Master tempo & time sig
        w.line(f"TEMPO {tempo.tempo_bpm:.4f} {tempo.beats_per_bar} 4")

        # Master track stub (Reaper auto-creates one if missing, but this is cleaner)
        w.open("MASTERPLAYSPEEDENV")
        w.line("EGUID " + _guid())
        w.line("ACT 0 -1")
        w.line("VIS 0 1 1")
        w.line("LANEHEIGHT 0 0")
        w.line("ARM 0")
        w.line("DEFSHAPE 0 -1 -1")
        w.close()

        # Marker at t=0 with the detected key
        key_label = f"key: {key.root} {key.scale} (sharps {key.sharps:+d})"
        w.line(f"MARKER 1 0 {_quote(key_label)} 0")

        # Tracks: stem audio first, then MIDI (so audio is on top in the project view)
        for i, track in enumerate(tracks):
            wav_rel = os.path.relpath(track.wav_path.resolve(), rpp_path.parent)
            _write_audio_track(w, track.name, wav_rel, duration_s, is_first=(i == 0))
        for track in tracks:
            if track.midi_path is None:
                continue
            midi_rel = os.path.relpath(track.midi_path.resolve(), rpp_path.parent)
            _write_midi_track(w, track.name, midi_rel, duration_s)

        w.close()  # REAPER_PROJECT

    return rpp_path
