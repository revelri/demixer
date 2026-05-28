"""FL Studio project (`.flp`) writer + piano-roll script generator.

FL Studio's `.flp` is proprietary and binary; there is no published spec and no
permissively-licensed library that can *create* one from scratch (PyFLP can only
mutate existing files). So we emit the bytes directly. The format is two RIFF-like
chunks — an `FLhd` header and an `FLdt` event stream — where each event is
type-length-value with the value size implied by the ID byte's range:

    0-63    BYTE   (1-byte value)
    64-127  WORD   (uint16 LE)
    128-191 DWORD  (uint32 LE)
    192-207 TEXT   (varint length + UTF-16-LE string, ASCII for the version)
    208-255 DATA   (varint length + raw bytes: structs / struct arrays)

The event IDs, struct layouts, and required ordering below were reverse-engineered
from PyFLP 2.2.1's source (the de-facto reference parser); we round-trip our output
back through PyFLP in the test suite to confirm validity without needing FL Studio.

We emit a full project: tempo + key, one Sampler channel per audio stem pointing at
its WAV, a silent Sampler channel hosting each transcribed stem's piano-roll pattern,
and an arrangement placing every stem (audio clips) and pattern (pattern clips) on the
playlist. `write_flpianoroll_scripts` additionally emits FL Studio 21+ `.pyscript`
files for the supported in-DAW note-injection path (Piano roll ▸ Tools ▸ Scripting).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pretty_midi

from demixer.core.analysis.key import KeyEstimate
from demixer.core.analysis.tempo_beats import TempoBeats

PPQ = 96  # ticks per quarter note; must be one of FL's VALID_PPQS (96 is the classic default)
FL_VERSION = "21.0.3.3517"  # sets the TEXT codec (UTF-16 for >= 11.5) and version gates
FL_BUILD = 3517
_MAX_TRACKS = 500           # FL >= 12.9.1 expects exactly this many playlist tracks
_MAX_TRACK_IDX = 499        # track indices are stored reversed: rvidx = _MAX_TRACK_IDX - idx
_PATTERN_BASE = 20480       # playlist item_index offset distinguishing pattern vs channel clips

# --- event IDs (see module docstring for the ID-range → value-size mapping) ---
_TIMESIG_NUM = 17    # BYTE
_TIMESIG_BEAT = 18   # BYTE
_CHAN_TYPE = 21      # BYTE  (0 = Sampler)
_CHAN_NEW = 64       # WORD  channel iid; hard delimiter for a channel's events
_CHAN_GROUPNUM = 145  # DWORD (i32) channel's display-group index
_DISPLAYGROUP_NAME = 231  # TEXT  names a channel-rack display group
_PAT_NEW = 65        # WORD  pattern iid
_ARR_CURRENT = 100   # WORD  selected arrangement; also closes the arrangement block
_ARR_NEW = 99        # WORD  arrangement iid (1-based)
_PROJ_TEMPO = 156    # DWORD round(bpm * 1000)
_FL_BUILD_ID = 159   # DWORD
_CHAN_NAME = 192     # TEXT
_PROJ_COMMENTS = 195  # TEXT
_CHAN_SAMPLEPATH = 196  # TEXT  absolute path to the .wav
_FL_VERSION_ID = 199    # TEXT  ASCII; must come first
_PAT_NOTES = 224     # DATA  array of 24-byte note structs
_PLAYLIST = 233      # DATA  array of 32-byte playlist-item structs
_TRACK_DATA = 238    # DATA  66-byte track struct
_ARR_NAME = 241      # TEXT

_CHANNEL_TYPE_SAMPLER = 0
_DEFAULT_TRACK_COLOR = 0x485156


@dataclass(frozen=True)
class StemTrack:
    name: str
    wav_path: Path
    midi_path: Path | None  # None for stems with no transcription (e.g. drums)


class _EventStream:
    """Accumulates FLdt events, emitting the correct wire size per ID range."""

    def __init__(self) -> None:
        self._buf = BytesIO()

    def byte(self, event_id: int, value: int) -> None:
        self._buf.write(struct.pack("<BB", event_id, value & 0xFF))

    def word(self, event_id: int, value: int) -> None:
        self._buf.write(struct.pack("<BH", event_id, value & 0xFFFF))

    def dword(self, event_id: int, value: int) -> None:
        self._buf.write(struct.pack("<BI", event_id, value & 0xFFFFFFFF))

    def text(self, event_id: int, value: str, *, ascii_codec: bool = False) -> None:
        if ascii_codec:
            payload = value.encode("latin-1", "replace") + b"\x00"
        else:
            payload = value.encode("utf-16-le") + b"\x00\x00"
        self.data(event_id, payload)

    def data(self, event_id: int, payload: bytes) -> None:
        self._buf.write(bytes([event_id]))
        self._buf.write(_varint(len(payload)))
        self._buf.write(payload)

    def getvalue(self) -> bytes:
        return self._buf.getvalue()


def _varint(n: int) -> bytes:
    """LEB128 (base-128, low 7 bits first, high bit = continuation)."""
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | 0x80 if n else b)
        if not n:
            return bytes(out)


def _sec_to_ticks(seconds: float, bpm: float) -> int:
    return max(0, round(seconds * (bpm / 60.0) * PPQ))


def _note_struct(
    *, position: int, length: int, key: int, rack_channel: int, velocity: int
) -> bytes:
    # 24 bytes: position, flags, rack_channel, length, key, group, fine, _u1,
    # release, midi_ch, pan, velocity, mod_x, mod_y.
    return struct.pack(
        "<IHHIHHBBBBBBBB",
        position, 0, rack_channel & 0xFFFF, length,
        key & 0xFFFF, 0,            # key, group
        120, 0,                     # fine_pitch (120 = no detune), _u1
        64, 0,                      # release, midi_channel
        64, min(velocity, 128),     # pan (centre), velocity
        64, 64,                     # mod_x, mod_y
    )


def _playlist_item(*, position: int, item_index: int, length: int, track_idx: int) -> bytes:
    # 32-byte (pre-FL21) item; modern FL reads it fine. item_index <= _PATTERN_BASE
    # is a channel (audio) clip; > _PATTERN_BASE is a pattern clip.
    track_rvidx = _MAX_TRACK_IDX - track_idx
    return struct.pack(
        "<IHHIHHBBHBBBBff",
        position, _PATTERN_BASE, item_index & 0xFFFF, length,
        track_rvidx & 0xFFFF, 0,    # track_rvidx, group
        120, 0,                     # _u1
        0x0040,                     # item_flags
        64, 100, 128, 128,          # _u2
        0.0, 0.0,                   # start_offset, end_offset
    )


def _track_struct(iid: int) -> bytes:
    # 66-byte track struct (FL 20.9.1 layout), unused tail zero-padded.
    head = struct.pack(
        "<IIIBfiBIIIIIIBB",
        iid, _DEFAULT_TRACK_COLOR, 0,  # iid, color, icon
        1, 1.0, 0, 0,                  # enabled, height, locked_height, content_locked
        0, 0, 0, 0, 0, 0,              # motion, press, trigger_sync, queued, tolerant, pos_sync
        0, 0,                          # grouped, locked
    )
    return head + b"\x00" * (66 - len(head))


def _notes_payload(midi_path: Path, rack_channel: int, bpm: float) -> bytes:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    out = bytearray()
    for instrument in midi.instruments:
        for n in instrument.notes:
            position = _sec_to_ticks(n.start, bpm)
            length = max(1, _sec_to_ticks(n.end, bpm) - position)
            out += _note_struct(
                position=position, length=length, key=int(n.pitch),
                rack_channel=rack_channel, velocity=int(n.velocity),
            )
    return bytes(out)


def write_flp(
    out_path: str | Path,
    *,
    tracks: list[StemTrack],
    tempo: TempoBeats,
    key: KeyEstimate,
    duration_s: float,
    project_name: str = "demixer",
) -> Path:
    """Write an FL Studio project referencing the given stems and MIDI."""
    out_path = Path(out_path)
    if out_path.suffix != ".flp":
        out_path = out_path.with_suffix(".flp")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bpm = tempo.tempo_bpm
    duration_ticks = _sec_to_ticks(duration_s, bpm)

    es = _EventStream()

    # 1. Project — version must come first (sets the TEXT codec for later events).
    es.text(_FL_VERSION_ID, FL_VERSION, ascii_codec=True)
    es.dword(_FL_BUILD_ID, FL_BUILD)
    es.dword(_PROJ_TEMPO, round(bpm * 1000))
    es.text(_PROJ_COMMENTS,
            f"{project_name} — key: {key.root} {key.scale} (sharps {key.sharps:+d})")
    es.byte(_TIMESIG_NUM, tempo.beats_per_bar)
    es.byte(_TIMESIG_BEAT, 4)

    # One channel-rack display group ("demixer") that every channel belongs to.
    es.text(_DISPLAYGROUP_NAME, project_name)

    # 2. Channel rack. Each stem gets a Sampler channel pointing at its WAV; each
    #    transcribed stem also gets a silent Sampler channel hosting its notes.
    #    We record the iid of every channel so playlist items can reference them.
    channel_count = 0
    audio_iid: dict[str, int] = {}
    notes_iid: dict[str, int] = {}

    def _new_channel(name: str, sample_path: Path | None) -> int:
        nonlocal channel_count
        iid = channel_count
        es.word(_CHAN_NEW, iid)
        es.byte(_CHAN_TYPE, _CHANNEL_TYPE_SAMPLER)
        es.dword(_CHAN_GROUPNUM, 0)  # index into the single display group above
        es.text(_CHAN_NAME, name)
        if sample_path is not None:
            es.text(_CHAN_SAMPLEPATH, str(sample_path.resolve()))
        channel_count += 1
        return iid

    for t in tracks:
        audio_iid[t.name] = _new_channel(t.name, t.wav_path)
        if t.midi_path is not None:
            notes_iid[t.name] = _new_channel(f"{t.name} notes", None)

    # 3. Patterns — one per transcribed stem, notes routed to its notes channel.
    pattern_iid: dict[str, int] = {}
    next_pat = 1  # FL patterns are 1-based
    for t in tracks:
        if t.midi_path is None:
            continue
        es.word(_PAT_NEW, next_pat)
        es.data(_PAT_NOTES, _notes_payload(t.midi_path, notes_iid[t.name], bpm))
        pattern_iid[t.name] = next_pat
        next_pat += 1

    # 4. Arrangement — 500 tracks (FL requirement) and a playlist placing each
    #    audio clip and pattern clip on its own track at t=0.
    es.word(_ARR_NEW, 1)
    es.text(_ARR_NAME, project_name)
    for i in range(_MAX_TRACKS):
        es.data(_TRACK_DATA, _track_struct(i + 1))

    items = bytearray()
    track_idx = 0
    for t in tracks:
        items += _playlist_item(
            position=0, item_index=audio_iid[t.name],
            length=duration_ticks, track_idx=track_idx,
        )
        track_idx += 1
    for t in tracks:
        if t.midi_path is None:
            continue
        items += _playlist_item(
            position=0, item_index=_PATTERN_BASE + pattern_iid[t.name],
            length=duration_ticks, track_idx=track_idx,
        )
        track_idx += 1
    es.data(_PLAYLIST, bytes(items))
    es.word(_ARR_CURRENT, 0)

    events = es.getvalue()

    with out_path.open("wb") as fp:
        # FLhd: magic, size=6, format=0 (full song), channel_count, ppq
        fp.write(b"FLhd")
        fp.write(struct.pack("<IhHH", 6, 0, channel_count, PPQ))
        # FLdt: magic, event-stream byte length, then the stream
        fp.write(b"FLdt")
        fp.write(struct.pack("<I", len(events)))
        fp.write(events)

    return out_path


_PYSCRIPT_TEMPLATE = '''\
"""FL Studio piano-roll script — generated by demixer for stem "{stem}".

Open the target pattern in the Piano roll, then run via Tools ▸ Scripting.
Replaces the current piano-roll contents with demixer's transcribed notes.
Times are stored in beats and scaled by the project's live PPQ, so the script
is timebase-independent.
"""

import flpianoroll as flp

# (midi_note, start_beats, length_beats, velocity_0_1)
NOTES = [
{notes}
]

TEMPO_BPM = {bpm:.4f}
KEY = "{key}"


def apply(score):
    ppq = score.PPQ
    score.clearNotes()
    for number, start_beats, length_beats, velocity in NOTES:
        n = flp.Note()
        n.number = number
        n.time = round(start_beats * ppq)
        n.length = max(1, round(length_beats * ppq))
        n.velocity = velocity
        score.addNote(n)
    marker = flp.Marker()
    marker.time = 0
    marker.name = "demixer: %.2f BPM, key %s" % (TEMPO_BPM, KEY)
    score.addMarker(marker)


apply(flp.score)
'''


def write_flpianoroll_scripts(
    out_dir: str | Path,
    *,
    tracks: list[StemTrack],
    tempo: TempoBeats,
    key: KeyEstimate,
) -> list[Path]:
    """Emit one `<stem>.pyscript` per transcribed stem (FL Studio 21+ piano-roll API)."""
    out_dir = Path(out_dir)
    bpm = tempo.tempo_bpm
    key_label = f"{key.root} {key.scale}"
    written: list[Path] = []

    for t in tracks:
        if t.midi_path is None:
            continue
        midi = pretty_midi.PrettyMIDI(str(t.midi_path))
        rows: list[str] = []
        for instrument in midi.instruments:
            for n in instrument.notes:
                start_beats = n.start * (bpm / 60.0)
                length_beats = max(n.end - n.start, 1e-4) * (bpm / 60.0)
                rows.append(
                    f"    ({int(n.pitch)}, {start_beats:.6f}, "
                    f"{length_beats:.6f}, {n.velocity / 127.0:.4f}),"
                )
        script = _PYSCRIPT_TEMPLATE.format(
            stem=t.name, notes="\n".join(rows), bpm=bpm, key=key_label,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{t.name}.pyscript"
        dst.write_text(script, encoding="utf-8")
        written.append(dst)

    return written
