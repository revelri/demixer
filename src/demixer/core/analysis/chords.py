"""Chord recognition via autochord (NNLS-Chroma + BiLSTM-CRF).

autochord ships a pretrained model and wraps the standard chord-recognition
pipeline (chroma → sequence model → segmentation). Real-music test on ABBA's
Super Trouper returned a clean diatonic-to-C-major progression
(G→C→Em→Dm→F→G→C…), matching the song.

API:    autochord.recognize(wav_path) -> list[(start_s, end_s, "ROOT:quality")]
Quality strings observed: "maj", "min", "N" (no-chord). We translate to the
short form ("C", "Em", "N") for human-readable chord charts.

Limitations vs. madmom's CNN+CRF: autochord's vocabulary is mostly triads
(no 7ths, no sus, no slash chords). Good enough for v1; can be revisited.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace

import soundfile as sf

from demixer.core.analysis.key import KeyEstimate
from demixer.core.ingest import IngestedAudio


@dataclass(frozen=True)
class ChordSegment:
    start_s: float
    end_s: float
    label: str   # short form: "C", "Em", "G", "N" (no chord)


# --- Enharmonic respelling to match the key signature -----------------------
# BTC/autochord emit sharp-spelled roots regardless of key, so flat keys read
# with clashing accidentals (Cm shown as D#/C#/A# instead of Eb/Db/Bb). We
# respell a chord root ONLY when its pitch class is diatonic to the detected
# key — then the spelling is unambiguously key-determined. Chromatic roots
# (pitch class not in the key's scale) are left exactly as the model emitted
# them, since their correct spelling depends on harmonic function we don't know.

_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SHARP_ORDER = ("F", "C", "G", "D", "A", "E", "B")
_FLAT_ORDER = ("B", "E", "A", "D", "G", "C", "F")


def _diatonic_pc_to_name(sharps: int) -> dict[int, str]:
    """Map each of the 7 diatonic pitch classes to its key-correct spelling.

    `sharps` is the signed key-signature count (same for a minor key as its
    relative major), which fully determines the diatonic note spellings.
    """
    acc = {letter: "" for letter in _LETTER_PC}
    if sharps > 0:
        for letter in _SHARP_ORDER[:sharps]:
            acc[letter] = "#"
    elif sharps < 0:
        for letter in _FLAT_ORDER[: abs(sharps)]:
            acc[letter] = "b"
    pc_to_name: dict[int, str] = {}
    for letter, base in _LETTER_PC.items():
        shift = 1 if acc[letter] == "#" else -1 if acc[letter] == "b" else 0
        pc_to_name[(base + shift) % 12] = letter + acc[letter]
    return pc_to_name


def _root_pc(root: str) -> int:
    pc = _LETTER_PC[root[0]]
    if len(root) > 1 and root[1] == "#":
        pc += 1
    elif len(root) > 1 and root[1] in ("b", "♭"):
        pc -= 1
    return pc % 12


def _respell_label(label: str, pc_to_name: dict[int, str]) -> str:
    if not label or label[0] not in "ABCDEFG":
        return label  # "N" (no chord) or anything non-pitched
    root_len = 2 if len(label) > 1 and label[1] in "#b♭" else 1
    root, suffix = label[:root_len], label[root_len:]
    pc = _root_pc(root)
    key_name = pc_to_name.get(pc)
    if key_name is not None and key_name != root:
        return key_name + suffix
    return label


def respell_to_key(chords: list[ChordSegment], key: KeyEstimate) -> list[ChordSegment]:
    """Respell diatonic chord roots to match the key signature's accidentals."""
    pc_to_name = _diatonic_pc_to_name(key.sharps)
    return [replace(c, label=_respell_label(c.label, pc_to_name)) for c in chords]


def _short_label(autochord_label: str) -> str:
    """Convert 'C:maj' -> 'C', 'E:min' -> 'Em', 'N' -> 'N' (no chord)."""
    if ":" not in autochord_label:
        return autochord_label  # "N" or unrecognized — pass through
    root, quality = autochord_label.split(":", 1)
    if quality == "maj":
        return root
    if quality == "min":
        return root + "m"
    return f"{root}{quality}"  # 7, sus, etc — preserve verbatim for future models


def estimate(audio: IngestedAudio) -> list[ChordSegment]:
    """Run autochord on the audio and return a list of chord segments."""
    import autochord  # heavy: pulls TF + the model on first call

    # autochord wants a path; we already have float32 stereo @ 44.1k in memory.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, audio.samples.T, audio.sample_rate, subtype="FLOAT")
        raw = autochord.recognize(f.name)

    return [
        ChordSegment(start_s=float(s), end_s=float(e), label=_short_label(label))
        for s, e, label in raw
    ]
