"""Global key estimation via Essentia's KeyExtractor (Krumhansl + Temperley profiles).

Essentia outperforms librosa for tonal pieces in our internal listening tests.
KeyExtractor returns root + scale + strength; we expose those plus a derived
MIDI sharps count (negative = flats) suitable for music21 KeySignature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from demixer.core.ingest import IngestedAudio

# Number of sharps (+) or flats (−) for each major / minor key on the circle of fifths.
# music21.key.KeySignature expects this integer directly.
_MAJOR_SHARPS = {
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
    "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7,
}
_MINOR_SHARPS = {
    "A": 0, "E": 1, "B": 2, "F#": 3, "C#": 4, "G#": 5, "D#": 6, "A#": 7,
    "D": -1, "G": -2, "C": -3, "F": -4, "Bb": -5, "Eb": -6, "Ab": -7,
}


@dataclass(frozen=True)
class KeyEstimate:
    root: str          # e.g. "C", "F#", "Bb"
    scale: str         # "major" | "minor"
    strength: float    # 0..1 confidence
    sharps: int        # signed: positive = sharps, negative = flats


def _sharps_for(root: str, scale: str) -> int:
    table = _MAJOR_SHARPS if scale == "major" else _MINOR_SHARPS
    if root not in table:
        raise ValueError(f"unknown root {root!r} for scale {scale!r}")
    return table[root]


def _prefer_fewer_accidentals(root: str, scale: str, sharps: int) -> tuple[str, int]:
    """Respell a 7-accidental key to its 5-accidental enharmonic equivalent.

    Essentia returns e.g. C# major (+7 sharps); Db major (−5 flats) is the same
    pitch set with two fewer accidentals and is the conventional spelling — and
    it keeps the engraved key signature readable (7 sharps is brutal). The
    enharmonic flips the circle-of-fifths position by 12 (±7 → ∓5). Keys with
    ≤6 accidentals are left untouched (6♯/6♭ is a genuine tie).
    """
    if abs(sharps) != 7:
        return root, sharps
    target = sharps - 12 if sharps > 0 else sharps + 12  # +7→−5, −7→+5
    table = _MAJOR_SHARPS if scale == "major" else _MINOR_SHARPS
    for name, s in table.items():
        if s == target:
            return name, target
    return root, sharps  # no equivalent found (shouldn't happen) — keep original


def harmonic_mono(stems: dict[str, np.ndarray]) -> np.ndarray:
    """Sum all non-drum stems to a mono signal for key/harmony estimation.

    Drums are inharmonic percussion that smear Essentia's chroma; excluding them
    sharpens key detection. A/B eval (2026-05-24) on the drums-excluded mix fixed
    ABBA "Super Trouper" (full-mix → G major, the dominant; drums-excluded → the
    correct C major), kept Billie Jean correct (F# minor), and gave a more
    plausible E minor on Metallica "One"'s distorted outro (vs full-mix A minor).
    """
    non_drum = [s for name, s in stems.items() if name != "drums"]
    if not non_drum:
        return np.zeros(1, dtype=np.float32)
    mix = np.sum(non_drum, axis=0)
    return np.asarray(mix).mean(axis=0).astype(np.float32, copy=False)


def estimate(audio: IngestedAudio, *, mono: np.ndarray | None = None) -> KeyEstimate:
    """Estimate the global key.

    By default uses the full-mix mono. Pass `mono` (e.g. from `harmonic_mono`)
    to estimate on a drums-excluded stem mix, which is more accurate — see
    `harmonic_mono`. Falls back to the full mix when stems aren't available.
    """
    # Imported lazily — essentia init logs a bunch of stuff.
    import essentia.standard as es

    if mono is None:
        mono = audio.samples.mean(axis=0).astype(np.float32, copy=False)
    # KeyExtractor expects a 1D float32 array at its configured sample rate (default 44100).
    extractor = es.KeyExtractor(sampleRate=audio.sample_rate)
    root, scale, strength = extractor(mono)
    sharps = _sharps_for(root, scale)
    root, sharps = _prefer_fewer_accidentals(root, scale, sharps)
    return KeyEstimate(root=root, scale=scale, strength=float(strength), sharps=sharps)
