"""Tests for analysis stages — tempo/beats (librosa) and key (essentia).

Both run end-to-end on a small synthesized fixture (no heavy ML downloads),
so they're not gated behind DEMIXER_RUN_HEAVY.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from demixer.core.analysis import key, tempo_beats
from demixer.core.analysis.key import _MAJOR_SHARPS, _prefer_fewer_accidentals, _sharps_for
from demixer.core.ingest import ingest


def _synth_click_track(path: Path, bpm: float, seconds: float = 6.0, sr: int = 44_100) -> None:
    """Stereo click track at `bpm` — tight onsets give librosa a clean signal."""
    n = int(seconds * sr)
    samples_per_beat = int(round(60.0 / bpm * sr))
    audio = np.zeros((n, 2), dtype=np.float32)
    # 5-ms exponentially-decaying clicks on each beat
    click_len = int(0.005 * sr)
    env = np.exp(-np.linspace(0, 6, click_len)).astype(np.float32)
    for i in range(0, n, samples_per_beat):
        end = min(i + click_len, n)
        click = env[: end - i]
        audio[i:end, 0] += click
        audio[i:end, 1] += click
    sf.write(path, audio, sr, subtype="FLOAT")


def _synth_triad(path: Path, freqs_hz: list[float], seconds: float = 4.0, sr: int = 44_100) -> None:
    """Stereo sustained triad — distinct tonal content for key extraction."""
    t = np.arange(int(seconds * sr)) / sr
    wave = np.zeros_like(t, dtype=np.float32)
    for f in freqs_hz:
        wave += 0.15 * np.sin(2 * np.pi * f * t).astype(np.float32)
    stereo = np.stack([wave, wave], axis=1)
    sf.write(path, stereo, sr, subtype="FLOAT")


# ---------- tempo_beats ----------

def test_tempo_beats_estimates_close_to_truth(tmp_path: Path) -> None:
    src = tmp_path / "click_120.wav"
    _synth_click_track(src, bpm=120.0, seconds=8.0)
    audio = ingest(src)
    result = tempo_beats.estimate(audio, beats_per_bar_hint=4)

    # Either backend (beat_this or librosa fallback) should land near 120 BPM
    assert 110 < result.tempo_bpm < 130, f"expected ~120 BPM, got {result.tempo_bpm}"
    # 8 s @ 120 BPM = 16 beats; allow generous slack for boundary effects
    # and beat_this's tendency to detect slightly more beats than librosa.
    assert 10 <= len(result.beat_times_s) <= 22
    assert result.method in {"beat_this", "librosa"}


def test_tempo_from_beats_subgrid_resolution() -> None:
    # beat_this-style 0.02s-grid beats whose mean period is 0.464s (129.3 BPM).
    # median snaps to 0.46 (130.43); the robust mean should recover ~129.3.
    from demixer.core.analysis.tempo_beats import _tempo_from_beats
    diffs = np.array([0.46, 0.48, 0.44, 0.46, 0.48, 0.46, 0.44, 0.48, 0.46])
    beats = np.concatenate([[0.0], np.cumsum(diffs)])
    bpm = _tempo_from_beats(beats)
    assert 128.0 < bpm < 131.0
    # And it must NOT be the coarse median value 130.43
    assert abs(bpm - 130.43) > 0.3


def test_tempo_from_beats_robust_to_missed_beat() -> None:
    # A dropped beat creates a ~2x gap; it must be filtered, not halve the tempo.
    from demixer.core.analysis.tempo_beats import _tempo_from_beats
    beats = np.array([0.0, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5])  # gap 1.0→2.0 (missed 1.5)
    bpm = _tempo_from_beats(beats)
    assert 115 < bpm < 125, f"expected ~120 BPM despite the gap, got {bpm}"


def test_beats_per_bar_median_robust_to_bad_intro() -> None:
    # Sturgill-shaped case: intro downbeats every 2 beats, then settled 4/4.
    # Global ratio would skew toward 3; the median must recover 4.
    from demixer.core.analysis.tempo_beats import _beats_per_bar
    beats = np.arange(0, 40) * 0.5  # 40 beats at 120 BPM
    # downbeats at beat indices: 0,2,4,6,8 (spacing 2) then 12,16,20,24,28,32,36 (spacing 4)
    db_idx = [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36]
    downbeats = beats[db_idx]
    assert _beats_per_bar(beats, downbeats) == 4


def test_beats_per_bar_clean_three_four() -> None:
    from demixer.core.analysis.tempo_beats import _beats_per_bar
    beats = np.arange(0, 30) * 0.5
    downbeats = beats[[0, 3, 6, 9, 12, 15, 18, 21, 24, 27]]  # every 3 beats
    assert _beats_per_bar(beats, downbeats) == 3


def test_beats_per_bar_defaults_to_four_without_downbeats() -> None:
    from demixer.core.analysis.tempo_beats import _beats_per_bar
    beats = np.arange(0, 10) * 0.5
    assert _beats_per_bar(beats, np.array([0.0])) == 4  # <2 downbeats → common-time


def test_tempo_beats_rejects_bad_hint(tmp_path: Path) -> None:
    src = tmp_path / "click.wav"
    _synth_click_track(src, bpm=100.0, seconds=3.0)
    audio = ingest(src)
    with pytest.raises(ValueError):
        tempo_beats.estimate(audio, beats_per_bar_hint=0)


def test_tempo_beats_falls_back_when_beat_this_unavailable(tmp_path: Path, monkeypatch) -> None:
    """If beat_this raises at import or call time, librosa picks up cleanly."""
    src = tmp_path / "click_100.wav"
    _synth_click_track(src, bpm=100.0, seconds=4.0)
    audio = ingest(src)

    def boom(*_a, **_kw):
        raise RuntimeError("simulated beat_this failure")
    monkeypatch.setattr(tempo_beats, "_estimate_beat_this", boom)

    result = tempo_beats.estimate(audio, beats_per_bar_hint=4)
    assert result.method == "librosa"
    assert 90 < result.tempo_bpm < 110


# ---------- key ----------

def test_sharps_table_consistency() -> None:
    # Sanity: C major has 0 sharps; G major has 1; F major has -1
    assert _sharps_for("C", "major") == 0
    assert _sharps_for("G", "major") == 1
    assert _sharps_for("F", "major") == -1
    assert _sharps_for("A", "minor") == 0  # A minor is relative to C major


def test_unknown_root_raises() -> None:
    with pytest.raises(ValueError, match="unknown root"):
        _sharps_for("H", "major")


def test_enharmonic_respelling_reduces_accidentals() -> None:
    # 7-sharp / 7-flat keys respell to their 5-accidental enharmonic equivalent.
    assert _prefer_fewer_accidentals("C#", "major", 7) == ("Db", -5)
    assert _prefer_fewer_accidentals("Cb", "major", -7) == ("B", 5)
    assert _prefer_fewer_accidentals("A#", "minor", 7) == ("Bb", -5)
    assert _prefer_fewer_accidentals("Ab", "minor", -7) == ("G#", 5)


def test_enharmonic_respelling_leaves_simple_keys_untouched() -> None:
    # ≤6 accidentals: no respelling (6 is a genuine tie, fewer is already minimal).
    assert _prefer_fewer_accidentals("C", "major", 0) == ("C", 0)
    assert _prefer_fewer_accidentals("F#", "major", 6) == ("F#", 6)
    assert _prefer_fewer_accidentals("G", "major", 1) == ("G", 1)
    assert _prefer_fewer_accidentals("Eb", "major", -3) == ("Eb", -3)


def test_all_majors_have_sharps_entry() -> None:
    # Smoke check — every major root we'd expect from essentia is mapped.
    for root in ["C", "G", "D", "A", "E", "B", "F#", "C#",
                 "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb"]:
        assert root in _MAJOR_SHARPS


def test_key_estimate_on_c_major_triad(tmp_path: Path) -> None:
    # C major triad: C4, E4, G4
    src = tmp_path / "c_major.wav"
    _synth_triad(src, freqs_hz=[261.63, 329.63, 392.00], seconds=4.0)
    audio = ingest(src)
    estimate = key.estimate(audio)

    # The exact root essentia returns can vary by ±a fifth on simple triads,
    # so we just confirm we got a structurally-valid result.
    assert estimate.scale in {"major", "minor"}
    assert 0.0 <= estimate.strength <= 1.0
    assert -7 <= estimate.sharps <= 7
    assert estimate.root in _MAJOR_SHARPS or estimate.root in {
        "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "D", "G", "C", "F", "Bb", "Eb", "Ab",
    }


def test_harmonic_mono_excludes_drums() -> None:
    from demixer.core.analysis.key import harmonic_mono
    # drums stem is loud; if excluded, result equals the sum of the rest.
    stems = {
        "drums":  np.ones((2, 100), dtype=np.float32) * 9.0,
        "bass":   np.ones((2, 100), dtype=np.float32) * 1.0,
        "other":  np.ones((2, 100), dtype=np.float32) * 2.0,
        "vocals": np.zeros((2, 100), dtype=np.float32),
    }
    mono = harmonic_mono(stems)
    # bass+other+vocals = 1+2+0 = 3 per sample, mono-averaged still 3; drums (9) excluded
    assert mono.shape == (100,)
    assert np.allclose(mono, 3.0), f"expected 3.0 (drums excluded), got {mono[0]}"


def test_harmonic_mono_empty_is_safe() -> None:
    from demixer.core.analysis.key import harmonic_mono
    assert harmonic_mono({"drums": np.ones((2, 10), dtype=np.float32)}).shape == (1,)
