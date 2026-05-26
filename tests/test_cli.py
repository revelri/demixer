"""Tests for CLI helpers that don't require running the full pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from demixer.cli.__main__ import _SILENT_STEM_PEAK, _stem_is_silent


def _write(path: Path, peak: float, sr: int = 44_100, dur_s: float = 0.5) -> Path:
    t = np.arange(int(dur_s * sr)) / sr
    y = (peak * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(path, np.stack([y, y], axis=1), sr, subtype="FLOAT")
    return path


def test_silent_stem_detected(tmp_path: Path) -> None:
    # An emptied Demucs stem (noise floor only) reads as silent.
    p = _write(tmp_path / "silent.wav", peak=_SILENT_STEM_PEAK / 5)
    assert _stem_is_silent(p) is True


def test_present_stem_not_silent(tmp_path: Path) -> None:
    p = _write(tmp_path / "loud.wav", peak=0.3)
    assert _stem_is_silent(p) is False


def test_threshold_boundary(tmp_path: Path) -> None:
    # Just above threshold → present; just below → silent.
    assert _stem_is_silent(_write(tmp_path / "above.wav", peak=_SILENT_STEM_PEAK * 2)) is False
    assert _stem_is_silent(_write(tmp_path / "below.wav", peak=_SILENT_STEM_PEAK / 2)) is True
