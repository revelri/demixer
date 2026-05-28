"""Unit tests for core.loop_detect.

Synthesizes loop / non-loop audio in-process — no Demucs, no ML.
"""

from __future__ import annotations

import numpy as np

from demixer.core.loop_detect import detect_loop_period


def _synth_loop(period_s: float, n_repeats: int, sr: int = 44100,
                seed: int = 0) -> np.ndarray:
    """One period of bandlimited noise, tiled `n_repeats` times → (channels, n)."""
    rng = np.random.default_rng(seed)
    n = int(period_s * sr)
    one = rng.standard_normal((2, n)).astype(np.float32) * 0.2
    return np.tile(one, (1, n_repeats))


def test_detects_clean_loop() -> None:
    sr = 44100
    audio = _synth_loop(period_s=2.0, n_repeats=5, sr=sr)
    det = detect_loop_period(audio, sr)
    assert det is not None
    assert abs(det.period_s - 2.0) < 0.02
    assert det.n_repeats == 5
    assert det.confidence > 0.99


def test_rejects_non_loop_noise() -> None:
    sr = 44100
    rng = np.random.default_rng(123)
    audio = rng.standard_normal((2, sr * 5)).astype(np.float32) * 0.2
    assert detect_loop_period(audio, sr) is None


def test_rejects_silence() -> None:
    sr = 44100
    audio = np.zeros((2, sr * 4), dtype=np.float32)
    assert detect_loop_period(audio, sr) is None


def test_rejects_uneven_residual() -> None:
    """A loop that doesn't divide the input length cleanly is not a loop."""
    sr = 44100
    audio = _synth_loop(period_s=1.5, n_repeats=3, sr=sr)
    # Append 1 s of unrelated noise so the period no longer divides the length
    rng = np.random.default_rng(7)
    tail = rng.standard_normal((2, sr)).astype(np.float32) * 0.2
    audio = np.concatenate([audio, tail], axis=1)
    assert detect_loop_period(audio, sr) is None


def test_mono_input_accepted() -> None:
    sr = 44100
    stereo = _synth_loop(period_s=1.5, n_repeats=4, sr=sr)
    det = detect_loop_period(stereo.mean(axis=0), sr)
    assert det is not None
    assert abs(det.period_s - 1.5) < 0.02


def test_rejects_period_below_minimum() -> None:
    """A 0.5 s pattern repeats inside the search range but falls under the
    1.5 s minimum and must not be reported as a loop."""
    sr = 44100
    audio = _synth_loop(period_s=0.5, n_repeats=20, sr=sr)
    assert detect_loop_period(audio, sr, min_period_s=1.5) is None
