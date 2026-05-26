"""Smoke tests for core.ingest using a synthesized fixture (no checked-in audio)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from demixer.core.ingest import TARGET_CHANNELS, TARGET_SR, ingest, write_wav


def _synth_sine_wav(path: Path, seconds: float = 2.0, freq: float = 440.0, sr: int = 48000) -> None:
    t = np.arange(int(seconds * sr)) / sr
    # Quiet sine on left, slightly louder on right — gives loudness normalization
    # something to do and keeps stereo distinct.
    left = 0.10 * np.sin(2 * np.pi * freq * t)
    right = 0.20 * np.sin(2 * np.pi * freq * t)
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    sf.write(path, stereo, sr, subtype="FLOAT")


def _have_ffmpeg() -> bool:
    return subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not available")
def test_ingest_resamples_and_normalizes(tmp_path: Path) -> None:
    src = tmp_path / "sine.wav"
    _synth_sine_wav(src, seconds=2.0, freq=440.0, sr=48000)

    audio = ingest(src)

    assert audio.sample_rate == TARGET_SR
    assert audio.samples.shape[0] == TARGET_CHANNELS
    assert audio.samples.dtype == np.float32
    assert 1.9 < audio.duration_s < 2.1
    # 64-char hex digest
    assert len(audio.sha256) == 64
    # Loudness moved toward -23 LUFS after normalization
    assert abs(audio.integrated_lufs_after - (-23.0)) < 0.5


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not available")
def test_ingest_is_deterministic(tmp_path: Path) -> None:
    src = tmp_path / "sine.wav"
    _synth_sine_wav(src)
    a = ingest(src)
    b = ingest(src)
    assert a.sha256 == b.sha256


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not available")
def test_write_wav_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "sine.wav"
    _synth_sine_wav(src)
    audio = ingest(src)
    out = write_wav(audio, tmp_path / "out.wav")
    assert out.exists() and out.stat().st_size > 0
    re_read, sr = sf.read(out, always_2d=True)
    assert sr == TARGET_SR
    assert re_read.shape[1] == TARGET_CHANNELS


def test_ingest_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ingest(tmp_path / "does-not-exist.wav")


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not available")
def test_ingest_empty_audio_raises_clear_error(tmp_path: Path) -> None:
    # A zero-length WAV (e.g. a trim seek past end-of-file) must raise a clear
    # error, not a cryptic pyloudnorm failure deep in the stack.
    src = tmp_path / "empty.wav"
    sf.write(src, np.zeros((0, 2), dtype=np.float32), 48000, subtype="FLOAT")
    with pytest.raises(ValueError, match="empty"):
        ingest(src)


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not available")
def test_ingest_short_clip_skips_loudness_norm(tmp_path: Path) -> None:
    # Clips shorter than the EBU R128 block size (0.4s) can't be loudness-measured;
    # ingest must pass them through (NaN loudness) rather than crash.
    src = tmp_path / "short.wav"
    _synth_sine_wav(src, seconds=0.2, sr=48000)
    audio = ingest(src)
    assert audio.samples.shape[0] == TARGET_CHANNELS
    assert audio.samples.shape[1] > 0
    assert np.isnan(audio.integrated_lufs_before)
    assert np.isnan(audio.integrated_lufs_after)
    assert len(audio.sha256) == 64


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not available")
def test_ingest_pure_silence_no_nan(tmp_path: Path) -> None:
    # Silence measures -inf LUFS; normalizing would give infinite gain → NaN.
    # ingest must pass silence through as finite zeros, not poison downstream.
    src = tmp_path / "silence.wav"
    sf.write(src, np.zeros((44_100, 2), dtype=np.float32), 44_100, subtype="FLOAT")
    audio = ingest(src)
    assert np.all(np.isfinite(audio.samples))
    assert not np.any(np.isnan(audio.samples))
    assert float(np.max(np.abs(audio.samples))) == 0.0
