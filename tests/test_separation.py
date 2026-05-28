"""Structural tests for core.separation.

A real end-to-end Demucs run downloads ~80MB of model weights and takes 30+s on
CPU even for a few seconds of audio, so the heavy test is gated behind the
DEMIXER_RUN_HEAVY env var. Structural tests run by default.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from demixer.core.ingest import ingest
from demixer.core.separation import (
    STEM_NAMES,
    SeparationResult,
    separate,
    write_stems,
)


@pytest.mark.parametrize("fmt,ext,subtype", [
    ("pcm24", ".wav",  "PCM_24"),
    ("pcm16", ".wav",  "PCM_16"),
    ("float", ".wav",  "FLOAT"),
    ("flac",  ".flac", "PCM_24"),
])
def test_write_stems_respects_stem_format(tmp_path: Path, fmt, ext, subtype) -> None:
    sr = 44100
    n = sr // 2
    rng = np.random.default_rng(0)
    loud = rng.standard_normal((2, n), dtype=np.float32) * 0.3
    result = SeparationResult(
        stems={"other": loud}, sample_rate=sr, model="htdemucs", device="cpu",
    )
    written = write_stems(result, tmp_path / "stems", stem_format=fmt)
    p = written["other"]
    assert p.suffix == ext, f"{fmt}: expected {ext}, got {p.suffix}"
    info = sf.info(str(p))
    assert info.subtype == subtype


def test_write_stems_rejects_unknown_format(tmp_path: Path) -> None:
    result = SeparationResult(
        stems={"other": np.ones((2, 1024), dtype=np.float32)},
        sample_rate=44100, model="htdemucs", device="cpu",
    )
    with pytest.raises(ValueError, match="unknown stem_format"):
        write_stems(result, tmp_path / "stems", stem_format="foo")  # type: ignore[arg-type]


def test_write_stems_skips_silent_sources(tmp_path: Path) -> None:
    """Silent stems (peak < −40 dBFS) must not be written to disk — otherwise
    they balloon every downstream DAW project and the .demixer archive."""
    sr = 44100
    n = sr // 2  # 0.5 s
    rng = np.random.default_rng(0)
    loud = rng.standard_normal((2, n), dtype=np.float32) * 0.3  # well above 0.01
    silence = np.zeros((2, n), dtype=np.float32)
    near_silence = rng.standard_normal((2, n), dtype=np.float32) * 1e-5  # ~−100 dBFS
    result = SeparationResult(
        stems={"vocals": silence, "drums": near_silence,
               "bass": silence, "other": loud},
        sample_rate=sr, model="htdemucs", device="cpu",
    )
    written = write_stems(result, tmp_path / "stems")
    assert set(written) == {"other"}
    assert written["other"].exists()
    for name in ("vocals", "drums", "bass"):
        assert not (tmp_path / "stems" / f"{name}.wav").exists()


def test_stem_names_match_variants() -> None:
    assert STEM_NAMES["htdemucs"] == ("drums", "bass", "other", "vocals")
    assert len(STEM_NAMES["htdemucs_6s"]) == 6
    assert "piano" in STEM_NAMES["htdemucs_6s"]
    assert "guitar" in STEM_NAMES["htdemucs_6s"]


def test_separate_rejects_unsupported_model() -> None:
    # Construct minimal IngestedAudio-like via a tiny ingest; cheaper than mocking.
    pytest.importorskip("demucs")
    from demixer.core.ingest import IngestedAudio
    fake = IngestedAudio(
        samples=np.zeros((2, 4096), dtype=np.float32),
        sample_rate=44_100,
        duration_s=4096 / 44_100,
        source_path=Path("/dev/null"),
        sha256="0" * 64,
        integrated_lufs_before=-70.0,
        integrated_lufs_after=-23.0,
    )
    with pytest.raises(ValueError, match="unsupported model"):
        separate(fake, model_name="not-a-real-model")  # type: ignore[arg-type]


def test_separate_rejects_wrong_sample_rate() -> None:
    pytest.importorskip("demucs")
    from demixer.core.ingest import IngestedAudio
    fake = IngestedAudio(
        samples=np.zeros((2, 4096), dtype=np.float32),
        sample_rate=48_000,
        duration_s=4096 / 48_000,
        source_path=Path("/dev/null"),
        sha256="0" * 64,
        integrated_lufs_before=-70.0,
        integrated_lufs_after=-23.0,
    )
    with pytest.raises(ValueError, match="44.1 kHz"):
        separate(fake)


@pytest.mark.skipif(
    not os.environ.get("DEMIXER_RUN_HEAVY"),
    reason="set DEMIXER_RUN_HEAVY=1 to run (downloads ~80MB model, takes 30+s on CPU)",
)
def test_separate_end_to_end_small_clip(tmp_path: Path) -> None:
    # 3-second synthetic stereo: sine + noise burst — gives demucs *something* to separate.
    sr = 48_000
    t = np.arange(int(3 * sr)) / sr
    sine = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    noise = 0.05 * np.random.RandomState(0).randn(len(t))
    stereo = np.stack([sine + noise, sine + noise * 0.8], axis=1).astype(np.float32)
    src = tmp_path / "synth.wav"
    sf.write(src, stereo, sr, subtype="FLOAT")

    audio = ingest(src)
    result = separate(audio, model_name="htdemucs", shifts=1)

    assert set(result.stems) == set(STEM_NAMES["htdemucs"])
    for name, samples in result.stems.items():
        assert samples.dtype == np.float32, name
        assert samples.shape[0] == 2, f"{name}: not stereo"
        assert samples.shape[1] == audio.samples.shape[1], f"{name}: length mismatch"

    written = write_stems(result, tmp_path / "stems")
    for name, path in written.items():
        assert path.exists() and path.stat().st_size > 0, name
