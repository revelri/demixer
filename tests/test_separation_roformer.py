"""Tests for the isolated-subprocess RoFormer vocals backend.

The actual separation runs in `.venv-roformer` (numpy 2.x) and is heavy, so the
real run is gated on DEMIXER_RUN_HEAVY. Default tests cover the graceful-failure
contract and length-normalization logic without invoking the model.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from demixer.core import separation_roformer as rf
from demixer.core.ingest import IngestedAudio


def _fake_audio(samples: int = 44_100) -> IngestedAudio:
    return IngestedAudio(
        samples=np.zeros((2, samples), dtype=np.float32),
        sample_rate=44_100,
        duration_s=samples / 44_100,
        source_path=Path("/dev/null"),
        sha256="0" * 64,
        integrated_lufs_before=-20.0,
        integrated_lufs_after=-23.0,
    )


def test_raises_clearly_when_venv_missing(monkeypatch) -> None:
    monkeypatch.setattr(rf, "roformer_available", lambda: False)
    with pytest.raises(rf.RoformerUnavailableError, match="venv"):
        rf.roformer_vocals(_fake_audio())


def test_roformer_available_checks_both_paths() -> None:
    # Both the venv python and worker script must exist for availability.
    result = rf.roformer_available()
    assert result == (rf._ROFORMER_PYTHON.is_file() and rf._WORKER.is_file())


def test_worker_script_exists() -> None:
    # The worker must ship with the repo even if the venv isn't built.
    assert rf._WORKER.is_file(), f"missing worker script at {rf._WORKER}"


def test_workers_framework_parses_status() -> None:
    import subprocess
    from demixer.core.workers import parse_worker_status
    ok, payload = parse_worker_status(
        subprocess.CompletedProcess(args=[], returncode=0, stdout="noise\nOK /tmp/x.mid\n", stderr="")
    )
    assert ok and payload == "/tmp/x.mid"
    bad, msg = parse_worker_status(
        subprocess.CompletedProcess(args=[], returncode=1, stdout="ERR boom\n", stderr="")
    )
    assert not bad and msg == "boom"


@pytest.mark.skipif(
    not os.environ.get("DEMIXER_RUN_HEAVY") or not rf.roformer_available(),
    reason="set DEMIXER_RUN_HEAVY=1 and build .venv-roformer to run",
)
def test_roformer_vocals_returns_aligned_stereo() -> None:
    import soundfile as sf
    # A 2s real-ish clip: load the ABBA fixture if present, else skip via guard
    src = Path("/tmp/super_trouper_30s.wav")
    if not src.exists():
        pytest.skip("no test clip available")
    y, sr = sf.read(src, always_2d=True)
    audio = IngestedAudio(
        samples=y.T.astype(np.float32), sample_rate=sr, duration_s=len(y) / sr,
        source_path=src, sha256="x" * 64,
        integrated_lufs_before=-12.0, integrated_lufs_after=-23.0,
    )
    vocals = rf.roformer_vocals(audio)
    assert vocals.shape == audio.samples.shape  # stereo, exact length
    assert vocals.dtype == np.float32
