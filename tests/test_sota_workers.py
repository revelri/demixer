"""Guard tests for the isolated SOTA worker clients (mt3, btc).

Actual inference runs in separate venvs and is heavy (gated on DEMIXER_RUN_HEAVY
elsewhere); here we check the worker-script presence and the availability /
graceful-failure contract without invoking the models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from demixer.core.analysis import chords_btc
from demixer.core.ingest import IngestedAudio
from demixer.core.transcription import mt3
from demixer.core.workers import Worker, WorkerUnavailableError


def _fake_audio() -> IngestedAudio:
    return IngestedAudio(
        samples=np.zeros((2, 44_100), dtype=np.float32),
        sample_rate=44_100, duration_s=1.0, source_path=Path("/dev/null"),
        sha256="0" * 64, integrated_lufs_before=-20.0, integrated_lufs_after=-23.0,
    )


def test_worker_scripts_ship_with_repo() -> None:
    # Worker scripts must exist even when their venvs aren't built.
    assert Worker("mt3").script.is_file()
    assert Worker("btc").script.is_file()


def test_mt3_client_raises_when_worker_missing(monkeypatch, tmp_path: Path) -> None:
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"x")
    # Worker is a frozen dataclass; patch the method at the class level.
    monkeypatch.setattr(Worker, "available", lambda self: False)
    with pytest.raises(WorkerUnavailableError):
        mt3.transcribe_mt3(wav)


def test_btc_client_raises_when_worker_missing(monkeypatch) -> None:
    monkeypatch.setattr(Worker, "available", lambda self: False)
    with pytest.raises(WorkerUnavailableError):
        chords_btc.estimate(_fake_audio())


def test_availability_reflects_venv_presence() -> None:
    assert mt3.available() == Worker("mt3").available()
    assert chords_btc.available() == Worker("btc").available()
