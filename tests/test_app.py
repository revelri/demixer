"""Tests for the GUI layer. RunOptions.to_argv is pure; MainWindow construction
runs under an offscreen Qt platform. All gated on PySide6 being installed."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")


def test_run_options_to_argv() -> None:
    from demixer.app.pipeline_runner import RunOptions
    o = RunOptions(input_path=Path("/tmp/in.wav"), output_dir=Path("/tmp/out"),
                   model="htdemucs_6s", transcriber="mt3", chords="btc", drums="adtof",
                   roformer_vocals=True, harmony=True, reharmonize="smoothest")
    argv = o.to_argv()
    assert argv[1:4] == ["-m", "demixer.cli", "process"]
    for tok in ("--model", "htdemucs_6s", "--transcriber", "mt3", "--chords", "btc",
                "--drums", "adtof", "--roformer-vocals", "--harmony",
                "--reharmonize", "smoothest", "--verbose"):
        assert tok in argv


def test_run_options_minimal_argv_omits_flags() -> None:
    from demixer.app.pipeline_runner import RunOptions
    argv = RunOptions(input_path=Path("/tmp/a.wav"), output_dir=Path("/tmp/o")).to_argv()
    assert "--roformer-vocals" not in argv
    assert "--harmony" not in argv
    assert "--reharmonize" not in argv


def test_milestones_monotonic() -> None:
    from demixer.app.pipeline_runner import _MILESTONES
    pcts = [p for _, p in _MILESTONES]
    assert pcts == sorted(pcts) and all(0 < p < 100 for p in pcts)


def test_mainwindow_constructs_offscreen() -> None:
    # Construct the GUI in a FRESH subprocess. PySide6's QApplication segfaults
    # when it initializes in a process that already loaded torch/TF/essentia
    # (which earlier tests in the suite do) — in isolation it's fine, but the
    # full-suite run crashed with a signal. A clean interpreter avoids the
    # native-library conflict and keeps the assertion meaningful.
    import subprocess
    import sys

    script = (
        "import os; os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')\n"
        "from pathlib import Path\n"
        "from PySide6.QtWidgets import QApplication\n"
        "from demixer.app.gui import MainWindow\n"
        "app = QApplication.instance() or QApplication([])\n"
        "w = MainWindow()\n"
        "w._set_input(Path('/tmp/song.wav'))\n"
        "assert w._run.isEnabled(), 'Process button should enable once input is set'\n"
        "assert w._bar.value() == 0, 'progress bar should start at 0'\n"
        "w.close()\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, f"GUI construction failed:\n{proc.stderr[-2000:]}"
