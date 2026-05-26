"""Threaded pipeline runner for the GUI.

Spawns `python -m demixer.cli process …` as a subprocess and streams its log to
the UI, mapping known stage markers to a progress percentage. The heavy ML runs
in the child process, so the Qt event loop stays responsive and torch never
shares the GUI thread.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QThread, Signal

# Ordered (substring, percent) milestones. The first matching substring on a log
# line advances the bar. Monotonic — we never move backwards.
_MILESTONES: list[tuple[str, int]] = [
    ("ingesting", 4),
    ("separating stems", 12),
    ("wrote", 30),
    ("estimating tempo", 40),
    ("estimating key", 48),
    ("recognizing chords", 55),
    ("transcribe", 62),
    ("bundle dir ready", 80),
    ("reaper project", 84),
    ("dawproject", 87),
    ("drag-in bundle", 89),
    ("quantizing", 91),
    ("score:", 94),
    ("harmony:", 96),
    ("bundle archive", 99),
]


@dataclass(frozen=True)
class RunOptions:
    input_path: Path
    output_dir: Path
    model: str = "htdemucs"
    transcriber: str = "basic-pitch"
    chords: str = "autochord"
    drums: str = "spectral"
    roformer_vocals: bool = False
    harmony: bool = False
    reharmonize: str | None = None

    def to_argv(self) -> list[str]:
        argv = [sys.executable, "-m", "demixer.cli", "process",
                str(self.input_path), "-o", str(self.output_dir),
                "--model", self.model, "--transcriber", self.transcriber,
                "--chords", self.chords, "--drums", self.drums, "--verbose"]
        if self.roformer_vocals:
            argv.append("--roformer-vocals")
        if self.harmony:
            argv.append("--harmony")
        if self.reharmonize:
            argv += ["--reharmonize", self.reharmonize]
        return argv


class PipelineWorker(QThread):
    """Runs the pipeline subprocess; emits log lines, progress, and completion."""

    log_line = Signal(str)
    progress = Signal(int)        # 0..100
    stage = Signal(str)           # human-readable current stage
    finished_ok = Signal(str)     # bundle dir path
    failed = Signal(str)          # error message

    def __init__(self, opts: RunOptions) -> None:
        super().__init__()
        self._opts = opts
        self._proc: subprocess.Popen[str] | None = None
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def run(self) -> None:  # QThread entrypoint
        last_pct = 0
        try:
            self._proc = subprocess.Popen(
                self._opts.to_argv(),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"failed to launch pipeline: {exc}")
            return

        assert self._proc.stdout is not None
        for raw in self._proc.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            self.log_line.emit(line)
            for substr, pct in _MILESTONES:
                if substr in line and pct > last_pct:
                    last_pct = pct
                    self.progress.emit(pct)
                    self.stage.emit(substr.rstrip(":").strip())
                    break

        code = self._proc.wait()
        if self._cancelled:
            self.failed.emit("cancelled")
        elif code == 0:
            self.progress.emit(100)
            self.stage.emit("done")
            self.finished_ok.emit(str(self._opts.output_dir))
        else:
            self.failed.emit(f"pipeline exited with code {code} (see log)")
