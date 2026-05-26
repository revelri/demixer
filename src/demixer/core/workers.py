"""Subprocess orchestration for SOTA models that can't share the main venv.

Several best-in-class models have mutually-exclusive dependency constraints
(numpy<2 vs numpy>=2, Keras 2 vs Keras 3, pinned transformers versions, py3.10
vs py3.11). Rather than forcing one fragile environment, each lives in its own
`.venv-<name>` and runs as a subprocess worker that exchanges files on disk.

This module is the thin main-env side: it locates a worker venv's python +
script and runs it, returning the worker's exit status and captured output. The
main env never imports any isolated model's packages.

Convention per model `<name>`:
  .venv-<name>/bin/python        — the isolated interpreter
  scripts/<name>_worker.py       — the worker entrypoint (run inside that venv)
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Worker:
    name: str

    @property
    def python(self) -> Path:
        return _REPO_ROOT / f".venv-{self.name}" / "bin" / "python"

    @property
    def script(self) -> Path:
        return _REPO_ROOT / "scripts" / f"{self.name}_worker.py"

    def available(self) -> bool:
        return self.python.is_file() and self.script.is_file()

    def run(self, *args: str, timeout: float = 600.0) -> subprocess.CompletedProcess[str]:
        """Run the worker with positional args. Raises if the venv/script is missing."""
        if not self.available():
            raise WorkerUnavailableError(
                f"worker '{self.name}' not set up (need {self.python} and {self.script})"
            )
        return subprocess.run(
            [str(self.python), str(self.script), *args],
            capture_output=True, text=True, timeout=timeout,
        )


class WorkerUnavailableError(RuntimeError):
    """Raised when an isolated worker venv/script is missing."""


def parse_worker_status(proc: subprocess.CompletedProcess[str]) -> tuple[bool, str]:
    """Workers print 'OK <payload>' or 'ERR <message>' as their last stdout line.

    Returns (ok, payload_or_message). Falls back to returncode if no marker found.
    """
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("OK "):
            return True, ln[3:].strip()
        if ln.startswith("ERR "):
            return False, ln[4:].strip()
    if proc.returncode == 0:
        return True, ""
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
    return False, " | ".join(tail)
