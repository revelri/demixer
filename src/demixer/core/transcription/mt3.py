"""Main-env client for MT3 transcription (runs in the isolated `.venv-mt3`).

MR-MT3 is a multi-instrument transformer transcriber — stronger on polyphony
than basic-pitch. It runs in its own venv (pinned transformers 4.56.2) via the
worker subprocess; this client shells out and loads the resulting MIDI.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pretty_midi

from demixer.core.workers import Worker, parse_worker_status

_WORKER = Worker("mt3")


def available() -> bool:
    return _WORKER.available()


def transcribe_mt3(wav_path: str | Path, *, model: str = "mr_mt3") -> pretty_midi.PrettyMIDI:
    """Transcribe an audio file to MIDI via the isolated MT3 worker."""
    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise FileNotFoundError(wav_path)

    with tempfile.TemporaryDirectory() as tmp:
        out_mid = Path(tmp) / "mt3.mid"
        proc = _WORKER.run(str(wav_path), str(out_mid), model, timeout=600.0)
        ok, msg = parse_worker_status(proc)
        if not ok or not out_mid.is_file():
            raise RuntimeError(f"MT3 worker failed: {msg}")
        return pretty_midi.PrettyMIDI(str(out_mid))


def transcribe_mt3_batch(
    wav_paths: dict[str, Path], *, model: str = "mr_mt3"
) -> dict[str, pretty_midi.PrettyMIDI]:
    """Transcribe several stems in ONE worker invocation.

    MT3's model load dominates per-run cost and is cached in-process, so batching
    all stems into a single subprocess loads the model once instead of once per
    stem. Returns name → PrettyMIDI for every stem that transcribed successfully
    (a per-file failure is skipped, not fatal — matches the CLI's best-effort
    transcription).
    """
    if not wav_paths:
        return {}

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        manifest = []
        out_for: dict[str, Path] = {}
        for name, wav in wav_paths.items():
            out_mid = tmpdir / f"{name}.mid"
            out_for[name] = out_mid
            manifest.append({"in": str(Path(wav)), "out": str(out_mid)})
        manifest_path = tmpdir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        # generous timeout: one model load + N transcriptions
        self_proc = _WORKER.run("--manifest", str(manifest_path), model, timeout=1800.0)
        _ = parse_worker_status(self_proc)  # summary only; we load by file existence

        results: dict[str, pretty_midi.PrettyMIDI] = {}
        for name, out_mid in out_for.items():
            if out_mid.is_file():
                results[name] = pretty_midi.PrettyMIDI(str(out_mid))
        if not results:
            raise RuntimeError(f"MT3 batch worker produced no output: {self_proc.stderr[-300:]}")
        return results
