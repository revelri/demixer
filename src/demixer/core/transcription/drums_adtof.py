"""Main-env client for the ADTOF (PyTorch) drum transcriber in `.venv-adtof`.

ADTOF Frame-RNN is a learned drum-transcription model (5 GM classes: kick 35,
snare 38, tom 47, hi-hat 42, crash 49) trained on 114 h of real annotated drums
— richer and more accurate than the built-in librosa onset+centroid classifier
(kick/snare/hi-hat only). Runs via the isolated-venv worker; this client shells
out and loads the resulting MIDI.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pretty_midi

from demixer.core.workers import Worker, parse_worker_status

_WORKER = Worker("adtof")


def available() -> bool:
    return _WORKER.available()


def transcribe_drums_adtof(wav_path: str | Path) -> pretty_midi.PrettyMIDI:
    """Transcribe a clean drum stem to GM drum MIDI via the isolated ADTOF worker."""
    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise FileNotFoundError(wav_path)

    with tempfile.TemporaryDirectory() as tmp:
        out_mid = Path(tmp) / "drums.mid"
        proc = _WORKER.run(str(wav_path), str(out_mid), timeout=600.0)
        ok, msg = parse_worker_status(proc)
        if not ok or not out_mid.is_file():
            raise RuntimeError(f"ADTOF worker failed: {msg}")
        return pretty_midi.PrettyMIDI(str(out_mid))
