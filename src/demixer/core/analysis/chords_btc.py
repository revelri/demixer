"""Main-env client for BTC chord recognition (runs in the isolated `.venv-btc`).

BTC does large-vocabulary chords (170 classes, incl. 7ths/extensions) vs
autochord's triads-only. Runs in its own venv via the worker subprocess; this
client shells out and parses the JSON segments.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import soundfile as sf

from demixer.core.analysis.chords import ChordSegment, _short_label
from demixer.core.ingest import IngestedAudio
from demixer.core.workers import Worker, parse_worker_status

_WORKER = Worker("btc")


def available() -> bool:
    return _WORKER.available()


def estimate(audio: IngestedAudio) -> list[ChordSegment]:
    """Recognize chords via the isolated BTC worker. Labels normalized to short form."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "mix.wav"
        out_json = Path(tmp) / "chords.json"
        sf.write(wav, audio.samples.T, audio.sample_rate, subtype="FLOAT")

        proc = _WORKER.run(str(wav), str(out_json), timeout=600.0)
        ok, msg = parse_worker_status(proc)
        if not ok or not out_json.is_file():
            raise RuntimeError(f"BTC worker failed: {msg}")
        raw = json.loads(out_json.read_text())

    return [
        ChordSegment(start_s=float(s["start_s"]), end_s=float(s["end_s"]),
                     label=_short_label(s["label"]))
        for s in raw
    ]
