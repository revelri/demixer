"""Isolated ADTOF (PyTorch) drum-transcription worker — runs in `.venv-adtof`.

ADTOF-pytorch (github.com/xavriley/ADTOF-pytorch) is a torch-only port of the
ADTOF Frame-RNN drum transcriber (5 GM classes: kick 35, snare 38, tom 47,
hi-hat 42, crash 49) — a learned model trained on 114 h of real annotated drums,
far richer than the main env's librosa onset+centroid 3-class classifier. It runs
here because the TF/Keras ADTOF and madmom conflict with the main env; the torch
port avoids all of that.

Usage:
    single:  adtof_worker.py <input.wav> <output.mid>
    batch:   adtof_worker.py --manifest <manifest.json>
             manifest = [{"in": "a.wav", "out": "a.mid"}, ...]

Prints per-file "OK <out>" / "ERR <file>: <msg>", then "OK <done>/<total>".
"""

import json
import sys
from pathlib import Path


def _load_specs() -> list[tuple[Path, Path]]:
    if sys.argv[1] == "--manifest":
        manifest = json.loads(Path(sys.argv[2]).read_text())
        return [(Path(s["in"]), Path(s["out"])) for s in manifest]
    return [(Path(sys.argv[1]), Path(sys.argv[2]))]


def main() -> int:
    if len(sys.argv) < 3:
        print("ERR usage: adtof_worker.py <in.wav> <out.mid> | --manifest <json>")
        return 2

    import adtof_pytorch  # torch-only; model weights ship with the package

    specs = _load_specs()
    done = 0
    for in_wav, out_mid in specs:
        if not in_wav.is_file():
            print(f"ERR {in_wav}: input not found")
            continue
        try:
            out_mid.parent.mkdir(parents=True, exist_ok=True)
            adtof_pytorch.transcribe_to_midi(str(in_wav), str(out_mid), device="cpu")
            print(f"OK {out_mid}")
            done += 1
        except Exception as e:  # noqa: BLE001
            print(f"ERR {in_wav}: {type(e).__name__}: {e}")

    print(f"OK {done}/{len(specs)}")
    return 0 if done == len(specs) else 1


if __name__ == "__main__":
    sys.exit(main())
