"""Isolated MT3 transcription worker — runs in `.venv-mt3`.

mt3-infer needs a pinned transformers (4.56.2) that conflicts with the main
env, so it runs here. Transcribes audio to MIDI via MR-MT3.

mt3_infer caches the loaded model in-process, so transcribing several files in a
single invocation loads the model once (the expensive part) and reuses it — far
cheaper than one subprocess per stem. Hence the batch mode.

Usage:
    single:  mt3_worker.py <input.wav> <output.mid> [model]
    batch:   mt3_worker.py --manifest <manifest.json> [model]
             manifest = [{"in": "a.wav", "out": "a.mid"}, ...]

Prints per-file "OK <out>" / "ERR <file>: <msg>" lines, then a final
"OK <done>/<total>" summary line.
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def _load_specs() -> tuple[list[tuple[Path, Path]], str]:
    if sys.argv[1] == "--manifest":
        manifest = json.loads(Path(sys.argv[2]).read_text())
        model = sys.argv[3] if len(sys.argv) > 3 else "mr_mt3"
        return [(Path(s["in"]), Path(s["out"])) for s in manifest], model
    in_wav, out_mid = Path(sys.argv[1]), Path(sys.argv[2])
    model = sys.argv[3] if len(sys.argv) > 3 else "mr_mt3"
    return [(in_wav, out_mid)], model


def main() -> int:
    if len(sys.argv) < 3:
        print("ERR usage: mt3_worker.py <in.wav> <out.mid> [model] | --manifest <json> [model]")
        return 2

    specs, model = _load_specs()

    import librosa
    import mt3_infer  # model is cached in-process across transcribe() calls

    done = 0
    for in_wav, out_mid in specs:
        if not in_wav.is_file():
            print(f"ERR {in_wav}: input not found")
            continue
        try:
            y, _ = librosa.load(str(in_wav), sr=16000, mono=True)
            mf = mt3_infer.transcribe(y, model=model, sr=16000)
            out_mid.parent.mkdir(parents=True, exist_ok=True)
            mf.save(str(out_mid))
            print(f"OK {out_mid}")
            done += 1
        except Exception as e:  # noqa: BLE001
            print(f"ERR {in_wav}: {type(e).__name__}: {e}")

    print(f"OK {done}/{len(specs)}")
    return 0 if done == len(specs) else 1


if __name__ == "__main__":
    sys.exit(main())
