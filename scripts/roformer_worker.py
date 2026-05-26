"""Isolated BS-RoFormer worker — runs in `.venv-roformer` (numpy 2.x).

audio-separator requires numpy>=2, which is irreconcilable with the main
environment's autochord (needs Keras 2 / numpy<2). So RoFormer runs here, in
its own venv, invoked as a subprocess by demixer.core.separation_roformer.

Usage:
    .venv-roformer/bin/python scripts/roformer_worker.py <input.wav> <output_vocals.wav>

Prints "OK <path>" on success, "ERR <message>" on failure.
"""

import shutil
import sys
import tempfile
from pathlib import Path

ROFORMER_VOCALS_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def main() -> int:
    if len(sys.argv) != 3:
        print("ERR usage: roformer_worker.py <input.wav> <output_vocals.wav>")
        return 2
    in_wav, out_wav = Path(sys.argv[1]), Path(sys.argv[2])
    if not in_wav.is_file():
        print(f"ERR input not found: {in_wav}")
        return 2

    from audio_separator.separator import Separator

    with tempfile.TemporaryDirectory() as tmp:
        sep = Separator(output_dir=tmp)
        sep.load_model(model_filename=ROFORMER_VOCALS_MODEL)
        files = sep.separate(str(in_wav))
        vocals = next((f for f in files if "Vocals" in f), None)
        if vocals is None:
            print(f"ERR no vocals stem in outputs: {files}")
            return 1
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(tmp) / vocals, out_wav)

    print(f"OK {out_wav}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
