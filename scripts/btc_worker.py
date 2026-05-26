"""Isolated BTC chord-recognition worker — runs in `.venv-btc`.

BTC (Bi-directional Transformer for Chords, ISMIR 2019) does large-vocabulary
chord recognition (170 classes incl. 7ths/extensions), beating autochord's
triads-only. It's 2019 research code (vendored + patched under third_party/BTC
for PyYAML 6 / numpy-2 / torch-2), so it runs in its own venv.

Usage:
    .venv-btc/bin/python scripts/btc_worker.py <input.wav> <output.json>

Output JSON: [{"start_s": float, "end_s": float, "label": "C:maj"}, ...]
Prints "OK <json_path>" or "ERR <message>".
"""

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_BTC_DIR = Path(__file__).resolve().parent.parent / "third_party" / "BTC"


def main() -> int:
    if len(sys.argv) != 3:
        print("ERR usage: btc_worker.py <input.wav> <output.json>")
        return 2
    in_wav, out_json = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    if not in_wav.is_file():
        print(f"ERR input not found: {in_wav}")
        return 2

    try:
        os.chdir(_BTC_DIR)
        sys.path.insert(0, str(_BTC_DIR))
        import numpy as np
        import torch
        from btc_model import BTC_model
        from utils.hparams import HParams
        from utils.mir_eval_modules import audio_file_to_features, idx2voca_chord

        config = HParams.load("run_config.yaml")
        config.feature["large_voca"] = True
        config.model["num_chords"] = 170
        idx_to_chord = idx2voca_chord()

        model = BTC_model(config=config.model)
        ckpt = torch.load("./test/btc_model_large_voca.pt", map_location="cpu", weights_only=False)
        mean, std = ckpt["mean"], ckpt["std"]
        model.load_state_dict(ckpt["model"])
        model.eval()

        # BTC's audio_file_to_features leaves `feature` unbound when the clip is
        # shorter than inst_len (its while-loop never runs) → UnboundLocalError.
        # Pad short inputs above inst_len so the loop runs once; clamp the emitted
        # segments back to the real duration afterward so we don't report chords
        # past the end of the audio.
        import soundfile as sf
        inst_len = float(config.mp3["inst_len"])
        info = sf.info(str(in_wav))
        orig_dur = info.frames / float(info.samplerate)
        feat_input = str(in_wav)
        if orig_dur <= inst_len:
            y, sr = sf.read(str(in_wav))
            if y.ndim > 1:
                y = y.mean(axis=1)
            need = int((inst_len + 1.0) * sr)
            if len(y) < need:
                y = np.pad(y, (0, need - len(y)))
            padded = in_wav.with_name("_btc_padded.wav")
            sf.write(str(padded), y, sr)
            feat_input = str(padded)

        feature, feature_per_second, _ = audio_file_to_features(feat_input, config)
        feature = feature.T
        feature = (feature - mean) / std
        n_timestep = config.model["timestep"]
        num_pad = n_timestep - (feature.shape[0] % n_timestep)
        feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
        num_instance = feature.shape[0] // n_timestep

        segments = []
        start_time = 0.0
        prev_chord = None
        with torch.no_grad():
            ft = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
            for t in range(num_instance):
                attn, _ = model.self_attn_layers(ft[:, n_timestep * t:n_timestep * (t + 1), :])
                pred, _ = model.output_layer(attn)
                pred = pred.squeeze()
                for i in range(n_timestep):
                    cur = pred[i].item()
                    if t == 0 and i == 0:
                        prev_chord = cur
                        continue
                    abs_t = feature_per_second * (n_timestep * t + i)
                    if cur != prev_chord:
                        segments.append({"start_s": round(start_time, 3),
                                         "end_s": round(abs_t, 3),
                                         "label": idx_to_chord[prev_chord]})
                        start_time = abs_t
                        prev_chord = cur
                    if t == num_instance - 1 and i + num_pad == n_timestep:
                        if start_time != abs_t:
                            segments.append({"start_s": round(start_time, 3),
                                             "end_s": round(abs_t, 3),
                                             "label": idx_to_chord[prev_chord]})
                        break

        # Clamp to the real (pre-pad) duration: drop segments that start at/after
        # the audio end, and trim a final segment's end to the true duration.
        clamped = []
        for s in segments:
            if s["start_s"] >= orig_dur:
                continue
            s["end_s"] = round(min(s["end_s"], orig_dur), 3)
            clamped.append(s)
        out_json.write_text(json.dumps(clamped))
    except Exception as e:  # noqa: BLE001
        print(f"ERR {type(e).__name__}: {e}")
        return 1

    print(f"OK {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
