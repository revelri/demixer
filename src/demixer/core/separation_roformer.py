"""Optional high-quality vocals separation via BS-RoFormer (audio-separator).

BS-RoFormer (SDR 12.97 on MUSDB18-HQ vocals) substantially out-separates
htdemucs on vocals — the published gap is ~2.5-3 dB SDR. We use it in a hybrid:
RoFormer supplies the vocals stem, htdemucs supplies drums/bass/other. RoFormer
is a real-vocal specialist (it won't isolate synthetic/GM "voices"), so it only
helps on genuine recordings.

**Subprocess isolation.** audio-separator requires numpy>=2, which is
irreconcilable with the main env's autochord (needs Keras 2 / numpy<2). So
RoFormer lives in a dedicated venv (`.venv-roformer`) and we invoke it via
`scripts/roformer_worker.py`, exchanging WAV files on disk. This module (in the
numpy<2 main env) never imports audio_separator.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from demixer.core.ingest import IngestedAudio

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ROFORMER_PYTHON = _REPO_ROOT / ".venv-roformer" / "bin" / "python"
_WORKER = _REPO_ROOT / "scripts" / "roformer_worker.py"


class RoformerUnavailableError(RuntimeError):
    """Raised when the isolated RoFormer venv / worker isn't set up."""


def roformer_available() -> bool:
    return _ROFORMER_PYTHON.is_file() and _WORKER.is_file()


def roformer_vocals(audio: IngestedAudio) -> np.ndarray:
    """Return the BS-RoFormer vocals stem as (channels, samples) at audio.sample_rate.

    Shells out to the isolated `.venv-roformer`. Length is trimmed/padded to
    exactly match the input so it lines up with the htdemucs stems it's combined
    with.
    """
    if not roformer_available():
        raise RoformerUnavailableError(
            f"RoFormer venv/worker missing ({_ROFORMER_PYTHON}). "
            "Create it with: uv venv --python 3.11 .venv-roformer && "
            "VIRTUAL_ENV=.venv-roformer uv pip install 'audio-separator[cpu]'"
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        mix_wav = tmpdir / "mix.wav"
        out_wav = tmpdir / "vocals.wav"
        sf.write(mix_wav, audio.samples.T, audio.sample_rate, subtype="FLOAT")

        proc = subprocess.run(
            [str(_ROFORMER_PYTHON), str(_WORKER), str(mix_wav), str(out_wav)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not out_wav.is_file():
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
            raise RuntimeError(f"RoFormer worker failed: {' | '.join(tail)}")

        vocals, sr = sf.read(out_wav, always_2d=True)

    if sr != audio.sample_rate:
        raise RuntimeError(f"RoFormer returned {sr} Hz; expected {audio.sample_rate}")

    # (samples, channels) -> (channels, samples); force stereo + exact length
    vocals = vocals.T.astype(np.float32, copy=False)
    if vocals.shape[0] == 1:
        vocals = np.repeat(vocals, 2, axis=0)
    target_len = audio.samples.shape[1]
    if vocals.shape[1] < target_len:
        vocals = np.pad(vocals, ((0, 0), (0, target_len - vocals.shape[1])))
    elif vocals.shape[1] > target_len:
        vocals = vocals[:, :target_len]
    return vocals
