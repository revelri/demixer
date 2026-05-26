"""Stem separation via Demucs v4 (htdemucs / htdemucs_6s / htdemucs_ft).

Wraps the demucs Python API into a single `separate()` call that takes an
IngestedAudio and returns a dict of stem name → float32 waveform at the
ingest sample rate (44.1 kHz).

Demucs models output at 44.1 kHz natively, matching our ingest target, so no
resampling is needed on either end.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch

from demixer.core.ingest import IngestedAudio

ModelName = Literal["htdemucs", "htdemucs_ft", "htdemucs_6s"]

# Canonical stem orderings each Demucs variant outputs. We surface them as a
# dict keyed by stem name, so callers never have to remember the order.
STEM_NAMES: dict[ModelName, tuple[str, ...]] = {
    "htdemucs":    ("drums", "bass", "other", "vocals"),
    "htdemucs_ft": ("drums", "bass", "other", "vocals"),
    "htdemucs_6s": ("drums", "bass", "other", "vocals", "guitar", "piano"),
}


@dataclass(frozen=True)
class SeparationResult:
    stems: dict[str, np.ndarray]   # stem_name -> (channels, samples) float32 @ sample_rate
    sample_rate: int
    model: str                     # e.g. "htdemucs" or "htdemucs+bs-roformer-vocals"
    device: str                    # "cuda" | "cpu"


def _pick_device(prefer_cuda: bool) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def separate(
    audio: IngestedAudio,
    model_name: ModelName = "htdemucs",
    *,
    prefer_cuda: bool = True,
    shifts: int = 1,
    overlap: float = 0.25,
    roformer_vocals: bool = False,
) -> SeparationResult:
    """Separate `audio` into stems using a pretrained Demucs model.

    `shifts > 1` improves quality at linear time cost (Demucs averages N shifted
    passes). `overlap` controls inter-chunk overlap during streaming inference.
    Defaults match Demucs's `--shifts 1 --overlap 0.25` recommended speed/quality.

    `roformer_vocals=True` replaces the htdemucs vocals stem with a BS-RoFormer
    vocals stem (higher SDR on real recordings; requires the `audio-separator`
    extra). drums/bass/other still come from Demucs.
    """
    # Imported lazily so test discovery / `--help` don't pay the torch+demucs cost
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    if model_name not in STEM_NAMES:
        raise ValueError(f"unsupported model {model_name!r}; choose from {list(STEM_NAMES)}")

    if audio.sample_rate != 44_100:
        raise ValueError(
            f"separate() expects 44.1 kHz input; got {audio.sample_rate}. "
            "Ingest via core.ingest first."
        )

    model = get_model(model_name)
    if model.samplerate != audio.sample_rate:
        # Sanity check against future demucs versions
        raise RuntimeError(
            f"model.samplerate ({model.samplerate}) != audio.sample_rate ({audio.sample_rate})"
        )

    device = _pick_device(prefer_cuda)
    model.to(device)
    model.eval()

    # demucs expects (batch, channels, samples)
    waveform = torch.from_numpy(audio.samples).unsqueeze(0).to(device)
    with torch.no_grad():
        sources = apply_model(
            model,
            waveform,
            shifts=shifts,
            overlap=overlap,
            progress=False,
        )
    # sources: (batch=1, n_sources, channels, samples) -> (n_sources, channels, samples)
    sources = sources.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    expected = STEM_NAMES[model_name]
    if sources.shape[0] != len(expected):
        raise RuntimeError(
            f"demucs returned {sources.shape[0]} sources; expected {len(expected)} for {model_name}"
        )

    stems = {name: sources[i] for i, name in enumerate(expected)}

    model_label: str = model_name
    if roformer_vocals:
        from demixer.core.separation_roformer import roformer_vocals as _roformer_vox
        stems["vocals"] = _roformer_vox(audio)
        model_label = f"{model_name}+bs-roformer-vocals"

    return SeparationResult(
        stems=stems, sample_rate=audio.sample_rate, model=model_label, device=device,
    )


def write_stems(result: SeparationResult, out_dir: str | Path) -> dict[str, Path]:
    """Write each stem to `out_dir/<stem>.wav` (float32 WAV). Returns name → path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, samples in result.stems.items():
        path = out_dir / f"{name}.wav"
        sf.write(path, samples.T, result.sample_rate, subtype="FLOAT")
        written[name] = path
    return written
