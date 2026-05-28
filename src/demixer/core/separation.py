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


# Stems whose peak amplitude is below this are treated as empty. Demucs zeroes
# out absent sources (e.g. drums on a vocal-only track), so writing them is pure
# I/O waste — a 3-minute float32 stereo stem of silence is ~60 MB on disk and
# the same bytes again inside every downstream DAW project. −40 dBFS sits well
# above dither/noise-floor and well below any musically present part.
_SILENT_STEM_PEAK = 0.01

# Container/subtype per stem format. FLAC compresses to ~30–60 % of PCM_16 for
# music and to near-zero for silent regions; PCM_24 is the mastering-quality
# default; PCM_16 is CD-quality and a quarter the size of FLOAT; FLOAT preserves
# bit-exact ML output for downstream re-feeding.
StemFormat = Literal["pcm24", "pcm16", "float", "flac"]

_STEM_FORMAT_SPEC: dict[str, tuple[str, str]] = {
    # name -> (file extension, soundfile subtype)
    "pcm24": (".wav",  "PCM_24"),
    "pcm16": (".wav",  "PCM_16"),
    "float": (".wav",  "FLOAT"),
    "flac":  (".flac", "PCM_24"),  # FLAC supports up to 24-bit losslessly
}


def write_stems(
    result: SeparationResult,
    out_dir: str | Path,
    stem_format: StemFormat = "pcm24",
) -> dict[str, Path]:
    """Write each non-silent stem to `out_dir/<stem>.<ext>`.

    Silent stems (peak < −40 dBFS) are skipped — they would otherwise propagate
    full-size copies of silence into every DAW project and the .demixer archive.

    `stem_format` controls the on-disk container/precision:
      - "pcm24" (default): 24-bit WAV, mastering quality, ~3/4 the size of float32
      - "pcm16":            16-bit WAV, CD quality, ~1/4 the size of float32
      - "float":            32-bit float WAV, bit-exact Demucs output (legacy)
      - "flac":             24-bit FLAC, lossless, typically 30–60 % of PCM_24

    Returns name → path for the stems that were actually written.
    """
    if stem_format not in _STEM_FORMAT_SPEC:
        raise ValueError(
            f"unknown stem_format {stem_format!r}; "
            f"expected one of {sorted(_STEM_FORMAT_SPEC)}"
        )
    ext, subtype = _STEM_FORMAT_SPEC[stem_format]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, samples in result.stems.items():
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak < _SILENT_STEM_PEAK:
            continue
        path = out_dir / f"{name}{ext}"
        sf.write(path, samples.T, result.sample_rate, subtype=subtype)
        written[name] = path
    return written
