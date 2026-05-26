"""Audio ingest: decode → 44.1 kHz stereo float32 → loudness-normalize → sha256.

The ingest stage is the head of the pipeline. Every downstream stage
(separation, transcription, analysis) keys its cache on the sha256 of the
*normalized* float32 PCM, not the source file — so re-running the pipeline on
a re-encoded copy of the same audio still hits cache.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

TARGET_SR = 44_100
TARGET_CHANNELS = 2
TARGET_LUFS = -23.0  # EBU R128 reference


@dataclass(frozen=True)
class IngestedAudio:
    samples: np.ndarray  # shape (channels, n_samples), float32, range ~[-1, 1]
    sample_rate: int
    duration_s: float
    source_path: Path
    sha256: str  # hex digest of the normalized PCM bytes — pipeline cache key
    integrated_lufs_before: float
    integrated_lufs_after: float


def _ffmpeg_decode(path: Path) -> np.ndarray:
    """Decode any ffmpeg-supported format to float32 stereo at TARGET_SR.

    Returns shape (channels, n_samples). Uses ffmpeg's f32le pipe — robust across
    MP3, FLAC, OGG, M4A, WAV without per-format codecs in Python.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel", "error",
        "-i", str(path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SR),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    interleaved = np.frombuffer(proc.stdout, dtype=np.float32)
    return interleaved.reshape(-1, TARGET_CHANNELS).T.copy()


def ingest(path: str | Path) -> IngestedAudio:
    src = Path(path).resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    samples = _ffmpeg_decode(src)

    if samples.shape[1] == 0:
        raise ValueError(
            f"decoded audio is empty (0 samples): {src}. "
            "Check the file isn't corrupt and any trim/seek range is within its duration."
        )

    # pyloudnorm wants shape (n_samples, channels)
    meter = pyln.Meter(TARGET_SR)
    interleaved = samples.T

    # EBU R128 integrated loudness is undefined for clips shorter than the meter's
    # block size (0.4 s). Rather than let pyloudnorm raise a cryptic error deep in
    # the stack, skip normalization for such clips and pass the audio through.
    if samples.shape[1] < meter.block_size * TARGET_SR:
        lufs_before = lufs_after = float("nan")
        out = samples.astype(np.float32, copy=False)
    else:
        lufs_before = float(meter.integrated_loudness(interleaved))
        # Silent / near-silent audio measures -inf (or nan) LUFS; normalizing it
        # computes an infinite gain → inf*0 = NaN, which poisons every downstream
        # stage. Loudness is undefined for silence, so pass it through unchanged.
        if not np.isfinite(lufs_before):
            lufs_after = lufs_before
            out = samples.astype(np.float32, copy=False)
        else:
            normalized = pyln.normalize.loudness(interleaved, lufs_before, TARGET_LUFS)
            # Re-measure for the cache (and to report)
            lufs_after = float(meter.integrated_loudness(normalized))
            out = normalized.T.astype(np.float32, copy=False)

    digest = hashlib.sha256(out.tobytes()).hexdigest()
    duration_s = out.shape[1] / TARGET_SR

    return IngestedAudio(
        samples=out,
        sample_rate=TARGET_SR,
        duration_s=duration_s,
        source_path=src,
        sha256=digest,
        integrated_lufs_before=lufs_before,
        integrated_lufs_after=lufs_after,
    )


def write_wav(audio: IngestedAudio, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, audio.samples.T, audio.sample_rate, subtype="FLOAT")
    return dest
