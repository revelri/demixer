"""Detect whether the input audio is a tight loop and, if so, the loop period.

Cross-correlation of a downsampled mono mix is robust to small amplitude
variations and cheap (FFT-based, ~25 ms for a 3-minute clip). When the input
genuinely loops — game-music corpora, drum-loop libraries, ambient beds —
this enables `separate_loop_aware` to run Demucs on a single period and tile
the stems back to full length, cutting separation cost by Nx where N is the
loop count.

The detector is deliberately conservative: a false positive corrupts every
downstream artifact, so the confidence floor sits well above what generic
music achieves on autocorrelation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LoopDetection:
    period_samples: int   # detected loop period at the input sample rate
    period_s: float
    confidence: float     # normalized autocorrelation peak height in [0, 1]
    n_repeats: int        # number of full periods that fit in the input


def detect_loop_period(
    samples: np.ndarray,
    sample_rate: int,
    *,
    min_period_s: float = 1.5,
    max_period_s: float = 60.0,
    min_confidence: float = 0.92,
    target_sr: int = 8000,
) -> LoopDetection | None:
    """Return loop-period info if `samples` is a tight loop, else None.

    `samples` is `(channels, n)` or `(n,)` float32 audio. Confidence is the
    normalized autocorrelation peak; the default 0.92 threshold rejects
    generic music (which sits around 0.4–0.7) and accepts clean loops (>0.95
    on the Encarta MindMaze QUESTION/BGLOOPx renders).

    Also requires the detected period to evenly divide the input duration
    within 1 % — true loops repeat an integer number of times.
    """
    if samples.ndim == 2:
        mono = samples.mean(axis=0)
    else:
        mono = samples
    mono = mono.astype(np.float32, copy=False)
    n_full = mono.size
    if n_full == 0:
        return None

    # Downsample by integer stride for cheap autocorrelation. 8 kHz is plenty
    # to localize a loop period to ~125 µs at the downsampled rate, which
    # rescales to ~14 ms at 44.1 kHz — well below a 16th note even at 240 BPM.
    step = max(1, sample_rate // target_sr)
    y = mono[::step]
    sr_ds = sample_rate // step
    n = y.size
    if n < int(min_period_s * sr_ds) * 2:
        return None  # too short to host a loop

    # Energy-normalize so the autocorrelation peak height is in [0, 1].
    y = y - y.mean()
    energy = float(np.dot(y, y))
    if energy < 1e-9:
        return None

    # FFT autocorrelation. Zero-pad to next power of two ≥ 2n to avoid wraparound.
    fft_size = 1 << int(np.ceil(np.log2(2 * n)))
    Y = np.fft.rfft(y, fft_size)
    raw_ac = np.fft.irfft(Y * np.conj(Y), fft_size).real[:n]

    # Coverage correction: raw_ac[k] sums over (N-k) pairs, so a perfectly
    # periodic signal still drops as k grows. Dividing by (N-k) gives a
    # per-pair correlation that hits ~1.0 for true repetition regardless of
    # how many periods fit. Normalize against per-window energy so amplitude
    # variation between cycles doesn't bias the score.
    lags = np.arange(n, dtype=np.float64)
    overlap = np.clip(n - lags, 1.0, None)
    # Per-pair correlation; bounded above by per-window energy / overlap.
    ac = raw_ac / (overlap * (energy / n))

    min_lag = int(min_period_s * sr_ds)
    # Cap max_lag at n//2 so every candidate keeps ≥ one full period of overlap.
    # Without this, the coverage-corrected ratio blows up at the tail where
    # only a handful of samples overlap and noise can correlate by accident.
    max_lag = min(int(max_period_s * sr_ds), n // 2)
    if max_lag <= min_lag:
        return None
    region = ac[min_lag:max_lag]
    peak_rel = int(np.argmax(region))
    confidence = float(region[peak_rel])
    if confidence < min_confidence:
        return None

    period_ds = min_lag + peak_rel
    period_samples = int(round(period_ds * step))
    period_s = period_samples / sample_rate
    n_repeats = int(round(n_full / period_samples))

    # True loops repeat ≥ 2× and divide the input within ~1 %.
    if n_repeats < 2:
        return None
    residual = abs(n_full - n_repeats * period_samples) / max(n_full, 1)
    if residual > 0.01:
        return None

    return LoopDetection(
        period_samples=period_samples,
        period_s=period_s,
        confidence=confidence,
        n_repeats=n_repeats,
    )
