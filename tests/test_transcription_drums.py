"""Tests for drum transcription on synthesized + (gated) real drum stems."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from demixer.core.transcription.drums import (
    _HIHAT_CLOSED,
    _KICK,
    _SNARE,
    _classify_onset,
    transcribe_drums,
)


def _synth_kick(sr: int = 44_100, dur_s: float = 0.1) -> np.ndarray:
    """50 Hz sine decaying fast — classic synth-kick spectrum."""
    t = np.arange(int(dur_s * sr)) / sr
    env = np.exp(-30 * t).astype(np.float32)
    return (0.8 * np.sin(2 * np.pi * 50 * t).astype(np.float32) * env)


def _synth_hihat(sr: int = 44_100, dur_s: float = 0.05) -> np.ndarray:
    """White noise + short envelope, high-pass-like via FFT mask."""
    rng = np.random.default_rng(0)
    n = rng.standard_normal(int(dur_s * sr)).astype(np.float32) * 0.4
    # Crude high-pass: zero out lower bins
    spec = np.fft.rfft(n)
    freqs = np.fft.rfftfreq(len(n), 1.0 / sr)
    spec[freqs < 5_000] = 0
    hp = np.fft.irfft(spec, n=len(n)).astype(np.float32)
    env = np.exp(-60 * np.arange(len(hp)) / sr).astype(np.float32)
    return hp * env


def _synth_snare(sr: int = 44_100, dur_s: float = 0.08) -> np.ndarray:
    """Snare: 200 Hz body tone + band-limited noise concentrated 200–4 kHz.

    Real snares have most of their energy in this band (drum body resonance +
    snare wires) — pure white noise overweights ultrasonic content and reads
    as a hi-hat to a centroid-based classifier.
    """
    t = np.arange(int(dur_s * sr)) / sr
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(len(t)).astype(np.float32)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(len(noise), 1.0 / sr)
    # Band-pass: keep 200-4000 Hz, zero out everything else
    spec[(freqs < 200) | (freqs > 4_000)] = 0
    bp_noise = (np.fft.irfft(spec, n=len(noise)) * 0.5).astype(np.float32)
    tone = 0.4 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
    env = np.exp(-25 * t).astype(np.float32)
    return (bp_noise + tone) * env


def test_classify_kick_is_low_dominant() -> None:
    sr = 44_100
    note, vel, dur = _classify_onset(_synth_kick(sr), sr)
    assert note == _KICK
    assert 20 <= vel <= 127
    assert dur > 0


def test_classify_hihat_is_high_dominant() -> None:
    sr = 44_100
    note, _, _ = _classify_onset(_synth_hihat(sr), sr)
    assert note in (_HIHAT_CLOSED, _HIHAT_CLOSED + 4)  # closed or open


def test_classify_snare_is_mid_dominant() -> None:
    sr = 44_100
    note, _, _ = _classify_onset(_synth_snare(sr), sr)
    assert note == _SNARE


def test_transcribe_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        transcribe_drums(tmp_path / "nope.wav")


def test_transcribe_produces_drum_instrument(tmp_path: Path) -> None:
    """Synthesize a 2-second drum loop (kick-snare-kick-snare at 120 BPM) and transcribe."""
    sr = 44_100
    out = np.zeros(int(2.0 * sr), dtype=np.float32)
    beat_samples = int(0.5 * sr)  # 120 BPM
    kick = _synth_kick(sr)
    snare = _synth_snare(sr)
    for i, sound in enumerate([kick, snare, kick, snare]):
        start = i * beat_samples
        out[start : start + len(sound)] += sound
    src = tmp_path / "drums.wav"
    sf.write(src, out, sr, subtype="FLOAT")

    midi = transcribe_drums(src)
    assert isinstance(midi, pretty_midi.PrettyMIDI)
    assert len(midi.instruments) == 1
    inst = midi.instruments[0]
    assert inst.is_drum

    notes = inst.notes
    assert len(notes) >= 3, f"expected ≥3 onsets detected, got {len(notes)}"
    pitches = {n.pitch for n in notes}
    # We should see both kick and snare
    assert _KICK in pitches, f"no kick detected; pitches={pitches}"
    assert _SNARE in pitches, f"no snare detected; pitches={pitches}"


@pytest.mark.skipif(
    not os.environ.get("DEMIXER_RUN_HEAVY") or
    not Path("/tmp/super_trouper_demix/stems/drums.wav").exists(),
    reason="run after `demixer process` with DEMIXER_RUN_HEAVY=1",
)
def test_transcribe_real_drum_stem() -> None:
    midi = transcribe_drums("/tmp/super_trouper_demix/stems/drums.wav")
    notes = midi.instruments[0].notes
    # ABBA Super Trouper has ~120 BPM steady 4/4 over 30s — expect dozens of hits
    assert 30 <= len(notes) <= 300, f"unexpected onset count: {len(notes)}"


@pytest.mark.skipif(
    not os.environ.get("DEMIXER_RUN_HEAVY"),
    reason="set DEMIXER_RUN_HEAVY=1 to run (needs the isolated .venv-adtof worker)",
)
def test_adtof_worker_transcribes(tmp_path: Path) -> None:
    """ADTOF (PyTorch) client returns a drum PrettyMIDI with GM drum pitches."""
    from demixer.core.transcription import drums_adtof
    if not drums_adtof.available():
        pytest.skip("ADTOF worker venv not set up")
    sr = 44_100
    out = np.zeros(int(2.0 * sr), dtype=np.float32)
    beat = int(0.5 * sr)
    kick = _synth_kick(sr); snare = _synth_snare(sr)
    for i, snd in enumerate([kick, snare, kick, snare]):
        out[i * beat:i * beat + len(snd)] += snd
    src = tmp_path / "drums.wav"
    sf.write(src, out, sr, subtype="FLOAT")
    midi = drums_adtof.transcribe_drums_adtof(src)
    assert midi.instruments and any(i.notes for i in midi.instruments)
    # ADTOF emits GM drum-kit pitches
    pitches = {n.pitch for i in midi.instruments for n in i.notes}
    assert pitches <= {35, 36, 38, 40, 42, 46, 47, 49, 51, 45, 48}, f"unexpected pitches {pitches}"
