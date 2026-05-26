"""Render MIDI → WAV via FluidSynth + a General MIDI SoundFont.

Used by the synthetic accuracy eval to turn known-ground-truth MIDI into audio
the pipeline can transcribe, so we can score transcription against the source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

# Full General MIDI SoundFont shipped on the system (148 MB).
DEFAULT_SOUNDFONT = "/usr/share/soundfonts/FluidR3_GM.sf2"
RENDER_SR = 44_100


def render_midi_to_wav(
    midi: pretty_midi.PrettyMIDI,
    dest_wav: str | Path,
    *,
    soundfont: str = DEFAULT_SOUNDFONT,
    sample_rate: int = RENDER_SR,
) -> Path:
    """Synthesize `midi` to a stereo WAV using FluidSynth + the given SoundFont."""
    dest_wav = Path(dest_wav)
    dest_wav.parent.mkdir(parents=True, exist_ok=True)

    if not Path(soundfont).is_file():
        raise FileNotFoundError(f"SoundFont not found: {soundfont}")

    # pretty_midi.fluidsynth() returns mono float waveform at `fs`.
    audio = midi.fluidsynth(fs=sample_rate, sf2_path=soundfont)
    if audio.size == 0:
        # Silence guard — fluidsynth returns empty for note-less MIDI
        audio = np.zeros(sample_rate, dtype=np.float32)
    # Normalize to avoid clipping, then make stereo
    peak = float(np.max(np.abs(audio))) or 1.0
    audio = (audio / peak * 0.9).astype(np.float32)
    stereo = np.stack([audio, audio], axis=1)
    sf.write(dest_wav, stereo, sample_rate, subtype="FLOAT")
    return dest_wav
