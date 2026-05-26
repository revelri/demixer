"""Controlled SI-SDR comparison: RoFormer vs htdemucs vocals recovery.

We can't get a real MUSDB SDR number without the MUSDB dataset, but we *can*
build a controlled apples-to-apples test: synthesize a "vocals" line and an
"instrumental" backing separately (these are ground truth), mix them, separate
the mix with each backend, and measure how well each recovers the vocals via
scale-invariant SDR.

Absolute numbers are synthetic-optimistic (GM-voice ≠ real vocals), but the
*delta* between separators on the identical mix is the informative signal.

Run:
    uv run python -m demixer.eval.separation_sdr
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

from demixer.eval.render import render_midi_to_wav

VOCAL_PROGRAM = 53  # Voice Oohs
BACKING_PROGRAMS = (0, 33)  # piano + finger bass


def _si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR in dB. Higher is better. Both 1-D, same length."""
    n = min(len(reference), len(estimate))
    ref, est = reference[:n], estimate[:n]
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.dot(target, target) + 1e-12) / (np.dot(noise, noise) + 1e-12)))


def _mono(path: Path) -> np.ndarray:
    y, _ = sf.read(path, always_2d=True)
    return y.mean(axis=1)


def _build_mix(tmpdir: Path) -> tuple[Path, Path]:
    """Return (mix_wav, ground_truth_vocals_wav)."""
    # Vocals: a simple melody
    vox = pretty_midi.PrettyMIDI()
    vi = pretty_midi.Instrument(program=VOCAL_PROGRAM)
    for i, p in enumerate([67, 69, 71, 72, 71, 69, 67, 64]):
        vi.notes.append(pretty_midi.Note(velocity=100, pitch=p, start=i * 0.6, end=i * 0.6 + 0.55))
    vox.instruments.append(vi)
    vox_wav = render_midi_to_wav(vox, tmpdir / "gt_vocals.wav")

    # Backing: chords + bass
    backing = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=BACKING_PROGRAMS[0])
    bass = pretty_midi.Instrument(program=BACKING_PROGRAMS[1])
    for i, root in enumerate([48, 53, 55, 48]):
        for off in (0, 4, 7):
            piano.notes.append(pretty_midi.Note(velocity=70, pitch=root + 12 + off,
                                                start=i * 1.2, end=i * 1.2 + 1.1))
        bass.notes.append(pretty_midi.Note(velocity=90, pitch=root - 12, start=i * 1.2,
                                           end=i * 1.2 + 1.1))
    backing.instruments.extend([piano, bass])
    backing_wav = render_midi_to_wav(backing, tmpdir / "gt_backing.wav")

    # Mix at equal RMS-ish
    v = _mono(vox_wav)
    b = _mono(backing_wav)
    n = max(len(v), len(b))
    v = np.pad(v, (0, n - len(v)))
    b = np.pad(b, (0, n - len(b)))
    mix = 0.6 * v + 0.6 * b
    peak = float(np.max(np.abs(mix))) or 1.0
    mix = (mix / peak * 0.9).astype(np.float32)
    mix_wav = tmpdir / "mix.wav"
    sf.write(mix_wav, np.stack([mix, mix], axis=1), 44_100, subtype="FLOAT")
    return mix_wav, vox_wav


def _htdemucs_vocals(mix_wav: Path, tmpdir: Path) -> np.ndarray:
    from demixer.core.ingest import ingest
    from demixer.core.separation import separate, write_stems
    audio = ingest(mix_wav)
    res = separate(audio, model_name="htdemucs")
    paths = write_stems(res, tmpdir / "htdemucs_stems")
    return _mono(paths["vocals"])


def _roformer_vocals(mix_wav: Path) -> np.ndarray:
    # Route through the same isolated-venv worker the production path uses
    # (audio_separator needs numpy>=2 and can't be imported in this env).
    #
    # NOTE: RoFormer is a *real-vocal* specialist and will not isolate the
    # synthetic GM "Voice Oohs" used in this controlled mix — expect it to score
    # poorly here. Use a real recording (library track) to see its true edge.
    from demixer.core.ingest import ingest
    from demixer.core.separation_roformer import roformer_vocals
    audio = ingest(mix_wav)
    vocals = roformer_vocals(audio)  # (channels, samples) @ 44.1k
    return vocals.mean(axis=0)


def run(verbose: bool = True) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        mix_wav, gt_vocals_wav = _build_mix(tmpdir)
        gt = _mono(gt_vocals_wav)

        ht = _htdemucs_vocals(mix_wav, tmpdir)
        rf = _roformer_vocals(mix_wav)

        sdr_ht = _si_sdr(gt, ht)
        sdr_rf = _si_sdr(gt, rf)

    if verbose:
        print(f"  htdemucs   vocals SI-SDR: {sdr_ht:6.2f} dB")
        print(f"  BS-RoFormer vocals SI-SDR: {sdr_rf:6.2f} dB")
        print(f"  delta (RoFormer - htdemucs): {sdr_rf - sdr_ht:+.2f} dB")
    return {"htdemucs": sdr_ht, "roformer": sdr_rf, "delta": sdr_rf - sdr_ht}


if __name__ == "__main__":
    print("Controlled vocals-recovery SI-SDR (synthetic mix, RoFormer vs htdemucs):\n")
    run()
