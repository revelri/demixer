"""Tempo, beat, and downbeat tracking — beat_this (primary) with librosa fallback.

Real-music eval (2026-05-24, 4 diverse tracks) showed beat_this:
  - matches librosa on tempo (±3 BPM of published BPM on all 4 tracks)
  - tracks micro-timing (~20 ms IBI IQR vs librosa's 0 ms rigid grid)
  - produces real downbeats from harmonic context (correctly identified 4/4 on
    standard pop; reasonable failures on polymetric/syncopated tracks).

beat_this is ~7× slower than librosa on CPU but absolute cost (~0.7s / 30s clip)
is negligible alongside Demucs separation.

librosa kicks in only if beat_this fails to load (missing checkpoint, no
network on first run, etc.). It produces naive `beats[::beats_per_bar]`
downbeats and lacks beat_this's micro-timing fidelity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import soundfile as sf

from demixer.core.ingest import IngestedAudio

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TempoBeats:
    tempo_bpm: float                  # global tempo (median of inter-beat intervals)
    beat_times_s: np.ndarray          # shape (n_beats,)
    downbeat_times_s: np.ndarray      # shape (n_downbeats,)
    beats_per_bar: int                # inferred from downbeats (beat_this) or assumed (librosa)
    method: str                       # "beat_this" | "librosa"
    confidence: float = 1.0           # 0..1 — see `_beat_confidence`
    reliable: bool = True             # confidence >= _RELIABLE_THRESHOLD


# Below this confidence the tempo/beat grid is not trustworthy (e.g. ambient or
# free-time material with no steady pulse). Callers should flag/suppress it
# rather than present a number. Tuned so steady-pulse pop/rock lands ~0.8-1.0
# and beatless ambient (the audit's Brian Eno "Lizard Point" → 7.9 BPM) → ~0.
_RELIABLE_THRESHOLD = 0.5

# A musical pulse lives roughly in this range; beat_this can emit a handful of
# widely-spaced "beats" on beatless audio, yielding absurd sub-40 BPM tempi.
_TEMPO_PLAUSIBLE_LO = 50.0
_TEMPO_PLAUSIBLE_HI = 200.0


def _beat_confidence(beat_times: np.ndarray, tempo_bpm: float, duration_s: float) -> float:
    """Heuristic 0..1 confidence that the beat grid reflects a real, steady pulse.

    Three independent factors, multiplied (any one failing kills confidence):
      • regularity  — fraction of inter-beat intervals near the median (a steady
                      pulse is near-uniform; scattered onsets are not)
      • plausibility — is the tempo in a musical range (ramps to 0 outside ~50-200)
      • coverage    — do the beats span most of the track (a few clustered beats
                      over a long ambient piece shouldn't read as a confident tempo)

    Beatless/free-time audio fails on plausibility (absurd tempo) and usually
    coverage too, driving the product toward 0.
    """
    if len(beat_times) < 8 or duration_s <= 0:
        return 0.0
    diffs = np.diff(beat_times)
    med = float(np.median(diffs))
    if med <= 0:
        return 0.0
    regularity = float(np.mean((diffs > 0.5 * med) & (diffs < 1.5 * med)))

    if _TEMPO_PLAUSIBLE_LO <= tempo_bpm <= _TEMPO_PLAUSIBLE_HI:
        plausibility = 1.0
    elif tempo_bpm < _TEMPO_PLAUSIBLE_LO:
        plausibility = max(0.0, tempo_bpm / _TEMPO_PLAUSIBLE_LO)
    else:
        plausibility = max(0.0, 1.0 - (tempo_bpm - _TEMPO_PLAUSIBLE_HI) / 100.0)

    coverage = min(1.0, float(beat_times[-1] - beat_times[0]) / (0.5 * duration_s))
    return float(regularity * plausibility * coverage)


def from_midi(midi_path: str | Path, *, audio_duration_s: float) -> TempoBeats:
    """Build an authoritative TempoBeats from a ground-truth MIDI file.

    Reads the tempo map and time signature from the MIDI and synthesizes a beat
    grid spanning `audio_duration_s`. Use this when the input is a render of a
    known MIDI (e.g. game-music corpora, MIDI-rendered demos) — the MIDI's
    tempo and meter are exact, where audio-side beat tracking is approximate.

    Falls back to a single 120 BPM / 4-4 grid for MIDIs missing tempo events.
    Confidence is 1.0; method is 'midi-groundtruth'.
    """
    import pretty_midi  # lazy — pretty_midi is a heavy import

    pm = pretty_midi.PrettyMIDI(str(Path(midi_path)))

    # Tempo: average BPM weighted by audio duration (handles tempo maps).
    times, tempi = pm.get_tempo_changes()
    if len(tempi):
        if len(tempi) == 1:
            tempo_bpm = float(tempi[0])
        else:
            segment_starts = np.asarray(times, dtype=np.float64)
            segment_ends = np.append(segment_starts[1:], audio_duration_s)
            durations = np.clip(segment_ends - segment_starts, 0.0, None)
            tempo_bpm = float(
                np.average(np.asarray(tempi, dtype=np.float64),
                           weights=durations + 1e-9)
            )
    else:
        tempo_bpm = 120.0

    # Time signature: first one in the MIDI wins; default 4/4.
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        beats_per_bar = int(ts.numerator)
    else:
        beats_per_bar = 4

    # Synthesize a beat grid spanning the audio. pretty_midi.get_beats() can be
    # used, but it stops at the last note — we want coverage of the whole
    # render, including trailing silence.
    beat_period = 60.0 / tempo_bpm if tempo_bpm > 0 else 0.5
    beat_times = np.arange(0.0, audio_duration_s, beat_period, dtype=np.float64)
    downbeat_times = beat_times[::max(1, beats_per_bar)].copy()

    return TempoBeats(
        tempo_bpm=tempo_bpm,
        beat_times_s=beat_times,
        downbeat_times_s=downbeat_times,
        beats_per_bar=beats_per_bar,
        method="midi-groundtruth",
        confidence=1.0,
        reliable=True,
    )


def find_sibling_midi(audio_path: str | Path) -> Path | None:
    """Return a sibling MIDI file matching `audio_path` (same stem, .mid/.midi),
    or None. Used by the CLI's --midi-hint auto mode."""
    p = Path(audio_path)
    for ext in (".mid", ".MID", ".midi", ".MIDI"):
        candidate = p.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def estimate(audio: IngestedAudio, *, beats_per_bar_hint: int = 4) -> TempoBeats:
    """Estimate tempo, beats, and downbeats from the ingested mix.

    `beats_per_bar_hint` is only consulted by the librosa fallback. beat_this
    infers meter from the audio.
    """
    if beats_per_bar_hint < 1:
        raise ValueError("beats_per_bar_hint must be >= 1")

    try:
        return _estimate_beat_this(audio)
    except Exception as exc:
        log.warning("beat_this failed (%s); falling back to librosa", exc)
        return _estimate_librosa(audio, beats_per_bar=beats_per_bar_hint)


def _tempo_from_beats(beat_times: np.ndarray) -> float:
    """Robust, sub-grid-resolution tempo from a beat sequence.

    beat_this emits beat times on a ~0.02 s frame grid, so `60/median(diffs)`
    snaps the reported BPM to coarse values (111.11, 120, 130.43 …) and can't
    express, say, 129 or 126. Averaging the inter-beat intervals recovers fine
    resolution. We first drop outlier intervals (missed/extra beats show up as
    ~2× or ~0.5× the median) so the mean isn't dragged by gaps, then take the
    mean of the survivors.
    """
    diffs = np.diff(beat_times)
    med = float(np.median(diffs))
    if med <= 0:
        return float(60.0 / max(med, 1e-6))
    inliers = diffs[(diffs > 0.5 * med) & (diffs < 1.5 * med)]
    period = float(np.mean(inliers)) if inliers.size else med
    return float(60.0 / period)


def _beats_per_bar(beat_times: np.ndarray, downbeat_times: np.ndarray) -> int:
    """Infer meter as the MEDIAN beats between consecutive downbeats.

    A global `round(len(beats)/len(downbeats))` ratio is skewed by sections
    where beat_this mis-spaces downbeats (e.g. detecting them every 2 beats in
    an intro), which dragged 4/4 tracks down to a reported 3. The median of the
    actual downbeat-to-downbeat spacings (in beat-index space) is robust to those
    local errors — on the Sturgill "Life of Sin" case it recovers 4 where the
    ratio gave 3.
    """
    if len(downbeat_times) < 2 or len(beat_times) < 2:
        return 4  # not enough downbeats to infer — assume common-time
    # Map each downbeat to its nearest beat index, then take spacings between them.
    idx = np.array([int(np.argmin(np.abs(beat_times - d))) for d in downbeat_times])
    spacings = np.diff(idx)
    spacings = spacings[spacings > 0]
    if spacings.size == 0:
        return 4
    return int(round(float(np.median(spacings))))


def _estimate_beat_this(audio: IngestedAudio) -> TempoBeats:
    from beat_this.inference import File2Beats

    # beat_this's File2Beats accepts a file path only; write a temp WAV. The
    # ingest pipeline already gives us 44.1 kHz stereo float32, matching what
    # the model expects.
    with NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, audio.samples.T, audio.sample_rate, subtype="FLOAT")
        f2b = File2Beats(device="cpu")  # CPU is fast enough; GPU optional later
        beats_arr, downbeats_arr = f2b(f.name)

    beat_times = np.asarray(beats_arr, dtype=np.float64)
    downbeat_times = np.asarray(downbeats_arr, dtype=np.float64)

    if len(beat_times) < 2:
        raise RuntimeError(f"beat_this returned only {len(beat_times)} beats; cannot infer tempo")

    tempo_bpm = _tempo_from_beats(beat_times)
    beats_per_bar_raw = _beats_per_bar(beat_times, downbeat_times)
    confidence = _beat_confidence(beat_times, tempo_bpm, audio.duration_s)

    # A meter < 2 is structurally meaningless — every downstream barline,
    # chord-rhythm boundary, and reharmonization unit inherits it. beat_this
    # reports bpb=1 when downbeats coincide with every beat (slow/simple loops,
    # ambient material). Fall back to common time and demote reliability.
    if beats_per_bar_raw < 2:
        log.warning(
            "beat_this inferred beats_per_bar=%d (downbeats on every beat); "
            "falling back to 4/4 — tempo flagged unreliable",
            beats_per_bar_raw,
        )
        beats_per_bar = 4
        reliable = False
    else:
        beats_per_bar = beats_per_bar_raw
        reliable = confidence >= _RELIABLE_THRESHOLD

    return TempoBeats(
        tempo_bpm=tempo_bpm,
        beat_times_s=beat_times,
        downbeat_times_s=downbeat_times,
        beats_per_bar=beats_per_bar,
        method="beat_this",
        confidence=confidence,
        reliable=reliable,
    )


def _estimate_librosa(audio: IngestedAudio, *, beats_per_bar: int) -> TempoBeats:
    import librosa

    mono = audio.samples.mean(axis=0).astype(np.float32, copy=False)
    tempo, beat_frames = librosa.beat.beat_track(y=mono, sr=audio.sample_rate, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=audio.sample_rate)
    tempo_bpm = float(np.asarray(tempo).reshape(-1)[0])
    downbeats = beat_times[::beats_per_bar].copy()
    confidence = _beat_confidence(beat_times, tempo_bpm, audio.duration_s)
    return TempoBeats(
        tempo_bpm=tempo_bpm,
        beat_times_s=beat_times,
        downbeat_times_s=downbeats,
        beats_per_bar=beats_per_bar,
        method="librosa",
        confidence=confidence,
        reliable=confidence >= _RELIABLE_THRESHOLD,
    )
