"""Grounded accuracy matrix against human-annotated ground truth (Isophonics).

Unlike the synthetic evals, this scores the real analysis pipeline (key, chords,
tempo) against *hand-labelled* references for the Beatles + Queen catalogue, using
standard MIREX scoring (mir_eval). Annotations are vendored under
tests/groundtruth/isophonics/ (see fetch_isophonics.sh).

Why human ground truth: scoring against algorithm-derived sources (TuneBat/Spotify)
measures agreement-with-a-peer, not correctness, and hides shared errors. Isophonics
labels are human, so a match means we're actually right.

Scope (v1):
  - chords: mir_eval majmin + thirds (root + third quality). Our model emits only
    maj/min/N, so extensions aren't scored — majmin/thirds is the honest ceiling.
  - key:    mir_eval weighted score (partial credit for fifth/relative/parallel).
            Reference = longest-duration key segment; modal refs are skipped.
  - tempo:  octave-equivalent accuracy (±4%) vs median inter-beat-interval of the
            reference beat track.

Run on demand (slow — full-track autochord over ~200 tracks), not in pytest:
    uv run python -m demixer.eval.groundtruth_matrix --music-root /path/to/Music
    uv run python -m demixer.eval.groundtruth_matrix --artist Queen --limit 5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ISO_ROOT = Path(__file__).resolve().parents[3] / "tests" / "groundtruth" / "isophonics"
AUDIO_EXTS = {".flac", ".mp3", ".m4a", ".wav", ".ogg", ".opus", ".aiff", ".wma"}
TEMPO_TOL = 0.04  # fractional tolerance for octave-equivalent tempo match
TEMPO_RATIOS = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)


def normalize_title(name: str) -> str:
    """Reduce a filename or annotation stem to a comparable track-title key.

    Strips extension, leading track number, the artist name, and all punctuation:
    '01 The Beatles - I Saw Her Standing There.flac' and
    '01_-_I_Saw_Her_Standing_There.lab' both -> 'isawherstandingthere'.
    """
    stem = Path(name).stem
    stem = stem.replace("_", " ")
    stem = re.sub(r"^\s*\d+\s*[-.]*\s*", "", stem)  # leading track number
    stem = re.sub(r"\b(the\s+)?beatles\b|\bqueen\b", "", stem, flags=re.I)
    stem = re.sub(r"[^a-z0-9]", "", stem.lower())
    return stem


def build_annotation_index() -> dict[str, dict[str, Path]]:
    """Map normalized track title -> {chord, key, beat} annotation paths."""
    index: dict[str, dict[str, Path]] = {}
    for kind, sub, ext in (("chord", "chordlab", ".lab"), ("key", "keylab", ".lab"), ("beat", "beat", ".txt")):
        root = ISO_ROOT / sub
        if not root.is_dir():
            continue
        for path in root.rglob(f"*{ext}"):
            index.setdefault(normalize_title(path.name), {})[kind] = path
    return index


_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _transpose_harte(label: str, k: int) -> str:
    """Shift a Harte chord root by k semitones, keeping the quality. 'N'/'X' pass through.

    Used to detect a global pitch mismatch between our audio and the reference master
    (some Beatles mono remasters are pitched a semitone off the annotated master).
    """
    if label in ("N", "X") or k == 0:
        return label
    m = re.match(r"([A-G][#b]?)(:.*)?$", label)
    if not m:
        return label
    import mir_eval

    pc = mir_eval.chord.pitch_class_to_semitone(m.group(1))
    return _SHARP_NAMES[(pc + k) % 12] + (m.group(2) or "")


def _short_to_harte(label: str) -> str:
    """ChordSegment short label -> Harte for mir_eval: 'Em'->'E:min', 'C'->'C:maj'.

    autochord emits only maj/min/N in practice; the qualifier branch keeps richer
    labels ('G7'->'G:7') valid for mir_eval should a future model produce them.
    """
    if label in ("N", "X") or not label:
        return "N"
    if ":" in label:
        return label  # already Harte
    m = re.match(r"([A-G][#b]?)(.*)", label)
    if not m:
        return "N"
    root, rest = m.group(1), m.group(2)
    if rest == "":
        return root + ":maj"
    if rest == "m":
        return root + ":min"
    return f"{root}:{rest}"


def _parse_keylab(path: Path) -> str | None:
    """Longest-duration 'Key <tonic>[:mode]' segment as a mir_eval key string."""
    best_dur, best_key = 0.0, None
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[2].lower() != "key":
            continue
        start, end, tonic = float(parts[0]), float(parts[1]), parts[3]
        if ":" in tonic:
            root, mode = tonic.split(":", 1)
            mode = mode.lower()
        else:
            root, mode = tonic, "major"
        if mode not in ("major", "minor"):  # modal — mir_eval can't score it
            continue
        if end - start > best_dur:
            best_dur, best_key = end - start, f"{root} {mode}"
    return best_key


def _parse_beat_tempo(path: Path) -> float | None:
    """Tempo (BPM) from the median inter-beat interval of a reference beat track."""
    times = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts:
            try:
                times.append(float(parts[0]))
            except ValueError:
                pass
    if len(times) < 4:
        return None
    arr = np.sort(np.asarray(times, dtype=float))
    ibis = np.diff(arr)
    ibis = ibis[ibis > 0.05]  # drop spurious near-zero gaps
    if ibis.size == 0:
        return None
    return float(60.0 / np.median(ibis))


def _tempo_matches(est: float, ref: float) -> bool:
    return any(abs(est - ref * r) <= TEMPO_TOL * ref * r for r in TEMPO_RATIOS)


@dataclass
class TrackScore:
    title: str
    chord_majmin: float | None = None
    chord_thirds: float | None = None
    chord_sevenths: float | None = None  # rewards correct 7ths/extensions (BTC's edge over autochord)
    chord_majmin_ti: float | None = None  # best over 12 transpositions (shape, master-pitch invariant)
    pitch_shift: int = 0  # semitones to best-match the reference master (0 = aligned)
    key_score: float | None = None
    tempo_ok: bool | None = None
    detail: str = ""


def score_track(
    audio_path: Path, ann: dict[str, Path], *, separate: bool = False, chord_engine: str = "autochord"
) -> TrackScore:
    import mir_eval

    from demixer.core import ingest as ingest_mod
    from demixer.core.analysis import chords as chords_mod
    from demixer.core.analysis import key as key_mod
    from demixer.core.analysis import tempo_beats as tempo_mod

    if chord_engine == "btc":
        from demixer.core.analysis import chords_btc
        estimate_chords = chords_btc.estimate
    else:
        estimate_chords = chords_mod.estimate

    ts = TrackScore(title=audio_path.stem)
    audio = ingest_mod.ingest(audio_path)

    # When --separate, feed the key stage the drums-excluded harmonic mix, matching
    # the production pipeline (more accurate than full-mix — see key.harmonic_mono).
    key_mono = None
    if separate and "key" in ann:
        from demixer.core import separation as sep_mod

        stems = sep_mod.separate(audio).stems
        key_mono = key_mod.harmonic_mono(stems)

    if "chord" in ann:
        ref_int, ref_lab = mir_eval.io.load_labeled_intervals(str(ann["chord"]))
        segs = estimate_chords(audio)
        if segs:
            est_int = np.array([[s.start_s, s.end_s] for s in segs])
            est_lab = [_short_to_harte(s.label) for s in segs]
            res = mir_eval.chord.evaluate(ref_int, ref_lab, est_int, est_lab)
            ts.chord_majmin = float(res["majmin"])
            ts.chord_thirds = float(res["thirds"])
            ts.chord_sevenths = float(res["sevenths"])
            # best majmin over all 12 transpositions isolates harmonic-shape
            # correctness from absolute-pitch mismatch with the reference master
            ti = [
                (k, float(mir_eval.chord.evaluate(ref_int, ref_lab, est_int, [_transpose_harte(l, k) for l in est_lab])["majmin"]))
                for k in range(12)
            ]
            ts.pitch_shift, ts.chord_majmin_ti = max(ti, key=lambda x: x[1])

    if "key" in ann:
        ref_key = _parse_keylab(ann["key"])
        if ref_key:
            ke = key_mod.estimate(audio, mono=key_mono)
            est_key = f"{ke.root} {ke.scale}"
            try:
                ts.key_score = float(mir_eval.key.evaluate(ref_key, est_key)["Weighted Score"])
            except ValueError:
                ts.key_score = None  # unparseable enharmonic — leave n/a

    if "beat" in ann:
        ref_tempo = _parse_beat_tempo(ann["beat"])
        if ref_tempo:
            tb = tempo_mod.estimate(audio)
            ts.tempo_ok = _tempo_matches(tb.tempo_bpm, ref_tempo)
            ts.detail = f"tempo {tb.tempo_bpm:.1f} vs ref {ref_tempo:.1f}"
    return ts


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def run(
    music_root: Path,
    *,
    artist: str | None,
    album: str | None,
    limit: int | None,
    separate: bool = False,
    chord_engine: str = "autochord",
) -> int:
    index = build_annotation_index()
    if not index:
        print(f"no annotations under {ISO_ROOT} — run tests/groundtruth/fetch_isophonics.sh", file=sys.stderr)
        return 1

    matched: list[tuple[Path, dict[str, Path]]] = []
    for audio_path in sorted(music_root.rglob("*")):
        if audio_path.suffix.lower() not in AUDIO_EXTS:
            continue
        if artist and artist.lower() not in str(audio_path).lower():
            continue
        if album and album.lower() not in str(audio_path).lower():
            continue
        ann = index.get(normalize_title(audio_path.name))
        if ann:
            matched.append((audio_path, ann))

    # de-dup by title (mono/stereo/compilation copies map to the same annotation)
    seen: set[str] = set()
    unique = []
    for p, a in matched:
        key = normalize_title(p.name)
        if key not in seen:
            seen.add(key)
            unique.append((p, a))
    if limit:
        unique = unique[:limit]

    mode = "drums-excluded key (--separate)" if separate else "full-mix key"
    print(f"matched {len(unique)} annotated tracks under {music_root}  [chords={chord_engine}, {mode}]")
    scores: list[TrackScore] = []
    for i, (path, ann) in enumerate(unique, 1):
        try:
            ts = score_track(path, ann, separate=separate, chord_engine=chord_engine)
        except Exception as exc:  # one bad file shouldn't sink the matrix
            print(f"  [{i}/{len(unique)}] SKIP {path.name}: {type(exc).__name__}: {exc}")
            continue
        scores.append(ts)
        mm = f"{ts.chord_majmin:.2f}" if ts.chord_majmin is not None else " - "
        th = f"{ts.chord_thirds:.2f}" if ts.chord_thirds is not None else " - "
        ky = f"{ts.key_score:.2f}" if ts.key_score is not None else " - "
        tp = {True: "ok", False: "X", None: "-"}[ts.tempo_ok]
        shift = f" shift+{ts.pitch_shift}" if ts.pitch_shift else ""
        print(f"  [{i}/{len(unique)}] majmin={mm} thirds={th} key={ky} tempo={tp:2}{shift}  {ts.title}  ({ts.detail})")

    print("\n=== AGGREGATE (Isophonics ground truth) ===")
    mm = _mean([s.chord_majmin for s in scores if s.chord_majmin is not None])
    ti = _mean([s.chord_majmin_ti for s in scores if s.chord_majmin_ti is not None])
    th = _mean([s.chord_thirds for s in scores if s.chord_thirds is not None])
    sv = _mean([s.chord_sevenths for s in scores if s.chord_sevenths is not None])
    ky = _mean([s.key_score for s in scores if s.key_score is not None])
    tempo_vals = [s.tempo_ok for s in scores if s.tempo_ok is not None]
    shifted = [s for s in scores if s.pitch_shift]
    print(f"  chord majmin (absolute):           {mm:.3f}" if mm is not None else "  chord majmin: n/a")
    print(f"  chord majmin (transposition-inv):  {ti:.3f}" if ti is not None else "  chord majmin TI: n/a")
    print(f"  chord thirds (absolute):           {th:.3f}" if th is not None else "  chord thirds: n/a")
    print(f"  chord sevenths (absolute):         {sv:.3f}" if sv is not None else "  chord sevenths: n/a")
    print(f"  key (MIREX weighted):              {ky:.3f}" if ky is not None else "  key: n/a")
    if tempo_vals:
        print(f"  tempo accuracy (octave-equiv):     {sum(tempo_vals)/len(tempo_vals):.3f}  ({sum(tempo_vals)}/{len(tempo_vals)})")
    print(f"  pitch-mismatched tracks (master):  {len(shifted)}  {[ (s.title[:18], f'+{s.pitch_shift}') for s in shifted ][:8]}")
    print(f"  tracks scored: {len(scores)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Grounded accuracy matrix vs Isophonics ground truth")
    ap.add_argument(
        "--music-root",
        type=Path,
        default=Path(os.environ.get("DEMIXER_MUSIC_ROOT", "")),
        help="root of your local music library (or set DEMIXER_MUSIC_ROOT)",
    )
    ap.add_argument("--artist", help="substring filter, e.g. Queen")
    ap.add_argument("--album", help="substring filter, e.g. 'Rubber Soul'")
    ap.add_argument("--limit", type=int, help="cap number of tracks (for quick runs)")
    ap.add_argument(
        "--chords",
        choices=["autochord", "btc"],
        default="autochord",
        help="chord engine to score (autochord = default pipeline; btc = large-vocab transformer)",
    )
    ap.add_argument(
        "--separate",
        action="store_true",
        help="run separation and feed drums-excluded harmonic mix to the key stage "
        "(matches the production pipeline; slower but faithful key accuracy)",
    )
    args = ap.parse_args(argv)
    return run(
        args.music_root,
        artist=args.artist,
        album=args.album,
        limit=args.limit,
        separate=args.separate,
        chord_engine=args.chords,
    )


if __name__ == "__main__":
    raise SystemExit(main())
