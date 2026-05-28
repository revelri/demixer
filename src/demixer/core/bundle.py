"""`.demixer` bundle — the unified output of a pipeline run.

Layout (under bundle_dir/):

    manifest.json          # bundle schema version, demixer version, model versions
    analysis.json          # tempo, beats, downbeats, key, source metadata
    stems/<name>.wav       # separated stems at 44.1k stereo float32
    midi/<name>.mid        # per-stem polyphonic MIDI (skipped for drums until wired)

The bundle is also zipped to `bundle_dir.with_suffix('.demixer')` so it can be
distributed as a single file. The directory form is the source of truth during
the run; the zip is a snapshot.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from demixer import __version__ as DEMIXER_VERSION
from demixer.core.analysis.chords import ChordSegment
from demixer.core.analysis.key import KeyEstimate
from demixer.core.analysis.tempo_beats import TempoBeats
from demixer.core.ingest import IngestedAudio

BUNDLE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BundleMetadata:
    audio: IngestedAudio
    tempo_beats: TempoBeats
    key: KeyEstimate
    chords: list[ChordSegment] | None  # None when chord recognition was skipped
    separation_model: str           # "htdemucs" | "htdemucs_6s" | …
    transcription_model: str        # e.g. "basic-pitch-icassp-2022"


def _analysis_dict(meta: BundleMetadata) -> dict[str, Any]:
    audio = meta.audio
    tb = meta.tempo_beats
    k = meta.key
    return {
        "schema": BUNDLE_SCHEMA_VERSION,
        "source": {
            "path": str(audio.source_path),
            "sha256": audio.sha256,
            "duration_s": audio.duration_s,
            "sample_rate": audio.sample_rate,
            "integrated_lufs_before": audio.integrated_lufs_before,
            "integrated_lufs_after": audio.integrated_lufs_after,
        },
        "tempo": {
            "bpm": tb.tempo_bpm,
            "beats_per_bar": tb.beats_per_bar,
            "method": tb.method,
            "confidence": tb.confidence,
            "reliable": tb.reliable,
            "beat_times_s": tb.beat_times_s.tolist(),
            "downbeat_times_s": tb.downbeat_times_s.tolist(),
        },
        "key": {
            "root": k.root,
            "scale": k.scale,
            "sharps": k.sharps,
            "strength": k.strength,
        },
        "chords": [
            {"start_s": c.start_s, "end_s": c.end_s, "label": c.label}
            for c in (meta.chords or [])
        ] if meta.chords is not None else None,
    }


def _manifest_dict(meta: BundleMetadata, stem_files: list[str], midi_files: list[str]) -> dict[str, Any]:
    return {
        "schema": BUNDLE_SCHEMA_VERSION,
        "demixer_version": DEMIXER_VERSION,
        "models": {
            "separation": meta.separation_model,
            "transcription": meta.transcription_model,
            "tempo_beats": meta.tempo_beats.method,
        },
        "files": {
            "analysis": "analysis.json",
            "stems": stem_files,
            "midi": midi_files,
        },
    }


def write_bundle(
    bundle_dir: str | Path,
    meta: BundleMetadata,
    stem_paths: dict[str, Path],
    midi_paths: dict[str, Path],
    *,
    zip_output: bool = True,
) -> tuple[Path, Path | None]:
    """Materialize the bundle directory and (optionally) zip it.

    `stem_paths` / `midi_paths` are existing files produced by separation /
    transcription; they're copied into the bundle layout. Returns
    `(bundle_dir, zip_path_or_None)`.
    """
    bundle_dir = Path(bundle_dir)
    stems_dir = bundle_dir / "stems"
    midi_dir = bundle_dir / "midi"
    stems_dir.mkdir(parents=True, exist_ok=True)
    midi_dir.mkdir(parents=True, exist_ok=True)

    relative_stems: list[str] = []
    for name, src in stem_paths.items():
        dst = stems_dir / f"{name}.wav"
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())
        relative_stems.append(f"stems/{dst.name}")

    relative_midi: list[str] = []
    for name, src in midi_paths.items():
        dst = midi_dir / f"{name}.mid"
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())
        relative_midi.append(f"midi/{dst.name}")

    (bundle_dir / "analysis.json").write_text(
        json.dumps(_analysis_dict(meta), indent=2)
    )
    (bundle_dir / "manifest.json").write_text(
        json.dumps(_manifest_dict(meta, relative_stems, relative_midi), indent=2)
    )

    zip_path = zip_bundle(bundle_dir) if zip_output else None
    return bundle_dir, zip_path


def zip_bundle(bundle_dir: str | Path, *, archive_stems: bool = True) -> Path:
    """Zip the entire bundle directory into a sibling `.demixer` archive.

    Call this AFTER all artifacts (stems, MIDI, analysis, DAW projects, score)
    have been written so the single-file bundle is complete. Re-running it
    overwrites any earlier partial archive.

    `archive_stems=False` excludes the loose `stems/` subtree from the archive
    *iff* a `.dawproject` is present in the bundle — the dawproject already
    embeds the same stem audio, so including the loose copy duplicates it
    (typically the dominant cost of the archive). Loose stems are kept on
    disk in the bundle dir either way; this only affects the zipped form.
    Falls back to including stems when no dawproject exists, so RPP / FL
    Studio projects extracted from the archive still resolve their audio.
    """
    bundle_dir = Path(bundle_dir)
    zip_path = bundle_dir.with_suffix(".demixer")
    # File extensions whose payload is already compressed — re-deflating wastes
    # CPU for ~0 % gain. Store them; deflate the rest.
    _PRECOMPRESSED = {".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac",
                      ".png", ".jpg", ".jpeg", ".webp",
                      ".dawproject", ".mscz", ".demixer", ".zip"}

    has_dawproject = any(bundle_dir.glob("*.dawproject"))
    skip_stems = (not archive_stems) and has_dawproject

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle_dir.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(bundle_dir)
            if skip_stems and rel.parts and rel.parts[0] == "stems":
                continue
            comp = (zipfile.ZIP_STORED if p.suffix.lower() in _PRECOMPRESSED
                    else zipfile.ZIP_DEFLATED)
            z.write(p, arcname=rel, compress_type=comp)
    return zip_path


def read_manifest(bundle_dir_or_zip: str | Path) -> dict[str, Any]:
    """Return the manifest from a bundle directory or zipped .demixer."""
    path = Path(bundle_dir_or_zip)
    if path.is_dir():
        return json.loads((path / "manifest.json").read_text())
    with zipfile.ZipFile(path) as z, z.open("manifest.json") as f:
        return json.loads(f.read())
