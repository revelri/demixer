"""Tests for bundle writing — uses fake stems/MIDI to avoid the heavy pipeline."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from demixer.core.analysis.chords import ChordSegment
from demixer.core.analysis.key import KeyEstimate
from demixer.core.analysis.tempo_beats import TempoBeats
from demixer.core.bundle import (
    BUNDLE_SCHEMA_VERSION,
    BundleMetadata,
    read_manifest,
    write_bundle,
    zip_bundle,
)
from demixer.core.ingest import IngestedAudio


def _fake_audio() -> IngestedAudio:
    return IngestedAudio(
        samples=np.zeros((2, 44_100), dtype=np.float32),
        sample_rate=44_100,
        duration_s=1.0,
        source_path=Path("/dev/null"),
        sha256="a" * 64,
        integrated_lufs_before=-30.0,
        integrated_lufs_after=-23.0,
    )


def _fake_meta() -> BundleMetadata:
    return BundleMetadata(
        audio=_fake_audio(),
        tempo_beats=TempoBeats(
            tempo_bpm=120.0,
            beat_times_s=np.array([0.0, 0.5, 1.0, 1.5]),
            downbeat_times_s=np.array([0.0, 2.0]),
            beats_per_bar=4,
            method="beat_this",
        ),
        key=KeyEstimate(root="C", scale="major", strength=0.85, sharps=0),
        chords=[
            ChordSegment(start_s=0.0, end_s=2.0, label="C"),
            ChordSegment(start_s=2.0, end_s=4.0, label="Am"),
        ],
        separation_model="htdemucs",
        transcription_model="basic-pitch-icassp-2022",
    )


def _write_fake_stem(path: Path) -> None:
    sf.write(path, np.zeros((44_100, 2), dtype=np.float32), 44_100, subtype="FLOAT")


def _write_fake_midi(path: Path) -> None:
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5))
    midi.instruments.append(inst)
    midi.write(str(path))


@pytest.fixture
def populated_inputs(tmp_path: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    src_stems_dir = tmp_path / "src_stems"
    src_midi_dir = tmp_path / "src_midi"
    src_stems_dir.mkdir()
    src_midi_dir.mkdir()
    stems = {}
    midis = {}
    for name in ("vocals", "bass", "other"):
        w = src_stems_dir / f"{name}.wav"
        _write_fake_stem(w)
        stems[name] = w
        m = src_midi_dir / f"{name}.mid"
        _write_fake_midi(m)
        midis[name] = m
    return stems, midis


def test_write_bundle_layout(tmp_path: Path, populated_inputs) -> None:
    stems, midis = populated_inputs
    bundle_dir, zip_path = write_bundle(tmp_path / "out", _fake_meta(), stems, midis)

    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "analysis.json").exists()
    assert sorted(p.name for p in (bundle_dir / "stems").iterdir()) == [
        "bass.wav", "other.wav", "vocals.wav"
    ]
    assert sorted(p.name for p in (bundle_dir / "midi").iterdir()) == [
        "bass.mid", "other.mid", "vocals.mid"
    ]
    assert zip_path is not None and zip_path.exists() and zip_path.suffix == ".demixer"

    # Manifest reflects what we wrote
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest["schema"] == BUNDLE_SCHEMA_VERSION
    assert manifest["models"]["separation"] == "htdemucs"
    assert manifest["models"]["tempo_beats"] == "beat_this"
    assert sorted(manifest["files"]["stems"]) == [
        "stems/bass.wav", "stems/other.wav", "stems/vocals.wav"
    ]


def test_analysis_json_round_trip_values(tmp_path: Path, populated_inputs) -> None:
    stems, midis = populated_inputs
    meta = _fake_meta()
    bundle_dir, _ = write_bundle(tmp_path / "out", meta, stems, midis)

    analysis = json.loads((bundle_dir / "analysis.json").read_text())
    assert analysis["source"]["sha256"] == "a" * 64
    assert analysis["tempo"]["bpm"] == 120.0
    assert analysis["tempo"]["method"] == "beat_this"
    assert analysis["tempo"]["beat_times_s"] == [0.0, 0.5, 1.0, 1.5]
    assert analysis["key"]["root"] == "C"
    assert analysis["key"]["scale"] == "major"
    assert analysis["chords"] == [
        {"start_s": 0.0, "end_s": 2.0, "label": "C"},
        {"start_s": 2.0, "end_s": 4.0, "label": "Am"},
    ]


def test_zip_contains_all_files(tmp_path: Path, populated_inputs) -> None:
    stems, midis = populated_inputs
    _, zip_path = write_bundle(tmp_path / "out", _fake_meta(), stems, midis)
    assert zip_path is not None

    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
    assert {"manifest.json", "analysis.json"} <= names
    assert {f"stems/{s}.wav" for s in ("vocals", "bass", "other")} <= names
    assert {f"midi/{s}.mid" for s in ("vocals", "bass", "other")} <= names


def test_read_manifest_works_from_zip(tmp_path: Path, populated_inputs) -> None:
    stems, midis = populated_inputs
    _, zip_path = write_bundle(tmp_path / "out", _fake_meta(), stems, midis)
    manifest = read_manifest(zip_path)
    assert manifest["schema"] == BUNDLE_SCHEMA_VERSION


def test_read_manifest_works_from_dir(tmp_path: Path, populated_inputs) -> None:
    stems, midis = populated_inputs
    bundle_dir, _ = write_bundle(tmp_path / "out", _fake_meta(), stems, midis, zip_output=False)
    manifest = read_manifest(bundle_dir)
    assert manifest["schema"] == BUNDLE_SCHEMA_VERSION


def test_zip_can_be_disabled(tmp_path: Path, populated_inputs) -> None:
    stems, midis = populated_inputs
    bundle_dir, zip_path = write_bundle(
        tmp_path / "out", _fake_meta(), stems, midis, zip_output=False
    )
    assert zip_path is None
    assert (bundle_dir / "manifest.json").exists()


def test_cli_help_doesnt_crash(capsys) -> None:
    from demixer.cli.__main__ import main
    with pytest.raises(SystemExit) as excinfo:
        main(["process", "--help"])
    # argparse exits 0 on --help
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "demixer" in out.lower()


def test_zip_bundle_captures_files_added_after_core_write(tmp_path: Path, populated_inputs) -> None:
    """Regression: the .demixer archive must include DAW projects + score written
    AFTER the core bundle (they used to be omitted by zipping mid-pipeline)."""
    stems, midis = populated_inputs
    bundle_dir, _ = write_bundle(tmp_path / "out", _fake_meta(), stems, midis, zip_output=False)
    # Simulate the later pipeline stages dropping files into the bundle dir
    (bundle_dir / "song.rpp").write_text("<REAPER_PROJECT>")
    (bundle_dir / "song.dawproject").write_bytes(b"PK\x03\x04fake")
    (bundle_dir / "score.musicxml").write_text("<score-partwise/>")
    (bundle_dir / "score").mkdir()
    (bundle_dir / "score" / "page-01.svg").write_text("<svg/>")

    zip_path = zip_bundle(bundle_dir)
    assert zip_path.suffix == ".demixer" and zip_path.exists()
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
    # Core + all later artifacts present
    assert {"manifest.json", "analysis.json"} <= names
    assert {"song.rpp", "song.dawproject", "score.musicxml", "score/page-01.svg"} <= names
    assert {f"stems/{s}.wav" for s in ("vocals", "bass", "other")} <= names


def test_compact_archive_drops_stems_when_dawproject_present(
    tmp_path: Path, populated_inputs,
) -> None:
    stems, midis = populated_inputs
    bundle_dir, _ = write_bundle(tmp_path / "out", _fake_meta(), stems, midis, zip_output=False)
    (bundle_dir / "song.dawproject").write_bytes(b"PK\x03\x04fake")

    zip_path = zip_bundle(bundle_dir, archive_stems=False)
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
    assert not any(n.startswith("stems/") for n in names), \
        "loose stems must be excluded from a compact archive when a dawproject is present"
    assert "song.dawproject" in names
    assert {"manifest.json", "analysis.json"} <= names


def test_compact_archive_keeps_stems_when_no_dawproject(
    tmp_path: Path, populated_inputs,
) -> None:
    """If no dawproject is in the bundle, stems must stay in the archive even
    when archive_stems=False — otherwise RPP / FL stem references would dangle."""
    stems, midis = populated_inputs
    bundle_dir, _ = write_bundle(tmp_path / "out", _fake_meta(), stems, midis, zip_output=False)
    # No .dawproject written

    zip_path = zip_bundle(bundle_dir, archive_stems=False)
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
    assert any(n.startswith("stems/") for n in names), \
        "stems must still be archived when no dawproject is present"
