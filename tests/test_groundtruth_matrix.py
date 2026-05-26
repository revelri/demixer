"""Unit tests for the Isophonics ground-truth matrix harness (logic only, no audio)."""

from __future__ import annotations

from demixer.eval import groundtruth_matrix as gm


def test_normalize_title_matches_library_and_annotation():
    lib = gm.normalize_title("01 The Beatles - I Saw Her Standing There.flac")
    ann = gm.normalize_title("01_-_I_Saw_Her_Standing_There.lab")
    assert lib == ann == "isawherstandingthere"


def test_normalize_title_strips_queen_and_punctuation():
    assert gm.normalize_title("05 Queen - Don't Stop Me Now.mp3") == gm.normalize_title(
        "07_-_Don't_Stop_Me_Now.lab"
    )


def test_short_to_harte():
    assert gm._short_to_harte("C") == "C:maj"
    assert gm._short_to_harte("Em") == "E:min"
    assert gm._short_to_harte("F#m") == "F#:min"
    assert gm._short_to_harte("N") == "N"
    assert gm._short_to_harte("") == "N"
    assert gm._short_to_harte("G7") == "G:7"  # richer quality -> valid Harte


def test_transpose_harte():
    assert gm._transpose_harte("C:maj", 0) == "C:maj"
    assert gm._transpose_harte("C:maj", 1) == "C#:maj"
    assert gm._transpose_harte("B:min", 1) == "C:min"  # wraps
    assert gm._transpose_harte("A#:maj", 1) == "B:maj"  # Bb spelled sharp
    assert gm._transpose_harte("N", 5) == "N"


def test_tempo_matches_octave_equivalent():
    assert gm._tempo_matches(120.0, 120.0)
    assert gm._tempo_matches(60.0, 120.0)  # half-time
    assert gm._tempo_matches(240.0, 120.0)  # double-time
    assert not gm._tempo_matches(100.0, 120.0)


def test_parse_keylab(tmp_path):
    f = tmp_path / "k.lab"
    f.write_text("0.000\t10.000\tKey\tD\n10.000\t150.000\tKey\tA:minor\n")
    assert gm._parse_keylab(f) == "A minor"  # longest segment wins


def test_parse_keylab_skips_modal(tmp_path):
    f = tmp_path / "k.lab"
    f.write_text("0.000\t150.000\tKey\tA:dorian\n")
    assert gm._parse_keylab(f) is None


def test_parse_beat_tempo(tmp_path):
    f = tmp_path / "b.txt"
    # 0.5s spacing -> 120 BPM
    f.write_text("\n".join(f"{i*0.5:.3f}\t{(i % 4) + 1}" for i in range(16)))
    tempo = gm._parse_beat_tempo(f)
    assert tempo is not None and abs(tempo - 120.0) < 1.0


def test_vendored_annotations_present_and_indexable():
    index = gm.build_annotation_index()
    assert len(index) > 100  # ~200 Beatles+Queen tracks vendored
    # Bohemian Rhapsody should be in the index with a chord annotation
    bohemian = index.get(gm.normalize_title("Bohemian Rhapsody.lab"))
    assert bohemian is not None and "chord" in bohemian
