"""Tests for parametric harmony analysis (hand-worked music-theory examples)."""

from __future__ import annotations

import pretty_midi

from demixer.core.analysis.chords import ChordSegment
from demixer.core.analysis.harmony import (
    _chord_pcs,
    _parse_label,
    _tension,
    analyze,
    describe,
    suggest_substitutions,
)
from demixer.core.analysis.key import KeyEstimate

C_MAJOR = KeyEstimate(root="C", scale="major", strength=0.9, sharps=0)
A_MINOR = KeyEstimate(root="A", scale="minor", strength=0.9, sharps=0)


def _seg(label, a=0.0, b=1.0):
    return ChordSegment(start_s=a, end_s=b, label=label)


# ---------- parsing ----------

def test_parse_label() -> None:
    assert _parse_label("C") == (0, "")
    assert _parse_label("C#m7") == (1, "m7")
    assert _parse_label("Bb7") == (10, "7")
    assert _parse_label("G7") == (7, "7")
    assert _parse_label("N") is None
    assert _parse_label("X") is None


def test_chord_pcs() -> None:
    assert _chord_pcs(0, "") == (0, 4, 7)          # C major
    assert _chord_pcs(9, "m") == (0, 4, 9)          # A minor → {9,0,4}
    assert _chord_pcs(7, "7") == (2, 5, 7, 11)      # G7 → {7,11,2,5}


def test_tension_orders_dissonance() -> None:
    # A major triad (no semitone/tritone) is less tense than a dom7 (has a tritone)
    assert _tension(_chord_pcs(0, "")) < _tension(_chord_pcs(7, "7"))
    # maj7 (has a semitone 11–0) is tenser than the plain triad
    assert _tension(_chord_pcs(0, "maj7")) > _tension(_chord_pcs(0, ""))


# ---------- descriptors: roman / function ----------

def test_roman_and_function_major_key() -> None:
    descs = describe([_seg("C"), _seg("Dm"), _seg("G7"), _seg("Am"), _seg("F")], C_MAJOR)
    by_label = {d.label: d for d in descs}
    assert by_label["C"].roman == "I" and by_label["C"].function == "T"
    assert by_label["Dm"].roman == "ii" and by_label["Dm"].function == "S"
    assert by_label["G7"].roman == "V7" and by_label["G7"].function == "D"
    assert by_label["Am"].roman == "vi" and by_label["Am"].function == "T"
    assert by_label["F"].roman == "IV" and by_label["F"].function == "S"


def test_chromatic_root_flagged() -> None:
    # Eb is not diatonic to C major → chromatic, bIII-ish
    d = describe([_seg("Eb")], C_MAJOR)[0]
    assert d.diatonic is False
    assert d.function == "chromatic"
    assert d.roman.startswith("b")


# ---------- transitions ----------

def test_transition_common_tones_and_root_motion() -> None:
    # C → Am shares {0,4} (2 tones); C → G shares {7} (1)
    _, trans = analyze([_seg("C"), _seg("Am"), _seg("G")], C_MAJOR)
    assert trans[0].common_tones == 2          # C{0,4,7}→Am{9,0,4} share {0,4}
    assert trans[1].common_tones == 0          # Am{9,0,4}→G{7,11,2} share nothing
    # root movement C(0)→Am(9): (9-0)=9 → signed -3
    assert trans[0].root_movement == -3


# ---------- substitutions ----------

def test_tritone_sub_of_dominant() -> None:
    g7 = describe([_seg("G7")], C_MAJOR)[0]
    subs = suggest_substitutions(g7, C_MAJOR)
    tri = [s for s in subs if s.kind == "tritone"]
    assert tri, "expected a tritone substitution for a dominant chord"
    # G7 {7,11,2,5} and its tritone sub (Db7 {1,5,8,11}) share the tritone {11,5}
    assert tri[0].common_tones == 2


def test_relative_and_modal_for_major_triad() -> None:
    c = describe([_seg("C")], C_MAJOR)[0]
    kinds = {s.kind for s in suggest_substitutions(c, C_MAJOR)}
    assert "relative" in kinds            # C → Am
    assert "modal-interchange" in kinds   # C → Cm
    assert "secondary-dominant" in kinds  # V7/I = G7


def test_substitutions_ranked_by_common_tones() -> None:
    c = describe([_seg("C")], C_MAJOR)[0]
    subs = suggest_substitutions(c, C_MAJOR)
    cts = [s.common_tones for s in subs]
    assert cts == sorted(cts, reverse=True)  # smoothest (most shared tones) first


# ---------- voicing from MIDI ----------

def test_voicing_from_midi() -> None:
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for pitch in (48, 60, 64, 67):  # C3..G4 sounding through the segment
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=0.0, end=1.0))
    midi.instruments.append(inst)
    d = describe([_seg("C", 0.0, 1.0)], C_MAJOR, stem_midis={"other": midi})[0]
    assert d.span == 67 - 48          # 19 semitones
    assert d.g_center == round((48 + 60 + 64 + 67) / 4, 1)
    assert d.thickness is not None and d.thickness > 0


def test_voicing_none_without_midi() -> None:
    d = describe([_seg("C")], C_MAJOR)[0]
    assert d.span is None and d.thickness is None and d.g_center is None


# ---------- reharmonization (generation) ----------

def test_reharmonize_preserves_timing_and_changes_labels() -> None:
    from demixer.core.analysis.harmony import reharmonize
    orig = [_seg("C", 0, 1), _seg("G7", 1, 2), _seg("Am", 2, 3)]
    out = reharmonize(orig, C_MAJOR, strategy="smoothest")
    assert [(c.start_s, c.end_s) for c in out] == [(0, 1), (1, 2), (2, 3)]  # timing kept
    assert len(out) == 3


def test_reharmonize_tritone_only_subs_dominants() -> None:
    from demixer.core.analysis.harmony import reharmonize
    out = reharmonize([_seg("G7", 0, 1), _seg("C", 1, 2)], C_MAJOR, strategy="tritone")
    labels = [c.label for c in out]
    assert labels[0] != "G7"        # dominant got a tritone sub
    assert labels[1] == "C"         # non-dominant unchanged (no tritone candidate)


def test_reharmonize_passes_through_no_chord() -> None:
    from demixer.core.analysis.harmony import reharmonize
    out = reharmonize([_seg("N")], C_MAJOR, strategy="smoothest")
    assert out[0].label == "N"


def test_reharmonize_rejects_bad_strategy() -> None:
    import pytest
    from demixer.core.analysis.harmony import reharmonize
    with pytest.raises(ValueError):
        reharmonize([_seg("C")], C_MAJOR, strategy="bogus")


def test_render_chord_midi() -> None:
    from demixer.core.analysis.harmony import render_chord_midi
    midi = render_chord_midi([_seg("C", 0, 1), _seg("Am", 1, 2), _seg("N", 2, 3)])
    notes = midi.instruments[0].notes
    assert len(notes) == 6                              # 3 + 3, "N" silent
    cmaj = sorted(n.pitch % 12 for n in notes if n.start == 0)
    assert cmaj == [0, 4, 7]                            # C E G
    assert all(n.end > n.start for n in notes)          # sustained blocks
