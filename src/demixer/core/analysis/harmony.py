"""Parametric harmony analysis + reharmonization suggestions over recognized chords.

The next layer above chord *recognition*: enrich each `ChordSegment` with harmonic
descriptors (function, tension, voicing) and transition metrics (common tones,
voice-leading, root motion), and suggest rule-based substitutions. Inspired by
ChordNova's "Parametric Harmony" indicators (tension / chroma / span / thickness /
common_note / sv / root_movement / g_center — from its `chorddata.h`), but
reimplemented as deterministic music theory over data demixer already produces:
the chords, the key, and the per-stem MIDI. No model, no external dependency.

Read-only analysis (v1): suggestions are ranked and labeled, never auto-applied.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pretty_midi

from demixer.core.analysis.chords import ChordSegment, _diatonic_pc_to_name, _root_pc
from demixer.core.analysis.key import KeyEstimate

# Chord-quality → semitone offsets from the root. Covers the BTC/autochord short
# labels (maj="", min="m", plus appended vocab qualities). Unknown → triad.
_QUALITY_INTERVALS: dict[str, tuple[int, ...]] = {
    "":        (0, 4, 7),          # major triad
    "m":       (0, 3, 7),          # minor triad
    "min":     (0, 3, 7),
    "maj":     (0, 4, 7),
    "dim":     (0, 3, 6),
    "aug":     (0, 4, 8),
    "sus2":    (0, 2, 7),
    "sus4":    (0, 5, 7),
    "7":       (0, 4, 7, 10),      # dominant 7
    "min7":    (0, 3, 7, 10),
    "m7":      (0, 3, 7, 10),
    "maj7":    (0, 4, 7, 11),
    "minmaj7": (0, 3, 7, 11),
    "6":       (0, 4, 7, 9),
    "min6":    (0, 3, 7, 9),
    "m6":      (0, 3, 7, 9),
    "dim7":    (0, 3, 6, 9),
    "hdim7":   (0, 3, 6, 10),      # half-diminished
    "9":       (0, 4, 7, 10, 2),
    "maj9":    (0, 4, 7, 11, 2),
    "min9":    (0, 3, 7, 10, 2),
}

# Dissonance weight per interval-class (1..6 semitones, octave-folded).
_IC_DISSONANCE = {1: 1.0, 2: 0.55, 3: 0.20, 4: 0.20, 5: 0.10, 6: 0.80}

_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)  # natural minor
_MAJOR_ROMAN = ("I", "ii", "iii", "IV", "V", "vi", "vii")
_MINOR_ROMAN = ("i", "ii", "III", "iv", "v", "VI", "VII")
# scale-degree (0-based) → harmonic function
_MAJOR_FUNC = ("T", "S", "T", "S", "D", "T", "D")
_MINOR_FUNC = ("T", "S", "T", "S", "D", "S", "D")


@dataclass(frozen=True)
class ChordDescriptors:
    label: str
    start_s: float
    end_s: float
    root_pc: int
    quality: str
    pitch_classes: tuple[int, ...]
    diatonic: bool
    roman: str                  # key-relative, e.g. "I", "V7", "bVI"
    function: str               # "T" | "S" | "D" | "chromatic"
    tension: float              # 0..1, interval-content dissonance
    span: int | None            # semitone range of sounding notes (needs MIDI)
    thickness: float | None     # mean simultaneous voices (needs MIDI)
    g_center: float | None      # mean MIDI pitch (needs MIDI)


@dataclass(frozen=True)
class ChordTransition:
    common_tones: int
    voice_leading: int          # Σ minimal pc movement (semitones)
    root_movement: int          # signed, −6..+6


@dataclass(frozen=True)
class Substitution:
    label: str
    kind: str                   # "tritone" | "relative" | "secondary-dominant" | "modal-interchange"
    common_tones: int
    note: str


def _parse_label(label: str) -> tuple[int, str] | None:
    """('C#m7') → (root_pc, quality). None for 'N'/'X'/non-pitched."""
    if not label or label[0] not in "ABCDEFG":
        return None
    root_len = 2 if len(label) > 1 and label[1] in "#b♭" else 1
    return _root_pc(label[:root_len]), label[root_len:]


def _chord_pcs(root_pc: int, quality: str) -> tuple[int, ...]:
    intervals = _QUALITY_INTERVALS.get(quality, (0, 4, 7))
    return tuple(sorted({(root_pc + i) % 12 for i in intervals}))


def _tension(pcs: tuple[int, ...]) -> float:
    if len(pcs) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):
            ic = abs(pcs[i] - pcs[j]) % 12
            ic = min(ic, 12 - ic)  # interval class 1..6
            total += _IC_DISSONANCE.get(ic, 0.0)
            pairs += 1
    return round(total / pairs, 3) if pairs else 0.0


def _degree_roman_function(root_pc: int, quality: str, key: KeyEstimate) -> tuple[bool, str, str]:
    tonic = _root_pc(key.root)
    scale = _MAJOR_SCALE if key.scale == "major" else _MINOR_SCALE
    roman_tbl = _MAJOR_ROMAN if key.scale == "major" else _MINOR_ROMAN
    func_tbl = _MAJOR_FUNC if key.scale == "major" else _MINOR_FUNC
    rel = (root_pc - tonic) % 12
    is_minor = quality.startswith(("m", "dim", "hdim")) and not quality.startswith("maj")
    seventh = "7" if "7" in quality else ""
    if rel in scale:
        deg = scale.index(rel)
        rn = roman_tbl[deg]
        # case the roman to the chord's own quality, append 7 if present
        rn = rn.lower() if is_minor else rn.upper()
        if quality in ("dim", "dim7", "hdim7"):
            rn = rn.lower() + "°"
        return True, rn + seventh, func_tbl[deg]
    # chromatic: spell as flat/sharp alteration of the nearest lower degree
    flat_deg = max(d for d in range(7) if scale[d] <= rel) if rel >= scale[0] else 0
    base = roman_tbl[flat_deg]
    base = base.lower() if is_minor else base.upper()
    return False, "b" + base + seventh, "chromatic"


def _segment_notes(seg: ChordSegment, stem_midis: dict[str, pretty_midi.PrettyMIDI]) -> list[int]:
    """MIDI pitches sounding within the segment across harmonic (non-drum) stems."""
    pitches: list[int] = []
    for name, midi in stem_midis.items():
        if name == "drums":
            continue
        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for n in inst.notes:
                if n.start < seg.end_s and n.end > seg.start_s:
                    pitches.append(int(n.pitch))
    return pitches


def _voicing(seg: ChordSegment, stem_midis: dict[str, pretty_midi.PrettyMIDI] | None
             ) -> tuple[int | None, float | None, float | None]:
    if not stem_midis:
        return None, None, None
    pitches = _segment_notes(seg, stem_midis)
    if not pitches:
        return None, None, None
    span = max(pitches) - min(pitches)
    g_center = round(sum(pitches) / len(pitches), 1)
    dur = max(seg.end_s - seg.start_s, 1e-6)
    # thickness ≈ average simultaneous voices = total sounding-time / segment length
    total_time = 0.0
    for name, midi in stem_midis.items():
        if name == "drums":
            continue
        for inst in midi.instruments:
            for n in inst.notes:
                ov = min(n.end, seg.end_s) - max(n.start, seg.start_s)
                if ov > 0:
                    total_time += ov
    return span, round(total_time / dur, 2), g_center


def _pc_circular_distance(a: int, b: int) -> int:
    d = abs(a - b) % 12
    return min(d, 12 - d)


def describe(chords: list[ChordSegment], key: KeyEstimate,
             stem_midis: dict[str, pretty_midi.PrettyMIDI] | None = None) -> list[ChordDescriptors]:
    out: list[ChordDescriptors] = []
    for seg in chords:
        parsed = _parse_label(seg.label)
        if parsed is None:
            continue  # skip "N"/"X"
        root_pc, quality = parsed
        pcs = _chord_pcs(root_pc, quality)
        diatonic, roman, func = _degree_roman_function(root_pc, quality, key)
        span, thickness, g_center = _voicing(seg, stem_midis)
        out.append(ChordDescriptors(
            label=seg.label, start_s=seg.start_s, end_s=seg.end_s,
            root_pc=root_pc, quality=quality, pitch_classes=pcs,
            diatonic=diatonic, roman=roman, function=func, tension=_tension(pcs),
            span=span, thickness=thickness, g_center=g_center,
        ))
    return out


def transitions(descs: list[ChordDescriptors]) -> list[ChordTransition]:
    out: list[ChordTransition] = []
    for a, b in zip(descs, descs[1:]):
        common = len(set(a.pitch_classes) & set(b.pitch_classes))
        vl = sum(min(_pc_circular_distance(pa, pb) for pb in b.pitch_classes)
                 for pa in a.pitch_classes)
        rm = (b.root_pc - a.root_pc) % 12
        if rm > 6:
            rm -= 12
        out.append(ChordTransition(common_tones=common, voice_leading=vl, root_movement=rm))
    return out


def _spell_pc(pc: int, key: KeyEstimate) -> str:
    diatonic = _diatonic_pc_to_name(key.sharps)
    if pc in diatonic:
        return diatonic[pc]
    # chromatic: flats for flat keys, sharps for sharp keys
    sharp_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    flat_names = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    return (flat_names if key.sharps < 0 else sharp_names)[pc]


def suggest_substitutions(desc: ChordDescriptors, key: KeyEstimate) -> list[Substitution]:
    """Rule-based reharmonization candidates, ranked by common tones (smoothest first)."""
    subs: list[Substitution] = []
    pcs = set(desc.pitch_classes)
    is_dominant = "7" in desc.quality and (desc.root_pc + 4) % 12 in pcs and \
        (desc.root_pc + 10) % 12 in pcs  # maj3 + b7
    is_major_triad = desc.quality in ("", "maj")
    is_minor_triad = desc.quality in ("m", "min")

    def _ct(other_pcs: tuple[int, ...]) -> int:
        return len(pcs & set(other_pcs))

    # Tritone substitution for dominant chords (V7 → bII7)
    if is_dominant:
        r = (desc.root_pc + 6) % 12
        sub_pcs = _chord_pcs(r, "7")
        subs.append(Substitution(f"{_spell_pc(r, key)}7", "tritone", _ct(sub_pcs),
                                 "tritone sub of the dominant (shares the guide-tone tritone)"))
    # Relative major/minor swap
    if is_major_triad:
        r = (desc.root_pc + 9) % 12
        subs.append(Substitution(f"{_spell_pc(r, key)}m", "relative", _ct(_chord_pcs(r, "m")),
                                 "relative minor (vi) — 2 shared tones"))
    if is_minor_triad:
        r = (desc.root_pc + 3) % 12
        subs.append(Substitution(f"{_spell_pc(r, key)}", "relative", _ct(_chord_pcs(r, "")),
                                 "relative major (III) — 2 shared tones"))
    # Secondary dominant: V7 of this chord (tonicize it)
    v_of = (desc.root_pc + 7) % 12
    subs.append(Substitution(f"{_spell_pc(v_of, key)}7", "secondary-dominant",
                             _ct(_chord_pcs(v_of, "7")),
                             f"V7/{desc.roman} — tonicizes this chord"))
    # Modal interchange: borrow the parallel-mode chord of the same root
    if is_major_triad:
        subs.append(Substitution(f"{_spell_pc(desc.root_pc, key)}m", "modal-interchange",
                                 _ct(_chord_pcs(desc.root_pc, "m")),
                                 "borrow from parallel minor (maj→min)"))
    elif is_minor_triad:
        subs.append(Substitution(f"{_spell_pc(desc.root_pc, key)}", "modal-interchange",
                                 _ct(_chord_pcs(desc.root_pc, "")),
                                 "borrow from parallel major (min→maj)"))

    subs.sort(key=lambda s: -s.common_tones)
    return subs


def analyze(chords: list[ChordSegment], key: KeyEstimate,
            stem_midis: dict[str, pretty_midi.PrettyMIDI] | None = None
            ) -> tuple[list[ChordDescriptors], list[ChordTransition]]:
    descs = describe(chords, key, stem_midis)
    return descs, transitions(descs)


# --- Reharmonization (generation): apply substitutions to the progression -----

_STRATEGIES = ("tritone", "relative", "modal-interchange", "secondary-dominant", "smoothest")


def reharmonize(chords: list[ChordSegment], key: KeyEstimate,
                strategy: str = "smoothest") -> list[ChordSegment]:
    """Return a NEW progression with substitutions applied per `strategy`.

    Timing is preserved; only labels change. 'smoothest' picks, per chord, the
    substitution sharing the most common tones with the original (falling back to
    the original if no sub improves smoothness). A named strategy applies only
    substitutions of that kind. Non-pitched segments ("N"/"X") pass through.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {_STRATEGIES}")
    out: list[ChordSegment] = []
    for seg in chords:
        seg_desc = describe([seg], key)  # per-segment → robust to shared timings
        if not seg_desc:
            out.append(seg)  # "N"/"X"
            continue
        subs = suggest_substitutions(seg_desc[0], key)
        if strategy != "smoothest":
            subs = [s for s in subs if s.kind == strategy]
        # 'smoothest' already ranked by common tones; require ≥1 shared tone so we
        # never produce a jarring, unrelated substitution.
        chosen = next((s for s in subs if s.common_tones >= 1), None)
        out.append(replace(seg, label=chosen.label) if chosen else seg)
    return out


def render_chord_midi(chords: list[ChordSegment], *, program: int = 0,
                      octave: int = 4) -> pretty_midi.PrettyMIDI:
    """Render a progression as block chords → a MIDI track (audible / DAW-loadable).

    Each segment becomes a sustained block chord (root-position chord tones placed
    around `octave`). Non-pitched segments are silent gaps. Lets the user hear and
    load a (possibly reharmonized) progression directly in a DAW.
    """
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program, name="chords")
    base = 12 * (octave + 1)  # MIDI octave convention: C4 = 60
    for seg in chords:
        parsed = _parse_label(seg.label)
        if parsed is None or seg.end_s <= seg.start_s:
            continue
        root_pc, quality = parsed
        intervals = _QUALITY_INTERVALS.get(quality, (0, 4, 7))
        for iv in intervals:
            pitch = base + (root_pc % 12) + iv
            inst.notes.append(pretty_midi.Note(
                velocity=70, pitch=max(0, min(127, pitch)),
                start=seg.start_s, end=seg.end_s,
            ))
    midi.instruments.append(inst)
    return midi
