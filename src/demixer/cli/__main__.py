"""`demixer process INPUT -o OUTDIR` — end-to-end Phase 1 pipeline.

Stages:
  1. ingest        — decode + 44.1k stereo float32 + EBU R128 normalize
  2. separate      — Demucs htdemucs (or htdemucs_6s) → stems
  3. analyze       — tempo/beats/downbeats (beat_this) + key (essentia)
  4. transcribe    — basic-pitch per pitched stem → MIDI (drums skipped — stub)
  5. bundle        — write `.demixer` directory + zip

Any stage can be skipped via `--skip <stage>` to iterate faster while
developing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from demixer.core.analysis import chords as chords_mod
from demixer.core.analysis import key as key_mod
from demixer.core.analysis import tempo_beats as tb_mod
from demixer.core.bundle import BundleMetadata, write_bundle, zip_bundle
from demixer.core.ingest import ingest
from demixer.core.project.dawproject import StemTrack as DPStemTrack
from demixer.core.project.dawproject import write_dawproject
from demixer.core.project.dragin import StemTrack as DIStemTrack
from demixer.core.project.dragin import write_dragin
from demixer.core.project.flstudio import StemTrack as FLStemTrack
from demixer.core.project.flstudio import write_flp, write_flpianoroll_scripts
from demixer.core.project.reaper import StemTrack, write_rpp
from demixer.core.score.musicxml import build_score, write_musicxml
from demixer.core.score.quantize import quantize_midi
from demixer.core.score.render import (
    musescore_available,
    render_audio,
    render_mscz,
    render_pdf,
    render_png,
    render_svg,
)
from demixer.core.separation import STEM_NAMES, separate, write_stems
from demixer.core.transcription.drums import transcribe_drums
from demixer.core.transcription.pitched import (
    InstrumentHint,
    transcribe_pitched,
    write_midi,
)

log = logging.getLogger("demixer")

# Stems whose peak amplitude is below this are treated as empty (Demucs zeroes
# out absent sources). −40 dBFS — well above dither/noise-floor, well below any
# genuinely present part.
_SILENT_STEM_PEAK = 0.01


def _stem_is_silent(wav_path: Path) -> bool:
    import soundfile as sf
    y, _ = sf.read(wav_path, always_2d=True)
    return bool(np.max(np.abs(y)) < _SILENT_STEM_PEAK) if y.size else True

# Map stem names to basic-pitch instrument hints
_HINT_FOR_STEM: dict[str, InstrumentHint] = {
    "vocals": "vocals",
    "bass":   "bass",
    "drums":  "drums",  # signals "skip / not yet wired"
    "other":  "other",
    "piano":  "piano",
    "guitar": "guitar",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="demixer", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    proc = sub.add_parser("process", help="run the full pipeline on a single audio file")
    proc.add_argument("input", help="path to source audio (mp3/flac/wav/m4a/ogg/…)")
    proc.add_argument("-o", "--output", required=True, help="output bundle directory")
    proc.add_argument(
        "--model",
        choices=list(STEM_NAMES.keys()),
        default="htdemucs",
        help="Demucs model variant (default: %(default)s)",
    )
    proc.add_argument(
        "--roformer-vocals",
        action="store_true",
        help="use BS-RoFormer for a higher-SDR vocals stem (drums/bass/other still Demucs; "
             "needs the .venv-roformer worker)",
    )
    proc.add_argument(
        "--transcriber",
        choices=["basic-pitch", "mt3"],
        default="basic-pitch",
        help="pitched transcription engine. 'mt3' uses MR-MT3 for polyphonic stems "
             "(piano/guitar/other) via the .venv-mt3 worker; bass/vocals stay on basic-pitch "
             "(default: %(default)s)",
    )
    proc.add_argument(
        "--chords",
        choices=["autochord", "btc"],
        default="autochord",
        help="chord recognition engine. 'btc' uses the BTC large-vocab transformer "
             "(170 chords incl. 7ths/extensions) via the .venv-btc worker; 'autochord' is "
             "triads-only in-env (default: %(default)s)",
    )
    proc.add_argument(
        "--score-renders",
        nargs="+",
        choices=["pdf", "mscz", "png", "audio"],
        default=["pdf", "mscz"],
        metavar="FMT",
        help="MuseScore output formats to render (default: pdf mscz). PDF + "
             "MSCZ alone cover viewing and editing; 'png' duplicates the PDF "
             "as ~1 MB raster pages and 'audio' synthesizes a low-fi MP3 "
             "preview — opt in only when you actually need them",
    )
    proc.add_argument(
        "--midi-hint",
        metavar="PATH|auto",
        default=None,
        help="treat a ground-truth MIDI as authoritative for tempo + time "
             "signature, bypassing audio-side beat tracking. 'auto' looks for "
             "a sibling .mid/.MID next to the input file. Useful for MIDI-"
             "rendered corpora (e.g. game music) where the audio is a synth "
             "rendering of a known MIDI",
    )
    proc.add_argument(
        "--compact-archive",
        action="store_true",
        help="when zipping the .demixer archive, drop the loose stems/ subtree "
             "if a .dawproject is present (the dawproject already embeds the "
             "same audio). Cuts archive size ~50%% for the typical 4-stem run. "
             "Loose stems remain on disk in the bundle dir.",
    )
    proc.add_argument(
        "--stem-format",
        choices=["pcm24", "pcm16", "float", "flac"],
        default="pcm24",
        help="on-disk stem container/precision (default: %(default)s). "
             "'flac' yields the smallest archive; 'float' preserves bit-exact "
             "Demucs output if you plan to re-feed stems into other ML models",
    )
    proc.add_argument(
        "--drums",
        choices=["spectral", "adtof"],
        default="spectral",
        help="drum transcription engine. 'adtof' uses the ADTOF Frame-RNN (learned, "
             "5 GM classes incl. tom/crash, far better onset recall) via the .venv-adtof "
             "worker; 'spectral' is the in-env librosa onset+centroid 3-class (default: %(default)s)",
    )
    proc.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=["separate", "analyze", "transcribe", "zip", "rpp", "dawproject", "dragin",
                 "flstudio", "score"],
        help="skip a stage (can be passed multiple times; useful during dev)",
    )
    proc.add_argument(
        "--harmony",
        action="store_true",
        help="emit harmony.json — parametric chord descriptors (function/roman/tension/"
             "voicing), transition metrics, and reharmonization suggestions (read-only analysis)",
    )
    proc.add_argument(
        "--reharmonize",
        choices=["tritone", "relative", "modal-interchange", "secondary-dominant", "smoothest"],
        default=None,
        help="apply a reharmonization strategy and emit reharmonization.mid + the new "
             "progression in harmony.json (implies --harmony)",
    )
    proc.add_argument("--verbose", "-v", action="store_true")
    return p


def cmd_process(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
    )

    src = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = set(args.skip)

    log.info("ingesting %s", src)
    audio = ingest(src)
    log.info(
        "ingested %.1fs (sha256=%s loudness %.1f → %.1f LUFS)",
        audio.duration_s, audio.sha256[:12],
        audio.integrated_lufs_before, audio.integrated_lufs_after,
    )

    # --- separation ---
    stem_paths: dict[str, Path] = {}
    sep_label: str = args.model
    sep_stems: dict[str, np.ndarray] | None = None  # in-memory stems for key estimation
    if "separate" not in skip:
        log.info("separating stems with %s%s (this is the slow part)", args.model,
                 " + BS-RoFormer vocals" if args.roformer_vocals else "")
        result = separate(audio, model_name=args.model, roformer_vocals=args.roformer_vocals)
        sep_label = result.model
        sep_stems = result.stems
        stem_paths = write_stems(result, out_dir / "stems", stem_format=args.stem_format)
        log.info("wrote %d stems to %s", len(stem_paths), out_dir / "stems")
    else:
        log.warning("--skip separate: no stems will be produced")

    # --- analysis ---
    tempo_beats = None
    key_estimate = None
    chords = None
    if "analyze" not in skip:
        midi_hint_path: Path | None = None
        if args.midi_hint == "auto":
            midi_hint_path = tb_mod.find_sibling_midi(src)
            if midi_hint_path is None:
                log.info("--midi-hint auto: no sibling .mid found next to %s", src.name)
        elif args.midi_hint:
            midi_hint_path = Path(args.midi_hint).expanduser().resolve()
            if not midi_hint_path.exists():
                log.warning("--midi-hint %s: file not found; falling back to beat tracking",
                            midi_hint_path)
                midi_hint_path = None

        if midi_hint_path is not None:
            log.info("tempo from ground-truth MIDI: %s", midi_hint_path.name)
            tempo_beats = tb_mod.from_midi(midi_hint_path, audio_duration_s=audio.duration_s)
        else:
            log.info("estimating tempo + beats + downbeats")
            tempo_beats = tb_mod.estimate(audio)
        log.info(
            "tempo=%.2f BPM (%s), %d beats, %d downbeats (~%d beats/bar), confidence=%.2f",
            tempo_beats.tempo_bpm, tempo_beats.method,
            len(tempo_beats.beat_times_s), len(tempo_beats.downbeat_times_s),
            tempo_beats.beats_per_bar, tempo_beats.confidence,
        )
        if not tempo_beats.reliable:
            log.warning(
                "tempo confidence %.2f below threshold — no steady pulse detected "
                "(beatless/free-time material?); treat BPM=%.1f as unreliable",
                tempo_beats.confidence, tempo_beats.tempo_bpm,
            )
        # Estimate key on the drums-excluded stem mix when stems are available
        # (sharper than the full mix — drums smear the chroma). Falls back to the
        # full mix when separation was skipped.
        key_mono = key_mod.harmonic_mono(sep_stems) if sep_stems else None
        log.info("estimating key (%s)", "drums-excluded stems" if key_mono is not None else "full mix")
        key_estimate = key_mod.estimate(audio, mono=key_mono)
        log.info(
            "key=%s %s (sharps=%+d, strength=%.2f)",
            key_estimate.root, key_estimate.scale,
            key_estimate.sharps, key_estimate.strength,
        )
        log.info("recognizing chords (%s)", args.chords)
        try:
            if args.chords == "btc":
                from demixer.core.analysis import chords_btc
                chords = chords_btc.estimate(audio)
            else:
                chords = chords_mod.estimate(audio)
            # Respell diatonic chord roots to the key's accidentals (e.g. in Cm,
            # D#/A# → Eb/Bb) so the chord chart reads consistently with the score
            # key signature. Chromatic roots are left as-is.
            chords = chords_mod.respell_to_key(chords, key_estimate)
            preview = " ".join(c.label for c in chords[:8])
            log.info("chords: %d segments (first 8: %s)", len(chords), preview)
        except Exception:
            log.exception("chord recognition failed; continuing without chords")
    else:
        log.warning("--skip analyze: bundle will be missing tempo/key/chord info")

    # --- transcription ---
    midi_paths: dict[str, Path] = {}
    midis: dict[str, object] = {}  # raw PrettyMIDI objects for downstream score
    if "transcribe" not in skip and stem_paths:
        midi_dir = out_dir / "midi"
        midi_dir.mkdir(parents=True, exist_ok=True)
        # MT3 wins big on polyphonic stems (piano/guitar/other) but octave-doubles
        # bass and is unreliable on lead vocals — so the mt3 transcriber routes
        # those two to basic-pitch. Measured on the synth eval.
        mt3_stems = {"other", "piano", "guitar"}
        pending_mt3: dict[str, Path] = {}  # batched in one worker invocation
        for name, wav in stem_paths.items():
            hint = _HINT_FOR_STEM.get(name, "other")
            # Skip near-silent stems: Demucs empties absent sources (e.g. the
            # vocals stem of an instrumental track), but transcribers will
            # otherwise hallucinate phantom notes from the noise floor — which
            # pollute the score and DAW project with a meaningless part.
            if _stem_is_silent(wav):
                log.info("transcribe %s: skipped (stem is near-silent)", name)
                continue
            if args.transcriber == "mt3" and name in mt3_stems:
                pending_mt3[name] = wav  # defer — batch below for one model load
                continue
            engine = (f"drums:{args.drums}" if hint == "drums" else "basic-pitch")
            log.info("transcribe %s (hint=%s, engine=%s)", name, hint, engine)
            try:
                if hint == "drums":
                    if args.drums == "adtof":
                        from demixer.core.transcription.drums_adtof import transcribe_drums_adtof
                        midi = transcribe_drums_adtof(wav)
                    else:
                        midi = transcribe_drums(wav)
                else:
                    midi = transcribe_pitched(wav, hint=hint)
                midi_paths[name] = write_midi(midi, midi_dir / f"{name}.mid")
                midis[name] = midi
            except Exception:
                log.exception("transcribe %s failed; continuing", name)

        # MT3 stems in a single worker call — loads the model once (cached
        # in-process) instead of cold-loading it per stem.
        if pending_mt3:
            log.info("transcribe %s via MT3 (batched, one model load)", ", ".join(pending_mt3))
            try:
                from demixer.core.transcription.mt3 import transcribe_mt3_batch
                for name, midi in transcribe_mt3_batch(pending_mt3).items():
                    midi_paths[name] = write_midi(midi, midi_dir / f"{name}.mid")
                    midis[name] = midi
            except Exception:
                log.exception("MT3 batch transcription failed; continuing")
    elif "transcribe" in skip:
        log.warning("--skip transcribe: bundle will contain no MIDI")
    else:
        log.warning("transcribe requires stems; nothing to do")

    # --- bundle ---
    if tempo_beats is None or key_estimate is None:
        log.warning("analysis was skipped; writing partial bundle without analysis.json")
        # Drop a minimal manifest so the dir is at least introspectable
        (out_dir / "manifest.json").write_text(json.dumps({
            "schema": 0, "partial": True, "skipped": sorted(skip),
        }, indent=2))
        return 0

    meta = BundleMetadata(
        audio=audio,
        tempo_beats=tempo_beats,
        key=key_estimate,
        chords=chords,
        separation_model=sep_label,
        transcription_model=("mr-mt3+basic-pitch" if args.transcriber == "mt3"
                             else "basic-pitch-icassp-2022"),
    )
    # Write the directory now; defer zipping until the DAW projects + score are
    # also written, so the single-file .demixer archive is complete.
    bundle_dir, _ = write_bundle(out_dir, meta, stem_paths, midi_paths, zip_output=False)
    log.info("bundle dir ready: %s", bundle_dir)

    if stem_paths and ("rpp" not in skip or "dawproject" not in skip or "flstudio" not in skip):
        stem_tracks_rpp = [
            StemTrack(name=name, wav_path=wav, midi_path=midi_paths.get(name))
            for name, wav in stem_paths.items()
        ]
        stem_tracks_dp = [
            DPStemTrack(name=name, wav_path=wav, midi_path=midi_paths.get(name))
            for name, wav in stem_paths.items()
        ]
        if "rpp" not in skip:
            rpp = write_rpp(
                bundle_dir / f"{src.stem}.rpp",
                tracks=stem_tracks_rpp,
                tempo=tempo_beats,
                key=key_estimate,
                duration_s=audio.duration_s,
                project_name=src.stem,
            )
            log.info("reaper project: %s", rpp)
        if "dawproject" not in skip:
            dp = write_dawproject(
                bundle_dir / f"{src.stem}.dawproject",
                tracks=stem_tracks_dp,
                tempo=tempo_beats,
                key=key_estimate,
                duration_s=audio.duration_s,
                project_name=src.stem,
            )
            log.info("dawproject: %s", dp)
        if "flstudio" not in skip:
            fl_tracks = [
                FLStemTrack(name=name, wav_path=wav, midi_path=midi_paths.get(name))
                for name, wav in stem_paths.items()
            ]
            flp = write_flp(
                bundle_dir / f"{src.stem}.flp",
                tracks=fl_tracks,
                tempo=tempo_beats,
                key=key_estimate,
                duration_s=audio.duration_s,
                project_name=src.stem,
            )
            log.info("fl studio project: %s", flp)
            scripts = write_flpianoroll_scripts(
                bundle_dir / "flstudio_scripts",
                tracks=fl_tracks,
                tempo=tempo_beats,
                key=key_estimate,
            )
            if scripts:
                log.info("fl studio piano-roll scripts: %d", len(scripts))
        if "dragin" not in skip:
            di_tracks = [
                DIStemTrack(name=name, wav_path=wav, midi_path=midi_paths.get(name))
                for name, wav in stem_paths.items()
            ]
            di = write_dragin(
                bundle_dir / "dragin",
                tracks=di_tracks,
                tempo=tempo_beats,
                key=key_estimate,
                duration_s=audio.duration_s,
                # Reference the bundle's top-level stems/midi instead of copying,
                # so the .demixer archive doesn't duplicate the audio.
                copy_media=False,
            )
            log.info("drag-in bundle: %s", di)

    # --- score ---
    # The whole stage is best-effort: music21's MusicXML export can throw on
    # quantized multi-part scores (tuplet/fractional-offset StreamExceptions on
    # dense material). A score failure must not abort the run — stems, MIDI,
    # analysis, and all DAW exports have already shipped.
    if "score" not in skip and midis:
        try:
            _engrave_score(bundle_dir, midis, tempo_beats, key_estimate, src.stem,
                           renders=tuple(args.score_renders))
        except Exception as exc:  # noqa: BLE001
            log.warning("score engraving failed (%s); bundle ships without a score", exc)

    # --- harmony (opt-in, read-only analysis + optional reharmonization) ---
    if (args.harmony or args.reharmonize) and chords and key_estimate is not None:
        try:
            _write_harmony(bundle_dir, chords, key_estimate, midis, reharm=args.reharmonize)
        except Exception as exc:  # noqa: BLE001
            log.warning("harmony analysis failed (%s); skipping", exc)

    # Zip last, so the single-file .demixer contains everything (stems, MIDI,
    # analysis, DAW projects, and the score) — not just the core written above.
    if "zip" not in skip:
        zip_path = zip_bundle(bundle_dir, archive_stems=not args.compact_archive)
        log.info("bundle archive: %s", zip_path)

    return 0


def _write_harmony(bundle_dir, chords, key_estimate, midis, reharm=None):  # type: ignore[no-untyped-def]
    """Emit harmony.json (descriptors + transitions + substitution suggestions) and,
    if `reharm` is a strategy, the reharmonized progression + a renderable
    reharmonization.mid (block chords, DAW-loadable)."""
    from dataclasses import asdict

    from demixer.core.analysis import harmony as harmony_mod

    descs, trans = harmony_mod.analyze(chords, key_estimate, stem_midis=midis or None)
    payload = {
        "key": {"root": key_estimate.root, "scale": key_estimate.scale},
        "chords": [
            {**asdict(d),
             "substitutions": [asdict(s) for s in harmony_mod.suggest_substitutions(d, key_estimate)]}
            for d in descs
        ],
        "transitions": [asdict(t) for t in trans],
    }
    log_msg = "harmony: %d chords analyzed → harmony.json" % len(descs)

    if reharm:
        reharmed = harmony_mod.reharmonize(chords, key_estimate, strategy=reharm)
        payload["reharmonization"] = {
            "strategy": reharm,
            "progression": [{"start_s": c.start_s, "end_s": c.end_s, "label": c.label}
                            for c in reharmed],
        }
        harmony_mod.render_chord_midi(reharmed).write(str(bundle_dir / "reharmonization.mid"))
        changed = sum(1 for o, r in zip(chords, reharmed) if o.label != r.label)
        log_msg += f"; reharmonized ({reharm}): {changed}/{len(reharmed)} chords changed → reharmonization.mid"

    (bundle_dir / "harmony.json").write_text(json.dumps(payload, indent=2))
    log.info(log_msg)


def _engrave_score(
    bundle_dir: Path,
    midis: dict[str, object],
    tempo_beats: "tb_mod.TempoBeats",
    key_estimate: "key_mod.KeyEstimate",
    project_stem: str,
    *,
    renders: tuple[str, ...] = ("pdf", "mscz"),
) -> None:
    """Quantize → MusicXML → Verovio SVG → (opportunistic) MuseScore renders.

    Raises on a hard failure (e.g. music21 MusicXML export StreamException); the
    caller treats the whole score stage as best-effort and continues.
    """
    log.info("quantizing %d MIDI parts and engraving score", len(midis))
    quantized = {
        name: quantize_midi(m, tempo_beats.beat_times_s, subdivisions_per_beat=4)
        for name, m in midis.items()
    }
    score_obj = build_score(quantized, tempo=tempo_beats, key=key_estimate,
                            project_name=project_stem)
    xml_path = write_musicxml(score_obj, bundle_dir / "score.musicxml")
    svg_paths = render_svg(xml_path, bundle_dir / "score")
    log.info("score: %s (%d Verovio SVG pages)", xml_path.name, len(svg_paths))

    if not musescore_available():
        log.info("(MuseScore CLI not on PATH — skipping PDF/.mscz/PNG/audio renders)")
        return

    # Each MuseScore render is opportunistic: some ML-transcribed scores trip
    # MuseScore even with -f. A failed render must not abort the others — the
    # .musicxml + Verovio SVG already shipped.
    def _try(label: str, fn) -> None:  # type: ignore[no-untyped-def]
        try:
            out = fn()
        except Exception as exc:  # noqa: BLE001
            log.warning("score %s render failed (%s); skipping", label, exc)
            return
        if out:
            log.info("score %s: %s", label, out if not isinstance(out, list)
                     else f"{len(out)} pages in {out[0].parent.name}/")
        else:
            log.warning("score %s render produced nothing (MuseScore rejected the score)", label)

    if "pdf" in renders:
        _try("PDF",   lambda: render_pdf(xml_path, bundle_dir / "score.pdf"))
    if "mscz" in renders:
        _try("MSCZ",  lambda: render_mscz(xml_path, bundle_dir / "score.mscz"))
    if "png" in renders:
        _try("PNG",   lambda: render_png(xml_path, bundle_dir / "score_png"))
    if "audio" in renders:
        _try("audio", lambda: render_audio(xml_path, bundle_dir / "score_preview.mp3", format="mp3"))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "process":
        return cmd_process(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
