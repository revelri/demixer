"""Render an engraved score: Verovio for in-process SVG, MuseScore CLI for the
rest (PDF, native .mscz, MS-engraved SVG, PNG previews, audio renders).

Verovio is always available (pure-Python wheel) and produces clean preview
SVGs without system dependencies. MuseScore CLI is invoked via subprocess for
everything else; each render function returns `None` when MuseScore isn't on
PATH so callers degrade gracefully.

MuseScore infers the output format from the destination file extension:
  .pdf .png .svg .mscz .mscx .mxl .musicxml .mid .midi .mp3 .wav .ogg .flac
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Literal

import verovio

# WAV is omitted on purpose — MuseScore 4 has a known bug where WAV export
# silently exits 0 without producing a file. mp3/ogg/flac route through the
# compressed-audio encoder and work reliably.
AudioFormat = Literal["mp3", "ogg", "flac"]

# Bundled SoundFont profile that doesn't require Muse Hub / the cloud-backed
# MuseSounds library. Always pass this on audio renders so they work headless.
_HEADLESS_SOUND_PROFILE = "MuseScore Basic"


# ---------- Verovio (always available) ----------

def render_svg(musicxml_path: str | Path, dest_dir: str | Path) -> list[Path]:
    """Render every page of the MusicXML to dest_dir/page-NN.svg via Verovio."""
    musicxml_path = Path(musicxml_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    tk = verovio.toolkit()
    tk.loadFile(str(musicxml_path))
    page_count = tk.getPageCount()
    paths: list[Path] = []
    for page in range(1, page_count + 1):
        svg = tk.renderToSVG(page)
        out = dest_dir / f"page-{page:02d}.svg"
        out.write_text(svg)
        paths.append(out)
    return paths


# ---------- MuseScore CLI ----------

def _find_musescore() -> str | None:
    """Return the absolute path to MuseScore 4's CLI if installed, else None."""
    for name in ("mscore", "mscore4", "musescore", "musescore4"):
        path = shutil.which(name)
        if path:
            return path
    return None


def musescore_available() -> bool:
    return _find_musescore() is not None


def _mscore_convert(
    musicxml_path: Path,
    dest: Path,
    *,
    extra_args: list[str] | None = None,
    output_glob: str | None = None,
) -> Path | None:
    """Generic `mscore -o dest input.musicxml [extras...]`. Returns None if mscore missing.

    Forces `QT_QPA_PLATFORM=offscreen` so the call survives on headless systems.
    Verifies the output actually materialized: MuseScore 4 sometimes exits 0
    without producing the file (notably the WAV-export bug), so we check before
    declaring success.

    `output_glob` lets multi-page formats (PNG, SVG) confirm at least one of
    several auto-numbered output files appeared.
    """
    mscore = _find_musescore()
    if mscore is None:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    # `-f` makes MuseScore proceed past "score corrupted" warnings instead of
    # silently exiting 0 without producing output — common on MusicXML generated
    # from polyphonic ML transcription (overlapping notes, missing voice tags).
    cmd = [mscore, "-f", "-o", str(dest), str(musicxml_path), *(extra_args or [])]
    subprocess.run(cmd, check=True, capture_output=True, env=env)
    if dest.exists():
        return dest
    if output_glob and any(dest.parent.glob(output_glob)):
        return dest  # caller will re-glob; we just verify something landed
    return None


def render_pdf(musicxml_path: str | Path, dest_pdf: str | Path) -> Path | None:
    """Engraved PDF via MuseScore. Best print quality."""
    return _mscore_convert(Path(musicxml_path), Path(dest_pdf))


def render_mscz(musicxml_path: str | Path, dest_mscz: str | Path) -> Path | None:
    """MuseScore's native compressed format (.mscz) — opens directly in MuseScore for editing."""
    return _mscore_convert(Path(musicxml_path), Path(dest_mscz))


def render_svg_mscore(musicxml_path: str | Path, dest_dir: str | Path) -> list[Path] | None:
    """MuseScore-engraved SVG. Produces dest_dir/score-NN.svg (one per page).

    MuseScore writes a single SVG per page when given a `.svg` destination —
    it auto-numbers via `-N` suffix. We pass `score.svg` and MS produces
    `score.svg`, `score-1.svg`, etc.
    """
    musicxml_path = Path(musicxml_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = _mscore_convert(musicxml_path, dest_dir / "score.svg", output_glob="score*.svg")
    if out is None:
        return None
    return sorted(dest_dir.glob("score*.svg"))


def render_png(musicxml_path: str | Path, dest_dir: str | Path) -> list[Path] | None:
    """PNG thumbnails per page (good for inline UI previews)."""
    musicxml_path = Path(musicxml_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = _mscore_convert(musicxml_path, dest_dir / "page.png", output_glob="page*.png")
    if out is None:
        return None
    return sorted(dest_dir.glob("page*.png"))


def render_audio(
    musicxml_path: str | Path,
    dest_audio: str | Path,
    *,
    format: AudioFormat = "mp3",
) -> Path | None:
    """Render the score to audio (synthesized playback via MuseScore Basic soundfont).

    Headless-safe — passes `--sound-profile "MuseScore Basic"` so MuseScore
    doesn't try to reach Muse Hub for MuseSounds. WAV is unsupported (MuseScore
    4 has a known bug where WAV export silently fails); use mp3/ogg/flac.
    """
    dest_audio = Path(dest_audio)
    if dest_audio.suffix.lstrip(".") != format:
        dest_audio = dest_audio.with_suffix(f".{format}")
    return _mscore_convert(
        Path(musicxml_path),
        dest_audio,
        extra_args=["--sound-profile", _HEADLESS_SOUND_PROFILE],
    )


def open_in_musescore(score_path: str | Path) -> bool:
    """Spawn MuseScore's GUI editor on the given score. Returns False if mscore missing.

    Non-blocking: returns once the process is spawned. Useful for "edit the
    auto-transcription in MuseScore" workflows.
    """
    mscore = _find_musescore()
    if mscore is None:
        return False
    subprocess.Popen(  # noqa: S603
        [mscore, str(score_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True
