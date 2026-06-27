End-to-end music deconstruction tool: separates a song into stems (Demucs), transcribes each to MIDI, analyzes tempo/key/chords/harmony, engraves a score, and exports Reaper/.dawproject/FL Studio projects — all packed into a single `.demixer` archive. Python 3.11 CLI with an optional PySide6 desktop shell.

## Current phase

Post-audit hardening — the full pipeline runs end-to-end (exit 0 across 20+ genres, all engine backends); 14 bugs found and fixed in a deep audit pass (2026-05-24); accuracy validated against Isophonics ground truth (key 0.763, tempo 0.916, chord majmin 0.752 with BTC). Version 0.0.1, AGPL-3.0, 129 passing tests.

Outstanding work:
- **PySide6 GUI**: shell constructs and imports cleanly (headless smoke-tested); interaction logic (file picker, pipeline progress, output view) is incomplete.
- **ADTOF drums**: shipped opt-in (`--drums adtof`, onset-F1 0.975 vs 0.825 spectral); not promoted to default pending an ear-check on real-stem class accuracy.
- **YourMT3+ polyphony**: upgrade path researched and attempted; blocked by upstream breakage (inference adapter broken, pytorch-port LFS smudge broken as of 2026-05-24).
- **Score engraving quality**: current fixed 16th-grid quantization is robust but naive; learned rhythm quantization (arXiv:2508.19262) identified as the highest perceived-quality gain — medium-high effort, deferred.

## revoy ledger block

<!-- revoy:begin -->
```toml
phase = "post-audit hardening"

[[todo]]
line = "Complete PySide6 GUI interaction logic: file picker, pipeline stage progress display, and output file tree"
difficulty = 45
priority = "MED"

[[todo]]
line = "Promote ADTOF drums to default after an ear-check on real-stem class accuracy vs the spectral classifier"
difficulty = 15
priority = "MED"

[[todo]]
line = "Unblock YourMT3+ worker once upstream inference adapter or pytorch-port LFS is fixed; add --transcriber yourmt3 option and A/B vs basic-pitch on polyphonic stems"
difficulty = 40
priority = "LOW"

[[todo]]
line = "Replace fixed-16th-grid quantize.py with a learned rhythm-quantization stage (2508.19262 approach) for human-readable scores; keep grid-snap as fallback"
difficulty = 65
priority = "LOW"

[[todo]]
line = "CLAP/VST3 companion plugin for one-click .demixer load in any DAW (nih-plug; design documented in docs/DAW_INTEGRATION_PLAN.md)"
difficulty = 80
priority = "VLOW"
```
<!-- revoy:end -->
