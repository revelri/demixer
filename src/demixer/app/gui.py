"""demixer desktop GUI — drag-drop an audio file, watch it become stems + MIDI +
sheet music + DAW projects, with a live progress bar and status log.

Run:  demixer-gui   (or  uv run python -m demixer.app.gui )
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from demixer.app.pipeline_runner import PipelineWorker, RunOptions

_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".opus"}


class DropZone(QFrame):
    """Click-or-drag target for the input audio file."""

    def __init__(self, on_file) -> None:
        super().__init__()
        self._on_file = on_file
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumHeight(110)
        self.setStyleSheet(
            "DropZone { border: 2px dashed #888; border-radius: 10px; background: #2b2b2b; }"
        )
        lay = QVBoxLayout(self)
        self._label = QLabel("Drop an audio file here\nor click to browse")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("color: #bbb; border: none;")
        lay.addWidget(self._label)

    def set_file(self, path: Path) -> None:
        self._label.setText(f"🎵 {path.name}\n{path.parent}")

    def mousePressEvent(self, _event) -> None:  # noqa: N802
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose audio", "", "Audio (*.wav *.flac *.mp3 *.m4a *.ogg *.aac *.opus)")
        if path:
            self._on_file(Path(path))

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.suffix.lower() in _AUDIO_EXTS:
                self._on_file(p)
                return


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("demixer — audio → stems · MIDI · sheet music · DAW")
        self.resize(820, 720)
        self._input: Path | None = None
        self._worker: PipelineWorker | None = None
        self._out_dir: Path | None = None

        root = QVBoxLayout(self)

        self._drop = DropZone(self._set_input)
        root.addWidget(self._drop)

        # --- options ---
        opts = QFormLayout()
        self._model = self._combo(["htdemucs", "htdemucs_ft", "htdemucs_6s"])
        self._transcriber = self._combo(["basic-pitch", "mt3"])
        self._chords = self._combo(["autochord", "btc"])
        self._drums = self._combo(["spectral", "adtof"])
        self._reharm = self._combo(["(none)", "smoothest", "tritone", "relative",
                                    "modal-interchange", "secondary-dominant"])
        self._roformer = QCheckBox("BS-RoFormer vocals (real-vocal tracks)")
        self._harmony = QCheckBox("Harmony analysis (function/tension/substitutions)")
        opts.addRow("Separation model", self._model)
        opts.addRow("Pitched transcriber", self._transcriber)
        opts.addRow("Chords", self._chords)
        opts.addRow("Drums", self._drums)
        opts.addRow("Reharmonize", self._reharm)
        opts.addRow("", self._roformer)
        opts.addRow("", self._harmony)
        root.addLayout(opts)

        # --- run / cancel ---
        btn_row = QHBoxLayout()
        self._run = QPushButton("▶  Process")
        self._run.setEnabled(False)
        self._run.clicked.connect(self._start)
        self._cancel = QPushButton("Cancel")
        self._cancel.setEnabled(False)
        self._cancel.clicked.connect(self._cancel_run)
        btn_row.addWidget(self._run)
        btn_row.addWidget(self._cancel)
        root.addLayout(btn_row)

        # --- progress ---
        self._stage = QLabel("idle")
        self._stage.setStyleSheet("color: #8ab4f8;")
        root.addWidget(self._stage)
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        root.addWidget(self._bar)

        # --- status log ---
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("monospace", 9))
        self._log.setStyleSheet("background: #1e1e1e; color: #ccc;")
        root.addWidget(self._log, stretch=1)

        # --- results ---
        self._results = QListWidget()
        self._results.itemDoubleClicked.connect(self._open_result_item)
        self._results.setMaximumHeight(150)
        root.addWidget(QLabel("Results (double-click to open):"))
        root.addWidget(self._results)
        self._open_dir_btn = QPushButton("Open output folder")
        self._open_dir_btn.setEnabled(False)
        self._open_dir_btn.clicked.connect(self._open_out_dir)
        root.addWidget(self._open_dir_btn)

    @staticmethod
    def _combo(items: list[str]) -> QComboBox:
        c = QComboBox()
        c.addItems(items)
        return c

    def _set_input(self, path: Path) -> None:
        self._input = path
        self._drop.set_file(path)
        self._run.setEnabled(True)
        self._log.appendPlainText(f"# input: {path}")

    def _start(self) -> None:
        if not self._input:
            return
        reharm = self._reharm.currentText()
        out = self._input.with_suffix("").parent / f"{self._input.stem}_demix"
        self._out_dir = out
        opts = RunOptions(
            input_path=self._input, output_dir=out,
            model=self._model.currentText(),
            transcriber=self._transcriber.currentText(),
            chords=self._chords.currentText(),
            drums=self._drums.currentText(),
            roformer_vocals=self._roformer.isChecked(),
            harmony=self._harmony.isChecked() or reharm != "(none)",
            reharmonize=None if reharm == "(none)" else reharm,
        )
        self._log.appendPlainText(f"# running: {' '.join(opts.to_argv()[2:])}\n")
        self._results.clear()
        self._bar.setValue(0)
        self._run.setEnabled(False)
        self._cancel.setEnabled(True)

        self._worker = PipelineWorker(opts)
        self._worker.log_line.connect(self._append_log)
        self._worker.progress.connect(self._bar.setValue)
        self._worker.stage.connect(lambda s: self._stage.setText(f"▶ {s}"))
        self._worker.finished_ok.connect(self._done)
        self._worker.failed.connect(self._error)
        self._worker.start()

    def _cancel_run(self) -> None:
        if self._worker:
            self._worker.cancel()
            self._stage.setText("cancelling…")

    def _append_log(self, line: str) -> None:
        self._log.appendPlainText(line)

    def _done(self, out_dir: str) -> None:
        self._stage.setText("✓ done")
        self._run.setEnabled(True)
        self._cancel.setEnabled(False)
        self._open_dir_btn.setEnabled(True)
        self._populate_results(Path(out_dir))

    def _error(self, msg: str) -> None:
        self._stage.setText(f"✗ {msg}")
        self._stage.setStyleSheet("color: #f28b82;")
        self._run.setEnabled(True)
        self._cancel.setEnabled(False)

    def _populate_results(self, out_dir: Path) -> None:
        self._results.clear()
        wanted = [
            ("stems/", "🎚  stems (WAV)"),
            ("midi/", "🎹  MIDI"),
            ("score.pdf", "🎼  sheet music (PDF)"),
            ("score.musicxml", "🎼  MusicXML"),
            ("harmony.json", "🎵  harmony analysis"),
            ("reharmonization.mid", "🔁  reharmonization MIDI"),
        ]
        for rel, label in wanted:
            p = out_dir / rel
            if p.exists():
                self._results.addItem(f"{label}   —   {rel}")
        # DAW projects + archive
        for pat, label in ((".rpp", "🎛  Reaper project"), (".dawproject", "🎛  DAWproject"),
                          (".demixer", "📦  bundle archive")):
            for f in sorted(out_dir.parent.glob(f"*{pat}")) + sorted(out_dir.glob(f"*{pat}")):
                self._results.addItem(f"{label}   —   {f.name}")
        self._results.addItem(f"📁  {out_dir}")

    def _open_out_dir(self) -> None:
        if self._out_dir:
            _open_path(self._out_dir)

    def _open_result_item(self, item) -> None:
        if not self._out_dir:
            return
        text = item.text()
        rel = text.split("—")[-1].strip()
        target = self._out_dir / rel
        _open_path(target if target.exists() else self._out_dir)


def _open_path(path: Path) -> None:
    """Open a file/folder with the platform handler."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:  # noqa: BLE001, S110
        pass


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
