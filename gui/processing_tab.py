"""
ProcessingTab: the main pipeline control and results panel.

Layout:
  ┌──────────────────────────────────────────────────────────────────┐
  │  [Input controls]          [Intermediate results]                │
  │  Upload / size / SNR /     Original | Clean | Noisy previews     │
  │  mode radios               Noise var / Compile / Sim status      │
  ├──────────────────────────────────────────────────────────────────┤
  │  [Run Pipeline]  [Clear]   ████████████ 65%                      │
  ├──────────────────────────────────────────────────────────────────┤
  │  Log ─────────────────────────────────────────────────────────── │
  │  step 1 ... step 2 ... step 3 ...                                │
  ├──────────────────────────────────────────────────────────────────┤
  │  [5×5 Result]              [3×3 Result]                          │
  │  restored image            restored image                        │
  │  Metric | Before | After   Metric | Before | After               │
  └──────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

_SAVE_STYLE = (
    "QPushButton { background:#111111; color:#ffffff; border:1px solid #111111; border-radius:4px;"
    " padding:4px 10px; font-size:12px; font-weight:600; }"
    "QPushButton:disabled { background:#d0d0d0; color:#777777; border-color:#d0d0d0; }"
    "QPushButton:hover:!disabled { background:#2a2a2a; border-color:#2a2a2a; }"
)

from gui.image_panel import ImagePanel
from pipeline.constants import DEFAULT_SIZE, DEFAULT_SNR, RTL_HARDCODED_SIZE


# ── tiny helper: before/after metrics table ──────────────────────────────────

class _MetricsTable(QWidget):
    """
    3-column table showing before/after metric values.

    Metric  │ Before (vs noisy) │ After (vs restored)
    ────────┼───────────────────┼────────────────────
    SNR     │        —          │         —
    PSNR    │        —          │         —
    SSIM    │        —          │         —
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from PySide6.QtWidgets import QGridLayout, QFrame

        grid = QGridLayout(self)
        grid.setSpacing(4)
        grid.setContentsMargins(4, 4, 4, 4)

        # Header row
        for col, text in enumerate(["Metric", "Before\n(vs noisy)", "After\n(vs restored)"]):
            lbl = QLabel(f"<b>{text}</b>")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background:#f1f1f1; color:#111111; padding:3px; border-radius:3px; border:1px solid #d0d0d0;")
            grid.addWidget(lbl, 0, col)

        # Data rows: (key, label)
        self._cells: dict[str, tuple[QLabel, QLabel]] = {}
        rows = [("snr", "SNR"), ("psnr", "PSNR"), ("ssim", "SSIM")]
        for r, (key, label) in enumerate(rows, start=1):
            name_lbl   = QLabel(f"<b>{label}</b>")
            before_lbl = QLabel("—")
            after_lbl  = QLabel("—")
            for lbl in (name_lbl, before_lbl, after_lbl):
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet("padding:2px;")
            # Alternating row background
            bg = "#fafafa" if r % 2 else "#f4f4f4"
            for lbl in (name_lbl, before_lbl, after_lbl):
                lbl.setStyleSheet(f"padding:2px; background:{bg}; color:#222222;")
            grid.addWidget(name_lbl,   r, 0)
            grid.addWidget(before_lbl, r, 1)
            grid.addWidget(after_lbl,  r, 2)
            self._cells[key] = (before_lbl, after_lbl)

    def update_metrics(self, metrics: dict) -> None:
        """Populate the table from a parse_metrics() dict."""
        def fmt_db(v: float | None) -> str:
            return f"{v:.3f} dB" if v is not None else "—"
        def fmt_f(v: float | None) -> str:
            return f"{v:.4f}" if v is not None else "—"

        snr_n  = metrics.get("snr_noisy")
        psnr_n = metrics.get("psnr_noisy")
        snr_r  = metrics.get("snr")
        psnr_r = metrics.get("psnr")
        ssim_r = metrics.get("ssim")

        b_snr,  a_snr  = self._cells["snr"]
        b_psnr, a_psnr = self._cells["psnr"]
        b_ssim, a_ssim = self._cells["ssim"]

        b_snr.setText(fmt_db(snr_n))
        a_snr.setText(fmt_db(snr_r))
        b_psnr.setText(fmt_db(psnr_n))
        a_psnr.setText(fmt_db(psnr_r))
        b_ssim.setText("—")           # SSIM vs noisy not computed
        a_ssim.setText(fmt_f(ssim_r))

    def clear_values(self) -> None:
        for before, after in self._cells.values():
            before.setText("—")
            after.setText("—")


# ── main tab widget ───────────────────────────────────────────────────────────

class ProcessingTab(QWidget):
    """Main pipeline control tab."""

    # Emitted when Run is clicked; connected to PipelineWorker.run()
    run_requested = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image_path: str | None = None
        self._build_ui()

    # ── public API (called by MainWindow) ─────────────────────────────────────

    def set_input_image(self, path: str) -> None:
        """Load *path* as the active input (used by generator tab)."""
        self._image_path = path
        self._file_label.setText(Path(path).name)
        self._preview_panel.set_image(Path(path))

    # ── slots wired up by MainWindow ─────────────────────────────────────────

    @Slot(str)
    def on_log(self, message: str) -> None:
        self._log.append(message)
        self._log.moveCursor(QTextCursor.End)

    @Slot(int)
    def on_progress(self, value: int) -> None:
        self._progress_bar.setValue(value)

    @Slot(str, dict)
    def on_step_done(self, step_id: str, data: dict) -> None:
        if step_id == "hex":
            self._orig_panel.set_image(Path(self._image_path))
            self._clean_panel.set_image(Path(data["clean_image_path"]))
            self._noisy_panel.set_image(Path(data["noisy_image_path"]))

        elif step_id == "noise":
            nv = data["noise_var"]
            self._noise_var_lbl.setText(f"Noise Variance: {nv:.4f}")

        elif step_id == "compile":
            self._compile_lbl.setText('<span style="color:#111111">&#10003; Compile: OK</span>')
            self._compile_lbl.setTextFormat(Qt.RichText)

        elif step_id == "sim":
            self._sim_lbl.setText('<span style="color:#111111">&#10003; Simulation: OK</span>')
            self._sim_lbl.setTextFormat(Qt.RichText)

        elif step_id in ("reconstruct_5x5", "reconstruct_3x3"):
            kernel = "5x5" if "5x5" in step_id else "3x3"
            grp = self._result_groups[kernel]
            grp["panel"].set_image(Path(data["image_path"]))
            grp["metrics_table"].update_metrics(data.get("metrics", {}))
            grp["save_btn"].setEnabled(True)

    @Slot(str, str)
    def on_error(self, step_id: str, message: str) -> None:
        self._append_log(f"[ERROR in {step_id}] {message}", color="#333333")
        self._run_btn.setEnabled(True)

    @Slot()
    def on_done(self) -> None:
        self._append_log("=== Pipeline complete ===", color="#111111")
        self._run_btn.setEnabled(True)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(6, 6, 6, 6)

        # ── top splitter: controls left, intermediate right ──────────────────
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.setChildrenCollapsible(False)
        top_splitter.addWidget(self._build_input_panel())
        top_splitter.addWidget(self._build_intermediate_panel())
        top_splitter.setSizes([320, 600])
        top_splitter.setFixedHeight(340)
        root.addWidget(top_splitter)

        # ── run bar ──────────────────────────────────────────────────────────
        run_bar = QWidget()
        run_layout = QHBoxLayout(run_bar)
        run_layout.setContentsMargins(0, 0, 0, 0)
        run_layout.setSpacing(6)

        self._run_btn = QPushButton("▶  Run Pipeline")
        self._run_btn.setFixedHeight(34)
        self._run_btn.setStyleSheet(
            "QPushButton { background:#111111; color:#ffffff; font-weight:700;"
            " border-radius:4px; border:1px solid #111111; padding:0 12px; }"
            "QPushButton:disabled { background:#d0d0d0; color:#777777; border-color:#d0d0d0; }"
            "QPushButton:hover:!disabled { background:#2a2a2a; border-color:#2a2a2a; }"
        )
        self._run_btn.clicked.connect(self._on_run)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setFixedHeight(34)
        self._clear_btn.clicked.connect(self._on_clear)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)

        run_layout.addWidget(self._run_btn)
        run_layout.addWidget(self._clear_btn)
        run_layout.addWidget(self._progress_bar, stretch=1)
        root.addWidget(run_bar)

        # ── log ──────────────────────────────────────────────────────────────
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(4, 4, 4, 4)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(110)
        font = self._log.font()
        font.setFamily("Menlo,Courier New,monospace")
        font.setPointSize(10)
        self._log.setFont(font)
        log_layout.addWidget(self._log)
        root.addWidget(log_group)

        # ── results ──────────────────────────────────────────────────────────
        results_splitter = QSplitter(Qt.Horizontal)
        results_splitter.setChildrenCollapsible(False)
        self._result_groups: dict[str, dict] = {}
        for kernel in ("5x5", "3x3"):
            grp_widget = self._build_result_panel(kernel)
            results_splitter.addWidget(grp_widget)
        root.addWidget(results_splitter, stretch=1)

        self._on_mode_changed()   # set initial visibility

    def _build_input_panel(self) -> QGroupBox:
        group = QGroupBox("Input")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        # Upload row
        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        self._upload_btn = QPushButton("Upload Image…")
        self._upload_btn.clicked.connect(self._on_upload)
        self._file_label = QLabel("No file selected")
        self._file_label.setWordWrap(True)
        self._file_label.setStyleSheet("color:#666666; font-style:italic; font-size:12px;")
        hl.addWidget(self._upload_btn)
        hl.addWidget(self._file_label, stretch=1)
        layout.addWidget(row)

        # Preview
        self._preview_panel = ImagePanel(max_size=130, placeholder="Upload an image")
        layout.addWidget(self._preview_panel, alignment=Qt.AlignHCenter)

        # Size
        size_row = QWidget()
        sl = QHBoxLayout(size_row)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(QLabel("Image Size:"))
        self._size_spin = QSpinBox()
        self._size_spin.setRange(32, 4096)
        self._size_spin.setValue(DEFAULT_SIZE)
        self._size_spin.setSingleStep(32)
        sl.addWidget(self._size_spin)
        sl.addStretch()
        layout.addWidget(size_row)

        # SNR
        snr_row = QWidget()
        nr = QHBoxLayout(snr_row)
        nr.setContentsMargins(0, 0, 0, 0)
        nr.addWidget(QLabel("Target SNR dB:"))
        self._snr_spin = QSpinBox()
        self._snr_spin.setRange(1, 60)
        self._snr_spin.setValue(DEFAULT_SNR)
        nr.addWidget(self._snr_spin)
        nr.addStretch()
        layout.addWidget(snr_row)

        # Mode radios
        mode_box = QGroupBox("Mode")
        mode_box.setFlat(True)
        ml = QHBoxLayout(mode_box)
        self._radio_5x5  = QRadioButton("5×5")
        self._radio_3x3  = QRadioButton("3×3")
        self._radio_both = QRadioButton("Both")
        self._radio_both.setChecked(True)
        for rb in (self._radio_5x5, self._radio_3x3, self._radio_both):
            rb.toggled.connect(self._on_mode_changed)
            ml.addWidget(rb)
        layout.addWidget(mode_box)
        layout.addStretch()
        return group

    def _build_intermediate_panel(self) -> QGroupBox:
        group = QGroupBox("Intermediate Results")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        # Three image previews side by side
        img_row = QWidget()
        img_hl = QHBoxLayout(img_row)
        img_hl.setSpacing(8)
        img_hl.setContentsMargins(0, 0, 0, 0)

        for attr, title in (
            ("_orig_panel",  "Original"),
            ("_clean_panel", "Clean"),
            ("_noisy_panel", "Noisy"),
        ):
            col = QVBoxLayout()
            lbl = QLabel(title)
            lbl.setAlignment(Qt.AlignHCenter)
            lbl.setStyleSheet("font-weight:bold; font-size:11px;")
            panel = ImagePanel(max_size=150, placeholder="—")
            setattr(self, attr, panel)
            col.addWidget(lbl)
            col.addWidget(panel)
            img_hl.addLayout(col)

        layout.addWidget(img_row)

        # Status labels
        self._noise_var_lbl = QLabel("Noise Variance: —")
        self._compile_lbl   = QLabel("Compile: —")
        self._sim_lbl       = QLabel("Simulation: —")
        for lbl in (self._noise_var_lbl, self._compile_lbl, self._sim_lbl):
            lbl.setStyleSheet("padding: 2px 4px;")
            layout.addWidget(lbl)
        layout.addStretch()
        return group

    def _build_result_panel(self, kernel: str) -> QGroupBox:
        """Build a result group for 5×5 or 3×3, store refs in self._result_groups."""
        group = QGroupBox(f"{kernel} Result")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        panel = ImagePanel(max_size=360, placeholder="—")
        metrics_table = _MetricsTable()

        # Save button (disabled until an image is available)
        save_btn = QPushButton("💾  Save Image…")
        save_btn.setEnabled(False)
        save_btn.setStyleSheet(_SAVE_STYLE)
        save_btn.setFixedHeight(28)
        save_btn.clicked.connect(lambda checked=False, k=kernel: self._on_save_result(k))

        layout.addWidget(panel, stretch=1)
        layout.addWidget(metrics_table)
        layout.addWidget(save_btn)

        self._result_groups[kernel] = {
            "group":         group,
            "panel":         panel,
            "metrics_table": metrics_table,
            "save_btn":      save_btn,
        }
        return group

    # ── slots ─────────────────────────────────────────────────────────────────

    @Slot()
    def _on_upload(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)",
        )
        if path:
            self.set_input_image(path)

    @Slot()
    def _on_mode_changed(self) -> None:
        mode = self._get_mode()
        for kernel, d in self._result_groups.items():
            visible = (mode == kernel) or (mode == "both")
            d["group"].setVisible(visible)

    @Slot()
    def _on_run(self) -> None:
        if not self._image_path:
            self._append_log("Please upload an image first.", color="#333333")
            return
        self._run_btn.setEnabled(False)
        self._progress_bar.setValue(0)
        self._log.clear()
        config = {
            "image_path": self._image_path,
            "size":       self._size_spin.value(),
            "snr":        float(self._snr_spin.value()),
            "mode":       self._get_mode(),
        }
        self.run_requested.emit(config)

    @Slot()
    def _on_clear(self) -> None:
        self._log.clear()
        self._progress_bar.setValue(0)
        for panel in (self._orig_panel, self._clean_panel, self._noisy_panel):
            panel.clear("—")
        self._noise_var_lbl.setText("Noise Variance: —")
        self._compile_lbl.setText("Compile: —")
        self._sim_lbl.setText("Simulation: —")
        for d in self._result_groups.values():
            d["panel"].clear("—")
            d["metrics_table"].clear_values()
            d["save_btn"].setEnabled(False)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _on_save_result(self, kernel: str) -> None:
        grp = self._result_groups.get(kernel)
        if grp is None:
            return
        px = grp["panel"].pixmap()
        if px is None:
            return
        default = f"restored_{kernel}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, f"Save {kernel} Result", default,
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All Files (*)",
        )
        if path:
            px.save(path)

    def _get_mode(self) -> str:
        if self._radio_5x5.isChecked():
            return "5x5"
        if self._radio_3x3.isChecked():
            return "3x3"
        return "both"

    def _append_log(self, text: str, color: str | None = None) -> None:
        if color:
            self._log.append(f'<span style="color:{color}">{text}</span>')
        else:
            self._log.append(text)
        self._log.moveCursor(QTextCursor.End)
