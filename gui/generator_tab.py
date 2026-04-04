"""
GeneratorTab: create synthetic grayscale test images and feed them into the pipeline.

Layout:
  ┌─────────────────────┬──────────────────────────────┐
  │ Pattern:  [combo]   │                              │
  │ Size:     [512   ]  │     preview image            │
  │ Seed:     [42    ]  │     (400 × 400)              │
  │                     │                              │
  │ [Generate Preview]  │                              │
  │                     │                              │
  │ [Use as Input →]    │                              │
  └─────────────────────┴──────────────────────────────┘
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui.image_panel import ImagePanel
from pipeline.constants import DEFAULT_SIZE, FRONTEND_DIR
from utils.test_image_gen import PATTERN_NAMES, save_image

_EDGE_CASES = {
    "All Black", "All White", "Uniform Gray", "Step Edge",
    "Single Bright Pixel", "Salt & Pepper (Sparse)", "High Freq Grid", "Low Contrast",
}
_OBJ_SET = {
    "Airplane", "Bird in Flight", "House", "Tree",
    "Car (Side)", "Star", "Human Figure", "Rocket",
}

# Three groups shown in the combo box
_GEO_PATTERNS  = [p for p in PATTERN_NAMES if p not in _OBJ_SET and p not in _EDGE_CASES]
_OBJ_PATTERNS  = [p for p in PATTERN_NAMES if p in _OBJ_SET]
_EDGE_PATTERNS = [p for p in PATTERN_NAMES if p in _EDGE_CASES]


class GeneratorTab(QWidget):
    """Synthetic test image generator tab."""

    # Emits the path of a saved image; connected to MainWindow
    image_ready = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_saved_path: str | None = None
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── left: controls ────────────────────────────────────────────────────
        ctrl_group = QGroupBox("Test Image Settings")
        ctrl_group.setFixedWidth(280)
        ctrl_layout = QVBoxLayout(ctrl_group)
        ctrl_layout.setSpacing(10)

        form = QFormLayout()
        form.setSpacing(8)

        # Pattern combo — grouped with separators
        self._pattern_combo = QComboBox()
        self._pattern_combo.addItems(_GEO_PATTERNS)
        self._pattern_combo.insertSeparator(self._pattern_combo.count())
        self._pattern_combo.addItems(_OBJ_PATTERNS)
        self._pattern_combo.insertSeparator(self._pattern_combo.count())
        self._pattern_combo.addItems(_EDGE_PATTERNS)
        form.addRow("Pattern:", self._pattern_combo)

        # Size
        self._size_spin = QSpinBox()
        self._size_spin.setRange(32, 4096)
        self._size_spin.setValue(DEFAULT_SIZE)
        self._size_spin.setSingleStep(32)
        form.addRow("Size (px):", self._size_spin)

        # Seed
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 99999)
        self._seed_spin.setValue(42)
        form.addRow("Random Seed:", self._seed_spin)

        ctrl_layout.addLayout(form)
        ctrl_layout.addSpacing(6)

        # Generate button
        self._gen_btn = QPushButton("Generate Preview")
        self._gen_btn.setStyleSheet(
            "QPushButton { padding: 6px; border-radius:4px; }"
        )
        self._gen_btn.clicked.connect(self._on_generate)
        ctrl_layout.addWidget(self._gen_btn)

        # Status label
        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet("color:#666666; font-size:11px;")
        ctrl_layout.addWidget(self._status_lbl)

        ctrl_layout.addSpacing(10)

        # Divider label
        sep = QLabel("── Send to Pipeline ──")
        sep.setAlignment(Qt.AlignCenter)
        sep.setStyleSheet("color:#888888; font-size:11px;")
        ctrl_layout.addWidget(sep)
        ctrl_layout.addSpacing(4)

        # Use as Input button
        self._use_btn = QPushButton("▶  Use as Pipeline Input")
        self._use_btn.setEnabled(False)
        self._use_btn.setStyleSheet(
            "QPushButton { background:#111111; color:#ffffff; font-weight:700;"
            " border-radius:4px; border:1px solid #111111; padding:6px; }"
            "QPushButton:disabled { background:#d0d0d0; color:#777777; border-color:#d0d0d0; }"
            "QPushButton:hover:!disabled { background:#2a2a2a; border-color:#2a2a2a; }"
        )
        self._use_btn.clicked.connect(self._on_use_as_input)
        ctrl_layout.addWidget(self._use_btn)
        ctrl_layout.addStretch()

        # Help note
        note = QLabel(
            "Tip: choose a geometric pattern, an object silhouette, or an edge-case "
            "(all-black, all-white, step edge, etc.). Generate a preview, "
            "then send it straight to the pipeline."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#888888; font-size:10px;")
        ctrl_layout.addWidget(note)

        root.addWidget(ctrl_group)

        # ── right: preview ────────────────────────────────────────────────────
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(6, 6, 6, 6)

        self._preview_panel = ImagePanel(max_size=520, placeholder="Click 'Generate Preview'")
        preview_layout.addWidget(self._preview_panel, stretch=1)

        self._info_lbl = QLabel("")
        self._info_lbl.setAlignment(Qt.AlignCenter)
        self._info_lbl.setStyleSheet("color:#666666; font-size:11px;")
        preview_layout.addWidget(self._info_lbl)

        root.addWidget(preview_group, stretch=1)

    # ── slots ─────────────────────────────────────────────────────────────────

    @Slot()
    def _on_generate(self) -> None:
        pattern = self._pattern_combo.currentText()
        size    = self._size_spin.value()
        seed    = self._seed_spin.value()

        # Save into frontEnd/ so the pipeline scripts can access it
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = FRONTEND_DIR / f"generated_{ts}.png"

        try:
            save_image(pattern, size, seed, path)
        except Exception as exc:
            QMessageBox.critical(self, "Generator Error", str(exc))
            return

        self._last_saved_path = str(path)
        self._preview_panel.set_image(path)
        self._status_lbl.setText(f"Saved: {path.name}")
        self._info_lbl.setText(f"{pattern}  |  {size}×{size}  |  seed {seed}")
        self._use_btn.setEnabled(True)

    @Slot()
    def _on_use_as_input(self) -> None:
        if self._last_saved_path:
            self.image_ready.emit(self._last_saved_path)
