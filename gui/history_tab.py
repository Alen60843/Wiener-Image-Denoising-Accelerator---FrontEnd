"""
HistoryTab: scrollable gallery of all past pipeline runs.

Each card shows:
  ┌──────────────────────────────────────────────────────┐
  │  2026-03-08 19:04  │  512×512  SNR 14 dB  Mode: both │
  │  [original] [clean] [noisy] [5x5] [3x3]             │
  │  Noise σ²: 867.5                                     │
  │  Metric  Before  After (5×5) After (3×3)             │
  │  SNR     14.3    25.1        24.9                    │
  │  PSNR    28.1    33.5        33.2                    │
  │  SSIM    —       0.8734      0.8611                  │
  └──────────────────────────────────────────────────────┘
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# ── single history card ───────────────────────────────────────────────────────

class _ThumbLabel(QLabel):
    """Clickable thumbnail that opens a ZoomDialog."""

    def __init__(self, path_str: str, size: int = 90, parent=None) -> None:
        super().__init__(parent)
        self._path = path_str
        self._px = QPixmap(path_str)
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "border:1px solid #d0d0d0; background:#f7f7f7;"
            " border-radius:3px;"
        )
        if not self._px.isNull():
            self.setPixmap(
                self._px.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            self.setCursor(QCursor(Qt.PointingHandCursor))
            self.setToolTip("Double-click to zoom")

    def mouseDoubleClickEvent(self, event) -> None:
        if not self._px.isNull():
            from gui.zoom_dialog import ZoomDialog
            title = Path(self._path).name
            dlg = ZoomDialog(self._px, title=title, source_path=self._path, parent=self)
            dlg.exec()
        super().mouseDoubleClickEvent(event)


class _HistoryCard(QFrame):
    """One card per pipeline run."""

    THUMB = 90   # thumbnail pixel size

    def __init__(self, entry: Dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame {"
            "  background: #ffffff;"
            "  border: 1px solid #d7d7d7;"
            "  border-left: 3px solid #111111;"
            "  border-radius: 6px;"
            "  margin: 2px;"
            "}"
        )
        self._build(entry)

    def _build(self, e: Dict[str, Any]) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(10, 8, 10, 8)

        # ── header row ────────────────────────────────────────────────────────
        header = QLabel(
            f"<span style='color:#111111;font-size:12px;font-weight:600;'>"
            f"{e.get('timestamp', '—')}</span>"
            f"<span style='color:#888888;'> &nbsp;│&nbsp; </span>"
            f"<span style='color:#222222;'>"
            f"{e.get('size', '?')}×{e.get('size', '?')}"
            f" &nbsp;·&nbsp; SNR {e.get('snr_target', '?')} dB"
            f" &nbsp;·&nbsp; Mode: <b>{e.get('mode', '?')}</b></span>"
        )
        header.setTextFormat(Qt.RichText)
        root.addWidget(header)

        noise_var = e.get("noise_var")
        if noise_var is not None:
            nv_lbl = QLabel(
                f"<span style='color:#666666;font-size:11px;'>"
                f"Estimated noise variance: "
                f"<b style='color:#111111;'>{noise_var:.4f}</b></span>"
            )
            nv_lbl.setTextFormat(Qt.RichText)
            root.addWidget(nv_lbl)

        # ── thumbnail row ────────────────────────────────────────────────────
        thumb_row = QWidget()
        thumb_hl = QHBoxLayout(thumb_row)
        thumb_hl.setSpacing(8)
        thumb_hl.setContentsMargins(0, 0, 0, 0)

        images = e.get("images", {})
        results = e.get("results", {})

        for label, path_str in [
            ("Original", images.get("original_image")),
            ("Clean",    images.get("clean_image")),
            ("Noisy",    images.get("noisy_image")),
            ("5×5",      results.get("5x5", {}).get("image")),
            ("3×3",      results.get("3x3", {}).get("image")),
        ]:
            if not path_str:
                continue
            col = QVBoxLayout()
            col.setSpacing(3)
            col.setAlignment(Qt.AlignHCenter)
            lbl = QLabel(label)
            lbl.setAlignment(Qt.AlignHCenter)
            lbl.setStyleSheet("font-size:10px; color:#888888; font-weight:600;")
            thumb = _ThumbLabel(path_str, size=self.THUMB)
            col.addWidget(lbl)
            col.addWidget(thumb)
            thumb_hl.addLayout(col)

        thumb_hl.addStretch()
        root.addWidget(thumb_row)

        # ── metrics table ─────────────────────────────────────────────────────
        kernels_present = [k for k in ("5x5", "3x3") if k in results]
        if kernels_present:
            root.addWidget(self._build_metrics_table(results, kernels_present))

    def _build_metrics_table(
        self, results: dict, kernels: list[str]
    ) -> QWidget:
        """Build a compact metrics comparison table."""
        w = QWidget()
        grid = QGridLayout(w)
        grid.setSpacing(2)
        grid.setContentsMargins(0, 2, 0, 0)

        # Header
        headers = ["Metric", "Before\n(vs noisy)"] + [f"After ({k})" for k in kernels]
        for col, h in enumerate(headers):
            lbl = QLabel(f"<b>{h}</b>")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                "background:#f1f1f1; color:#111111; padding:3px 6px;"
                " border-radius:3px; font-size:11px;"
            )
            grid.addWidget(lbl, 0, col)

        # Rows
        metric_defs = [
            ("snr",  "snr_noisy",  "SNR",  "{:.3f} dB"),
            ("psnr", "psnr_noisy", "PSNR", "{:.4f} dB"),
            ("ssim", None,         "SSIM", "{:.4f}"),
        ]
        for row_i, (after_key, before_key, label, fmt) in enumerate(metric_defs, start=1):
            bg = "#fafafa" if row_i % 2 else "#f4f4f4"
            style = f"font-size:11px; padding:2px 6px; background:{bg}; color:#222222;"

            name_lbl = QLabel(f"<b>{label}</b>")
            name_lbl.setAlignment(Qt.AlignCenter)
            name_lbl.setStyleSheet(
                f"font-size:11px; padding:2px 6px; background:{bg};"
                " color:#111111; font-weight:600;"
            )
            grid.addWidget(name_lbl, row_i, 0)

            # "Before" column
            before_val = "—"
            if before_key:
                for k in kernels:
                    v = results.get(k, {}).get("metrics", {}).get(before_key)
                    if v is not None:
                        before_val = fmt.format(v)
                        break
            b_lbl = QLabel(before_val)
            b_lbl.setAlignment(Qt.AlignCenter)
            b_lbl.setStyleSheet(style)
            grid.addWidget(b_lbl, row_i, 1)

            # "After" columns
            for col_i, k in enumerate(kernels, start=2):
                v = results.get(k, {}).get("metrics", {}).get(after_key)
                val_str = fmt.format(v) if v is not None else "—"
                a_lbl = QLabel(val_str)
                a_lbl.setAlignment(Qt.AlignCenter)
                a_style = f"font-size:11px; padding:2px 6px; background:{bg}; color:#111111; font-weight:600;" if v is not None else style
                a_lbl.setStyleSheet(a_style)
                grid.addWidget(a_lbl, row_i, col_i)

        return w


# ── tab widget ────────────────────────────────────────────────────────────────

class HistoryTab(QWidget):
    """Scrollable gallery of all past runs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def load_entries(self, entries: list[Dict[str, Any]]) -> None:
        """Populate the gallery from a list of history entries (newest first)."""
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for entry in entries:
            self._cards_layout.addWidget(_HistoryCard(entry))
        self._cards_layout.addStretch()
        self._update_empty_label(len(entries) == 0)
        self._update_count(len(entries))

    def add_entry(self, entry: Dict[str, Any]) -> None:
        """Prepend a new card at the top (called after each successful run)."""
        if self._cards_layout.count():
            last = self._cards_layout.itemAt(self._cards_layout.count() - 1)
            if last and last.spacerItem():
                self._cards_layout.removeItem(last)

        self._cards_layout.insertWidget(0, _HistoryCard(entry))
        self._cards_layout.addStretch()
        self._update_empty_label(False)
        # Update count
        total = self._cards_layout.count() - 1  # subtract stretch
        self._update_count(total)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Title bar
        title_row = QWidget()
        title_hl = QHBoxLayout(title_row)
        title_hl.setContentsMargins(0, 0, 0, 0)
        title_lbl = QLabel("Run History")
        title_lbl.setStyleSheet(
            "font-size:15px; font-weight:700; color:#111111;"
        )
        self._count_lbl = QLabel("")
        self._count_lbl.setStyleSheet(
            "color:#888888; font-size:12px; padding-left:6px;"
        )
        title_hl.addWidget(title_lbl)
        title_hl.addWidget(self._count_lbl)
        title_hl.addStretch()
        root.addWidget(title_row)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        container = QWidget()
        self._cards_layout = QVBoxLayout(container)
        self._cards_layout.setSpacing(8)
        self._cards_layout.setContentsMargins(2, 2, 2, 2)

        self._empty_lbl = QLabel("No runs yet — run the pipeline to populate history.")
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        self._empty_lbl.setStyleSheet(
            "color:#888888; font-size:13px; padding:60px;"
        )
        self._cards_layout.addWidget(self._empty_lbl)
        self._cards_layout.addStretch()

        scroll.setWidget(container)
        root.addWidget(scroll, stretch=1)

    def _update_empty_label(self, is_empty: bool) -> None:
        self._empty_lbl.setVisible(is_empty)

    def _update_count(self, n: int) -> None:
        self._count_lbl.setText(f"({n} run{'s' if n != 1 else ''})")
