"""
ZoomDialog: full-screen image viewer with smooth scroll-wheel zoom,
drag-to-pan, keyboard shortcuts, and a Save button.

Usage:
    dlg = ZoomDialog(pixmap, title="5×5 Result", parent=self)
    dlg.exec()
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QKeySequence,
    QPainter,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class _ZoomView(QGraphicsView):
    """A QGraphicsView with scroll-wheel zoom and drag-to-pan."""

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(Qt.black)
        self._zoom = 1.0

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom *= factor
        self._zoom = max(0.05, min(self._zoom, 40.0))
        self.scale(factor, factor)

    def zoom_in(self)  -> None: self.scale(1.25, 1.25)
    def zoom_out(self) -> None: self.scale(1 / 1.25, 1 / 1.25)

    def reset_zoom(self, item) -> None:
        self.fitInView(item, Qt.KeepAspectRatio)


class ZoomDialog(QDialog):
    """Modal image viewer with zoom/pan/save controls."""

    def __init__(
        self,
        pixmap: QPixmap,
        title: str = "Image",
        source_path: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 750)
        self._pixmap = pixmap
        self._source_path = source_path
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── graphics view ─────────────────────────────────────────────────────
        self._scene = QGraphicsScene(self)
        self._item  = QGraphicsPixmapItem(self._pixmap)
        self._scene.addItem(self._item)

        self._view = _ZoomView(self._scene, self)
        root.addWidget(self._view, stretch=1)

        # ── toolbar ───────────────────────────────────────────────────────────
        bar = QHBoxLayout()
        bar.setSpacing(6)

        def _btn(label: str, tip: str) -> QPushButton:
            b = QPushButton(label)
            b.setToolTip(tip)
            b.setFixedHeight(30)
            return b

        zoom_in_btn  = _btn("  +  ", "Zoom in  (scroll up)")
        zoom_out_btn = _btn("  −  ", "Zoom out  (scroll down)")
        reset_btn    = _btn("Fit", "Fit image to window")
        actual_btn   = _btn("1:1", "Actual pixel size")
        save_btn     = _btn("💾  Save Image…", "Save to file")
        save_btn.setStyleSheet(
            "QPushButton { background:#111111; color:#ffffff;"
            " border-radius:5px; border:1px solid #111111; padding:0 10px; font-weight:600; }"
            "QPushButton:hover { background:#2a2a2a; border-color:#2a2a2a; }"
        )

        zoom_in_btn.clicked.connect(self._view.zoom_in)
        zoom_out_btn.clicked.connect(self._view.zoom_out)
        reset_btn.clicked.connect(lambda: self._view.reset_zoom(self._item))
        actual_btn.clicked.connect(self._actual_size)
        save_btn.clicked.connect(self._on_save)

        # Size info
        self._info_lbl = QLabel(
            f"{self._pixmap.width()} × {self._pixmap.height()} px"
        )
        self._info_lbl.setStyleSheet("color:#666666; font-size:11px; font-weight:600;")

        bar.addWidget(zoom_in_btn)
        bar.addWidget(zoom_out_btn)
        bar.addWidget(reset_btn)
        bar.addWidget(actual_btn)
        bar.addStretch()
        bar.addWidget(self._info_lbl)
        bar.addStretch()
        bar.addWidget(save_btn)
        root.addLayout(bar)

        # Keyboard shortcuts
        from PySide6.QtGui import QShortcut
        QShortcut(QKeySequence("Ctrl+="), self).activated.connect(self._view.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self).activated.connect(self._view.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self).activated.connect(
            lambda: self._view.reset_zoom(self._item)
        )

        # Fit on show
        self._view.reset_zoom(self._item)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._view.reset_zoom(self._item)

    def _actual_size(self) -> None:
        self._view.resetTransform()

    def _on_save(self) -> None:
        default = Path(self._source_path).name if self._source_path else "image.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", default,
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All Files (*)",
        )
        if path:
            self._pixmap.save(path)
