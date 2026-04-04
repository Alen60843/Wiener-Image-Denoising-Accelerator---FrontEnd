from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class ImagePanel(QWidget):
    """
    Reusable image thumbnail widget.
    Double-click (or Ctrl+click) opens a full ZoomDialog.
    A hand cursor and tooltip hint appear once an image is loaded.
    """

    def __init__(self, max_size: int = 200, placeholder: str = "No image", parent=None):
        super().__init__(parent)
        self._max_size = max_size
        self._pixmap: QPixmap | None = None
        self._source_path: str | None = None

        self._label = QLabel(placeholder)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setMinimumSize(max_size, max_size)
        self._label.setStyleSheet(
            "border: 1px solid #cfcfcf; background: #f8f8f8; color: #666666; border-radius: 4px;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

    def set_image(self, path: Path) -> None:
        px = QPixmap(str(path))
        if px.isNull():
            self._label.setText(f"Cannot load:\n{path.name}")
            self._pixmap = None
            self._source_path = None
            self._label.setCursor(Qt.ArrowCursor)
            self._label.setToolTip("")
            return
        self._pixmap = px
        self._source_path = str(path)
        self._show_scaled()
        self._label.setCursor(QCursor(Qt.PointingHandCursor))
        self._label.setToolTip("Double-click to zoom")

    def pixmap(self) -> QPixmap | None:
        """Return the original (unscaled) pixmap, or None if no image is loaded."""
        return self._pixmap

    def clear(self, placeholder: str = "No image") -> None:
        self._pixmap = None
        self._source_path = None
        self._label.setPixmap(QPixmap())
        self._label.setText(placeholder)
        self._label.setCursor(Qt.ArrowCursor)
        self._label.setToolTip("")

    def _show_scaled(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self._max_size,
            self._max_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
        self._label.setText("")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._show_scaled()

    def mouseDoubleClickEvent(self, event) -> None:
        if self._pixmap is not None:
            self._open_zoom()
        super().mouseDoubleClickEvent(event)

    def _open_zoom(self) -> None:
        from gui.zoom_dialog import ZoomDialog
        title = Path(self._source_path).name if self._source_path else "Image"
        dlg = ZoomDialog(self._pixmap, title=title, source_path=self._source_path, parent=self)
        dlg.exec()
