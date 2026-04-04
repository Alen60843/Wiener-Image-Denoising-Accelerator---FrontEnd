import sys
from pathlib import Path

# Ensure frontEnd/ is on sys.path so that `gui` and `pipeline` packages resolve
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow

_STYLESHEET = """
/* ── Base ─────────────────────────────────────────────────────────────── */
QWidget {
    background-color: #ffffff;
    color: #111111;
    font-family: "SF Pro Text", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
}

/* ── Main window ─────────────────────────────────────────────────────── */
QMainWindow {
    background-color: #ffffff;
}

/* ── Tabs ────────────────────────────────────────────────────────────── */
QTabWidget::pane {
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    background: #ffffff;
}
QTabBar::tab {
    background: #f3f3f3;
    color: #555555;
    border: 1px solid #d0d0d0;
    border-bottom: none;
    padding: 6px 18px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #111111;
    border-bottom: 2px solid #111111;
}
QTabBar::tab:hover:!selected {
    background: #eaeaea;
    color: #111111;
}

/* ── Group boxes ─────────────────────────────────────────────────────── */
QGroupBox {
    border: 1px solid #d6d6d6;
    border-radius: 6px;
    background: #ffffff;
    margin-top: 18px;
    padding-top: 6px;
    font-weight: 600;
    color: #1a1a1a;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    left: 10px;
    color: #000000;
}

/* ── Push buttons ────────────────────────────────────────────────────── */
QPushButton {
    background-color: #111111;
    color: #ffffff;
    border: 1px solid #111111;
    border-radius: 5px;
    padding: 5px 12px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #2a2a2a;
    border-color: #2a2a2a;
}
QPushButton:pressed {
    background-color: #000000;
}
QPushButton:disabled {
    background-color: #d0d0d0;
    color: #7a7a7a;
    border-color: #d0d0d0;
}

/* ── Line edits / Spin boxes ─────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #ffffff;
    color: #111111;
    border: 1px solid #c8c8c8;
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: #111111;
    selection-color: #ffffff;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #111111;
}
QSpinBox::up-button, QSpinBox::down-button {
    background: #f0f0f0;
    border: none;
    width: 16px;
}
QSpinBox::up-arrow  { image: none; }
QSpinBox::down-arrow{ image: none; }

/* ── Combo boxes ─────────────────────────────────────────────────────── */
QComboBox {
    background-color: #ffffff;
    color: #111111;
    border: 1px solid #c8c8c8;
    border-radius: 4px;
    padding: 4px 8px;
    min-width: 60px;
}
QComboBox:focus {
    border-color: #111111;
}
QComboBox::drop-down {
    border: none;
    width: 22px;
}
QComboBox QAbstractItemView {
    background: #ffffff;
    color: #111111;
    border: 1px solid #c8c8c8;
    selection-background-color: #111111;
    selection-color: #ffffff;
    outline: none;
}

/* ── Radio buttons ───────────────────────────────────────────────────── */
QRadioButton {
    color: #1a1a1a;
    spacing: 6px;
}
QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border-radius: 7px;
    border: 2px solid #666666;
    background: #ffffff;
}
QRadioButton::indicator:checked {
    background: #111111;
    border-color: #111111;
}
QRadioButton::indicator:hover {
    border-color: #111111;
}

/* ── Progress bar ─────────────────────────────────────────────────────── */
QProgressBar {
    background: #f3f3f3;
    border: 1px solid #d0d0d0;
    border-radius: 5px;
    text-align: center;
    color: #000000;
    height: 16px;
    font-weight: 600;
}
QProgressBar::chunk {
    background: #90ee90;
    border-radius: 4px;
}

/* ── Text edit (log) ─────────────────────────────────────────────────── */
QTextEdit {
    background: #ffffff;
    color: #111111;
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    font-family: "Menlo", "Consolas", "Courier New", monospace;
    font-size: 12px;
    selection-background-color: #111111;
    selection-color: #ffffff;
}

/* ── Scroll bars ─────────────────────────────────────────────────────── */
QScrollBar:vertical {
    background: #f0f0f0;
    width: 10px;
    margin: 0;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #a5a5a5;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #7a7a7a;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }

QScrollBar:horizontal {
    background: #f0f0f0;
    height: 10px;
    margin: 0;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background: #a5a5a5;
    min-width: 20px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal:hover {
    background: #7a7a7a;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }

/* ── Splitter handle ─────────────────────────────────────────────────── */
QSplitter::handle {
    background: #d6d6d6;
}
QSplitter::handle:horizontal { width: 3px; }
QSplitter::handle:vertical   { height: 3px; }

/* ── Labels ──────────────────────────────────────────────────────────── */
QLabel {
    background: transparent;
    color: #111111;
}

/* ── Dialogs ─────────────────────────────────────────────────────────── */
QDialog {
    background: #ffffff;
}

/* ── Tool tips ───────────────────────────────────────────────────────── */
QToolTip {
    background: #111111;
    color: #ffffff;
    border: 1px solid #111111;
    padding: 4px;
    border-radius: 3px;
}

/* ── Message boxes ───────────────────────────────────────────────────── */
QMessageBox {
    background: #ffffff;
}
"""


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Wiener Pipeline")
    app.setStyleSheet(_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
