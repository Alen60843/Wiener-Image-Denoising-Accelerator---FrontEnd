"""
MainWindow: top-level application window.

Responsibilities:
  - Own the QThread + PipelineWorker (single thread, lives for app lifetime)
  - Own the HistoryManager
  - Host the QTabWidget with three tabs:
      0 – Pipeline   (ProcessingTab)
      1 – History    (HistoryTab)
      2 – Generate   (GeneratorTab)
  - Wire all signals between worker, tabs, and history manager
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Slot
from PySide6.QtWidgets import QMainWindow, QTabWidget, QWidget

from gui.generator_tab  import GeneratorTab
from gui.history_tab    import HistoryTab
from gui.processing_tab import ProcessingTab
from gui.worker         import PipelineWorker
from pipeline.constants import FRONTEND_DIR
from utils.history      import HistoryManager


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Wiener Denoising Pipeline")
        self.resize(1200, 860)

        # History manager (reads index.json on startup)
        self._history = HistoryManager(FRONTEND_DIR)

        self._setup_worker()
        self._build_tabs()
        self._connect_signals()

        # Populate history tab with persisted entries
        self._history_tab.load_entries(self._history.all_entries())

    # ── worker setup ─────────────────────────────────────────────────────────

    def _setup_worker(self) -> None:
        """Create worker and dedicate a long-lived thread to it."""
        self._thread = QThread(self)
        self._worker = PipelineWorker()
        self._worker.moveToThread(self._thread)
        self._thread.start()

    # ── UI structure ──────────────────────────────────────────────────────────

    def _build_tabs(self) -> None:
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)   # cleaner look on macOS
        self.setCentralWidget(self._tabs)

        self._proc_tab  = ProcessingTab()
        self._history_tab = HistoryTab()
        self._gen_tab   = GeneratorTab()

        self._tabs.addTab(self._proc_tab,    "⚙  Pipeline")
        self._tabs.addTab(self._history_tab, "🕒  History")
        self._tabs.addTab(self._gen_tab,     "🎲  Generate")

    # ── signal wiring ─────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        # ProcessingTab → Worker (queued cross-thread call)
        self._proc_tab.run_requested.connect(self._worker.run)

        # Worker → ProcessingTab (live updates)
        self._worker.log_message.connect(self._proc_tab.on_log)
        self._worker.progress.connect(self._proc_tab.on_progress)
        self._worker.step_done.connect(self._proc_tab.on_step_done)
        self._worker.error.connect(self._proc_tab.on_error)
        self._worker.done.connect(self._proc_tab.on_done)

        # Worker run_complete → save history + update history tab
        self._worker.run_complete.connect(self._on_run_complete)

        # GeneratorTab image_ready → load into ProcessingTab + switch tab
        self._gen_tab.image_ready.connect(self._on_generator_image_ready)

    # ── slots ─────────────────────────────────────────────────────────────────

    @Slot(dict)
    def _on_run_complete(self, run_data: dict) -> None:
        """Called (in main thread via queued connection) after a successful run."""
        entry = self._history.add_entry(run_data)
        self._history_tab.add_entry(entry)

    @Slot(str)
    def _on_generator_image_ready(self, path: str) -> None:
        """Load generated image into ProcessingTab and switch to it."""
        self._proc_tab.set_input_image(path)
        self._tabs.setCurrentIndex(0)

    # ── cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._thread.quit()
        self._thread.wait()
        super().closeEvent(event)
