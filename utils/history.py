"""
HistoryManager: persist pipeline run results to disk and load them back.

Disk layout:
    <frontend_dir>/history/
        index.json          – ordered list of run metadata (newest first)
        runs/
            <8-char hex id>/
                original_image.png
                clean_image.png
                noisy_image.png
                restored_5x5.png    (if 5×5 was run)
                restored_3x3.png    (if 3×3 was run)

Expected shape of a run_data dict (passed to add_entry):
    {
        "original_image": str,   # path to the source image
        "clean_image":    str,
        "noisy_image":    str,
        "size":           int,
        "snr_target":     float,
        "mode":           str,   # "5x5" | "3x3" | "both"
        "noise_var":      float,
        "results": {
            "5x5": { "image": str, "metrics": {...} },
            "3x3": { "image": str, "metrics": {...} },
        },
    }
"""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class HistoryManager:
    """
    Manages a persistent list of pipeline runs stored on disk.
    All write methods must be called from the main GUI thread to avoid
    concurrent JSON writes.
    """

    def __init__(self, base_dir: Path) -> None:
        self._runs_dir   = base_dir / "history" / "runs"
        self._index_path = base_dir / "history" / "index.json"
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._entries: List[Dict[str, Any]] = self._load_index()

    # ── public API ────────────────────────────────────────────────────────────

    def add_entry(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Copy relevant images into a new run folder, record metadata,
        persist the index, and return the stored entry dict.
        """
        run_id  = uuid.uuid4().hex[:8]
        run_dir = self._runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        entry: Dict[str, Any] = {
            "id":         run_id,
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "size":       run_data.get("size"),
            "snr_target": run_data.get("snr_target"),
            "mode":       run_data.get("mode"),
            "noise_var":  run_data.get("noise_var"),
            "images":     {},
            "results":    {},
        }

        # Copy source images (original upload, clean, noisy)
        for key in ("original_image", "clean_image", "noisy_image"):
            src = Path(run_data.get(key, ""))
            if src.is_file():
                dst = run_dir / f"{key}.png"
                shutil.copy2(src, dst)
                entry["images"][key] = str(dst)

        # Copy result images and store metrics per kernel
        for kernel in ("5x5", "3x3"):
            res = run_data.get("results", {}).get(kernel)
            if not res:
                continue
            img_src = Path(res.get("image", ""))
            stored_path: str | None = None
            if img_src.is_file():
                dst = run_dir / f"restored_{kernel}.png"
                shutil.copy2(img_src, dst)
                stored_path = str(dst)
            entry["results"][kernel] = {
                "image":   stored_path,
                "metrics": res.get("metrics", {}),
            }

        self._entries.insert(0, entry)   # newest first
        self._save_index()
        return entry

    def all_entries(self) -> List[Dict[str, Any]]:
        """Return all stored entries (newest first)."""
        return list(self._entries)

    def clear(self) -> None:
        """Delete all stored metadata (does NOT remove files from disk)."""
        self._entries = []
        self._save_index()

    # ── internals ─────────────────────────────────────────────────────────────

    def _load_index(self) -> List[Dict[str, Any]]:
        if self._index_path.exists():
            try:
                data = json.loads(self._index_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except Exception:
                pass
        return []

    def _save_index(self) -> None:
        self._index_path.write_text(
            json.dumps(self._entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
