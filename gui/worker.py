"""
PipelineWorker: runs the full Wiener denoising pipeline in a QThread.

Speed optimisations applied:
  1. img_to_hex.py is called *inline* (functions imported directly) to avoid
     Python subprocess startup + import overhead (~1 s saved).
  2. Noise estimation and iverilog compile run in PARALLEL (ThreadPoolExecutor)
     because they are completely independent after step 1.
  3. 5×5 and 3×3 reconstruction also run in PARALLEL for mode="both".

Signals emitted during a run:
    log_message(str)          – one line of live console output
    progress(int)             – 0-100 progress value
    step_done(str, dict)      – step_id + payload (see each step below)
    error(str, str)           – step_id + human-readable message
    run_complete(dict)        – full run data for history (on success)
    done()                    – fired last on success

step_done payloads:
    "hex"             → { clean_image_path, noisy_image_path }
    "noise"           → { noise_var }
    "compile"         → { status: "ok" }
    "sim"             → { status: "ok" }
    "reconstruct_5x5" → { image_path, metrics }
    "reconstruct_3x3" → { image_path, metrics }
"""
from __future__ import annotations

import importlib.util as _ilu
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from pipeline.constants import FRONTEND_DIR, RTL_HARDCODED_SIZE
from pipeline.steps import parse_metrics, subprocess_run


def _trickle(
    start: int,
    end: int,
    emit_fn,
    stop: threading.Event,
    interval: float = 0.4,
    expected_secs: float = 20.0,
) -> None:
    """
    Advance linearly from start toward (end-1) at a steady rate, never reaching end.

    Rate is computed so that 90 % of the range is filled in `expected_secs`.
    A hard floor guarantees at least one integer step every ~2 s regardless of
    range size, so the bar always visibly moves.
    """
    ceiling   = end - 1.0
    current   = float(start)
    ticks_90  = expected_secs / interval          # ticks to fill 90 % of range
    step      = (end - start) * 0.90 / ticks_90  # units per tick
    step      = max(step, interval / 2.0)         # floor: 1 unit every 2 s at most
    last_int  = start

    while not stop.is_set():
        time.sleep(interval)
        if stop.is_set():
            break
        current = min(current + step, ceiling)
        new_int = int(current)
        if new_int > last_int:
            emit_fn(new_int)
            last_int = new_int


# ── load img_to_hex functions once at import time (no subprocess overhead) ───

def _load_img_to_hex_module():
    """Import img_to_hex.py without executing its __main__ block."""
    spec = _ilu.spec_from_file_location(
        "img_to_hex", FRONTEND_DIR / "img_to_hex.py"
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_img_to_hex = _load_img_to_hex_module()


class PipelineWorker(QObject):
    log_message  = Signal(str)
    progress     = Signal(int)
    step_done    = Signal(str, dict)   # step_id, payload
    error        = Signal(str, str)    # step_id, message
    run_complete = Signal(dict)        # full run data → history manager
    done         = Signal()            # fired last

    # ── entry point ───────────────────────────────────────────────────────────

    @Slot(dict)
    def run(self, config: dict) -> None:
        image_path: str = config["image_path"]
        size: int       = config["size"]
        snr: float      = config["snr"]
        mode: str       = config["mode"]     # "5x5" | "3x3" | "both"

        cwd = FRONTEND_DIR

        # Accumulate results for history
        run_data: dict = {
            "original_image": image_path,
            "size":           size,
            "snr_target":     snr,
            "mode":           mode,
            "results":        {},
        }

        # ── Step 1: img_to_hex (inline — no subprocess overhead) ─────────────
        self.progress.emit(0)
        self.log_message.emit("=== Step 1: img_to_hex (inline) ===")

        if size != RTL_HARDCODED_SIZE:
            self.log_message.emit(
                f"[WARNING] Image size {size} != {RTL_HARDCODED_SIZE} — "
                "RTL testbench is hardcoded for 512×512"
            )

        try:
            from PIL import Image as _PIL

            clean = _img_to_hex.load_image(image_path, size, size)
            _img_to_hex.save_image_to_hex(clean, str(cwd / "clean_image.hex"))
            _PIL.fromarray(clean).save(str(cwd / "clean_image.png"))

            noisy, sigma = _img_to_hex.add_gaussian_noise(clean, snr)
            _img_to_hex.save_image_to_hex(noisy, str(cwd / "noisy_image.hex"))
            _PIL.fromarray(noisy).save(str(cwd / "noisy_image.png"))

            measured_snr = _img_to_hex.compute_snr_db(clean, noisy)
            self.log_message.emit(f"Image size: {size} x {size}")
            self.log_message.emit(f"Target SNR: {snr} dB")
            self.log_message.emit(f"Noise sigma: {sigma:.3f}")
            self.log_message.emit(f"Measured SNR: {measured_snr:.3f} dB")
        except Exception as exc:
            self.error.emit("hex", str(exc))
            return

        run_data["clean_image"] = str(cwd / "clean_image.png")
        run_data["noisy_image"] = str(cwd / "noisy_image.png")
        self.progress.emit(15)
        self.step_done.emit("hex", {
            "clean_image_path": run_data["clean_image"],
            "noisy_image_path": run_data["noisy_image"],
        })

        # ── Steps 2 + 3 in PARALLEL: noise estimation AND iverilog compile ───
        #    These are fully independent: noise_est reads noisy_image.png,
        #    iverilog compiles the static RTL files.  Running them together
        #    saves the compile time (typically 1-5 s).
        self.log_message.emit("=== Steps 2+3: noise estimation || iverilog compile ===")

        noise_cmd   = [
            sys.executable,
            str(cwd / "pipeline" / "noise_est.py"),
            str(cwd),
            str(cwd / "noisy_image.png"),
        ]
        compile_cmd = ["iverilog", "-g", "2012", "-o", "sim.out", "-f", "rtl.f"]

        # Thread-safe log: prefix each line so interleaved output is readable
        log_lock = threading.Lock()

        def tagged_log(tag: str):
            def _cb(line: str) -> None:
                with log_lock:
                    self.log_message.emit(f"[{tag}] {line}")
            return _cb

        noise_result  = {}   # populated by future
        compile_result = {}

        def _run_noise():
            rc, out = subprocess_run(noise_cmd, cwd, tagged_log("noise"))
            noise_result["rc"]     = rc
            noise_result["stdout"] = out

        def _run_compile():
            rc, out = subprocess_run(compile_cmd, cwd, tagged_log("compile"))
            compile_result["rc"] = rc

        _stop_trickle = threading.Event()
        _t = threading.Thread(
            target=_trickle, args=(15, 50, self.progress.emit, _stop_trickle, 0.4, 15.0),
            daemon=True,
        )
        _t.start()
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_noise   = pool.submit(_run_noise)
            f_compile = pool.submit(_run_compile)
            f_noise.result()
            f_compile.result()
        _stop_trickle.set()
        _t.join()

        # Check compile result
        if compile_result.get("rc", 1) != 0:
            self.error.emit("compile", "iverilog exited with non-zero status")
            return
        self.step_done.emit("compile", {"status": "ok"})

        # Check noise estimation result
        if noise_result.get("rc", 1) != 0:
            self.error.emit("noise", "noise_est.py exited with non-zero status")
            return
        try:
            stdout_lines = noise_result["stdout"].strip().splitlines()
            noise_var = float(stdout_lines[-1])
        except (ValueError, IndexError) as exc:
            self.error.emit("noise", f"Could not parse noise variance: {exc}")
            return

        run_data["noise_var"] = noise_var
        self.log_message.emit(f"Estimated noise variance: {noise_var:.4f}")
        self.progress.emit(50)
        self.step_done.emit("noise", {"noise_var": noise_var})

        # ── Step 4: vvp simulation ───────────────────────────────────────────
        self.log_message.emit("=== Step 4: vvp simulation ===")
        cmd = ["vvp", "sim.out", f"+NOISE_VAR={noise_var}"]

        # Drive progress 50→70 by counting vvp output lines.
        # The testbench writes one output pixel per line → size×size total lines.
        # We map line_count linearly to the 50-69 window (70 is emitted on completion).
        _sim_count   = [0]
        _sim_expected = size * size   # total expected output lines
        _sim_last    = [50]

        def _sim_log(line: str) -> None:
            self.log_message.emit(line)
            _sim_count[0] += 1
            pct = min(_sim_count[0] / _sim_expected, 0.99)
            val = 50 + int(pct * 19)   # 50 → 69 linearly
            if val > _sim_last[0]:
                _sim_last[0] = val
                self.progress.emit(val)

        try:
            rc, _ = subprocess_run(cmd, cwd, _sim_log)
        except Exception as exc:
            self.error.emit("sim", str(exc))
            return
        if rc != 0:
            self.error.emit("sim", f"vvp exited with code {rc}")
            return

        self.progress.emit(70)
        self.step_done.emit("sim", {"status": "ok"})

        # ── Step 5: reconstruct (5×5 and/or 3×3 in PARALLEL for mode=both) ──
        kernels: list[str] = []
        if mode in ("5x5", "both"):
            kernels.append("5x5")
        if mode in ("3x3", "both"):
            kernels.append("3x3")

        self.log_message.emit(
            f"=== Step 5: reconstruct {' || '.join(kernels)} ==="
        )

        # Check output files exist before trying to reconstruct
        for k in kernels:
            pf = cwd / f"output_pixels_{k}.txt"
            if not pf.is_file():
                self.error.emit(f"reconstruct_{k}", f"Expected file not found: {pf.name}")
                return

        # Build a reconstruct job for each kernel
        def _run_reconstruct(kernel: str) -> tuple[str, int, str]:
            pixel_file = f"output_pixels_{kernel}.txt"
            out_img    = f"restored_{kernel}.png"
            cmd = [
                sys.executable, "reconstruct_image.py",
                "--file",     pixel_file,
                "--out_img",  out_img,
                "--size",     str(size),
                "--clean",    "clean_image.hex",
                "--noisy",    "noisy_image.hex",
                "--restored", pixel_file,
            ]
            rc, stdout = subprocess_run(cmd, cwd, tagged_log(kernel))
            return kernel, rc, stdout

        _stop_rec = threading.Event()
        _t2 = threading.Thread(
            target=_trickle, args=(70, 100, self.progress.emit, _stop_rec, 0.4, 12.0),
            daemon=True,
        )
        _t2.start()
        with ThreadPoolExecutor(max_workers=len(kernels)) as pool:
            futures = {pool.submit(_run_reconstruct, k): k for k in kernels}
            for fut in as_completed(futures):
                try:
                    kernel, rc, stdout = fut.result()
                except Exception as exc:
                    _stop_rec.set()
                    self.error.emit("reconstruct", str(exc))
                    return

                if rc != 0:
                    _stop_rec.set()
                    self.error.emit(
                        f"reconstruct_{kernel}",
                        f"reconstruct_image.py exited with code {rc}",
                    )
                    return

                metrics = parse_metrics(stdout)
                run_data["results"][kernel] = {
                    "image":   str(cwd / f"restored_{kernel}.png"),
                    "metrics": metrics,
                }
                # Emit step_done for each kernel as it finishes
                self.step_done.emit(f"reconstruct_{kernel}", {
                    "image_path": str(cwd / f"restored_{kernel}.png"),
                    "metrics":    metrics,
                })

        _stop_rec.set()
        _t2.join()
        self.progress.emit(100)
        self.run_complete.emit(run_data)
        self.done.emit()
