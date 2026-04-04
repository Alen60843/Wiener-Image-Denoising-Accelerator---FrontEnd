"""
Pipeline helper functions.
  subprocess_run()  – stream a subprocess line-by-line to a callback
  parse_metrics()   – extract SNR / PSNR / SSIM from reconstruct_image.py output
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Callable


def subprocess_run(
    cmd: list,
    cwd: Path,
    line_callback: Callable[[str], None],
    env: dict | None = None,
) -> tuple[int, str]:
    """
    Run *cmd* as a subprocess, stream each stdout line to *line_callback*,
    accumulate all output, and return (returncode, full_stdout_string).
    stderr is merged into stdout.
    """
    if env is None:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    lines: list[str] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        lines.append(line)
        line_callback(line)

    proc.wait()
    return proc.returncode, "\n".join(lines)


def parse_metrics(output: str) -> dict:
    """
    Parse all metric lines from reconstruct_image.py stdout.

    reconstruct_image.py prints (subset may be present):
        SNR (original vs noisy)    = 14.321 dB
        SNR (original vs restored) = 25.123 dB
        PSNR (original vs noisy)   = 28.1234 dB
        PSNR (original vs restored)= 33.4567 dB
        SSIM (original vs restored)= 0.8734

    Returns dict with keys (only those present in *output*):
        snr_noisy   – SNR original vs noisy
        psnr_noisy  – PSNR original vs noisy
        snr         – SNR original vs restored
        psnr        – PSNR original vs restored
        ssim        – SSIM original vs restored
    """
    metrics: dict = {}

    _patterns = [
        ("snr_noisy",  r"SNR \(original vs noisy\)\s*=\s*([\d.]+)"),
        ("snr",        r"SNR \(original vs restored\)\s*=\s*([\d.]+)"),
        ("psnr_noisy", r"PSNR \(original vs noisy\)\s*=\s*([\d.]+)"),
        ("psnr",       r"PSNR \(original vs restored\)\s*=\s*([\d.]+)"),
        ("ssim",       r"SSIM \(original vs restored\)\s*=\s*([\d.]+)"),
    ]
    for key, pat in _patterns:
        m = re.search(pat, output)
        if m:
            metrics[key] = float(m.group(1))

    return metrics
