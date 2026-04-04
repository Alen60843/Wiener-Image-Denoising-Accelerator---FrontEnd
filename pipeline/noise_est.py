"""
Standalone script: estimate noise variance using Wiener_filter.estimate_noise_var_immerkaer.
Called as a subprocess by the GUI worker so that numpy/scipy/OpenBLAS run in a
fresh process with a full stack — not inside a QThread's restricted stack.

Usage:
    python noise_est.py <frontend_dir> <noisy_image_path>

Prints a single float (the estimated noise variance) to stdout.
"""
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

frontend_dir = Path(sys.argv[1])
noisy_image_path = Path(sys.argv[2])

# Stub GUI-only / unneeded modules so Wiener_filter.py loads without a display
for _mod in ("matplotlib", "matplotlib.pyplot",
             "skimage", "skimage.io", "skimage.color", "skimage.metrics"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)
# Wire parent → child so `from skimage import io, color` and
# `from skimage.metrics import structural_similarity` work on the stubs
_sk = sys.modules["skimage"]
_sk.io = sys.modules["skimage.io"]
_sk.color = sys.modules["skimage.color"]
_sk.metrics = sys.modules["skimage.metrics"]
sys.modules["skimage.metrics"].structural_similarity = lambda *a, **kw: 0.0

sys.path.insert(0, str(frontend_dir))
import importlib.util as ilu

spec = ilu.spec_from_file_location("Wiener_filter", frontend_dir / "Wiener_filter.py")
wf = ilu.module_from_spec(spec)
spec.loader.exec_module(wf)

img = np.array(Image.open(str(noisy_image_path)).convert("L"), dtype=np.float64)
noise_var = float(wf.estimate_noise_var_immerkaer(img))
print(noise_var)
