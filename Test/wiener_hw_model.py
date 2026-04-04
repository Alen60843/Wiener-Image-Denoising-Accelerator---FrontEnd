"""
Hardware-faithful Python simulation of wiener_core.sv + three optional enhancements.

─────────────────────────────────────────────────────────────────────────────
BASE PARAMETERS (all exist / map directly to wiener_core.sv / Wiener_filter.py)
─────────────────────────────────────────────────────────────────────────────
  noise_var_scale        : multiply Immerkaer estimate           (default 1.0)
  noise_var_bias         : additive offset on noise_var          (default 0.0)
  immerkaer_sigma_thresh : sigma threshold in Immerkaer          (default 20.0)
  immerkaer_high_scale   : multiplier for sigma > thresh         (default 1.25)
  sigma2_bias            : additive floor on local variance      (default 0.0)
  w_zero_thresh          : R threshold → weight = 0             (default 60293/65536)
  w_min                  : minimum weight floor                  (default 4096/65536)
  w_power                : exponent on raw weight                (default 1.0)
  correction_scale       : scale the correction term             (default 1.0)

─────────────────────────────────────────────────────────────────────────────
ENHANCEMENT 1 — Adaptive 3×3 / 5×5 blend  (blend_thresh, blend_soft)
─────────────────────────────────────────────────────────────────────────────
  The hardware already computes 3×3 OR 5×5 stats (mode_3x3 flag, stats_calc_5x5.sv).
  Idea: run BOTH window sizes in parallel and smoothly blend their outputs
  using the local 5×5 variance as an edge detector.

  alpha = sigmoid((sigma2_5x5 - blend_thresh) / blend_soft)   in [0,1]
  out   = (1-alpha)*restored_5x5 + alpha*restored_3x3
           ^ smooth regions (low var)     ^ edge regions (high var)

  Intuition: near edges local variance is large → prefer 3×3 (less blur).
             in smooth areas local variance is small → prefer 5×5 (more averaging).

  blend_thresh  : variance level where blend crosses 0.5   (default 9999 = disabled)
  blend_soft    : sigmoid steepness; small=sharp, large=gentle  (default 100.0)

─────────────────────────────────────────────────────────────────────────────
ENHANCEMENT 2 — Two-pass residual noise re-estimation  (residual_scale2)
─────────────────────────────────────────────────────────────────────────────
  Run the filter twice. After pass 1, the residual (noisy − restored_1) contains
  mostly noise with little image structure. Running Immerkaer on the residual
  gives a more accurate global noise estimate for pass 2.

  residual_scale2 = 0 → disabled (single-pass)
  residual_scale2 > 0 → enabled; noise_var_2 = immerkaer(residual) * residual_scale2

─────────────────────────────────────────────────────────────────────────────
ENHANCEMENT 3 — Local correction clipping  (local_clip_k)
─────────────────────────────────────────────────────────────────────────────
  After computing the correction term (w*(noisy-mu)), clip it to
  ±local_clip_k * sqrt(noise_var). This stops the filter from applying
  a correction larger than what the noise could plausibly have caused,
  preventing ringing artifacts at sharp edges.

  local_clip_k = 0 → disabled
  local_clip_k > 0 → correction = clip(w*(noisy-mu), ±k*sqrt(noise_var))
"""

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import convolve2d


# ── Immerkaer noise variance estimator ───────────────────────────────────────

def estimate_noise_var_immerkaer(img, sigma_thresh=20.0, high_scale=1.25):
    """Mirrors Wiener_filter.py / Wiener_test.py implementation."""
    img    = img.astype(np.float64)
    smooth = uniform_filter(img, size=5, mode="reflect")
    resid  = img - smooth

    k = np.array([[1, -2, 1],
                  [-2, 4, -2],
                  [1, -2, 1]], dtype=np.float64)
    conv = convolve2d(resid, k, mode="valid", boundary="symm")

    h, w     = conv.shape
    mean_abs = np.sum(np.abs(conv)) / (h * w)
    sigma    = np.sqrt(np.pi / 2.0) * (mean_abs / 6.0)

    return sigma ** 2 if sigma <= sigma_thresh else high_scale * sigma ** 2


def local_mean_var(img, win_size=5):
    img   = img.astype(np.float64)
    mu    = uniform_filter(img,      size=win_size, mode="reflect")
    mu_sq = uniform_filter(img ** 2, size=win_size, mode="reflect")
    return mu, np.maximum(mu_sq - mu ** 2, 0.0)


# ── Internal single-pass core (accepts pre-computed noise_var) ────────────────

def _wiener_core(noisy, noise_var, win_size,
                 sigma2_bias, w_zero_thresh, w_min, w_power,
                 correction_scale, local_clip_k):
    """
    Pure computation core — noise_var already supplied by caller.
    Returns (restored, sigma2_local).
    """
    mu, sigma2 = local_mean_var(noisy, win_size=win_size)

    sigma2_safe = np.maximum(sigma2 + sigma2_bias, 1.0 / 256.0)

    R     = noise_var / sigma2_safe
    w_raw = np.clip(1.0 - R, 0.0, None)
    w_raw = np.where(R >= w_zero_thresh, 0.0, w_raw)
    w_raw = np.where(w_raw > 0.0,
                     np.power(np.maximum(w_raw, 0.0), w_power),
                     0.0)
    w = np.maximum(w_raw, w_min)

    correction = w * (noisy - mu)

    # Enhancement 3: local correction clipping
    if local_clip_k > 0.0:
        max_corr = local_clip_k * np.sqrt(max(noise_var, 0.0))
        correction = np.clip(correction, -max_corr, max_corr)

    restored = mu + correction_scale * correction

    return restored, sigma2


# ── Public API: base single-pass filter ──────────────────────────────────────

def wiener_hw(noisy, win_size=5,
              # ─ noise variance ─
              noise_var_scale=1.0,
              noise_var_bias=0.0,
              # ─ Immerkaer knobs ─
              immerkaer_sigma_thresh=20.0,
              immerkaer_high_scale=1.25,
              # ─ local variance floor ─
              sigma2_bias=0.0,
              # ─ weight shaping ─
              w_zero_thresh=60293 / 65536,
              w_min=4096 / 65536,
              w_power=1.0,
              # ─ output scale ─
              correction_scale=1.0,
              # ─ (internal) override Immerkaer if already computed ─
              _noise_var_override=None):
    """
    Base hardware-faithful Wiener filter (single pass, one window size).
    Core formula unchanged from wiener_core.sv.
    """
    noisy = noisy.astype(np.float64)

    if _noise_var_override is None:
        raw_nv    = estimate_noise_var_immerkaer(noisy,
                                                 sigma_thresh=immerkaer_sigma_thresh,
                                                 high_scale=immerkaer_high_scale)
        noise_var = max(raw_nv * noise_var_scale + noise_var_bias, 0.0)
    else:
        noise_var = _noise_var_override

    restored, _ = _wiener_core(noisy, noise_var, win_size,
                                sigma2_bias, w_zero_thresh, w_min,
                                w_power, correction_scale, 0.0)
    return restored, noise_var


# ── Public API: enhanced filter (all three enhancements) ─────────────────────

def wiener_hw_enhanced(noisy, win_size=5,
                       # ─ base params ─
                       noise_var_scale=1.0,
                       noise_var_bias=0.0,
                       immerkaer_sigma_thresh=20.0,
                       immerkaer_high_scale=1.25,
                       sigma2_bias=0.0,
                       w_zero_thresh=60293 / 65536,
                       w_min=4096 / 65536,
                       w_power=1.0,
                       correction_scale=1.0,
                       # ─ Enhancement 1: adaptive 3x3/5x5 blend ─
                       blend_thresh=9999.0,    # high default = disabled
                       blend_soft=100.0,
                       # ─ Enhancement 2: two-pass residual ─
                       residual_scale2=0.0,    # 0 = disabled
                       # ─ Enhancement 3: local correction clipping ─
                       local_clip_k=0.0):      # 0 = disabled
    """
    Enhanced Wiener filter with up to three optional extras.

    All enhancements default to disabled so this is backward-compatible
    with wiener_hw() when called with default extra params.
    """
    noisy = noisy.astype(np.float64)

    # ── Noise variance estimation (shared across passes) ─────────────────────
    raw_nv    = estimate_noise_var_immerkaer(noisy,
                                             sigma_thresh=immerkaer_sigma_thresh,
                                             high_scale=immerkaer_high_scale)
    noise_var = max(raw_nv * noise_var_scale + noise_var_bias, 0.0)

    core_kwargs = dict(sigma2_bias=sigma2_bias,
                       w_zero_thresh=w_zero_thresh,
                       w_min=w_min,
                       w_power=w_power,
                       correction_scale=correction_scale,
                       local_clip_k=local_clip_k)

    # ── Enhancement 1: Adaptive 3×3/5×5 blend ────────────────────────────────
    blend_active = blend_thresh < 9000.0

    if blend_active:
        restored_5x5, sigma2_5x5 = _wiener_core(noisy, noise_var, 5, **core_kwargs)
        restored_3x3, _          = _wiener_core(noisy, noise_var, 3, **core_kwargs)

        # Smooth sigmoid blend: alpha=0 → 5×5 (smooth), alpha=1 → 3×3 (edge)
        _, sigma2_map = local_mean_var(noisy, win_size=5)
        # clip argument to avoid overflow in exp for extreme blend_thresh values
        z        = np.clip((sigma2_map - blend_thresh) / blend_soft, -500, 500)
        alpha    = 1.0 / (1.0 + np.exp(-z))
        restored = (1.0 - alpha) * restored_5x5 + alpha * restored_3x3
    else:
        restored, sigma2_map = _wiener_core(noisy, noise_var, win_size, **core_kwargs)

    # ── Enhancement 2: Two-pass residual re-estimation ───────────────────────
    if residual_scale2 > 0.0:
        residual  = noisy - np.clip(restored, 0.0, 255.0)
        # Immerkaer on the residual: contains mostly noise → better estimate
        nv_resid  = estimate_noise_var_immerkaer(residual,
                                                  sigma_thresh=immerkaer_sigma_thresh,
                                                  high_scale=immerkaer_high_scale)
        noise_var2 = max(nv_resid * residual_scale2, 0.0)

        if blend_active:
            restored2_5x5, _ = _wiener_core(noisy, noise_var2, 5, **core_kwargs)
            restored2_3x3, _ = _wiener_core(noisy, noise_var2, 3, **core_kwargs)
            restored = (1.0 - alpha) * restored2_5x5 + alpha * restored2_3x3
        else:
            restored, _ = _wiener_core(noisy, noise_var2, win_size, **core_kwargs)

        noise_var = noise_var2   # report the pass-2 noise_var

    return restored, noise_var
