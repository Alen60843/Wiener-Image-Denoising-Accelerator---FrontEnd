"""
Parameter Optimizer for the hardware Wiener filter — extended equations.

Searches 9 parameters using Latin Hypercube Sampling → MLP surrogate →
differential_evolution → Nelder-Mead fine-tune.

Usage
-----
    cd /Users/alenfaer/Technion/PROJECT/scripts/frontEnd/Test
    python optimize_params.py --image-path ../../../images/test_7.jpg --snr-db 14

Optional flags
--------------
    --n-samples N   LHS sample count for coarse search  (default 2000)
    --seeds     N   noise seeds averaged per evaluation (default 3)
    --no-show       suppress plots
"""

import argparse
import sys
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.metrics import structural_similarity as ssim_fn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc          # Latin Hypercube Sampler

sys.path.insert(0, os.path.dirname(__file__))
from wiener_hw_model import wiener_hw, wiener_hw_enhanced


# ── Metric helpers ────────────────────────────────────────────────────────────

def snr_db(ref, test):
    ref  = ref.astype(np.float64)
    test = test.astype(np.float64)
    num  = np.sum(ref ** 2)
    den  = np.sum((ref - test) ** 2)
    return 10.0 * np.log10(num / den) if den > 0 else np.inf


def psnr_db(ref, test):
    ref  = ref.astype(np.float64)
    test = test.astype(np.float64)
    I_max   = np.max(test) ** 2
    err_sum = np.sum((ref - test) ** 2)
    return 10.0 * np.log10((I_max ** 2) / err_sum) if err_sum > 0 else np.inf


def ssim_score(ref, restored):
    return ssim_fn(ref, restored, data_range=255.0)


# ── Image loading ─────────────────────────────────────────────────────────────

def load_gray(path):
    img = io.imread(path)
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.ndim == 3:
        if img.shape[0] == 1:          # (1, H, W) channel-first
            return img[0].astype(np.uint8)
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        return (color.rgb2gray(img) * 255).astype(np.uint8)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def add_gaussian_noise(original, snr_db_target, seed=0):
    rng        = np.random.default_rng(seed)
    original   = original.astype(np.float64)
    sig_pow    = np.mean(original ** 2)
    nv         = sig_pow / (10 ** (snr_db_target / 10.0))
    noisy      = np.clip(original + rng.normal(0, np.sqrt(nv), original.shape), 0, 255)
    return noisy, nv


# ── Parameter space (9 dimensions) ───────────────────────────────────────────
#
# Order must match the keyword arguments of wiener_hw().
#
PARAM_NAMES = [
    # ── original 9 ──────────────────────────────────────────────────────────
    "noise_var_scale",        # multiplier on Immerkaer estimate
    "noise_var_bias",         # additive offset on noise_var (before hardware)
    "immerkaer_sigma_thresh", # sigma threshold inside Immerkaer estimator
    "immerkaer_high_scale",   # scale factor when sigma > threshold
    "sigma2_bias",            # additive bias on local variance (lifts denominator)
    "w_zero_thresh",          # R threshold above which weight → 0
    "w_min",                  # minimum weight floor
    "w_power",                # exponent on raw weight  (1.0 = linear)
    "correction_scale",       # scale on the Wiener correction term
    # ── Enhancement 1: adaptive 3×3/5×5 blend ───────────────────────────────
    "blend_thresh",           # local-variance threshold for 3×3/5×5 blend
                              #   very large (≥9000) → blend disabled, pure 5×5
    "blend_soft",             # sigmoid transition width (larger = gentler blend)
    # ── Enhancement 2: two-pass residual re-estimation ──────────────────────
    "residual_scale2",        # scale on residual Immerkaer for pass 2
                              #   0 → single pass (disabled)
    # ── Enhancement 3: local correction clipping ────────────────────────────
    "local_clip_k",           # clip |correction| to k*sqrt(noise_var)
                              #   0 → disabled
]

PARAM_BOUNDS = [
    # ── original 9 ──
    (0.20,  4.00),    # noise_var_scale
    (-200., 400.),    # noise_var_bias
    (5.0,   50.0),    # immerkaer_sigma_thresh
    (0.80,  2.50),    # immerkaer_high_scale
    (0.0,   50.0),    # sigma2_bias
    (0.70,  1.50),    # w_zero_thresh
    (0.00,  0.20),    # w_min
    (0.50,  3.00),    # w_power
    (0.70,  1.50),    # correction_scale
    # ── enhancements ──
    (10.,  9999.),    # blend_thresh  (9999 = disabled; small = active blending)
    (1.0,  500.),     # blend_soft
    (0.0,   3.0),     # residual_scale2  (0 = disabled)
    (0.0,   5.0),     # local_clip_k     (0 = disabled)
]

# Default values — enhancements all disabled
DEFAULTS = [
    1.0,            # noise_var_scale
    0.0,            # noise_var_bias
    20.0,           # immerkaer_sigma_thresh
    1.25,           # immerkaer_high_scale
    0.0,            # sigma2_bias
    60293 / 65536,  # w_zero_thresh
    4096  / 65536,  # w_min
    1.0,            # w_power
    1.0,            # correction_scale
    9999.0,         # blend_thresh  (disabled)
    100.0,          # blend_soft
    0.0,            # residual_scale2  (disabled)
    0.0,            # local_clip_k     (disabled)
]


# ── Objective ─────────────────────────────────────────────────────────────────

def evaluate(p, original, snr_db_target, seeds=(0, 1, 2), win_size=5):
    """Average SNR (original vs restored) over several noise seeds."""
    kwargs = dict(zip(PARAM_NAMES, p))
    scores = []
    for seed in seeds:
        noisy, _ = add_gaussian_noise(original, snr_db_target, seed=seed)
        try:
            restored, _ = wiener_hw_enhanced(noisy, win_size=win_size, **kwargs)
            restored = np.clip(restored, 0, 255)
            scores.append(snr_db(original, restored))
        except Exception:
            scores.append(-np.inf)
    val = np.mean(scores)
    return float(val) if np.isfinite(val) else -999.0


# ── Phase 1: Latin Hypercube Sampling ────────────────────────────────────────

def lhs_search(original, snr_db_target, n_samples, seeds):
    print(f"\n[1/4] Latin Hypercube Sampling: {n_samples} points × {len(seeds)} seeds")
    print("      Explores the full 9-D parameter space efficiently.\n")

    lo = np.array([b[0] for b in PARAM_BOUNDS])
    hi = np.array([b[1] for b in PARAM_BOUNDS])

    sampler = qmc.LatinHypercube(d=len(PARAM_BOUNDS), seed=42)
    unit    = sampler.random(n=n_samples)
    X_raw   = qmc.scale(unit, lo, hi)

    # Always include the default point so we have a known baseline
    X_raw = np.vstack([np.array(DEFAULTS), X_raw])

    X, y = [], []
    t0   = time.time()
    for i, p in enumerate(X_raw):
        val = evaluate(p.tolist(), original, snr_db_target, seeds=seeds)
        X.append(p.tolist())
        y.append(val)
        if (i + 1) % max(1, len(X_raw) // 20) == 0 or (i + 1) == len(X_raw):
            elapsed = time.time() - t0
            pct     = 100 * (i + 1) / len(X_raw)
            eta     = elapsed / (i + 1) * (len(X_raw) - i - 1)
            print(f"  {i+1:>5}/{len(X_raw)}  ({pct:5.1f}%)  "
                  f"best so far: {max(y):.4f} dB  ETA: {eta:.0f}s")

    X = np.array(X)
    y = np.array(y)
    best_i = np.argmax(y)
    print(f"\n  LHS best  SNR: {y[best_i]:.4f} dB")
    print(f"  At params: {dict(zip(PARAM_NAMES, X[best_i]))}")
    return X, y


# ── Phase 2: Neural network surrogate ────────────────────────────────────────

def train_surrogate(X, y):
    print("\n[2/4] Training MLP surrogate …")
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    mlp    = MLPRegressor(
        hidden_layer_sizes=(256, 256, 128, 64),
        activation="relu",
        max_iter=3000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        learning_rate_init=1e-3,
    )
    mlp.fit(Xs, y)
    y_pred = mlp.predict(Xs)
    print(f"  Surrogate R²        = {mlp.score(Xs, y):.4f}")
    print(f"  Mean absolute error = {np.mean(np.abs(y - y_pred)):.4f} dB")
    return mlp, scaler


# ── Phase 3: Global search on surrogate ──────────────────────────────────────

def optimise_surrogate(mlp, scaler, X_lhs, y_lhs):
    print("\n[3/4] Global optimisation on surrogate (differential_evolution) …")

    def neg_pred(p):
        return -mlp.predict(scaler.transform([p]))[0]

    # Warm-start DE with the top-5 LHS points
    top5 = X_lhs[np.argsort(y_lhs)[-5:]]

    result = differential_evolution(
        neg_pred, PARAM_BOUNDS,
        seed=0, tol=1e-5,
        maxiter=1000, popsize=20,
        init=np.vstack([top5,
                        qmc.scale(qmc.LatinHypercube(d=len(PARAM_BOUNDS),
                                                     seed=1).random(n=95),
                                  [b[0] for b in PARAM_BOUNDS],
                                  [b[1] for b in PARAM_BOUNDS])]),
        workers=1,
    )
    print(f"  Surrogate best predicted SNR : {-result.fun:.4f} dB")
    print(f"  At params: {dict(zip(PARAM_NAMES, result.x))}")
    return result.x


# ── Phase 4: Fine-tune on real objective ─────────────────────────────────────

def fine_tune(p0, original, snr_db_target, seeds):
    print("\n[4/4] Fine-tuning on real objective (Nelder-Mead) …")
    lo = [b[0] for b in PARAM_BOUNDS]
    hi = [b[1] for b in PARAM_BOUNDS]

    def neg_obj(p):
        if any(not (lo[i] <= p[i] <= hi[i]) for i in range(len(p))):
            return 1e6
        return -evaluate(p, original, snr_db_target, seeds=seeds)

    result = minimize(neg_obj, p0, method="Nelder-Mead",
                      options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-5})
    return result.x, -result.fun


# ── Report helpers ────────────────────────────────────────────────────────────

def print_equations(p):
    """
    Print the optimised filter as a set of human-readable equations
    so each result shows the actual expression (e.g. w = 1.23 * w_raw).
    """
    kw            = dict(zip(PARAM_NAMES, p))
    nv_scale      = kw["noise_var_scale"]
    nv_bias       = kw["noise_var_bias"]
    s2_bias       = kw["sigma2_bias"]
    w_zero        = kw["w_zero_thresh"]
    w_min         = kw["w_min"]
    w_pwr         = kw["w_power"]
    c_scale       = kw["correction_scale"]
    imm_thresh    = kw["immerkaer_sigma_thresh"]
    imm_scale     = kw["immerkaer_high_scale"]
    W_MIN_Q1_16   = round(w_min  * 65536)
    W_ZERO_Q16_16 = round(w_zero * 65536)

    print("  ┌─ OPTIMISED EQUATIONS ──────────────────────────────────────────")
    print("  │")
    print("  │  [Immerkaer estimator]")
    if imm_scale == 1.0:
        print(f"  │  noise_var_raw = sigma²     (for all sigma, scale=1.0)")
    else:
        print(f"  │  noise_var_raw = sigma²               if sigma ≤ {imm_thresh:.2f}")
        print(f"  │               = {imm_scale:.4f} × sigma²     if sigma  > {imm_thresh:.2f}")
    print("  │")
    print("  │  [Noise variance loaded into hardware register]")
    if abs(nv_bias) < 1e-4:
        print(f"  │  noise_var = {nv_scale:.4f} × noise_var_raw")
    elif nv_bias >= 0:
        print(f"  │  noise_var = {nv_scale:.4f} × noise_var_raw  +  {nv_bias:.2f}")
    else:
        print(f"  │  noise_var = {nv_scale:.4f} × noise_var_raw  -  {abs(nv_bias):.2f}")
    print("  │")
    print("  │  [Local variance floor  (sigma2_safe)]")
    if abs(s2_bias) < 1e-4:
        print(f"  │  sigma2_safe = max(sigma2,  1/256)")
    else:
        print(f"  │  sigma2_safe = max(sigma2  +  {s2_bias:.2f},  1/256)")
    print("  │")
    print("  │  [Ratio]")
    print(f"  │  R = noise_var / sigma2_safe")
    print("  │")
    print("  │  [Weight  (mirrors w_fixed / w_clamped in wiener_core.sv)]")
    print(f"  │  w_raw = 0                    if R ≥ {w_zero:.5f}  "
          f"(W_ZERO_Q16_16 = {W_ZERO_Q16_16})")
    print(f"  │        = 1 - R                if R  < {w_zero:.5f}")
    if abs(w_pwr - 1.0) < 1e-4:
        print(f"  │  w     = max(w_raw,  {w_min:.5f})   "
              f"(W_MIN_Q1_16 = {W_MIN_Q1_16})")
    else:
        print(f"  │  w     = max(w_raw ^ {w_pwr:.4f},  {w_min:.5f})   "
              f"(W_MIN_Q1_16 = {W_MIN_Q1_16})")
    print("  │")
    print("  │  [Reconstruction]")
    if abs(c_scale - 1.0) < 1e-4:
        print(f"  │  restored = mu  +  w × (noisy - mu)")
    else:
        print(f"  │  restored = mu  +  {c_scale:.4f} × w × (noisy - mu)")

    # Enhancement 3: local clipping
    clip_k = kw["local_clip_k"]
    if clip_k > 0.0:
        print(f"  │")
        print(f"  │  [Enhancement 3 — correction clipped to ±{clip_k:.4f}×√noise_var]")

    # Enhancement 1: blend
    bt = kw["blend_thresh"]; bs = kw["blend_soft"]
    if bt < 9000.0:
        print(f"  │")
        print(f"  │  [Enhancement 1 — adaptive 3×3/5×5 blend]")
        print(f"  │  alpha = sigmoid((sigma2_5x5 − {bt:.1f}) / {bs:.1f})")
        print(f"  │  final = (1−alpha)×restored_5x5  +  alpha×restored_3x3")

    # Enhancement 2: two-pass
    rs2 = kw["residual_scale2"]
    if rs2 > 0.0:
        print(f"  │")
        print(f"  │  [Enhancement 2 — two-pass residual re-estimation]")
        print(f"  │  noise_var₂ = {rs2:.4f} × immerkaer(noisy − restored₁)")
        print(f"  │  then re-run the full filter with noise_var₂")

    print("  └────────────────────────────────────────────────────────────────")


def print_hw_flow_review(p):
    """
    Full hardware flow review for all three enhancements.
    Explains what changes are needed in wiener_top.sv / wiener_core.sv and
    the clock-synchronization implications of each.
    """
    kw              = dict(zip(PARAM_NAMES, p))
    blend_active    = kw["blend_thresh"] < 9000.0
    pass2_active    = kw["residual_scale2"] > 0.0
    clip_active     = kw["local_clip_k"]   > 0.0

    W = 512   # assume 512×512 image

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║          HARDWARE FLOW REVIEW (wiener_top.sv / wiener_core.sv)  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # ── Current pipeline latency ─────────────────────────────────────────────
    L = (4 * W          # four line-buffers (delay_line × 4)
         + 1            # v2_valid_r register
         + 1            # vert_mux5 output register
         + 1            # window5x5 shift → win_valid_raw
         + 3            # win_valid_pipe shift register
         + 1            # col_reflect_block output register
         + 1            # col_0_d register
         + 2            # stats_calc_5x5 (sum_d + output register)
         + 2            # center_pix_r, center_pix_r2
         + 1            # wiener_core output register
         + 1)           # final output register
    print(f"\n  Pipeline latency (L) = 4×{W} + 13 = {L} clock cycles")
    print(f"  First valid output after L cycles from first in_valid.")
    print(f"  Full frame: {W}×{W} + L = {W*W + L} cycles  "
          f"({(W*W+L)/1e6:.3f}M cycles)")

    print("\n──────────────────────────────────────────────────────────────────")

    # ── Enhancement 1: Adaptive 3×3/5×5 blend ────────────────────────────────
    status = "ACTIVE" if blend_active else "INACTIVE (blend_thresh ≥ 9000)"
    print(f"\n  [Enhancement 1 — Adaptive 3×3/5×5 Blend]  {status}")
    if blend_active:
        print(f"  Parameters: blend_thresh={kw['blend_thresh']:.1f}  "
              f"blend_soft={kw['blend_soft']:.1f}")
    print("""
  Current hardware behaviour:
    stats_calc_5x5.sv selects 3×3 or 5×5 via the static `mode_3x3` input.
    Only ONE set of stats (mu, var) is produced per pixel cycle.

  Required hardware changes:
    1. stats_calc_5x5.sv — duplicate the 3×3 accumulator path so BOTH
       (mu_3x3, var_3x3) and (mu_5x5, var_5x5) are produced in parallel.
       Cost: ~2× the adder tree for the inner 9-pixel subset (indices
       6,7,8,11,12,13,16,17,18 — already selected by the mode_3x3 flag).
       No new latency; runs inside the existing always_comb + 1-cycle register.

    2. wiener_top.sv — instantiate a SECOND wiener_core (u_wiener_core_3x3)
       fed with (mu_3x3, var_3x3) and the same noise_var_reg.
       Both cores run in parallel; out_valid is the same for both
       because they share the same pipeline stage.

    3. wiener_top.sv — add blend mux after the two cores:
         alpha_thresh = {blend_thresh as Q16.8 fixed-point constant}
         blend = (var_q16_8_5x5 > alpha_thresh) ? wien_3x3 : wien_5x5
       For a soft blend, replace the comparator with a fractional multiply
       (requires one 8-bit × 8-bit multiply — cheap).

  Clock synchronisation:
    ✓ No extra pipeline stages required.
    ✓ Both wiener_core outputs arrive at the SAME cycle (same latency).
    ✓ The blend decision (var_q16_8 threshold) is computed in the same
      always_ff block that already registers out_valid — add 0 cycles.
    ✓ mode_3x3 input to stats_calc_5x5 can be repurposed as "force 3×3"
      override for test/debug; normal operation ignores it.""")

    print("\n──────────────────────────────────────────────────────────────────")

    # ── Enhancement 2: Two-pass residual re-estimation ───────────────────────
    status = "ACTIVE" if pass2_active else "INACTIVE (residual_scale2 = 0)"
    print(f"\n  [Enhancement 2 — Two-Pass Residual Re-estimation]  {status}")
    if pass2_active:
        print(f"  Parameters: residual_scale2={kw['residual_scale2']:.4f}")
    print(f"""
  Current hardware behaviour:
    Single pass — in_valid stream → pipeline → out_valid stream.
    noise_var_reg is loaded ONCE via cfg_en/cfg_data before the frame.

  Required hardware changes:
    1. Output frame buffer:  {W}×{W}×8 bits = {W*W} bytes SRAM
       (or DRAM) to store pass-1 output pixels.

    2. Frame controller FSM (new top-level logic):
       State PASS1: assert in_valid, write out_pixel to frame buffer.
       State GAP  : deassert in_valid for at least {4*W} cycles (line
                    buffers must flush) OR assert rst_n briefly.
       State CFG  : shift new noise_var into cfg register (24 cycles
                    with cfg_en). The new value = immerkaer(residual)
                    computed in SOFTWARE between the two passes.
       State PASS2: re-assert in_valid, read pixels from frame buffer.

    3. Software (Python / C driver) between passes:
       a. Read all {W*W} output pixels from the frame buffer.
       b. Compute residual = noisy − pass1_output.
       c. Run Immerkaer on residual → multiply by residual_scale2.
       d. Serial-load 24-bit result via cfg_en / cfg_data.

  Clock synchronisation:
    ✓ Pass 1 finishes when out_valid deasserts after the last pixel.
      Last out_valid = cycle (H×W + L − 1) from frame start.
    ⚠ Gap between passes: in_valid must stay LOW for at least 4×{W} = {4*W}
      cycles to flush all four line-buffers completely.
    ⚠ cfg_en loading takes 24 cycles and must complete BEFORE in_valid
      for pass 2 is asserted (otherwise noise_var_reg is partial).
    ✓ Once PASS2 in_valid starts, timing is identical to pass 1.
    ✓ Total frame time: 2 × ({W*W} + L) + {4*W} flush + 24 cfg cycles
      = {2*(W*W+L) + 4*W + 24} cycles  ({(2*(W*W+L) + 4*W + 24)/1e6:.3f}M cycles).""")

    print("\n──────────────────────────────────────────────────────────────────")

    # ── Enhancement 3: Local correction clipping ─────────────────────────────
    status = "ACTIVE" if clip_active else "INACTIVE (local_clip_k = 0)"
    print(f"\n  [Enhancement 3 — Local Correction Clipping]  {status}")
    if clip_active:
        print(f"  Parameters: local_clip_k={kw['local_clip_k']:.4f}")
    print("""
  Current wiener_core.sv behaviour:
    prod_q8_8 = w_clamped * diff_extended  (the correction term)
    restored_full = mu_extended + prod_q8_8
    (no limit on how large prod_q8_8 can be)

  Required hardware changes — SMALLEST of the three enhancements:
    Inside wiener_core.sv, after computing prod_q8_8, add:

      logic signed [17:0] clip_limit;        // k * sqrt(noise_var)
      // sqrt(noise_var) can be pre-computed in software and loaded
      // via a second 16-bit shift register (cfg2_en / cfg2_data), OR
      // approximated as noise_var_reg >> SHIFT_K for a power-of-2 factor.

      assign clip_limit = CLIP_K_Q8_8;       // fixed constant if k is fixed

      wire signed [17:0] prod_clipped;
      assign prod_clipped =
          (prod_q8_8 >  clip_limit) ?  clip_limit :
          (prod_q8_8 < -clip_limit) ? -clip_limit :
                                       prod_q8_8;

      assign restored_full = mu_extended + prod_clipped;

    Cost: 2 signed comparators + 2-to-1 mux (< 50 LUT equivalents).

  Clock synchronisation:
    ✓ Fully combinational — zero extra clock cycles.
    ✓ Fits inside the existing always_comb path in wiener_core.sv.
    ✓ No change to in_valid / out_valid timing at all.
    ✓ If clip_limit is a loadable register (not hardcoded), load it via
      a second shift register similar to noise_var_reg — also 24 bits.""")

    print("\n──────────────────────────────────────────────────────────────────")
    active_count = sum([blend_active, pass2_active, clip_active])
    print(f"\n  Summary: {active_count}/3 enhancements active in the optimised solution.")
    if blend_active:
        print("  → Enhancement 1 (blend) is ACTIVE: needs parallel stats + second wiener_core")
    if pass2_active:
        print("  → Enhancement 2 (2-pass) is ACTIVE: needs frame buffer + FSM + software loop")
    if clip_active:
        print("  → Enhancement 3 (clip) is ACTIVE: trivial 2-comparator change in wiener_core.sv")
    print("╚══════════════════════════════════════════════════════════════════╝\n")


def hw_suggestions(p):
    kw            = dict(zip(PARAM_NAMES, p))
    W_MIN_Q1_16   = round(kw["w_min"]        * 65536)
    W_ZERO_Q16_16 = round(kw["w_zero_thresh"] * 65536)
    return {
        "noise_var_scale": f"{kw['noise_var_scale']:.6f}",
        "noise_var_bias ": f"{kw['noise_var_bias']:.4f}",
        "immerkaer_sigma_thresh": f"{kw['immerkaer_sigma_thresh']:.2f}",
        "immerkaer_high_scale  ": f"{kw['immerkaer_high_scale']:.4f}",
        "sigma2_bias": f"{kw['sigma2_bias']:.4f}",
        "W_MIN_Q1_16  (replace 4096  in wiener_core.sv)": W_MIN_Q1_16,
        "W_ZERO_Q16_16 (replace 60293 in wiener_core.sv)": W_ZERO_Q16_16,
        "w_power": f"{kw['w_power']:.4f}",
        "correction_scale": f"{kw['correction_scale']:.4f}",
    }


def metrics(ref, restored):
    r = np.clip(restored, 0, 255)
    return snr_db(ref, r), psnr_db(ref, r), ssim_score(ref, r)


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_plot(original, noisy, rest_def, rest_opt, snr_target, save_path):
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    for ax, img, title in zip(
        axs,
        [original, noisy, rest_def, rest_opt],
        ["Original",
         f"Noisy (SNR={snr_target} dB)",
         "Default params",
         "Optimised params"],
    ):
        ax.imshow(np.clip(img, 0, 255).astype(np.uint8), cmap="gray",
                  vmin=0, vmax=255)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    s_d, p_d, _ = metrics(original, rest_def)
    s_o, p_o, _ = metrics(original, rest_opt)
    fig.suptitle(
        f"Default  SNR={s_d:.2f} dB  PSNR={p_d:.2f} dB   │   "
        f"Optimised  SNR={s_o:.2f} dB  PSNR={p_o:.2f} dB",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved → {save_path}")


# ── Sensitivity plot ──────────────────────────────────────────────────────────

def sensitivity_plot(mlp, scaler, p_opt, save_path):
    """1-D sensitivity around the optimum for each parameter."""
    n_dim = len(PARAM_NAMES)
    fig, axs = plt.subplots(3, 3, figsize=(14, 10))
    axs = axs.flatten()

    for i, (name, (lo, hi)) in enumerate(zip(PARAM_NAMES, PARAM_BOUNDS)):
        sweep = np.linspace(lo, hi, 60)
        y_sw  = []
        for v in sweep:
            p = p_opt.copy()
            p[i] = v
            y_sw.append(mlp.predict(scaler.transform([p]))[0])
        axs[i].plot(sweep, y_sw)
        axs[i].axvline(p_opt[i], color="red", linestyle="--", label="optimum")
        axs[i].set_title(name, fontsize=8)
        axs[i].set_xlabel("value", fontsize=7)
        axs[i].set_ylabel("pred. SNR (dB)", fontsize=7)
        axs[i].legend(fontsize=6)

    plt.suptitle("Parameter Sensitivity (surrogate model)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    print(f"Sensitivity plot saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Wiener hw parameter optimizer (extended)")
    parser.add_argument("--image-path",  default="../../../images/test_7.jpg")
    parser.add_argument("--snr-db",      type=float, default=14.0)
    parser.add_argument("--win-size",    type=int,   default=5)
    parser.add_argument("--n-samples",   type=int,   default=2000,
                        help="LHS sample count for phase 1 (default 2000)")
    parser.add_argument("--seeds",       type=int,   default=3,
                        help="Noise seeds averaged per evaluation (default 3)")
    parser.add_argument("--no-show",     action="store_true")
    args = parser.parse_args()

    seeds = list(range(args.seeds))

    # ── Load image ────────────────────────────────────────────────────────────
    img_path = os.path.join(os.path.dirname(__file__), args.image_path)
    print(f"Loading image: {img_path}")
    original = load_gray(img_path).astype(np.float64)
    print(f"Image shape  : {original.shape}")

    noisy_0, true_nv = add_gaussian_noise(original, args.snr_db, seed=0)

    # ── Baseline ──────────────────────────────────────────────────────────────
    rest_def, est_nv = wiener_hw_enhanced(noisy_0, win_size=args.win_size,
                                          **dict(zip(PARAM_NAMES, DEFAULTS)))
    s_d, p_d, ss_d = metrics(original, rest_def)
    print("\n──────────────── Baseline (default params) ────────────────")
    print(f"  True noise variance  : {true_nv:.4f}")
    print(f"  Estimated noise var  : {est_nv:.4f}")
    print(f"  SNR  (orig vs rest)  : {s_d:.4f} dB")
    print(f"  PSNR (orig vs rest)  : {p_d:.4f} dB")
    print(f"  SSIM                 : {ss_d:.4f}")

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    X, y = lhs_search(original, args.snr_db, args.n_samples, seeds)

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    mlp, scaler = train_surrogate(X, y)

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    p_surr = optimise_surrogate(mlp, scaler, X, y)

    # ── Phase 4 ───────────────────────────────────────────────────────────────
    best_lhs = X[np.argmax(y)]
    p_start  = (p_surr
                if evaluate(p_surr,    original, args.snr_db, seeds) >
                   evaluate(best_lhs,  original, args.snr_db, seeds)
                else best_lhs)

    p_opt, snr_opt = fine_tune(p_start, original, args.snr_db, seeds)

    # ── Evaluate optimised (seed=0 for display) ───────────────────────────────
    rest_opt, _ = wiener_hw_enhanced(noisy_0, win_size=args.win_size,
                                     **dict(zip(PARAM_NAMES, p_opt)))
    s_o, p_o, ss_o = metrics(original, rest_opt)

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n══════════════════════════════════════════════════════════════")
    print("  OPTIMISED PARAMETERS")
    print("══════════════════════════════════════════════════════════════")
    for name, val in zip(PARAM_NAMES, p_opt):
        default = DEFAULTS[PARAM_NAMES.index(name)]
        print(f"  {name:35s}: {val:>10.5f}   (was {default:.5f})")

    print()
    print_equations(p_opt)
    print_hw_flow_review(p_opt)

    print("\n  Hardware register suggestions:")
    for k, v in hw_suggestions(p_opt).items():
        print(f"    {k}: {v}")

    print("\n  Quality metrics (seed=0):")
    print(f"    SNR  : {s_d:.4f} dB  →  {s_o:.4f} dB   (Δ={s_o-s_d:+.4f})")
    print(f"    PSNR : {p_d:.4f} dB  →  {p_o:.4f} dB   (Δ={p_o-p_d:+.4f})")
    print(f"    SSIM : {ss_d:.4f}     →  {ss_o:.4f}      (Δ={ss_o-ss_d:+.4f})")
    print(f"  (avg SNR over {args.seeds} seeds: {snr_opt:.4f} dB)")

    # ── Save plots ────────────────────────────────────────────────────────────
    here = os.path.dirname(__file__)
    make_plot(original, noisy_0, rest_def, rest_opt,
              args.snr_db, os.path.join(here, "optimization_result.png"))
    sensitivity_plot(mlp, scaler, p_opt,
                     os.path.join(here, "sensitivity.png"))

    # ── Save best params ──────────────────────────────────────────────────────
    out_path = os.path.join(here, "best_params.txt")
    kw = dict(zip(PARAM_NAMES, p_opt))
    with open(out_path, "w") as f:
        f.write("# Best parameters found by optimize_params.py\n")
        f.write(f"# Image: {args.image_path}   SNR target: {args.snr_db} dB\n\n")

        f.write("# ── Raw parameter values ────────────────────────────────────\n")
        for name, val in zip(PARAM_NAMES, p_opt):
            f.write(f"{name} = {val:.8f}\n")

        f.write("\n# ── Optimised equations ─────────────────────────────────────\n")
        nv_s = kw["noise_var_scale"]; nv_b = kw["noise_var_bias"]
        s2b  = kw["sigma2_bias"]
        wzt  = kw["w_zero_thresh"];   wm  = kw["w_min"]; wp = kw["w_power"]
        cs   = kw["correction_scale"]
        it   = kw["immerkaer_sigma_thresh"]; ihs = kw["immerkaer_high_scale"]

        if abs(nv_b) < 1e-4:
            f.write(f"noise_var = {nv_s:.4f} * immerkaer_estimate\n")
        else:
            sign = "+" if nv_b >= 0 else "-"
            f.write(f"noise_var = {nv_s:.4f} * immerkaer_estimate {sign} {abs(nv_b):.2f}\n")

        if abs(s2b) < 1e-4:
            f.write("sigma2_safe = max(sigma2, 1/256)\n")
        else:
            f.write(f"sigma2_safe = max(sigma2 + {s2b:.2f}, 1/256)\n")

        f.write(f"R = noise_var / sigma2_safe\n")

        f.write(f"w_raw = 0          if R >= {wzt:.5f}   "
                f"(W_ZERO_Q16_16 = {round(wzt*65536)})\n")
        f.write(f"      = 1 - R      if R  < {wzt:.5f}\n")
        if abs(wp - 1.0) < 1e-4:
            f.write(f"w = max(w_raw, {wm:.5f})   (W_MIN_Q1_16 = {round(wm*65536)})\n")
        else:
            f.write(f"w = max(w_raw ^ {wp:.4f}, {wm:.5f})   (W_MIN_Q1_16 = {round(wm*65536)})\n")

        if abs(cs - 1.0) < 1e-4:
            f.write("restored = mu + w * (noisy - mu)\n")
        else:
            f.write(f"restored = mu + {cs:.4f} * w * (noisy - mu)\n")

        f.write("\n# ── Hardware register suggestions ───────────────────────────\n")
        for k, v in hw_suggestions(p_opt).items():
            f.write(f"# {k} = {v}\n")

        f.write(f"\n# Results (seed=0):  SNR={s_o:.4f} dB  PSNR={p_o:.4f} dB  SSIM={ss_o:.4f}\n")
        f.write(f"# avg SNR over {args.seeds} seeds = {snr_opt:.4f} dB\n")
    print(f"\nBest params saved → {out_path}")
    print("══════════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
