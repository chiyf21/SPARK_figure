#!/usr/bin/env python3
"""
Registration metrics v4 — F260517 dataset.

Changes over v3:
    - ALL metrics computed against a FIXED reference (the initial ref_mem_adj
      calibrated on frames 0-4).  The registration pipeline still uses the
      updated/blended ref_mem_adj internally, but metric sampling always uses
      the fixed reference.
    - This eliminates artificial jumps in baseline metrics caused by reference
      intensity mapping changes.

Output:
    registration_metrics_ref20_v4.csv   — registered metrics (200 rows)
    baseline_metrics_ref20_v4.csv       — z_init-only baseline metrics (200 rows)
"""

import os, sys, time
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label as label_ndi
from skimage.measure import regionprops

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
cp.cuda.Device(1).use()

NOTEBOOK_ROOT = Path(
    "/home/cyf/wbi/Virginia/code/wbi_0123/wholistic_registration/src/"
    "wholistic_registration"
)
sys.path.insert(0, str(NOTEBOOK_ROOT))

COARSEFLOW_ROOT = Path("/home/cyf/wbi/Virginia/code/CoarseFlow").resolve()
sys.path.insert(0, str(COARSEFLOW_ROOT))

from utils import IO, calFlowCrossResolution, mask, preprocess as prep
from utils.calFlowCrossResolution import (
    generate_continuous_H_gpu,
    apply_H_to_matrix_gpu,
)

OUT_DIR = Path("/home/cyf/wbi/Virginia/code_for_paper")
os.makedirs(str(OUT_DIR), exist_ok=True)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
F260517_mov_path = "/home/cyf/wbi/Virginia/raw_data/f260517/260517_exp_00001_TZCYX.ome.tiff"
F260517_ref_path = "/home/cyf/wbi/Virginia/raw_data/f260517/260517_anat_00003_TZCYX.ome.tiff"

SOFT_BLEND_FRAMES = 5  # number of frames over which tgt_q is blended

# ===========================================================================
# Intensity-mapping helpers  (split into learn / apply / full-init)
# ===========================================================================

def learn_tgt_q_from_target(ref_mem, target_stack_zyx, z_idx, percentiles):
    """Learn only (src_q, tgt_q) from matched planes.  No mask change."""
    ref_source_zyx = ref_mem[z_idx].astype(np.float32, copy=False)
    src_q, tgt_q, used = prep.learn_quantile_mapping(
        source=ref_source_zyx, target=np.asarray(target_stack_zyx, dtype=np.float32),
        percentiles=percentiles,
    )
    return src_q, tgt_q, used


def apply_mapping_to_ref(ref_mem_zyx, src_q, tgt_q):
    """Apply quantile mapping: (Z,Y,X) -> (X,Y,Z).  Pure intensity transform."""
    return prep.apply_quantile_mapping(
        ref_mem_zyx, src_q, tgt_q,
    ).transpose(2, 1, 0).astype(np.float32, copy=False)  # → (X,Y,Zref)


def full_initial_ref_setup(ref_mem, target_stack_zyx, z_idx, option,
                            thresFactor, maskRange, smoothPenalty_raw, percentiles):
    """
    Initial setup: learn mapping, compute mask_ref & smoothPenalty ONCE.
    Returns (ref_adj, src_q, tgt_q).
    """
    src_q, tgt_q, _ = learn_tgt_q_from_target(
        ref_mem, target_stack_zyx, z_idx, percentiles,
    )
    ref_adj = apply_mapping_to_ref(ref_mem, src_q, tgt_q)

    option["mask_ref"] = mask.getMask(ref_adj, thresFactor)
    option["mask_ref"] = mask.bwareafilt3_wei(option["mask_ref"], maskRange)
    Pnltfactor = prep.getSmPnltNormFctr(ref_adj, option)
    option["smoothPenalty"] = Pnltfactor * smoothPenalty_raw

    return ref_adj, src_q, tgt_q


# ===========================================================================
# Edge detection
# ===========================================================================

def sobel_edge_magnitude(plane_2d):
    from scipy.ndimage import sobel
    p = np.asarray(plane_2d, dtype=np.float32)
    p = p - p.min()
    denom = p.max() - p.min()
    if denom > 1e-8:
        p = p / denom
    gx = sobel(p, axis=-1, mode="nearest")
    gy = sobel(p, axis=-2, mode="nearest")
    return np.sqrt(gx**2 + gy**2 + 1e-8).astype(np.float32)


def binarize_edge_magnitude(edge_mag, percentile=90):
    thresh = np.percentile(edge_mag, percentile)
    return (edge_mag >= thresh).astype(np.uint8)


def symmetric_edge_distance_2d(edges_a, edges_b):
    if not np.any(edges_a) or not np.any(edges_b):
        return np.nan
    dt_b = distance_transform_edt(1 - edges_b)
    dt_a = distance_transform_edt(1 - edges_a)
    return float(0.5 * (np.mean(dt_b[edges_a > 0]) + np.mean(dt_a[edges_b > 0])))


# ===========================================================================
# NCC
# ===========================================================================

def zncc_2d(a, b, eps=1e-8):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    a_c = a - np.mean(a)
    b_c = b - np.mean(b)
    numer = np.dot(a_c, b_c)
    denom = np.sqrt(np.dot(a_c, a_c) * np.dot(b_c, b_c) + eps)
    if denom < eps: return np.nan
    return float(numer / denom)


# ===========================================================================
# Sparse-cell centroid metrics
# ===========================================================================

def sparse_cell_centroid_metrics_2d(mov_plane, mapped_plane,
                                     thresh_factor=3.0, match_radius=5.0):
    mov = np.asarray(mov_plane, dtype=np.float32)
    mapped = np.asarray(mapped_plane, dtype=np.float32)

    def get_centroids(img):
        mu, sigma = float(np.mean(img)), float(np.std(img))
        bin_mask = img > (mu + thresh_factor * sigma)
        if not np.any(bin_mask):
            return np.empty((0, 2), dtype=np.float32)
        labeled, _ = label_ndi(bin_mask)
        props = regionprops(labeled)
        return np.array([p.centroid for p in props], dtype=np.float32)

    cents_mov = get_centroids(mov)
    cents_mapped = get_centroids(mapped)
    n_mov, n_mapped = len(cents_mov), len(cents_mapped)
    if n_mov == 0 or n_mapped == 0:
        return np.nan, np.nan, np.nan, n_mov, n_mapped

    dists_m2p = np.full(n_mov, np.nan, dtype=np.float32)
    for i, c in enumerate(cents_mov):
        dists_m2p[i] = np.min(np.sqrt(np.sum((cents_mapped - c)**2, axis=1)))
    dists_p2m = np.full(n_mapped, np.nan, dtype=np.float32)
    for i, c in enumerate(cents_mapped):
        dists_p2m[i] = np.min(np.sqrt(np.sum((cents_mov - c)**2, axis=1)))

    nn = float(0.5 * (np.nanmean(dists_m2p) + np.nanmean(dists_p2m)))
    rec = float(np.mean(dists_m2p <= match_radius))
    prec = float(np.mean(dists_p2m <= match_radius))
    return nn, rec, prec, n_mov, n_mapped


# ===========================================================================
# Per-frame metrics  (FIXED mask logic: exclude mask pixels from error)
# ===========================================================================

def compute_frame_metrics(
    mov_mem_zyx, mem_mapped_zyx,
    mov_sparse_zyx, sparse_mapped_zyx,
    mask_mov_zyx=None,          # True = outlier / should be EXCLUDED
    edge_percentile=90,
    sparse_thresh_factor=3.0,
):
    """
    mask_mov_zyx=True  →  outlier pixel, same convention as getMotion_v2.
    We compute errors on ~mask_mov (non-outlier) regions,
    and ZERO OUT the error in mask_mov regions.
    """
    K = mov_mem_zyx.shape[0]
    out = {
        "mem_MAE": [], "mem_MSE": [], "mem_NCC": [],
        "mem_edge_sym_dist": [],
        "mem_nMAE": [], "mem_nRMSE": [],
        "sparse_MAE": [], "sparse_MSE": [],
        "sparse_nMAE": [], "sparse_nRMSE": [],
        "sparse_centroid_NN": [], "sparse_recall": [], "sparse_precision": [],
    }

    for k in range(K):
        m_mov = mov_mem_zyx[k]
        m_map = mem_mapped_zyx[k]
        s_mov = mov_sparse_zyx[k]
        s_map = sparse_mapped_zyx[k]

        # --- Build eval mask: exclude outlier pixels, keep normal pixels ---
        if mask_mov_zyx is not None:
            exclude = mask_mov_zyx[k].astype(bool)   # True = exclude
        else:
            exclude = np.zeros_like(m_mov, dtype=bool)

        # Eval on ~exclude (normal pixels), but zero out masked region
        eval_mask = ~exclude

        # Fallback if everything is excluded
        if not np.any(eval_mask):
            for key in out:
                out[key].append(np.nan)
            continue

        # --- Robust dynamic range from mov (non-excluded pixels) -----------
        m_valid = m_mov[eval_mask]
        p1, p99 = np.percentile(m_valid, [1, 99])
        dyn_range_m = max(p99 - p1, 1e-8)

        s_valid = s_mov[eval_mask]
        sp1, sp99 = np.percentile(s_valid, [1, 99])
        dyn_range_s = max(sp99 - sp1, 1e-8)

        # --- Membrane MAE / MSE / nMAE / nRMSE ------------------------------
        diff_m = np.abs(m_mov.astype(np.float32) - m_map.astype(np.float32))
        diff_m[exclude] = 0.0   # <-- mask区域error置零
        n_total = m_mov.size
        mae_m = float(np.sum(diff_m) / n_total)
        mse_m = float(np.sum(diff_m**2) / n_total)

        out["mem_MAE"].append(mae_m)
        out["mem_MSE"].append(mse_m)
        out["mem_nMAE"].append(mae_m / dyn_range_m)
        out["mem_nRMSE"].append(np.sqrt(mse_m) / dyn_range_m)

        # --- Membrane NCC (on eval pixels only) -----------------------------
        out["mem_NCC"].append(zncc_2d(m_mov[eval_mask], m_map[eval_mask]))

        # --- Membrane edge distance -----------------------------------------
        e_mov = sobel_edge_magnitude(m_mov)
        e_map = sobel_edge_magnitude(m_map)
        b_mov = binarize_edge_magnitude(e_mov, percentile=edge_percentile)
        b_map = binarize_edge_magnitude(e_map, percentile=edge_percentile)
        out["mem_edge_sym_dist"].append(symmetric_edge_distance_2d(b_mov, b_map))

        # --- Sparse MAE / MSE / nMAE / nRMSE --------------------------------
        diff_s = np.abs(s_mov.astype(np.float32) - s_map.astype(np.float32))
        diff_s[exclude] = 0.0
        mae_s = float(np.sum(diff_s) / n_total)
        mse_s = float(np.sum(diff_s**2) / n_total)

        out["sparse_MAE"].append(mae_s)
        out["sparse_MSE"].append(mse_s)
        out["sparse_nMAE"].append(mae_s / dyn_range_s)
        out["sparse_nRMSE"].append(np.sqrt(mse_s) / dyn_range_s)

        # --- Sparse centroid ------------------------------------------------
        nn_d, recall, precision, _, _ = sparse_cell_centroid_metrics_2d(
            s_mov, s_map, thresh_factor=sparse_thresh_factor, match_radius=5.0,
        )
        out["sparse_centroid_NN"].append(nn_d)
        out["sparse_recall"].append(recall)
        out["sparse_precision"].append(precision)

    agg = {}
    for key, vals in out.items():
        agg[key] = float(np.nanmean(vals))
    return agg


# ===========================================================================
# Main pipeline
# ===========================================================================

def main():
    print("=" * 80)
    print("Registration Metrics Pipeline v4 — F260517")
    print("  Sol.1: soft tgt_q blend over 5 frames")
    print("  Sol.2: mask_ref & smoothPenalty fixed after init")
    print("  Mask:  excluded from error (same as getMotion_v2 data term)")
    print("  NEW:   ALL metrics sampled from FIXED initial reference")
    print(f"  Output: {OUT_DIR}")
    print("=" * 80)

    # -------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------
    print("\n[1/6] Loading data ...")
    t0 = time.time()
    F260517_mov, _ = IO.readTiff(F260517_mov_path)
    F260517_ref, _ = IO.readTiff(F260517_ref_path)

    ref_mem_raw = F260517_ref[90:310, 1, :, :].astype(np.float32)           # (Z,Y,X)
    ref_sparse_raw = F260517_ref[90:310, 0, :, :].astype(np.float32)
    mov_mem_all = F260517_mov[:, :, 1, :, :].astype(np.float32)              # (T,K,Y,X)
    mov_sparse_all = F260517_mov[:, :, 0, :, :].astype(np.float32)

    print(f"  ref_mem:      {ref_mem_raw.shape}")
    print(f"  mov_mem:      {mov_mem_all.shape}")
    print(f"  loaded in {time.time() - t0:.1f}s")

    # -------------------------------------------------------------------
    # 2. Setup
    # -------------------------------------------------------------------
    print("\n[2/6] Initial setup ...")
    option = {}
    option["r"] = 5
    option["layer"] = 3
    option["iter"] = 10
    option["movRange"] = 5.0
    option["tol"] = 1e-6
    option["zRatio_HR"] = 1
    option["wrong_region_enable"] = False

    thresFactor = 5.0
    maskRange = [5.0, 4000.0]
    smoothPenalty_raw = 0.01
    ref_update_every = 20
    percentiles = [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.8]

    # --- z_init ---------------------------------------------------------
    z_init = calFlowCrossResolution.FindInitZ_stack_global_fixed_spacing(
        mov_mem_all[0].transpose(2, 1, 0),
        ref_mem_raw.transpose(2, 1, 0),
        delta_ref_idx=10, use_gradient=False,
    )
    z_init = z_init.astype(np.float32)
    z_idx = np.rint(z_init).astype(np.int32)
    z_idx = np.clip(z_idx, 0, ref_mem_raw.shape[0] - 1)

    K, T = int(z_init.shape[0]), int(mov_mem_all.shape[0])

    # --- Identity phase for baseline -------------------------------------
    x = np.arange(mov_mem_all[0].shape[2], dtype=np.float32)
    y = np.arange(mov_mem_all[0].shape[1], dtype=np.float32)
    k = np.arange(K, dtype=np.int32)
    X_grid, Y_grid, K_grid = np.meshgrid(x, y, k, indexing="ij")
    coords_xyz = np.empty((len(x), len(y), K, 3), dtype=np.float32)
    coords_xyz[..., 0] = X_grid
    coords_xyz[..., 1] = Y_grid
    coords_xyz[..., 2] = z_init[K_grid]

    print(f"  K={K}, T={T}, z_init={z_init}")

    # -------------------------------------------------------------------
    # 3. Initial reference setup  (mask_ref & smoothPenalty set ONCE here)
    # -------------------------------------------------------------------
    print("\n[3/6] Initial reference setup (mask + smoothPenalty fixed) ...")
    init_target = np.mean(mov_mem_all[[0, 1, 2, 3, 4]], axis=0)  # (K,Y,X)
    ref_mem_adj, src_q_fixed, tgt_q_current = full_initial_ref_setup(
        ref_mem=ref_mem_raw, target_stack_zyx=init_target, z_idx=z_idx,
        option=option, thresFactor=thresFactor, maskRange=maskRange,
        smoothPenalty_raw=smoothPenalty_raw, percentiles=percentiles,
    )
    # mask_ref & smoothPenalty are now in option and will NOT be changed again.
    ref_mem_adj_fixed = ref_mem_adj.copy()  # ← FIXED reference for ALL metrics
    ref_mem_adj_fixed_gpu = cp.asarray(ref_mem_adj_fixed, dtype=cp.float32)
    print("  mask_ref and smoothPenalty fixed.")
    print("  Fixed reference saved — all metrics will use this.")

    # -------------------------------------------------------------------
    # GPU pre-allocations
    # -------------------------------------------------------------------
    ref_sparse_gpu = cp.asarray(ref_sparse_raw.transpose(2, 1, 0))  # (X,Y,Zref)

    def sample_ref_sparse(phase):
        ph = phase if hasattr(phase, "get") else cp.asarray(phase, dtype=cp.float32)
        H = generate_continuous_H_gpu(ref_sparse_gpu, zRatio=1)
        sm = apply_H_to_matrix_gpu(ph, H)
        if hasattr(sm, "get"): sm = sm.get()
        return np.asarray(sm, dtype=np.float32).transpose(2, 1, 0)

    def sample_ref_mem(phase, ref_xyz):
        ph = phase if hasattr(phase, "get") else cp.asarray(phase, dtype=cp.float32)
        H = generate_continuous_H_gpu(ref_xyz, zRatio=1)
        mp = apply_H_to_matrix_gpu(ph, H)
        if hasattr(mp, "get"): mp = mp.get()
        return np.asarray(mp, dtype=np.float32).transpose(2, 1, 0)

    # -------------------------------------------------------------------
    # 4. Registration loop  (with soft tgt_q blending)
    # -------------------------------------------------------------------
    print(f"\n[4/6] Starting loop (ref update every {ref_update_every}frames, "
          f"blend={SOFT_BLEND_FRAMES}frames) ...")

    registered_records = []
    baseline_records = []
    registered_mem_mapped_cache = {}
    frames_since_ref_update = 0
    ref_update_id = 0

    # Soft-blend state
    blend_frames_left = 0
    blend_old_tgt_q = None
    blend_new_tgt_q = None

    option["phase"] = coords_xyz.copy()
    option.pop("motion", None)

    total_start = time.time()

    for i in range(T):
        frame_start = time.time()

        # ================================================================
        # Soft blend: if we are in the middle of a transition, blend tgt_q
        # and recompute ref_mem_adj (mapping only, no mask change).
        # ================================================================
        if blend_frames_left > 0:
            alpha = (SOFT_BLEND_FRAMES - blend_frames_left + 1) / SOFT_BLEND_FRAMES
            blended_tgt_q = (alpha * blend_new_tgt_q +
                             (1.0 - alpha) * blend_old_tgt_q)
            ref_mem_adj = apply_mapping_to_ref(ref_mem_raw, src_q_fixed, blended_tgt_q)
            blend_frames_left -= 1
        elif blend_frames_left == 0 and blend_new_tgt_q is not None:
            # Transition complete
            tgt_q_current = blend_new_tgt_q
            blend_new_tgt_q = None
            ref_mem_adj = apply_mapping_to_ref(ref_mem_raw, src_q_fixed, tgt_q_current)

        # ================================================================
        # Registration
        # ================================================================
        if i == 0:
            option["phase"] = coords_xyz.copy()
            option.pop("motion", None)

        raw_mem_zyx = mov_mem_all[i]          # (K,Y,X) already float32
        raw_sparse_zyx = mov_sparse_all[i]    # (K,Y,X)
        mov_mem_xyk = raw_mem_zyx.transpose(2, 1, 0)  # (X,Y,K)

        option["mask_mov"] = mask.getMask(mov_mem_xyk, thresFactor)
        option["mask_mov"] = mask.bwareafilt3_wei(option["mask_mov"], maskRange)

        phase_new, motion_current, mem_mapped_xyk = calFlowCrossResolution.getMotion_v2(
            mov_mem_xyk, ref_mem_adj, option, verbose=False,
        )
        if hasattr(phase_new, "get"):       phase_new = phase_new.get()
        if hasattr(motion_current, "get"):  motion_current = motion_current.get()
        if hasattr(mem_mapped_xyk, "get"):  mem_mapped_xyk = mem_mapped_xyk.get()
        phase_new = np.asarray(phase_new, dtype=np.float32)
        motion_current = np.asarray(motion_current, dtype=np.float32)

        mem_mapped_zyx_reg = np.asarray(mem_mapped_xyk, dtype=np.float32).transpose(2, 1, 0)
        sparse_mapped_zyx_reg = sample_ref_sparse(cp.asarray(phase_new))

        # ---- FOR METRICS: re-sample from FIXED reference -----------------
        mem_mapped_zyx_reg_fixed = sample_ref_mem(
            cp.asarray(phase_new), ref_mem_adj_fixed_gpu,
        )

        registered_mem_mapped_cache[i] = mem_mapped_zyx_reg

        # ================================================================
        # Baseline: identity phase + z_init, FIXED reference
        # ================================================================
        baseline_mem_zyx = sample_ref_mem(cp.asarray(coords_xyz, dtype=cp.float32),
                                          ref_mem_adj_fixed_gpu)
        baseline_sparse_zyx = sample_ref_sparse(cp.asarray(coords_xyz, dtype=cp.float32))

        # ================================================================
        # Mask for metrics  (True = exclude from error, match getMotion_v2)
        # ================================================================
        mask_mov = option["mask_mov"]
        if hasattr(mask_mov, "get"): mask_mov = mask_mov.get()
        mask_mov_zyx = np.asarray(mask_mov, dtype=bool).transpose(2, 1, 0)

        # ================================================================
        # Compute metrics
        # ================================================================
        reg_metrics = compute_frame_metrics(
            mov_mem_zyx=raw_mem_zyx, mem_mapped_zyx=mem_mapped_zyx_reg_fixed,
            mov_sparse_zyx=raw_sparse_zyx, sparse_mapped_zyx=sparse_mapped_zyx_reg,
            mask_mov_zyx=mask_mov_zyx,
        )
        base_metrics = compute_frame_metrics(
            mov_mem_zyx=raw_mem_zyx, mem_mapped_zyx=baseline_mem_zyx,
            mov_sparse_zyx=raw_sparse_zyx, sparse_mapped_zyx=baseline_sparse_zyx,
            mask_mov_zyx=mask_mov_zyx,
        )

        elapsed = time.time() - frame_start

        registered_records.append(
            {"frame": i, "ref_update_id": ref_update_id, "elapsed_s": elapsed,
             **reg_metrics})
        baseline_records.append(
            {"frame": i, "ref_update_id": ref_update_id, "elapsed_s": elapsed,
             **base_metrics})

        # ================================================================
        # Log
        # ================================================================
        blend_tag = f" blend{SOFT_BLEND_FRAMES-blend_frames_left}/{SOFT_BLEND_FRAMES}" if blend_frames_left > 0 else ""
        print(
            f"[Frame {i:03d}/{T-1:03d}]{blend_tag} "
            f"reg: NCC={reg_metrics['mem_NCC']:.4f} nMAE={reg_metrics['mem_nMAE']:.4f} "
            f"edge={reg_metrics['mem_edge_sym_dist']:.3f}px "
            f"spR={reg_metrics['sparse_recall']:.3f} | "
            f"base: NCC={base_metrics['mem_NCC']:.4f} nMAE={base_metrics['mem_nMAE']:.4f} "
            f"{elapsed:.1f}s"
        )

        # ================================================================
        # Temporal init for next frame
        # ================================================================
        option["motion"] = (0.7 * motion_current).astype(np.float32, copy=False)

        # ================================================================
        # Ref update  (only intensity mapping; mask & smoothPenalty fixed)
        # ================================================================
        frames_since_ref_update += 1
        if frames_since_ref_update >= ref_update_every:
            calib_frames = sorted(registered_mem_mapped_cache.keys())[-5:]
            ref_update_id += 1

            stacks = [registered_mem_mapped_cache[fi]
                      for fi in calib_frames if fi in registered_mem_mapped_cache]
            if len(stacks) > 0:
                target = np.mean(np.stack(stacks, axis=0), axis=0).astype(np.float32)
                _, new_tgt_q, _ = learn_tgt_q_from_target(
                    ref_mem_raw, target, z_idx, percentiles,
                )

                # Start soft blend from current → new
                blend_old_tgt_q = tgt_q_current.copy()
                blend_new_tgt_q = new_tgt_q.copy()
                blend_frames_left = SOFT_BLEND_FRAMES

                print(f"\n  >>> Ref Update #{ref_update_id} — frames {calib_frames}")
                print(f"      old tgt_q[0]: {blend_old_tgt_q[0]:.2f}  "
                      f"new tgt_q[0]: {blend_new_tgt_q[0]:.2f}  "
                      f"blending over {SOFT_BLEND_FRAMES} frames")

            frames_since_ref_update = 0
            print()

    total_elapsed = time.time() - total_start
    print(f"\n  Done in {total_elapsed:.1f}s ({total_elapsed/T:.1f}s/frame)")

    # -------------------------------------------------------------------
    # 5. Save CSVs
    # -------------------------------------------------------------------
    print("\n[5/6] Saving CSVs ...")
    df_reg = pd.DataFrame(registered_records)
    df_base = pd.DataFrame(baseline_records)
    df_reg.to_csv(str(OUT_DIR / "registration_metrics_ref20_v4.csv"), index=False)
    df_base.to_csv(str(OUT_DIR / "baseline_metrics_ref20_v4.csv"), index=False)
    print(f"  Registered: {len(df_reg)} rows")
    print(f"  Baseline:   {len(df_base)} rows")

    # -------------------------------------------------------------------
    # 6. Summary
    # -------------------------------------------------------------------
    print("\n[6/6] Summary statistics ...")
    metric_cols = [
        "mem_MAE", "mem_nMAE", "mem_nRMSE", "mem_NCC", "mem_edge_sym_dist",
        "sparse_MAE", "sparse_nMAE", "sparse_nRMSE",
        "sparse_centroid_NN", "sparse_recall", "sparse_precision",
    ]

    for label, df in [("REGISTERED", df_reg), ("BASELINE", df_base)]:
        print(f"\n  --- {label} (global means) ---")
        for col in metric_cols:
            vals = df[col].dropna()
            print(f"    {col:25s}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
                  f"min={vals.min():.4f}  max={vals.max():.4f}")

    print("\n  --- Registered: per-ref-update-block ---")
    for ruid in sorted(df_reg["ref_update_id"].unique()):
        block = df_reg[df_reg["ref_update_id"] == ruid]
        print(
            f"    ru{ruid:02d} frames [{block['frame'].min():.0f}-{block['frame'].max():.0f}] "
            f"n={len(block)}: nMAE={block['mem_nMAE'].mean():.4f}±{block['mem_nMAE'].std():.4f}  "
            f"NCC={block['mem_NCC'].mean():.4f}±{block['mem_NCC'].std():.4f}  "
            f"edge={block['mem_edge_sym_dist'].mean():.3f}±{block['mem_edge_sym_dist'].std():.3f}"
        )

    print("\nDone.")
    return df_reg, df_base


if __name__ == "__main__":
    main()
