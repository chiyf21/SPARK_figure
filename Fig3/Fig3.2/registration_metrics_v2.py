#!/usr/bin/env python3
"""
Registration metrics v2 — F260517 dataset.

Additions over v1:
    1. Baseline metrics: mov vs. ref sampled at z_init (identity XY, no warp).
       Same ref_mem_adj sequence as registered version — fair comparison.
    2. Intensity-normalised metrics (nMAE, nRMSE) per plane, using mov's
       robust dynamic range (P99 - P1) as denominator.
    3. Ref update every 20 frames (same as v1).

Output:
    registration_metrics_ref20_v2.csv    — registered metrics (200 rows)
    baseline_metrics_ref20_v2.csv        — z_init-only baseline metrics (200 rows)
"""

import os
import sys
import time
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

# ---------------------------------------------------------------------------
# Reference intensity mapping helper
# ---------------------------------------------------------------------------
def update_reference_intensity_mapping_from_target_stack(
    F260517_ref_mem, target_stack_zyx, z_idx, option,
    thresFactor, maskRange, smoothPenalty_raw, percentiles,
):
    target_stack_zyx = np.asarray(target_stack_zyx, dtype=np.float32)
    if target_stack_zyx.shape[0] != len(z_idx):
        raise ValueError(f"K mismatch: {target_stack_zyx.shape[0]} vs {len(z_idx)}")

    ref_source_zyx = F260517_ref_mem[z_idx].astype(np.float32, copy=False)
    src_q, tgt_q, used_percentiles = prep.learn_quantile_mapping(
        source=ref_source_zyx, target=target_stack_zyx, percentiles=percentiles,
    )
    F260517_ref_mem_adj = prep.apply_quantile_mapping(
        F260517_ref_mem, src_q, tgt_q,
    ).transpose(2, 1, 0).astype(np.float32, copy=False)

    option["mask_ref"] = mask.getMask(F260517_ref_mem_adj, thresFactor)
    option["mask_ref"] = mask.bwareafilt3_wei(option["mask_ref"], maskRange)
    Pnltfactor = prep.getSmPnltNormFctr(F260517_ref_mem_adj, option)
    option["smoothPenalty"] = Pnltfactor * smoothPenalty_raw

    return F260517_ref_mem_adj, src_q, tgt_q, used_percentiles


# ---------------------------------------------------------------------------
# Edge detection
# ---------------------------------------------------------------------------
def sobel_edge_magnitude(plane_2d):
    from scipy.ndimage import sobel
    p = np.asarray(plane_2d, dtype=np.float32)
    p = p - p.min()
    denom = p.max() - p.min()
    if denom > 1e-8:
        p = p / denom
    gx = sobel(p, axis=-1, mode="nearest")
    gy = sobel(p, axis=-2, mode="nearest")
    return np.sqrt(gx ** 2 + gy ** 2 + 1e-8).astype(np.float32)


def binarize_edge_magnitude(edge_mag, percentile=90):
    thresh = np.percentile(edge_mag, percentile)
    return (edge_mag >= thresh).astype(np.uint8)


def symmetric_edge_distance_2d(edges_a, edges_b):
    if not np.any(edges_a) or not np.any(edges_b):
        return np.nan
    dt_b = distance_transform_edt(1 - edges_b)
    dt_a = distance_transform_edt(1 - edges_a)
    return float(0.5 * (np.mean(dt_b[edges_a > 0]) + np.mean(dt_a[edges_b > 0])))


# ---------------------------------------------------------------------------
# NCC
# ---------------------------------------------------------------------------
def zncc_2d(a, b, mask=None, eps=1e-8):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        ma = np.mean(a[m])
        mb = np.mean(b[m])
        numer = np.sum((a - ma) * (b - mb) * m)
        denom_a = np.sqrt(np.sum((a - ma) ** 2 * m) + eps)
        denom_b = np.sqrt(np.sum((b - mb) ** 2 * m) + eps)
    else:
        a_c = a - np.mean(a)
        b_c = b - np.mean(b)
        numer = np.sum(a_c * b_c)
        denom_a = np.sqrt(np.sum(a_c * a_c) + eps)
        denom_b = np.sqrt(np.sum(b_c * b_c) + eps)
    if denom_a < eps or denom_b < eps:
        return np.nan
    return float(numer / (denom_a * denom_b))


# ---------------------------------------------------------------------------
# Sparse-cell centroid metrics
# ---------------------------------------------------------------------------
def sparse_cell_centroid_metrics_2d(
    mov_plane, mapped_plane, thresh_factor=3.0, match_radius=5.0,
):
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
        dists_m2p[i] = np.min(np.sqrt(np.sum((cents_mapped - c) ** 2, axis=1)))

    dists_p2m = np.full(n_mapped, np.nan, dtype=np.float32)
    for i, c in enumerate(cents_mapped):
        dists_p2m[i] = np.min(np.sqrt(np.sum((cents_mov - c) ** 2, axis=1)))

    nn_dist = float(0.5 * (np.nanmean(dists_m2p) + np.nanmean(dists_p2m)))
    recall = float(np.mean(dists_m2p <= match_radius))
    precision = float(np.mean(dists_p2m <= match_radius))
    return nn_dist, recall, precision, n_mov, n_mapped


# ---------------------------------------------------------------------------
# Per-frame metrics computer  (shared by registered & baseline)
# ---------------------------------------------------------------------------
def compute_frame_metrics(
    mov_mem_zyx,
    mem_mapped_zyx,
    mov_sparse_zyx,
    sparse_mapped_zyx,
    mask_mov_zyx=None,
    edge_percentile=90,
    sparse_thresh_factor=3.0,
):
    K = mov_mem_zyx.shape[0]
    out = {
        "mem_MAE": [], "mem_MSE": [], "mem_NCC": [],
        "mem_edge_sym_dist": [],
        "mem_nMAE": [], "mem_nRMSE": [],
        "sparse_MAE": [], "sparse_MSE": [],
        "sparse_nMAE": [], "sparse_nRMSE": [],
        "sparse_centroid_NN": [],
        "sparse_recall": [], "sparse_precision": [],
    }

    for k in range(K):
        m_mov = mov_mem_zyx[k]
        m_map = mem_mapped_zyx[k]
        s_mov = mov_sparse_zyx[k]
        s_map = sparse_mapped_zyx[k]

        valid = np.ones_like(m_mov, dtype=bool)
        if mask_mov_zyx is not None:
            valid = mask_mov_zyx[k].astype(bool)
            if not np.any(valid):
                valid = np.ones_like(m_mov, dtype=bool)  # fallback if mask empty

        # -- robust dynamic range for normalisation (from mov only) ---------
        p1, p99 = np.percentile(m_mov[valid], [1, 99])
        dyn_range = max(p99 - p1, 1e-8)
        # also for sparse:
        sp1, sp99 = np.percentile(s_mov[valid], [1, 99])
        s_dyn_range = max(sp99 - sp1, 1e-8)

        # -- membrane MAE / MSE / nMAE / nRMSE ------------------------------
        diff_m = np.abs(m_mov.astype(np.float32) - m_map.astype(np.float32))
        sq_m = diff_m ** 2
        mae_m = float(np.mean(diff_m[valid]))
        mse_m = float(np.mean(sq_m[valid]))

        out["mem_MAE"].append(mae_m)
        out["mem_MSE"].append(mse_m)
        out["mem_nMAE"].append(mae_m / dyn_range)                     # fraction of dynamic range
        out["mem_nRMSE"].append(np.sqrt(mse_m) / dyn_range)

        # -- membrane NCC --------------------------------------------------
        out["mem_NCC"].append(zncc_2d(m_mov, m_map, mask=valid))

        # -- membrane edge distance ----------------------------------------
        e_mov = sobel_edge_magnitude(m_mov)
        e_map = sobel_edge_magnitude(m_map)
        b_mov = binarize_edge_magnitude(e_mov, percentile=edge_percentile)
        b_map = binarize_edge_magnitude(e_map, percentile=edge_percentile)
        out["mem_edge_sym_dist"].append(symmetric_edge_distance_2d(b_mov, b_map))

        # -- sparse MAE / MSE / nMAE / nRMSE --------------------------------
        diff_s = np.abs(s_mov.astype(np.float32) - s_map.astype(np.float32))
        sq_s = diff_s ** 2
        mae_s = float(np.mean(diff_s[valid]))
        mse_s = float(np.mean(sq_s[valid]))

        out["sparse_MAE"].append(mae_s)
        out["sparse_MSE"].append(mse_s)
        out["sparse_nMAE"].append(mae_s / s_dyn_range)
        out["sparse_nRMSE"].append(np.sqrt(mse_s) / s_dyn_range)

        # -- sparse centroid ------------------------------------------------
        nn_d, recall, precision, _, _ = sparse_cell_centroid_metrics_2d(
            s_mov, s_map, thresh_factor=sparse_thresh_factor, match_radius=5.0,
        )
        out["sparse_centroid_NN"].append(nn_d)
        out["sparse_recall"].append(recall)
        out["sparse_precision"].append(precision)

    # aggregate over K planes (nanmean handles missing centroids etc.)
    agg = {}
    for key, vals in out.items():
        agg[key] = float(np.nanmean(vals))
    return agg


# ===========================================================================
# Main pipeline
# ===========================================================================
def main():
    print("=" * 80)
    print("Registration Metrics Pipeline v2 — F260517")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 80)

    # -------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------
    print("\n[1/6] Loading data ...")
    t0 = time.time()
    F260517_mov, _ = IO.readTiff(F260517_mov_path)
    F260517_ref, _ = IO.readTiff(F260517_ref_path)

    F260517_ref_mem = F260517_ref[90:310, 1, :, :]
    F260517_ref_sparseCell = F260517_ref[90:310, 0, :, :]
    F260517_mov_mem = F260517_mov[:, :, 1, :, :]
    F260517_mov_sparseCell = F260517_mov[:, :, 0, :, :]

    print(f"  ref_mem:      {F260517_ref_mem.shape}")
    print(f"  mov_mem:      {F260517_mov_mem.shape}")
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

    # z_init
    z_init = calFlowCrossResolution.FindInitZ_stack_global_fixed_spacing(
        F260517_mov_mem[0].transpose(2, 1, 0),
        F260517_ref_mem.transpose(2, 1, 0),
        delta_ref_idx=10, use_gradient=False,
    )
    z_init = np.asarray(z_init, dtype=np.float32)
    z_idx = np.rint(z_init).astype(np.int32)
    z_idx = np.clip(z_idx, 0, F260517_ref_mem.shape[0] - 1)

    K = int(z_init.shape[0])
    T = int(F260517_mov_mem.shape[0])

    # Identity phase (XY identity + z_init Z) — used for baseline
    x = np.arange(F260517_mov_mem[0].shape[2], dtype=np.float32)
    y = np.arange(F260517_mov_mem[0].shape[1], dtype=np.float32)
    k = np.arange(K, dtype=np.int32)
    X_grid, Y_grid, K_grid = np.meshgrid(x, y, k, indexing="ij")
    coords_xyz = np.empty((len(x), len(y), K, 3), dtype=np.float32)
    coords_xyz[..., 0] = X_grid
    coords_xyz[..., 1] = Y_grid
    coords_xyz[..., 2] = z_init[K_grid]

    option["phase"] = coords_xyz.copy()

    print(f"  K={K}, T={T}, z_init={z_init}")

    # -------------------------------------------------------------------
    # 3. Initial reference intensity mapping
    # -------------------------------------------------------------------
    print("\n[3/6] Initial reference intensity mapping ...")
    init_target_stack = np.mean(
        F260517_mov_mem[[0, 1, 2, 3, 4]].astype(np.float32), axis=0,
    )
    F260517_ref_mem_adj, _, _, _ = update_reference_intensity_mapping_from_target_stack(
        F260517_ref_mem=F260517_ref_mem, target_stack_zyx=init_target_stack,
        z_idx=z_idx, option=option, thresFactor=thresFactor,
        maskRange=maskRange, smoothPenalty_raw=smoothPenalty_raw,
        percentiles=percentiles,
    )
    print("  Done.")

    # -------------------------------------------------------------------
    # GPU pre-allocations
    # -------------------------------------------------------------------
    ref_sparse_gpu = cp.asarray(
        F260517_ref_sparseCell.transpose(2, 1, 0).astype(np.float32, copy=False)
    )  # (X, Y, Zref)

    def sample_ref_sparse(phase):
        """Sample reference sparse-cell at phase coordinates, return (K,Y,X)."""
        ph = phase if hasattr(phase, "get") else cp.asarray(phase, dtype=cp.float32)
        H = generate_continuous_H_gpu(ref_sparse_gpu, zRatio=1)
        sm = apply_H_to_matrix_gpu(ph, H)
        if hasattr(sm, "get"):
            sm = sm.get()
        return np.asarray(sm, dtype=np.float32).transpose(2, 1, 0)  # (K,Y,X)

    def sample_ref_mem(phase, ref_xyz_gpu):
        """Sample reference membrane at phase coordinates, return (K,Y,X)."""
        ph = phase if hasattr(phase, "get") else cp.asarray(phase, dtype=cp.float32)
        H = generate_continuous_H_gpu(ref_xyz_gpu, zRatio=1)
        mp = apply_H_to_matrix_gpu(ph, H)
        if hasattr(mp, "get"):
            mp = mp.get()
        return np.asarray(mp, dtype=np.float32).transpose(2, 1, 0)  # (K,Y,X)

    # -------------------------------------------------------------------
    # 4. Registration loop + baseline
    # -------------------------------------------------------------------
    print("\n[4/6] Starting loop (registration + baseline) ...")
    print("      Ref update: every 20 frames")

    registered_records = []
    baseline_records = []

    registered_mem_mapped_cache = {}
    frames_since_ref_update = 0
    ref_update_id = 0

    def make_ref_mem_gpu():
        return cp.asarray(F260517_ref_mem_adj, dtype=cp.float32)  # (X,Y,Zref)

    def update_ref_from_recent(frames):
        stacks = [registered_mem_mapped_cache[fi] for fi in frames if fi in registered_mem_mapped_cache]
        if len(stacks) == 0:
            return None
        target = np.mean(np.stack(stacks, axis=0), axis=0).astype(np.float32, copy=False)
        return update_reference_intensity_mapping_from_target_stack(
            F260517_ref_mem=F260517_ref_mem, target_stack_zyx=target,
            z_idx=z_idx, option=option, thresFactor=thresFactor,
            maskRange=maskRange, smoothPenalty_raw=smoothPenalty_raw,
            percentiles=percentiles,
        )

    option["phase"] = coords_xyz.copy()
    option.pop("motion", None)

    total_start = time.time()

    for i in range(T):
        frame_start = time.time()

        # ---- Registration ------------------------------------------------
        if i == 0:
            option["phase"] = coords_xyz.copy()
            option.pop("motion", None)

        raw_mem_zyx = F260517_mov_mem[i].astype(np.float32, copy=False)
        raw_sparse_zyx = F260517_mov_sparseCell[i].astype(np.float32, copy=False)
        mov_mem_xyk = raw_mem_zyx.transpose(2, 1, 0).astype(np.float32, copy=False)  # (X,Y,K)

        option["mask_mov"] = mask.getMask(mov_mem_xyk, thresFactor)
        option["mask_mov"] = mask.bwareafilt3_wei(option["mask_mov"], maskRange)

        phase_new, motion_current, mem_mapped_xyk = calFlowCrossResolution.getMotion_v2(
            mov_mem_xyk, F260517_ref_mem_adj, option, verbose=False,
        )
        if hasattr(phase_new, "get"):   phase_new = phase_new.get()
        if hasattr(motion_current, "get"): motion_current = motion_current.get()
        if hasattr(mem_mapped_xyk, "get"): mem_mapped_xyk = mem_mapped_xyk.get()
        phase_new = phase_new.astype(np.float32, copy=False)
        motion_current = motion_current.astype(np.float32, copy=False)

        mem_mapped_zyx_reg = np.asarray(mem_mapped_xyk, dtype=np.float32).transpose(2, 1, 0)  # (K,Y,X)
        sparse_mapped_zyx_reg = sample_ref_sparse(cp.asarray(phase_new))

        registered_mem_mapped_cache[i] = mem_mapped_zyx_reg

        # ---- Baseline: identity phase + z_init ---------------------------
        baseline_mem_zyx = sample_ref_mem(
            cp.asarray(coords_xyz, dtype=cp.float32), make_ref_mem_gpu()
        )
        baseline_sparse_zyx = sample_ref_sparse(cp.asarray(coords_xyz, dtype=cp.float32))

        # ---- Mov mask for metrics -----------------------------------------
        mask_mov = option["mask_mov"]
        if hasattr(mask_mov, "get"): mask_mov = mask_mov.get()
        mask_mov_zyx = np.asarray(mask_mov, dtype=bool).transpose(2, 1, 0)

        # ---- Compute metrics (registered & baseline) ----------------------
        reg_metrics = compute_frame_metrics(
            mov_mem_zyx=raw_mem_zyx, mem_mapped_zyx=mem_mapped_zyx_reg,
            mov_sparse_zyx=raw_sparse_zyx, sparse_mapped_zyx=sparse_mapped_zyx_reg,
            mask_mov_zyx=mask_mov_zyx,
        )
        base_metrics = compute_frame_metrics(
            mov_mem_zyx=raw_mem_zyx, mem_mapped_zyx=baseline_mem_zyx,
            mov_sparse_zyx=raw_sparse_zyx, sparse_mapped_zyx=baseline_sparse_zyx,
            mask_mov_zyx=mask_mov_zyx,
        )

        elapsed = time.time() - frame_start

        reg_rec = {"frame": i, "ref_update_id": ref_update_id, "elapsed_s": elapsed}
        reg_rec.update(reg_metrics)
        registered_records.append(reg_rec)

        base_rec = {"frame": i, "ref_update_id": ref_update_id, "elapsed_s": elapsed}
        base_rec.update(base_metrics)
        baseline_records.append(base_rec)

        # ---- Log ----------------------------------------------------------
        print(
            f"[Frame {i:03d}/{T-1:03d}] "
            f"reg:  NCC={reg_metrics['mem_NCC']:.4f} nMAE={reg_metrics['mem_nMAE']:.4f} "
            f"edge={reg_metrics['mem_edge_sym_dist']:.3f}px "
            f"spR={reg_metrics['sparse_recall']:.3f} | "
            f"base: NCC={base_metrics['mem_NCC']:.4f} nMAE={base_metrics['mem_nMAE']:.4f} "
            f"edge={base_metrics['mem_edge_sym_dist']:.3f}px "
            f"{elapsed:.1f}s"
        )

        # ---- Temporal init for next frame ---------------------------------
        option["motion"] = (0.7 * motion_current).astype(np.float32, copy=False)

        # ---- Ref update ---------------------------------------------------
        frames_since_ref_update += 1
        if frames_since_ref_update >= ref_update_every:
            calib_frames = sorted(registered_mem_mapped_cache.keys())[-5:]
            ref_update_id += 1
            print(f"\n  >>> Ref Update #{ref_update_id} — frames {calib_frames}")
            new_ref = update_ref_from_recent(calib_frames)
            if new_ref is not None:
                F260517_ref_mem_adj = new_ref[0]
            frames_since_ref_update = 0
            print()

    total_elapsed = time.time() - total_start
    print(f"\n  Done in {total_elapsed:.1f}s ({total_elapsed / T:.1f}s/frame)")

    # -------------------------------------------------------------------
    # 5. Save CSVs
    # -------------------------------------------------------------------
    print("\n[5/6] Saving CSVs ...")
    df_reg = pd.DataFrame(registered_records)
    df_base = pd.DataFrame(baseline_records)

    csv_reg = OUT_DIR / "registration_metrics_ref20_v2.csv"
    csv_base = OUT_DIR / "baseline_metrics_ref20_v2.csv"
    df_reg.to_csv(str(csv_reg), index=False)
    df_base.to_csv(str(csv_base), index=False)
    print(f"  Registered: {len(df_reg)} rows → {csv_reg}")
    print(f"  Baseline:   {len(df_base)} rows → {csv_base}")

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
            if col not in df.columns:
                continue
            vals = df[col].dropna()
            print(f"    {col:25s}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
                  f"min={vals.min():.4f}  max={vals.max():.4f}")

    # Per-ref-update-block for registered
    print("\n  --- Registered: per-ref-update-block ---")
    for ruid in sorted(df_reg["ref_update_id"].unique()):
        block = df_reg[df_reg["ref_update_id"] == ruid]
        print(
            f"    ru{ruid:02d} frames [{block['frame'].min():.0f}-{block['frame'].max():.0f}]: "
            f"nMAE={block['mem_nMAE'].mean():.4f}±{block['mem_nMAE'].std():.4f}  "
            f"NCC={block['mem_NCC'].mean():.4f}±{block['mem_NCC'].std():.4f}  "
            f"edge={block['mem_edge_sym_dist'].mean():.3f}±{block['mem_edge_sym_dist'].std():.3f}"
        )

    print("\nDone.")
    return df_reg, df_base


if __name__ == "__main__":
    main()
