#!/usr/bin/env python3
"""
Registration metrics evaluation for F260517 dataset.

Performs forward-only registration (no projection), computes per-frame quality
metrics for both membrane and sparse-cell channels, with reference intensity
re-calibration every 20 frames.

Metrics computed per frame:
    - mem_MAE:              Mean absolute error |mov_mem - mem_mapped|
    - mem_edge_sym_dist:    Mean symmetric edge distance (pixels)
    - sparse_MAE:           Mean absolute error |mov_sparse - sparse_mapped|
    - sparse_centroid_NN:   Mean nearest-neighbor distance between centroids (pixels)

Output:
    metrics.csv  — one row per frame, columns:
        frame, ref_update_id, mem_MAE, mem_edge_sym_dist, sparse_MAE,
        sparse_centroid_NN, mem_MSE, sparse_MSE
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

NOTEBOOK_ROOT = Path("/home/cyf/wbi/Virginia/code/wbi_0123/wholistic_registration/src/wholistic_registration")
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
# Helper: reference intensity mapping (from notebook)
# ---------------------------------------------------------------------------
def update_reference_intensity_mapping_from_target_stack(
    F260517_ref_mem,
    target_stack_zyx,
    z_idx,
    option,
    thresFactor,
    maskRange,
    smoothPenalty_raw,
    percentiles,
):
    """Learn reference -> target intensity mapping, apply to full reference."""
    target_stack_zyx = np.asarray(target_stack_zyx, dtype=np.float32)

    if target_stack_zyx.shape[0] != len(z_idx):
        raise ValueError(
            f"target_stack_zyx K={target_stack_zyx.shape[0]} != z_idx len={len(z_idx)}"
        )

    ref_source_zyx = F260517_ref_mem[z_idx].astype(np.float32, copy=False)

    src_q, tgt_q, used_percentiles = prep.learn_quantile_mapping(
        source=ref_source_zyx,
        target=target_stack_zyx,
        percentiles=percentiles,
    )

    F260517_ref_mem_adj = prep.apply_quantile_mapping(
        F260517_ref_mem, src_q, tgt_q,
    ).transpose(2, 1, 0).astype(np.float32, copy=False)  # (X, Y, Zref)

    option["mask_ref"] = mask.getMask(F260517_ref_mem_adj, thresFactor)
    option["mask_ref"] = mask.bwareafilt3_wei(option["mask_ref"], maskRange)

    Pnltfactor = prep.getSmPnltNormFctr(F260517_ref_mem_adj, option)
    option["smoothPenalty"] = Pnltfactor * smoothPenalty_raw

    return F260517_ref_mem_adj, src_q, tgt_q, used_percentiles


# ---------------------------------------------------------------------------
# Edge detection (self-contained, per 2D plane)
# ---------------------------------------------------------------------------
def sobel_edge_magnitude(plane_2d):
    """
    Return per-pixel edge magnitude for one 2D plane (Y, X) using Sobel filters.
    The input is normalised to [0,1] internally.
    """
    from scipy.ndimage import sobel

    p = np.asarray(plane_2d, dtype=np.float32)
    p = p - p.min()
    denom = p.max() - p.min()
    if denom > 1e-8:
        p = p / denom

    gx = sobel(p, axis=-1, mode="nearest")
    gy = sobel(p, axis=-2, mode="nearest")
    mag = np.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    return mag.astype(np.float32)


def binarize_edge_magnitude(edge_mag, percentile=90):
    """Keep the top (100-percentile)% edge pixels as binary edge map."""
    thresh = np.percentile(edge_mag, percentile)
    return (edge_mag >= thresh).astype(np.uint8)


def symmetric_edge_distance_2d(edges_a, edges_b):
    """
    Mean symmetric edge distance (pixels) between two binary edge maps.
    For each edge pixel in A, compute distance to nearest edge in B, and vice versa.
    """
    if not np.any(edges_a) or not np.any(edges_b):
        return np.nan

    dt_b = distance_transform_edt(1 - edges_b)  # distance to nearest B edge
    dt_a = distance_transform_edt(1 - edges_a)  # distance to nearest A edge

    dist_a_to_b = dt_b[edges_a > 0]
    dist_b_to_a = dt_a[edges_b > 0]

    return float(0.5 * (np.mean(dist_a_to_b) + np.mean(dist_b_to_a)))


# ---------------------------------------------------------------------------
# NCC (Normalised Cross-Correlation) — per 2D plane, intensity-invariant
# ---------------------------------------------------------------------------
def zncc_2d(a, b, mask=None, eps=1e-8):
    """
    Zero-mean normalised cross-correlation (ZNCC) between two 2D arrays.

    Range: [-1, 1].  1 = perfect match.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        a = a.copy()
        b = b.copy()
        a[~m] = 0.0
        b[~m] = 0.0
        denom_a = np.sqrt(np.sum((a - np.mean(a[m])) ** 2 * m) + eps)
        denom_b = np.sqrt(np.sum((b - np.mean(b[m])) ** 2 * m) + eps)
        numer = np.sum((a - np.mean(a[m])) * (b - np.mean(b[m])) * m)
    else:
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        a_c = a - a_mean
        b_c = b - b_mean
        numer = np.sum(a_c * b_c)
        denom_a = np.sqrt(np.sum(a_c * a_c) + eps)
        denom_b = np.sqrt(np.sum(b_c * b_c) + eps)

    if denom_a < eps or denom_b < eps:
        return np.nan
    return float(numer / (denom_a * denom_b))


# ---------------------------------------------------------------------------
# Sparse-cell centroid NN distance (per 2D plane)
# ---------------------------------------------------------------------------
def sparse_cell_centroid_metrics_2d(mov_plane, mapped_plane, thresh_factor=3.0,
                                     match_radius=5.0):
    """
    Sparse-cell centroid metrics between bright spots in mov_plane and mapped_plane.

    Bright spots are defined as pixels > mean + thresh_factor * std.

    Returns
    -------
    nn_dist : float or nan
        Mean symmetric nearest-neighbour distance (pixels).
    recall : float or nan
        Fraction of mov centroids matched by a mapped centroid within match_radius.
    precision : float or nan
        Fraction of mapped centroids matched by a mov centroid within match_radius.
    n_mov, n_mapped : int
        Raw centroid counts (for debugging).
    """
    mov = np.asarray(mov_plane, dtype=np.float32)
    mapped = np.asarray(mapped_plane, dtype=np.float32)

    def get_centroids(img):
        mu, sigma = float(np.mean(img)), float(np.std(img))
        bin_mask = img > (mu + thresh_factor * sigma)
        if not np.any(bin_mask):
            return np.empty((0, 2), dtype=np.float32)
        labeled, nlab = label_ndi(bin_mask)
        props = regionprops(labeled)
        cents = np.array([p.centroid for p in props], dtype=np.float32)  # (N, 2), (y, x)
        return cents

    cents_mov = get_centroids(mov)
    cents_mapped = get_centroids(mapped)
    n_mov = len(cents_mov)
    n_mapped = len(cents_mapped)

    if n_mov == 0 or n_mapped == 0:
        return np.nan, np.nan, np.nan, n_mov, n_mapped

    # Nearest-neighbour distances: mov -> mapped
    dists_mov_to_mapped = np.full(n_mov, np.nan, dtype=np.float32)
    for i, c in enumerate(cents_mov):
        d = np.sqrt(np.sum((cents_mapped - c) ** 2, axis=1))
        dists_mov_to_mapped[i] = np.min(d)

    # mapped -> mov
    dists_mapped_to_mov = np.full(n_mapped, np.nan, dtype=np.float32)
    for i, c in enumerate(cents_mapped):
        d = np.sqrt(np.sum((cents_mov - c) ** 2, axis=1))
        dists_mapped_to_mov[i] = np.min(d)

    nn_dist = float(0.5 * (np.nanmean(dists_mov_to_mapped) + np.nanmean(dists_mapped_to_mov)))

    recall = float(np.mean(dists_mov_to_mapped <= match_radius))
    precision = float(np.mean(dists_mapped_to_mov <= match_radius))

    return nn_dist, recall, precision, n_mov, n_mapped


# ---------------------------------------------------------------------------
# Per-frame metrics computer
# ---------------------------------------------------------------------------
def compute_frame_metrics(
    mov_mem_zyx,          # (K, Y, X)
    mem_mapped_zyx,       # (K, Y, X)
    mov_sparse_zyx,       # (K, Y, X)
    sparse_mapped_zyx,    # (K, Y, X)
    mask_mov_zyx=None,    # optional (K, Y, X) bool mask
    edge_percentile=90,
    sparse_thresh_factor=3.0,
):
    """
    Compute all registration quality metrics for one frame.

    Returns dict with scalar means over all K planes.
    """
    K = mov_mem_zyx.shape[0]

    mem_mae_planes = []
    mem_mse_planes = []
    mem_ncc_planes = []
    edge_sym_planes = []
    sparse_mae_planes = []
    sparse_mse_planes = []
    sparse_nn_planes = []
    sparse_recall_planes = []
    sparse_precision_planes = []

    for k in range(K):
        mov_m = mov_mem_zyx[k]
        mapped_m = mem_mapped_zyx[k]
        mov_s = mov_sparse_zyx[k]
        mapped_s = sparse_mapped_zyx[k]

        # Valid mask for this plane
        if mask_mov_zyx is not None:
            valid = mask_mov_zyx[k].astype(bool)
        else:
            valid = np.ones_like(mov_m, dtype=bool)

        # -- membrane MAE / MSE -------------------------------------------------
        diff_m = np.abs(mov_m.astype(np.float32) - mapped_m.astype(np.float32))
        mem_mae_planes.append(float(np.mean(diff_m[valid])))

        sq_m = (mov_m.astype(np.float32) - mapped_m.astype(np.float32)) ** 2
        mem_mse_planes.append(float(np.mean(sq_m[valid])))

        # -- membrane NCC (intensity-invariant) ----------------------------------
        mem_ncc_planes.append(zncc_2d(mov_m, mapped_m, mask=valid))

        # -- membrane symmetric edge distance -----------------------------------
        edge_mov = sobel_edge_magnitude(mov_m)
        edge_mapped = sobel_edge_magnitude(mapped_m)
        bin_mov = binarize_edge_magnitude(edge_mov, percentile=edge_percentile)
        bin_mapped = binarize_edge_magnitude(edge_mapped, percentile=edge_percentile)
        edge_sym_planes.append(symmetric_edge_distance_2d(bin_mov, bin_mapped))

        # -- sparse cell MAE / MSE -----------------------------------------------
        diff_s = np.abs(mov_s.astype(np.float32) - mapped_s.astype(np.float32))
        sparse_mae_planes.append(float(np.mean(diff_s[valid])))

        sq_s = (mov_s.astype(np.float32) - mapped_s.astype(np.float32)) ** 2
        sparse_mse_planes.append(float(np.mean(sq_s[valid])))

        # -- sparse cell centroid metrics ---------------------------------------
        nn_d, recall, precision, n_mov_cells, n_mapped_cells = \
            sparse_cell_centroid_metrics_2d(
                mov_s, mapped_s, thresh_factor=sparse_thresh_factor, match_radius=5.0,
            )
        sparse_nn_planes.append(nn_d)
        sparse_recall_planes.append(recall)
        sparse_precision_planes.append(precision)

    return {
        "mem_MAE": float(np.nanmean(mem_mae_planes)),
        "mem_MSE": float(np.nanmean(mem_mse_planes)),
        "mem_NCC": float(np.nanmean(mem_ncc_planes)),
        "mem_edge_sym_dist": float(np.nanmean(edge_sym_planes)),
        "sparse_MAE": float(np.nanmean(sparse_mae_planes)),
        "sparse_MSE": float(np.nanmean(sparse_mse_planes)),
        "sparse_centroid_NN": float(np.nanmean(sparse_nn_planes)),
        "sparse_recall": float(np.nanmean(sparse_recall_planes)),
        "sparse_precision": float(np.nanmean(sparse_precision_planes)),
        # per-plane arrays (kept for debugging if needed)
        "mem_MAE_per_plane": mem_mae_planes,
        "mem_NCC_per_plane": mem_ncc_planes,
        "mem_edge_sym_per_plane": edge_sym_planes,
        "sparse_MAE_per_plane": sparse_mae_planes,
        "sparse_centroid_NN_per_plane": sparse_nn_planes,
        "sparse_recall_per_plane": sparse_recall_planes,
        "sparse_precision_per_plane": sparse_precision_planes,
    }


# ===========================================================================
# Main pipeline
# ===========================================================================
def main():
    print("=" * 80)
    print("Registration Metrics Pipeline — F260517")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    print("\n[1/6] Loading data ...")
    t0 = time.time()

    F260517_mov, F260517_mov_desc = IO.readTiff(F260517_mov_path)
    F260517_ref, F260517_ref_desc = IO.readTiff(F260517_ref_path)

    F260517_ref_mem = F260517_ref[90:310, 1, :, :]          # (Zref, Y, X)
    F260517_ref_sparseCell = F260517_ref[90:310, 0, :, :]   # (Zref, Y, X)

    F260517_mov_mem = F260517_mov[:, :, 1, :, :]             # (T, K, Y, X)
    F260517_mov_sparseCell = F260517_mov[:, :, 0, :, :]       # (T, K, Y, X)

    print(f"  ref_mem:          {F260517_ref_mem.shape}")
    print(f"  ref_sparseCell:   {F260517_ref_sparseCell.shape}")
    print(f"  mov_mem:          {F260517_mov_mem.shape}")
    print(f"  mov_sparseCell:   {F260517_mov_sparseCell.shape}")
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    # -----------------------------------------------------------------------
    # 2. Setup options & initial z_init
    # -----------------------------------------------------------------------
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

    # Compute initial z
    z_init = calFlowCrossResolution.FindInitZ_stack_global_fixed_spacing(
        F260517_mov_mem[0].transpose(2, 1, 0),     # (X, Y, K)
        F260517_ref_mem.transpose(2, 1, 0),         # (X, Y, Zref)
        delta_ref_idx=10,
        use_gradient=False,
    )
    z_init = np.asarray(z_init, dtype=np.float32)
    z_idx = np.rint(z_init).astype(np.int32)
    z_idx = np.clip(z_idx, 0, F260517_ref_mem.shape[0] - 1)

    K = z_init.shape[0]
    T = F260517_mov_mem.shape[0]

    # Build initial coordinate grid
    x = np.arange(F260517_mov_mem[0].shape[2], dtype=np.float32)
    y = np.arange(F260517_mov_mem[0].shape[1], dtype=np.float32)
    k = np.arange(K, dtype=np.int32)
    X_grid, Y_grid, K_grid = np.meshgrid(x, y, k, indexing="ij")

    coords_xyz = np.empty(
        (F260517_mov_mem[0].shape[2], F260517_mov_mem[0].shape[1], K, 3),
        dtype=np.float32,
    )
    coords_xyz[..., 0] = X_grid
    coords_xyz[..., 1] = Y_grid
    coords_xyz[..., 2] = z_init[K_grid]

    option["phase"] = coords_xyz

    print(f"  K (moving slices): {K}")
    print(f"  T (total frames):  {T}")
    print(f"  z_init:            {z_init}")

    # Quantile mapping percentiles
    percentiles = [
        0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.8,
    ]

    # -----------------------------------------------------------------------
    # 3. Initial reference intensity mapping (using mean of frames 0-4)
    # -----------------------------------------------------------------------
    print("\n[3/6] Initial reference intensity mapping ...")
    init_calibration_frames = [0, 1, 2, 3, 4]
    init_target_stack = np.mean(
        F260517_mov_mem[init_calibration_frames].astype(np.float32), axis=0
    )  # (K, Y, X)

    F260517_ref_mem_adj, src_q, tgt_q, used_percentiles = (
        update_reference_intensity_mapping_from_target_stack(
            F260517_ref_mem=F260517_ref_mem,
            target_stack_zyx=init_target_stack,
            z_idx=z_idx,
            option=option,
            thresFactor=thresFactor,
            maskRange=maskRange,
            smoothPenalty_raw=smoothPenalty_raw,
            percentiles=percentiles,
        )
    )
    print(f"  Initial ref mapping done.")

    # -----------------------------------------------------------------------
    # 4. Registration loop
    # -----------------------------------------------------------------------
    print("\n[4/6] Starting registration loop ...")
    print("      Ref update interval: 20 frames")

    # --- State ---
    registered_mem_mapped = {}   # frame_idx -> (K, Y, X)
    metrics_records = []

    ref_update_every = 20       # <-- changed from 40 to 20
    frames_since_ref_update = 0
    ref_update_id = 0           # increments each time we update the reference

    # Pre-build sparse-cell reference on GPU for sampling
    ref_sparse_xyz = cp.asarray(
        F260517_ref_sparseCell.transpose(2, 1, 0).astype(np.float32, copy=False)
    )  # (X, Y, Zref)

    def get_sparse_mapped(phase_new):
        """Sample reference sparse-cell volume at phase coordinates."""
        if hasattr(phase_new, "get"):
            phase = phase_new
        else:
            phase = cp.asarray(phase_new, dtype=cp.float32)
        H_ref_sparse = generate_continuous_H_gpu(ref_sparse_xyz, zRatio=1)
        sparse_mapped = apply_H_to_matrix_gpu(phase, H_ref_sparse)
        if hasattr(sparse_mapped, "get"):
            return cp.asnumpy(sparse_mapped).astype(np.float32)
        return np.asarray(sparse_mapped, dtype=np.float32)

    def process_single_frame(i, ref_mem_adj):
        """Register one frame. Returns dict with all relevant arrays."""
        raw_mem_zyx = F260517_mov_mem[i].astype(np.float32, copy=False)
        raw_sparse_zyx = F260517_mov_sparseCell[i].astype(np.float32, copy=False)

        mov_mem_i = raw_mem_zyx.transpose(2, 1, 0).astype(np.float32, copy=False)  # (X, Y, K)

        # Moving mask
        option["mask_mov"] = mask.getMask(mov_mem_i, thresFactor)
        option["mask_mov"] = mask.bwareafilt3_wei(option["mask_mov"], maskRange)

        # Registration
        phase_new, motion_current, mem_mapped = calFlowCrossResolution.getMotion_v2(
            mov_mem_i, ref_mem_adj, option, verbose=False,
        )

        # Convert from cupy if needed
        if hasattr(phase_new, "get"):
            phase_new = phase_new.get()
        if hasattr(motion_current, "get"):
            motion_current = motion_current.get()
        if hasattr(mem_mapped, "get"):
            mem_mapped = mem_mapped.get()

        phase_new = phase_new.astype(np.float32, copy=False)
        motion_current = motion_current.astype(np.float32, copy=False)
        mem_mapped = mem_mapped.astype(np.float32, copy=False)

        # mem_mapped is (X, Y, K) -> convert to (K, Y, X)
        mem_mapped_zyx = mem_mapped.transpose(2, 1, 0).astype(np.float32, copy=False)

        # Sample sparse cell
        sparse_mapped_zyx = get_sparse_mapped(cp.asarray(phase_new))

        # Sparse mapped is (X, Y, K) -> (K, Y, X)
        sparse_mapped_zyx = sparse_mapped_zyx.transpose(2, 1, 0).astype(np.float32, copy=False)

        # Get moving mask in zyx order for metrics
        mov_mask_zyx = option["mask_mov"]
        if hasattr(mov_mask_zyx, "get"):
            mov_mask_zyx = mov_mask_zyx.get()
        mov_mask_zyx = np.asarray(mov_mask_zyx, dtype=bool).transpose(2, 1, 0)

        mem_err = float(np.mean(np.abs(mov_mem_i - mem_mapped)))

        # Temporal initialization for next frame
        option["motion"] = (0.7 * motion_current).astype(np.float32, copy=False)

        return {
            "frame": i,
            "phase_new": phase_new,
            "motion_current": motion_current,
            "mem_mapped_zyx": mem_mapped_zyx,
            "sparse_mapped_zyx": sparse_mapped_zyx,
            "mem_err": mem_err,
            "raw_mem_zyx": raw_mem_zyx,
            "raw_sparse_zyx": raw_sparse_zyx,
            "mov_mask_zyx": mov_mask_zyx,
        }

    def update_ref_from_recent_frames(frame_indices):
        """Calibrate reference intensity using mean of specified registered frames."""
        stacks = []
        for fi in frame_indices:
            if fi in registered_mem_mapped:
                stacks.append(registered_mem_mapped[fi])

        if len(stacks) == 0:
            print("    WARNING: no cached frames, skipping ref update")
            return None

        target_stack = np.mean(np.stack(stacks, axis=0), axis=0).astype(np.float32, copy=False)

        ref_adj, sq, tq, up = update_reference_intensity_mapping_from_target_stack(
            F260517_ref_mem=F260517_ref_mem,
            target_stack_zyx=target_stack,
            z_idx=z_idx,
            option=option,
            thresFactor=thresFactor,
            maskRange=maskRange,
            smoothPenalty_raw=smoothPenalty_raw,
            percentiles=percentiles,
        )
        print(f"    src_q: {sq}")
        print(f"    tgt_q: {tq}")
        return ref_adj

    # --- Initialize for first frame ---
    option["phase"] = coords_xyz.copy()
    option.pop("motion", None)

    total_start = time.time()

    for i in range(T):
        frame_start = time.time()

        # ---- Registration ----
        if i == 0:
            option["phase"] = coords_xyz.copy()
            option.pop("motion", None)

        result = process_single_frame(i, F260517_ref_mem_adj)

        registered_mem_mapped[i] = result["mem_mapped_zyx"]

        # ---- Compute metrics ----
        metrics = compute_frame_metrics(
            mov_mem_zyx=result["raw_mem_zyx"],       # (K, Y, X)
            mem_mapped_zyx=result["mem_mapped_zyx"],  # (K, Y, X)
            mov_sparse_zyx=result["raw_sparse_zyx"],  # (K, Y, X)
            sparse_mapped_zyx=result["sparse_mapped_zyx"],  # (K, Y, X)
            mask_mov_zyx=result["mov_mask_zyx"],
            edge_percentile=90,
            sparse_thresh_factor=3.0,
        )

        record = {
            "frame": i,
            "ref_update_id": ref_update_id,
            "mem_MAE": metrics["mem_MAE"],
            "mem_MSE": metrics["mem_MSE"],
            "mem_NCC": metrics["mem_NCC"],
            "mem_edge_sym_dist": metrics["mem_edge_sym_dist"],
            "sparse_MAE": metrics["sparse_MAE"],
            "sparse_MSE": metrics["sparse_MSE"],
            "sparse_centroid_NN": metrics["sparse_centroid_NN"],
            "sparse_recall": metrics["sparse_recall"],
            "sparse_precision": metrics["sparse_precision"],
            "mem_err_reg": result["mem_err"],
            "elapsed_s": time.time() - frame_start,
        }
        metrics_records.append(record)

        frames_since_ref_update += 1

        # Logging
        print(
            f"[Frame {i:03d}/{T-1:03d}] "
            f"mem_MAE={metrics['mem_MAE']:.4f} | mem_NCC={metrics['mem_NCC']:.4f} | "
            f"mem_edge={metrics['mem_edge_sym_dist']:.3f}px | "
            f"sparse_MAE={metrics['sparse_MAE']:.4f} | "
            f"sparse_NN={metrics['sparse_centroid_NN']:.3f}px | "
            f"sparse_R={metrics['sparse_recall']:.3f} | "
            f"sparse_P={metrics['sparse_precision']:.3f} | "
            f"{record['elapsed_s']:.1f}s"
        )

        # ---- Reference update every 20 frames ----
        if frames_since_ref_update >= ref_update_every:
            calib_frames = sorted(registered_mem_mapped.keys())[-5:]
            ref_update_id += 1
            print(f"\n  >>> Ref Update #{ref_update_id} using frames {calib_frames}")

            new_ref = update_ref_from_recent_frames(calib_frames)
            if new_ref is not None:
                F260517_ref_mem_adj = new_ref
            frames_since_ref_update = 0
            print()

    total_elapsed = time.time() - total_start
    print(f"\n  Registration complete. Total time: {total_elapsed:.1f}s "
          f"({total_elapsed / T:.1f}s/frame)")

    # -----------------------------------------------------------------------
    # 5. Save metrics
    # -----------------------------------------------------------------------
    print("\n[5/6] Saving metrics ...")
    df = pd.DataFrame(metrics_records)
    csv_path = OUT_DIR / "registration_metrics_ref20.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"  Saved {len(df)} rows to {csv_path}")

    # Also save per-plane details for the first few and last few ref-update blocks
    per_plane_rows = []
    for rec in metrics_records:
        # Only save every 5th frame + frames right after ref update to keep file manageable
        fi = rec["frame"]
        if fi % 5 == 0 or fi == 0 or fi == T - 1:
            per_plane_rows.append({
                "frame": fi,
                "ref_update_id": rec["ref_update_id"],
                "mem_MAE": rec["mem_MAE"],
                "mem_MSE": rec["mem_MSE"],
                "mem_NCC": rec["mem_NCC"],
                "mem_edge_sym_dist": rec["mem_edge_sym_dist"],
                "sparse_MAE": rec["sparse_MAE"],
                "sparse_MSE": rec["sparse_MSE"],
                "sparse_centroid_NN": rec["sparse_centroid_NN"],
                "sparse_recall": rec["sparse_recall"],
                "sparse_precision": rec["sparse_precision"],
            })
    if per_plane_rows:
        df_pp = pd.DataFrame(per_plane_rows)
        df_pp.to_csv(str(OUT_DIR / "registration_metrics_ref20_subsample.csv"), index=False)

    # -----------------------------------------------------------------------
    # 6. Summary statistics
    # -----------------------------------------------------------------------
    print("\n[6/6] Summary statistics ...")

    # Group by ref_update_id to see error jumps after updates
    print("\n  --- Per-ref-update-block metrics ---")
    for ruid in sorted(df["ref_update_id"].unique()):
        block = df[df["ref_update_id"] == ruid]
        print(
            f"  ref_update {ruid:02d}: "
            f"frames [{block['frame'].min()}-{block['frame'].max()}], "
            f"n={len(block)}, "
            f"mem_MAE={block['mem_MAE'].mean():.4f} ± {block['mem_MAE'].std():.4f}, "
            f"mem_NCC={block['mem_NCC'].mean():.4f} ± {block['mem_NCC'].std():.4f}, "
            f"mem_edge={block['mem_edge_sym_dist'].mean():.3f} ± {block['mem_edge_sym_dist'].std():.3f}, "
            f"sparse_MAE={block['sparse_MAE'].mean():.4f} ± {block['sparse_MAE'].std():.4f}, "
            f"sparse_NN={block['sparse_centroid_NN'].mean():.3f} ± {block['sparse_centroid_NN'].std():.3f}, "
            f"sparse_R={block['sparse_recall'].mean():.3f} ± {block['sparse_recall'].std():.3f}, "
            f"sparse_P={block['sparse_precision'].mean():.3f} ± {block['sparse_precision'].std():.3f}"
        )

    print("\n  --- Global statistics ---")
    metric_cols = [
        "mem_MAE", "mem_MSE", "mem_NCC", "mem_edge_sym_dist",
        "sparse_MAE", "sparse_MSE", "sparse_centroid_NN",
        "sparse_recall", "sparse_precision",
    ]
    for col in metric_cols:
        vals = df[col].dropna()
        print(f"  {col:25s}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"min={vals.min():.4f}, max={vals.max():.4f}")

    print("\nDone.")
    return df


if __name__ == "__main__":
    main()
