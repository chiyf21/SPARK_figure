#!/usr/bin/env python3
"""
F260517 — Metrics-only re-run for Fig3 plotting.  (2025-06-26)

Same registration parameters as run_F260517_0625.py, but:
  - No projection / no tiff saving
  - NCC & MAE computed on ALL pixels (mask logic fixed)
  - Output: CSVs only → code_for_paper/Fig3/
"""

import os, sys, time
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
from scipy.ndimage import sobel, distance_transform_edt

cp.cuda.Device(1).use()

HERE = Path(__file__).resolve().parent
SRC_DIR = Path("/home/cyf/wbi/Virginia/code/wbi_0123/wholistic_registration/src")
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "wholistic_registration"))
sys.path.insert(0, str(SRC_DIR / "wholistic_registration" / "tests"))

from utils import IO, calFlowCrossResolution, mask, preprocess as prep
import f260517_helpers as fh

OUT_DIR = HERE  # code_for_paper/Fig3/

# ---------------------------------------------------------------------------
# Parameters (same as run_F260517_0625.py)
# ---------------------------------------------------------------------------
F260517_mov_path = "/home/cyf/wbi/Virginia/raw_data/f260517/260517_exp_00001_TZCYX.ome.tiff"
F260517_ref_path = "/home/cyf/wbi/Virginia/raw_data/f260517/260517_anat_00003_TZCYX.ome.tiff"

option = {}
option["r"] = 5; option["layer"] = 3; option["iter"] = 10
option["movRange"] = 5.0; option["tol"] = 1e-6; option["zRatio_HR"] = 1
option["wrong_region_enable"] = False

thresFactor = 5.0; maskRange = [5.0, 4000.0]
smoothPenalty_raw = 0.03; ref_update_every = 40; Z_WINDOW = 3.0
WARMUP_FRAMES = [0, 1, 2, 3, 4]
percentiles = [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.8]

# ===========================================================================
# 1. Load
# ===========================================================================
print("=" * 60)
print("F260517 — Metrics-only run for Fig3")
print("=" * 60)
print("[1/6] Loading ...")
t0 = time.time()
F260517_mov, _ = IO.readTiff(F260517_mov_path)
F260517_ref, _ = IO.readTiff(F260517_ref_path)
ref_mem_raw    = F260517_ref[90:310, 1, :, :].astype(np.float32)
mov_mem_all    = F260517_mov[:, :, 1, :, :].astype(np.float32)
ref_sparse_raw = F260517_ref[90:310, 0, :, :].astype(np.float32)
mov_sparse_all = F260517_mov[:, :, 0, :, :].astype(np.float32)
print(f"  loaded in {time.time()-t0:.1f}s")

# ===========================================================================
# 2. Setup
# ===========================================================================
print("[2/6] Setup ...")
z_init = calFlowCrossResolution.FindInitZ_stack_global_fixed_spacing(
    mov_mem_all[0].transpose(2, 1, 0), ref_mem_raw.transpose(2, 1, 0),
    delta_ref_idx=10, use_gradient=False)
z_init = z_init.astype(np.float32)
z_idx = np.rint(z_init).astype(np.int32)
z_idx = np.clip(z_idx, 0, ref_mem_raw.shape[0] - 1)
K, T = int(z_init.shape[0]), int(mov_mem_all.shape[0])

x_c = np.arange(mov_mem_all[0].shape[2], dtype=np.float32)
y_c = np.arange(mov_mem_all[0].shape[1], dtype=np.float32)
Xg, Yg, Kg = np.meshgrid(x_c, y_c, np.arange(K, dtype=np.int32), indexing="ij")
coords_xyz = np.empty((len(x_c), len(y_c), K, 3), dtype=np.float32)
coords_xyz[..., 0] = Xg; coords_xyz[..., 1] = Yg; coords_xyz[..., 2] = z_init[Kg]
option["phase"] = coords_xyz.copy()
print(f"  K={K}  T={T}")

# ===========================================================================
# 3. Initial ref calibration
# ===========================================================================
print("[3/6] Initial ref calibration ...")
init_target = np.mean(mov_mem_all[WARMUP_FRAMES].astype(np.float32), axis=0)
ref_mem_adj, src_q_fixed, _, _ = fh.update_reference_intensity_mapping_from_target_stack(
    F260517_ref_mem=ref_mem_raw, target_stack_zyx=init_target, z_idx=z_idx,
    option=option, thresFactor=thresFactor, maskRange=maskRange,
    smoothPenalty_raw=smoothPenalty_raw, percentiles=percentiles)

# ===========================================================================
# 4. Warmup
# ===========================================================================
print("[4/6] Warmup ...")
warmup_phase = {}
for idx, i in enumerate(WARMUP_FRAMES):
    if idx == 0: option["phase"] = coords_xyz.copy(); option.pop("motion", None)
    mov_i = mov_mem_all[i].transpose(2, 1, 0).astype(np.float32, copy=False)
    option["mask_mov"] = mask.getMask(mov_i, thresFactor)
    option["mask_mov"] = mask.bwareafilt3_wei(option["mask_mov"], maskRange)
    pn, mc, _ = calFlowCrossResolution.getMotion_v2(mov_i, ref_mem_adj, option, verbose=False)
    if hasattr(pn, "get"): pn = pn.get()
    if hasattr(mc, "get"): mc = mc.get()
    warmup_phase[i] = np.asarray(pn, dtype=np.float32)
    option["motion"] = (0.7 * np.asarray(mc, dtype=np.float32))

target_z_list = []
for i in WARMUP_FRAMES:
    tz, _ = fh.estimate_projection_z_from_phase_simple(
        phase_new=warmup_phase[i], z_init=z_init, ref_shape=ref_mem_raw.shape,
        ref_volume_order="zyx", method="trimmed_mean", trim_percentiles=(5,95), frame_idx=i)
    target_z_list.append(tz)
fixed_target_z = fh.robust_average_target_z(target_z_list, method="median")
fixed_target_z[~np.isfinite(fixed_target_z)] = z_init[~np.isfinite(fixed_target_z)]

# ===========================================================================
# Metrics helpers (NCC & MAE on ALL pixels, no mask)
# ===========================================================================

def zncc_2d(a, b, eps=1e-8):
    a, b = np.asarray(a, dtype=np.float32).ravel(), np.asarray(b, dtype=np.float32).ravel()
    ac, bc = a - np.mean(a), b - np.mean(b)
    n = np.dot(ac, bc); d = np.sqrt(np.dot(ac, ac) * np.dot(bc, bc) + eps)
    return float(n / d) if d >= eps else np.nan

def symmetric_edge_distance_2d(a, b):
    gxa, gya = sobel(a.astype(np.float32), axis=-1, mode='nearest'), sobel(a.astype(np.float32), axis=-2, mode='nearest')
    gxb, gyb = sobel(b.astype(np.float32), axis=-1, mode='nearest'), sobel(b.astype(np.float32), axis=-2, mode='nearest')
    ma, mb = np.sqrt(gxa**2+gya**2+1e-8), np.sqrt(gxb**2+gyb**2+1e-8)
    ea = (ma >= np.percentile(ma, 90)).astype(np.uint8)
    eb = (mb >= np.percentile(mb, 90)).astype(np.uint8)
    if not np.any(ea) or not np.any(eb): return np.nan
    dta, dtb = distance_transform_edt(1-ea), distance_transform_edt(1-eb)
    return float(0.5*(np.mean(dtb[ea>0])+np.mean(dta[eb>0])))

def compute_membrane_metrics(mov_zyx, mapped_zyx):
    """MAE, NCC, edge: all pixels, full image."""
    Kk = mov_zyx.shape[0]
    out = {"MAE": [], "nMAE": [], "NCC": [], "edge": []}
    for kk in range(Kk):
        mm, mp = mov_zyx[kk], mapped_zyx[kk]
        diff = np.abs(mm.astype(np.float32) - mp.astype(np.float32))
        mae = float(np.mean(diff))
        p1, p99 = np.percentile(mm, [1, 99])
        dyn = max(p99 - p1, 1e-8)
        out["MAE"].append(mae); out["nMAE"].append(mae/dyn)
        out["NCC"].append(zncc_2d(mm, mp))
        out["edge"].append(symmetric_edge_distance_2d(mm, mp))
    return {k: float(np.nanmean(v)) for k, v in out.items()}

# ===========================================================================
# 5. Forward loop
# ===========================================================================
print("[5/6] Forward loop ...")
error_mem, error_sparse, hole_records = [], [], []
registered_cache = {}
frames_since_ref_update, ref_update_id = 0, 0
option["phase"] = coords_xyz.copy(); option.pop("motion", None)
total_start = time.time()

from utils.calFlowCrossResolution import generate_continuous_H_gpu as genH, apply_H_to_matrix_gpu as applyH
sparse_ref_gpu = cp.asarray(ref_sparse_raw.transpose(2,1,0), dtype=cp.float32)

for i in range(0, T):
    fs = time.time()
    raw_mem_zyx = mov_mem_all[i]; raw_sparse_zyx = mov_sparse_all[i]
    mov_mem_xyk = raw_mem_zyx.transpose(2, 1, 0).astype(np.float32, copy=False)
    option["mask_mov"] = mask.getMask(mov_mem_xyk, thresFactor)
    option["mask_mov"] = mask.bwareafilt3_wei(option["mask_mov"], maskRange)

    phase_new, motion_current, mem_mapped_xyk = calFlowCrossResolution.getMotion_v2(
        mov_mem_xyk, ref_mem_adj, option, verbose=False)
    if hasattr(phase_new, "get"):      phase_new = phase_new.get()
    if hasattr(motion_current, "get"): motion_current = motion_current.get()
    if hasattr(mem_mapped_xyk, "get"): mem_mapped_xyk = mem_mapped_xyk.get()
    phase_new = np.asarray(phase_new, dtype=np.float32)
    motion_current = np.asarray(motion_current, dtype=np.float32)
    mem_mapped_zyx = np.asarray(mem_mapped_xyk, dtype=np.float32).transpose(2, 1, 0)

    # Sparse mapped
    H_sp = genH(sparse_ref_gpu, zRatio=1)
    sm_xyk = applyH(cp.asarray(phase_new, dtype=cp.float32), H_sp)
    if hasattr(sm_xyk, "get"): sm_xyk = sm_xyk.get()
    sparse_mapped_zyx = np.asarray(sm_xyk, dtype=np.float32).transpose(2, 1, 0)

    registered_cache[i] = mem_mapped_zyx

    # ---- Metrics (ALL pixels) ----
    mem_m = compute_membrane_metrics(raw_mem_zyx, mem_mapped_zyx)
    sparse_m = compute_membrane_metrics(raw_sparse_zyx, sparse_mapped_zyx)

    # ---- Hole fraction (fast pixel-wise) ----
    z_t_zyx = phase_new[..., 2].transpose(2, 1, 0)
    hole = np.ones_like(z_t_zyx, dtype=bool)
    for tz in fixed_target_z:
        hole &= (np.abs(z_t_zyx - tz) > Z_WINDOW)
    hole_frac = float(np.mean(hole))
    hole_per_k = [float(np.mean(hole[kk])) for kk in range(K)]

    elapsed = time.time() - fs

    error_mem.append({"frame": i, "ref_update_id": ref_update_id,
        "MAE": mem_m["MAE"], "nMAE": mem_m["nMAE"], "NCC": mem_m["NCC"],
        "edge": mem_m["edge"], "hole_frac": hole_frac, "elapsed_s": elapsed})
    error_sparse.append({"frame": i, "ref_update_id": ref_update_id,
        "MAE": sparse_m["MAE"], "nMAE": sparse_m["nMAE"], "NCC": sparse_m["NCC"],
        "edge": sparse_m["edge"], "hole_frac": hole_frac, "elapsed_s": elapsed})
    hole_records.append({"frame": i, "ref_update_id": ref_update_id,
        "hole_frac": hole_frac, "max_hole_frac_k": float(np.max(hole_per_k)),
        **{f"k{kk:02d}": hole_per_k[kk] for kk in range(K)}})

    print(f"[{i:03d}/{T-1:03d}] mem_MAE={mem_m['MAE']:.1f} mem_NCC={mem_m['NCC']:.4f} "
          f"mem_nMAE={mem_m['nMAE']:.4f} holes={hole_frac*100:.1f}% {elapsed:.1f}s")

    option["motion"] = (0.7 * motion_current).astype(np.float32, copy=False)

    frames_since_ref_update += 1
    if frames_since_ref_update >= ref_update_every:
        calib_frames = sorted(registered_cache.keys())[-5:]; ref_update_id += 1
        stacks = [mov_mem_all[fi].astype(np.float32, copy=False) for fi in calib_frames]
        if len(stacks) > 0:
            target = np.mean(np.stack(stacks, axis=0), axis=0).astype(np.float32)
            ref_source = ref_mem_raw[z_idx].astype(np.float32, copy=False)
            _, new_tgt_q, _ = prep.learn_quantile_mapping(
                source=ref_source, target=target, percentiles=percentiles)
            ref_mem_adj = prep.apply_quantile_mapping(
                ref_mem_raw, src_q_fixed, new_tgt_q).transpose(2,1,0).astype(np.float32, copy=False)
            print(f"\n  >>> Ref Update #{ref_update_id} — raw frames {calib_frames}  tgt_q[0]={new_tgt_q[0]:.1f}\n")
        frames_since_ref_update = 0

print(f"\n  Done in {time.time()-total_start:.1f}s ({(time.time()-total_start)/T:.1f}s/frame)")

# ===========================================================================
# 6. Save CSVs
# ===========================================================================
print("[6/6] Saving ...")
pd.DataFrame(error_mem).to_csv(str(OUT_DIR / "errors_membrane.csv"), index=False)
pd.DataFrame(error_sparse).to_csv(str(OUT_DIR / "errors_sparse.csv"), index=False)
pd.DataFrame(hole_records).to_csv(str(OUT_DIR / "hole_summary.csv"), index=False)

for label, df in [("Membrane", pd.DataFrame(error_mem)), ("Sparse", pd.DataFrame(error_sparse))]:
    print(f"\n  {label}:")
    for col in ["MAE", "nMAE", "NCC", "edge"]:
        v = df[col].dropna()
        print(f"    {col:8s}: mean={v.mean():.4f}  std={v.std():.4f}")
print("\nDone.")
