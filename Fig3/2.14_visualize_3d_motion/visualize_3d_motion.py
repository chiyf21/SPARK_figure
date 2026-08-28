#!/usr/bin/env python3
"""Visualize predicted 3-D surfaces for selected registration frames.

The input files are the ``phase_new_fXXXXXX.npy`` files produced by the
registration pipeline. Each file has shape ``(X, Y, K, 3)`` and stores
``(x_ref, y_ref, z_ref)`` coordinates for the K reference planes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.lines import Line2D
import numpy as np


DEFAULT_RESULT_DIR = Path(
    "/home/cyf/wbi/Virginia/registrated_data/f260517/f260517_0820/diagnostics"
)
DEFAULT_FRAME_LABELS = [1, 40, 80, 120, 160, 200]
DEFAULT_FILE_FRAMES = [0, 39, 79, 119, 159, 199]
DEFAULT_K_INDICES = list(range(0, 20, 2))


def load_phase_new(result_dir: Path, file_frame: int) -> np.ndarray:
    """Load one phase_new file and return ``(K, X, Y, 3)``."""
    path = result_dir / f"phase_new_f{file_frame:06d}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    phase = np.load(path, mmap_mode="r")
    if phase.ndim != 4 or phase.shape[-1] != 3:
        raise ValueError(f"Expected (X, Y, K, 3), got {phase.shape} from {path}")
    return np.transpose(np.asarray(phase), (2, 0, 1, 3))


def sample_surfaces(
    result_dir: Path,
    file_frames: list[int],
    k_indices: list[int],
    downsample_x: int,
    downsample_y: int,
) -> np.ndarray:
    """Load and downsample selected files, returning ``(F, K, Y, X, 3)``."""
    sampled = []
    for file_frame in file_frames:
        surfaces = load_phase_new(result_dir, file_frame)
        if max(k_indices) >= surfaces.shape[0]:
            raise IndexError(
                f"Requested K index {max(k_indices)}, but file has "
                f"{surfaces.shape[0]} planes: frame {file_frame}"
            )
        surfaces = surfaces[k_indices, ::downsample_x, ::downsample_y, :]
        sampled.append(surfaces)
    # Input indexing is (K, X, Y, xyz); matplotlib expects meshgrid (Y, X).
    return np.asarray(sampled).transpose(0, 1, 3, 2, 4)


def _surface_colors(
    base_color: tuple[float, float, float, float],
    distance: np.ndarray,
    norm: colors.Normalize,
) -> np.ndarray:
    """Make a same-hue face-color array whose value decreases with distance."""
    base_hsv = colors.rgb_to_hsv(np.asarray(base_color[:3]))
    intensity = norm(distance)
    hsv = np.empty(distance.shape + (3,), dtype=float)
    hsv[..., 0] = base_hsv[0]
    hsv[..., 1] = 0.30 + 0.70 * intensity
    hsv[..., 2] = 0.98 - 0.58 * intensity
    rgba = colors.hsv_to_rgb(hsv)
    return np.concatenate([rgba, np.full(distance.shape + (1,), 0.72)], axis=-1)


def _limits(sampled: np.ndarray, display_z: np.ndarray) -> tuple[tuple[float, float], ...]:
    values = sampled[..., :2].reshape(-1, 2)
    xlim = (float(np.nanmin(values[:, 0])), float(np.nanmax(values[:, 0])))
    ylim = (float(np.nanmin(values[:, 1])), float(np.nanmax(values[:, 1])))
    zlim = (float(np.nanmin(display_z)), float(np.nanmax(display_z)))

    def padded(pair: tuple[float, float]) -> tuple[float, float]:
        span = max(pair[1] - pair[0], 1.0)
        pad = 0.04 * span
        return pair[0] - pad, pair[1] + pad

    return padded(xlim), padded(ylim), padded(zlim)


def render_frame(
    surfaces: np.ndarray,
    fixed_z: np.ndarray,
    output_path: Path,
    *,
    mode: str,
    z_scale: float,
    grid_stride: int,
    elev: float,
    azim: float,
    limits: tuple[tuple[float, float], ...],
    distance_norm: colors.Normalize,
) -> None:
    fig = plt.figure(figsize=(8.5, 8.0), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    cmap = cm.get_cmap("tab10")

    for surface_index, (surface, fixed) in enumerate(zip(surfaces, fixed_z)):
        x = surface[..., 0]
        y = surface[..., 1]
        z = surface[..., 2]
        distance = np.abs(z - fixed)
        if mode == "motion":
            z_plot = fixed + z_scale * (z - fixed)
        else:
            z_plot = z

        base = cmap(surface_index % 10)
        facecolors = _surface_colors(base, distance, distance_norm)
        line_color = tuple(np.clip(np.asarray(base[:3]) * 0.42, 0.0, 1.0))
        ax.plot_surface(
            x,
            y,
            z_plot,
            facecolors=facecolors,
            linewidth=0,
            antialiased=True,
            shade=False,
            alpha=0.72,
        )
        ax.plot_wireframe(
            x,
            y,
            z_plot,
            rstride=grid_stride,
            cstride=grid_stride,
            color=line_color,
            linewidth=1.15,
            alpha=0.95,
        )

    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1.0, 1.7, 0.9))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_legend(fixed_z: np.ndarray, output_path: Path, *, columns: int = 2) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 5.6), dpi=180)
    cmap = cm.get_cmap("tab10")
    handles = [
        Line2D(
            [0],
            [0],
            color=cmap(index % 10),
            lw=5,
            label=f"reference plane {z:g}",
        )
        for index, z in enumerate(fixed_z)
    ]
    ax.legend(
        handles=handles,
        loc="center",
        ncol=columns,
        frameon=False,
        fontsize=10,
        handlelength=1.8,
        columnspacing=1.2,
        labelspacing=1.0,
    )
    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1, transparent=True)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frames", nargs="+", type=int, default=DEFAULT_FRAME_LABELS)
    parser.add_argument("--file-frames", nargs="+", type=int, default=DEFAULT_FILE_FRAMES)
    parser.add_argument("--k-indices", nargs="+", type=int, default=DEFAULT_K_INDICES)
    parser.add_argument("--fixed-z-start", type=float, default=20.0)
    parser.add_argument("--fixed-z-step", type=float, default=10.0)
    parser.add_argument("--downsample-x", type=int, default=25)
    parser.add_argument("--downsample-y", type=int, default=50)
    parser.add_argument("--grid-stride", type=int, default=2)
    parser.add_argument("--elev", type=float, default=25.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    parser.add_argument("--z-scale", type=float, default=3.0)
    parser.add_argument("--mode", choices=("motion", "absolute"), default="motion")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if len(args.frames) != len(args.file_frames):
        raise ValueError("--frames and --file-frames must have the same length")
    if not args.k_indices:
        raise ValueError("--k-indices cannot be empty")
    if args.downsample_x < 1 or args.downsample_y < 1 or args.grid_stride < 1:
        raise ValueError("downsampling and grid stride must be positive")

    output_dir = args.output_dir or args.result_dir / "coords_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_z_all = args.fixed_z_start + args.fixed_z_step * np.arange(21, dtype=float)
    if max(args.k_indices) >= len(fixed_z_all):
        raise IndexError("A requested K index is outside the fixed-plane list")
    fixed_z = fixed_z_all[args.k_indices]
    sampled = sample_surfaces(
        args.result_dir,
        args.file_frames,
        args.k_indices,
        args.downsample_x,
        args.downsample_y,
    )
    z = sampled[..., 2]
    fixed_broadcast = fixed_z[None, :, None, None]
    distances = np.abs(z - fixed_broadcast)
    max_distance = float(np.nanmax(distances))
    distance_norm = colors.Normalize(vmin=0.0, vmax=max(max_distance, 1e-6))
    if args.mode == "motion":
        display_z = fixed_broadcast + args.z_scale * (z - fixed_broadcast)
    else:
        display_z = z
    limits = _limits(sampled, display_z)

    suffix = f"z{args.z_scale:g}x" if args.mode == "motion" else "absolute"
    generated = []
    for index, frame_label in enumerate(args.frames):
        path = output_dir / f"coords_{args.mode}_frame{frame_label:03d}_{suffix}.png"
        render_frame(
            sampled[index],
            fixed_z,
            path,
            mode=args.mode,
            z_scale=args.z_scale,
            grid_stride=args.grid_stride,
            elev=args.elev,
            azim=args.azim,
            limits=limits,
            distance_norm=distance_norm,
        )
        generated.append(path)

    legend_path = output_dir / "coords_slice_legend.png"
    render_legend(fixed_z, legend_path)
    generated.append(legend_path)
    print(f"loaded shape: {sampled.shape}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
