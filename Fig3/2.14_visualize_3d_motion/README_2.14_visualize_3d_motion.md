# 2.14_visualize_3d_motion

This folder contains the code and outputs for the 3-D visualization of the
predicted surfaces in the F260517 registration experiment.

## What is visualized

The visualization reads the registration outputs
`phase_new_fXXXXXX.npy`. Each file is expected to have shape `(X, Y, K, 3)`;
the last dimension stores the predicted `(x_ref, y_ref, z_ref)` coordinate.

The default figures use:

- displayed frame labels: `1, 40, 80, 120, 160, 200`;
- corresponding file indices: `0, 39, 79, 119, 159, 199`;
- reference-plane indices: `K = 0, 2, 4, ..., 18`;
- reference z positions: `20, 40, 60, ..., 200`;
- color hue to identify the reference slice;
- color darkness proportional to `abs(z_ref - fixed_z)`;
- motion display height `fixed_z + 3 * (z_ref - fixed_z)`.

The factor of 3 is only a display amplification. It does not modify the
saved coordinates or the registration result. The figures use a filled
surface with a dark grid, an oblique view, no xyz tick labels, no axis labels,
and no colorbar. The slice legend is saved separately.

## Input and output locations

The default input directory is:

```text
/home/cyf/wbi/Virginia/registrated_data/f260517/f260517_0820/diagnostics
```

The default output directory is `coords_visualization` below that directory.
For the packaged paper-code copy, the generated PNG files are stored in the
`outputs/` subdirectory next to this README.

## Re-run the figures

From this folder, run:

```bash
/home/cyf/.conda/envs/Allen/bin/python visualize_3d_motion.py \
  --result-dir /home/cyf/wbi/Virginia/registrated_data/f260517/f260517_0820/diagnostics \
  --output-dir outputs
```

The script uses `numpy` and the non-interactive `matplotlib` backend, so it can
run on the server without a display. It prints the loaded array shape and all
generated files.

## Notebook

`visualize_coords_F260517_0820.ipynb` contains the same workflow in an
interactive form. It is useful for changing frames, plane selection,
downsampling, view angle, or z-axis display amplification before regenerating
figures.

## Files

- `visualize_3d_motion.py`: reproducible command-line renderer;
- `visualize_coords_F260517_0820.ipynb`: interactive version;
- `outputs/coords_motion_frame*.png`: one figure per selected frame;
- `outputs/coords_slice_legend.png`: separate slice-color legend.
