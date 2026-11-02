# MADMAX SBP

**SBP means spider bead pull.**

The first workflow, **SBP verification**, checks how well a few lines from a full
bead-pull grid reproduce its transverse field and coherent transverse integrals.
Open [sbp_verification.ipynb](sbp_verification.ipynb) and run it from top to bottom.

## Setup

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra notebook
uv run jupyter lab
```

Or use Python 3.11 or newer:

```bash
python -m pip install numpy scipy h5py matplotlib tqdm jupyterlab ipykernel
python -m jupyter lab
```

Run Jupyter from this repository's root.

## Files

- `src/sbp_verification.py`: gap and plane selection, running the analysis,
  comparison metrics and saving results.
- `src/verification_core.py`: the original time gating, background subtraction,
  field extraction, line selection, complex Gaussian fit and integration.
- `src/verification_plots.py`: field maps and transverse-integral comparisons.
- `sbp_verification.ipynb`: configuration, analysis steps and plots.

Functions live in `src/`; the notebook imports and uses them. There are no
scheduler, cache-manager or hardware dependencies in this workflow.

## Measurement input

Place the full-grid `.hdf5` files in `data/full_bead_pull/`, or set `DATA_DIR` in
the notebook. Measurement data and generated outputs are ignored by Git.

The input format is the same as in the original repository:

- `s11_map`: complex reflection array with axes `(x, y, z, frequency)`.
- `vna_config`: a JSON HDF5 attribute containing `start`, `stop`, `points`.
- `measurement_config`: a JSON HDF5 attribute containing
  `space = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]`, in **millimetres**.

The initial notebook settings use the original four OB300 fixed-disk grid
measurements over 18–24 GHz. It starts with nine frequency bins near 19.5 GHz;
set `FREQUENCY_INDICES=None` to process every acquired frequency. Time gating is
applied before frequency selection.

## Calculation

1. Read the full grids and order the gaps by acquisition z.
2. Time gate the reflection traces with the original 180 ns window, 55 ns offset
   and cosine edges.
3. Subtract a complex affine background estimated from the edge samples.
4. Take a spatially unwrapped complex square root of the reflection change and
   convert it to the finite-bead field estimate using a 2.93 mm alumina bead,
   relative permittivity 9.23 and the 1 W normalization convention.
5. Retain 1–4 lines and selected z planes; fit the original complex fundamental
   Gaussian with phase curvature.
6. Compare the fitted field and its transverse integral with the full grid.

`REFERENCE_ALIGNMENT=True` reproduces the original **reference-assisted**
calculation: full-grid values align the line-wise square-root signs and complex
amplitudes before fitting. The full reference therefore contributes information
to the reduced reconstruction. Set it to `False` to disable that assistance.

A single line does not independently determine the perpendicular beam centre.
The reconstructed transverse shape and its integral depend on the Gaussian model.
The field here includes the bead response; no longitudinal finite-bead
correction or boost fit is applied in this workflow.

## Output

The notebook writes `outputs/sbp_verification/ET_results.npz` and three plots.
The NPZ keeps the original downstream input keys:

| Array | Shape | Unit |
| --- | --- | --- |
| `ETs_full` | `(full_z, frequency)` | V m |
| `ETs_reduced` | `(reduced_z, frequency)` | V m |
| `z_full`, `z_reduced` | `(z,)` | m |
| `frequencies` | `(frequency,)` | Hz |

The z arrays retain the original positive acquisition-relative coordinate. An
absolute lab/mirror coordinate conversion belongs to the downstream model.
The file also records the configuration and whether reference alignment was used.

To compare other configurations, change `LINE_COUNT`, `Z_SLICE_COUNT` or
`GAP_COUNT` in the notebook. The original per-gap plane-selection rules are
retained and the selected indices are printed before fitting.

## Origin

The numerical routines are extracted from
[`spider-bead-pull-ai/src/adaptive_bead_pull.py`](https://github.com/nabil-2/spider-bead-pull-ai/blob/9f28201c14dbc6724984ee4ea5e8e05af2fecda3/src/adaptive_bead_pull.py).
The migration removes the old plotting, Slurm and cache orchestration while
preserving the numerical calculations and defaults. The original source SHA-256
is `aae7978eabf8ac849e58057220a7fed735580514e465677032192493ac4b1909`.

Migration check: on an original measurement grid, the old and new routines gave
identical gated traces, fields, fitted parameters and ET arrays for 1–4 lines
at 19.494, 19.500 and 19.506 GHz. The notebook was also executed on all four
original grids using its nine-frequency initial selection. No full-spectrum
reanalysis is claimed by this quick check.
