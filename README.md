# MADMAX SBP

**SBP means spider bead pull.**

The first workflow, **SBP verification**, checks how well a few lines from a full
bead-pull grid reproduce its transverse field and coherent transverse integrals.
Open [sbp_verification.ipynb](sbp_verification.ipynb) and run it from top to bottom.

For measurement preparation and acquisition:

- [sbp_calibration.ipynb](sbp_calibration.ipynb): record each sub-thread's XYZ
  endpoints and motor step anchors, then save the calibration.
- [sbp_acquisition.ipynb](sbp_acquisition.ipynb): preview the scan, acquire VNA
  traces and inspect the saved run.

Both notebooks start in simulation mode and run without instruments.

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
- `src/calibration.py`: per-sub-thread position conversion and calibration files.
- `src/motor.py`: real/simulated motor, stored position/home and travel limits.
- `src/acquisition.py`: configuration, VNA readout, scan execution and logging.

Functions live in `src/`; the notebooks import and use them. Hardware dependencies
are optional and are only needed when connecting real instruments.

## Calibration and data acquisition

Run the two notebooks in order: calibration, then acquisition. Their default
simulation uses four example sub-threads and writes under `outputs/`. The VNA
test mode stores zero traces and is only a check of scan execution and logging.

For a real setup, install the instrument dependencies and make a local config:

```bash
uv sync --extra notebook --extra hardware
cp config/measurement_config.example.json config/measurement_config.json
```

Set the serial port, VNA address and sweep, motor profile/direction, and measured
travel limits in that file. The example step limits are illustrative. In the
calibration notebook, replace the example XYZ coordinates with your measured
endpoints, choose their unit and set `SIMULATE=False`. Capture both motor step
anchors for every sub-thread and save the resulting calibration. Then use the
same local configuration in the acquisition notebook.

Each sub-thread's scale is its step span divided by its XYZ endpoint separation.
Coordinate fields saved as `*_xyz_m` and scan distances are in metres. The
calibration keeps the old schema (version 2), including start/end margins.

The real motor keeps its logical position and manually chosen home in
`~/.madmax_bead_pull/motor_state.json`, shared with the original software.
`stepper.state_file` can select another file. This rig has no hardware limit
switch; home is a manually established reference. The original travel-limit,
move-size, timeout and emergency-stop handling is retained. Background jogging
keeps the notebook STOP button responsive. After an emergency stop, re-establish
the position with `motor.set_position(known_steps)` before further motion.

Acquisition also runs from the command line. A quick instrument-free example:

```bash
uv run python -c "from src.acquisition import create_demo_config; create_demo_config('outputs/demo')"
uv run python -m src.acquisition --config outputs/demo/measurement_config.json --dry-run
uv run python -m src.acquisition --config outputs/demo/measurement_config.json --simulate --vna-testmode
```

For a real run, use your local config and omit the simulation flags. Each run
writes a timestamped folder under `output.root` unless `output.directory` is set.
Existing scan files are refused unless overwrite is explicitly enabled.

| File | Contents |
| --- | --- |
| `s11_traces.h5` | Complex `s_parameter` with axes `(trace, frequency)` and `frequencies_Hz` |
| `scan_log.csv`, `scan_log.jsonl` | Reached bead positions, XYZ coordinates, timestamps and trace indices |
| `scan_manifest.json` | Configuration, calibration, completion status and measured-point count |
| `vna_log.txt` | Communication log when a real VNA is connected |

The historical HDF5 filename is retained; attributes identify the actual
S-parameter (normally S22, ports 2/2). The instrument's sweep is read back before
writing the frequency axis. Kernel Interrupt aborts a scan and cleanup releases
the motor and VNA. Local configurations, measurement files and generated outputs
are excluded from Git.

Motor/calibration routines and the VNA writer are ported from
`spider-bead-pull-ai/src/bead_pull_controller.py` and
`spider-bead-pull-ai/run_bead_pull_measurement.py`. The implementation was checked
against the original with the same calibration and a short simulated scan;
physical instrument operation has not been tested here.

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
