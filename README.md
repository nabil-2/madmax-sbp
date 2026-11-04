# MADMAX SBP

**SBP means spider bead pull.**

[Documentation: verification, data taking and analysis](https://nabil-2.github.io/madmax-sbp/)
describes the code and mathematical steps. Sphinx sources live in `docs/` and
GitHub Actions publishes them to GitHub Pages. The 1D/3D booster models and the
bead-response calculation are treated as black boxes; the bead-response page
cites Egge et al.

The first workflow, **SBP verification**, checks how well a few lines from a full
bead-pull grid reproduce its transverse field and coherent transverse integrals.
Open [sbp_verification.ipynb](sbp_verification.ipynb) and run it from top to bottom.

Measurement analysis notebooks:

- [sbp_1d_analysis.ipynb](sbp_1d_analysis.ipynb): original measurement and 1D plot sequence.
- [sbp_3d_analysis.ipynb](sbp_3d_analysis.ipynb): disk and thread full-tilt analysis,
  using the original full-tilt report figures.

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
python -m pip install numpy scipy h5py matplotlib pandas tqdm jupyterlab ipykernel
python -m jupyter lab
```

Run Jupyter from this repository's root.

## Files

- `src/sbp_verification.py`: gap and plane selection, running the analysis,
  comparison metrics and saving results.
- `src/verification_core.py`: the original time gating, background subtraction,
  field extraction, line selection, complex Gaussian fit and integration.
- `src/verification_plots.py`: field maps and transverse-integral comparisons.
- `src/verification_diagnostics.py`: the original sampling, background,
  alignment, complex-field, integral-frequency and illustrative mode plots.
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

The notebook writes `outputs/sbp_verification/ET_results.npz` and includes all
original verification diagnostic families. Its default three-line configuration
has **19 rendered figures**, with editable labels and automatic PNG/PDF exports
under `plots/verification/L3_Z1_G4/reference_assisted/`. Sampling, preprocessing,
fields, integrals and mode checks have separate subfolders. Changing the line,
plane or gap count, or disabling reference alignment, selects a separate folder.
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

## 1D measurement analysis

The [1D documentation](https://nabil-2.github.io/madmax-sbp/analysis/one_dimensional.html)
explains the SBP trace-to-integral preparation and geometry extension. It treats
the calculations already present in the old `5_` notebook as the **existing 1D
analysis pipeline**: a black box for amplitude fitting, field integration and
boost calculation. Disk-gap fitting already exists there. The backend now uses
the notebook's functions and original MADBead routines directly, including
frequency-outlier rejection and Monte Carlo propagation. Our raw-to-transverse-
integral logic is retained, along with the shared geometry extensions.

Install the Julia dependencies once before enabling recomputation:

```bash
julia --project=julia/analysis_1d -e 'using Pkg; Pkg.instantiate()'
```

Julia must be on `PATH`, or selected through `MADMAX_SBP_JULIA`. The cluster's
`/software/julia/default/bin/julia` is detected automatically. No separate
MADBead checkout is needed; the required original routines are in `src/madbead/`.

Open [sbp_1d_analysis.ipynb](sbp_1d_analysis.ipynb). It retains the plots and their
order from `pipeline_overview.ipynb`: setup, raw complex traces, reference
subtraction, centre finding, radial fields and maps, transverse integration,
longitudinal fits, geometry shifts, boost/residual spectra and the full-grid
comparison. Figure titles and labels remain editable in each plot cell.
Rendered figures are saved in the notebook for viewing on GitHub.

Both analysis notebooks save every displayed figure as **PNG and PDF** under
the root `plots/` folder. They create the directories automatically. The figure
cells expose their titles, axis labels, legend entries, colorbar labels and
annotations as editable keyword arguments, matching the original notebook style.
Edit a string and rerun its cell to update the inline figure and both exports.
Templates such as `{frequency_GHz:.3f}` keep numerical values current; raw
strings such as `r"$|E|$ [V/m]"` support mathematical labels. `PLOT_ROOT` in each
notebook selects the export root.

```text
plots/
  verification/L3_Z1_G4/reference_assisted/
    sampling/
    preprocessing/background/
    fields/
    integrals/frequency/
    model_checks/
  1d/
    measurement/{setup,raw,transverse,incident_beam}/
    both/{fits,fields}/                   # selected GEOMETRY
    geometry_comparison/{fits,geometry,boost,history,residuals}/
    full_bead_comparison/both/
  3d/full_tilt/
    measurement/{raw,incident_beam}/
    boost/
    profiles/
    fields/{chords,transverse}/
    geometry/
    residuals/
    full_bead_comparison/
```

`plots/` is ignored by Git. All exports can be recreated by running the notebooks
with the local measurement/result files described below. The notebooks themselves
are committed with their rendered figures included; no external plotting script
is required.

- `src/sbp_analysis.py`: raw line data to radial fields and transverse integrals.
- `src/fit_1D.py`: numeric adapter for the existing Julia 1D analysis pipeline.
- `src/run_1d.jl`: shared geometry amendments and the existing analysis calls.
- `src/existing_1d_functions.jl`, `src/madbead/`: notebook/MADBead routines.
- `src/pipeline_plots.py`: the original measurement and 1D figures.
- `src/full_bead_comparison.py`, `src/full_bead_model.py`: read the saved full-pull
  results and reproduce their model/normalization for comparison.

The four cases keep geometry fixed, adjust disk gaps, adjust thread z, or adjust
both. This longitudinal model has no disk-tilt degrees of freedom; disk and
thread tilts belong to the 3D workflow. The corrected nominal mirror gap is
**7.803 mm**. The existing pipeline propagates its input uncertainties through
100 Monte Carlo draws by default, with an assumed 5% ET error. The shared
geometry fit does not supply an additional covariance estimate. The old
too-few-z fallback and reduced-grid configuration ranking are omitted.

The default notebook reads local copies of these original inputs/results:

```text
data/spider_bead_pull_runs/run_20261025_141003/
    scan_manifest.json, scan_log.csv, s11_traces.h5
data/outputs/run_20261025_141003/
    integrated_fields.npz, integrated_fields_fit_1D_{fixed,gaps,positions,both}.npz
    geometry_correction.json
    history/nominal_gap_8p803mm/   # comparison with previous nominal geometry
data/reference_data/beam_measurements/beam_fit_S31.csv
data/full_bead_pull/result_bf_21_0.jld2
```

These files are deliberately excluded from Git. They have been copied into the
local checkout; another checkout needs its own measurement files to rerun the
notebook. The saved figures remain viewable without them. Both measurement
pipelines discard raw frequencies outside **20–22 GHz before preprocessing**.
This includes baseline estimation, centre finding, phase unwrapping and fitting;
the 3D beam CSV is also restricted before interpolation.

The optional recompute cells near the top of the 1D notebook regenerate the
fields and all four fits before loading results. They remain disabled. Existing
saved 1D results and rendered figures still use the earlier 12–22 GHz analysis
and previous backend; no measurement reanalysis has been run. The updated
pipeline rejects these old products instead of silently reusing them.

The initial migration check, before changing the frequency restriction,
reproduced the original fitted fields at 20.7, 20.825 and 21 GHz to numerical
precision. The committed rendered figures come from that earlier execution.


## 3D analysis with disk and thread tilts

Open [sbp_3d_analysis.ipynb](sbp_3d_analysis.ipynb). It starts with boost spectra,
then shows measured profiles, the external beam input, complex fitted chords,
finite-bead line responses, model electric-field maps, nominal/fitted geometry,
residuals, and the full-pull comparison. The figures use the original corrected
full-tilt report plotting code, including separate disk projection and adjustment
figures. Notebook outputs are included; large measurements and banks are not.

The likelihood uses corrected **complex S22 samples directly**, at their actual
recorded x/y and fitted z. It fits three disk shifts, two slopes per disk and two
endpoint heights per straight thread: **17 shared geometry parameters**. Per
frequency it solves analytically for a common complex response coefficient and
per-thread complex affine backgrounds. Boost follows from the model's receive
coupling multiplied by the magnitude of that coefficient. It is not fitted as
an independent spectral curve.

- `src/analysis_3d_inputs.py`: raw run → corrected S22, profile diagnostics,
  selected-frequency mask; beam CSV → six-profile input parameters.
- `src/receive_field.py`, `src/tilted_disks.py`: modal propagation and tilted
  scattering interfaces, using the original BoostFractor matrix-bank format.
- `src/exterior_field.py`: variable-height angular-spectrum continuation for T0.
- `src/bead_response.py`: the original finite-sphere Mie response.
- `src/analysis_3d.py`: physical scan chords, fit constraints, variable projection,
  SLSQP fitting and saved result arrays.
- `src/analysis_3d_exports.py`: conditional residual-bootstrap bands and physical
  electric-field maps at horizontal thread midpoint sections.
- `src/analysis_3d_plots.py`: the original full-tilt report figures.
- `src/build_bank.jl`: CPU/CUDA propagation-bank construction.

The independent beam scan contains six Gaussian **magnitude** profiles at scan
z = 50–100 mm. Their coordinate origin is not registered to the spider mirror.
The model combines them at the outer reference face, assumes flat phase and
normalizes their whole-plane power to 1 W. Preliminary Gaussian fits to spider
chords select the 33 geometry-fit bins; they do not supply a rotated field to
the 3D model. The radial reconstruction in the 1D notebook is a separate path.

Local data in `data/3d/` comprise `measurement_inputs.npz`, `beam_prior.npz`,
`disk_tilts_threads.npz`, `conditional_bands.npz`, `field_maps.npz` and `bank/`.
Their original relative paths and SHA-256 hashes are recorded in
[config/analysis_3d_provenance.json](config/analysis_3d_provenance.json).
[config/full_tilt_geometry.json](config/full_tilt_geometry.json) keeps the fitted
17-vector, units, corrected nominal positions and measured relative slopes.
The notebook has optional cells to prepare inputs, run a new single-start fit,
and export results. These switches default to off; loading a notebook does not
start an optimization. The original saved result used multiple starts.

The 120-mode local bank contains the 33 geometry-fit frequencies. A bank with
all 401 bins is needed to recompute a 20–22 GHz dense curve; the included local
saved spectrum already covers that band. To build a bank with Julia 1.10+:

```bash
julia --project=julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/nabil-2/BoostFractor.jl", rev="acf4f491fe633ab05c3b8abbafe6d6288e42aba0"))'
julia --project=julia src/build_bank.jl config/3d_bank.toml
```

For GPU construction, add CUDA to that environment with
`julia --project=julia -e 'using Pkg; Pkg.add("CUDA")'`, then change `backend` to
`"cuda"` in the configuration and run on a GPU node. The config writes banks
under `data/3d/`; edit its frequency list for a dense export. Python optimization
uses the resulting small modal matrices on CPU. The builder preserves the
original matrix calculation; it was not rerun during this migration.

Quick numerical check: at all 33 geometry-fit frequencies, this port reproduces
the original full-tilt boost, common scale, predicted complex S22, directional
fields and pointwise heights. Relative L2 differences were below **2×10⁻¹⁴**.
Raw preprocessing also reproduced every saved input array, including the 33-bin
mask; the six-profile beam parameters were bit-identical. Both analysis notebooks
were executed with copied original measurements/results.
This migration does not establish a better physical fit: fitted bounds remain
active, weak-thread residuals remain large, and the shaded bootstrap interval
keeps geometry, beam and calibration fixed.
