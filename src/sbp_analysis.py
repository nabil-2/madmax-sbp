"""Minimal radial analysis of runs from run_bead_pull_measurement.py.

Each sub-thread crosses a radially symmetric beam. Its first and last 2.5 cm
are off-beam reference regions. Subtract a complex linear reference, estimate
one beam centre per thread from the symmetry of all frequency traces, fold the
two halves, unwrap the radial phase, and integrate the complex field with 2*pi*r.

Usage from the repository root::

    python -m src.sbp_analysis data/spider_bead_pull_runs/run_TIMESTAMP

The result is data/outputs/RUN_NAME/integrated_fields.npz.
ET has axes (sub-thread, frequency), z_m is
measured from the antenna towards the mirror, and z_from_mirror_m is the
positive distance in front of the mirror. Beam centres are distances from each
calibrated thread start. The field uses the existing 1 W normalization and bead
conversion convention; longitudinal finite-bead correction remains downstream.
The square-root branch is continuous in radius and frequency but retains an
overall sign ambiguity for each sub-thread.
This version performs no time gating, beam-shape fit, or uncertainty estimation.
Raw frequencies outside 20–22 GHz are discarded before all preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

if __package__:
    from .analysis_band import frequency_mask, load_raw_traces
    from .data_paths import analysis_output_path
else:
    from analysis_band import frequency_mask, load_raw_traces
    from data_paths import analysis_output_path


REFERENCE_EDGE_M = 0.025
BEAD_DIAMETER_M = 2.93e-3
# Measured alumina value, also used by adaptive_bead_pull.sqrt_dgamma_to_E.
# Egge et al., JCAP 04 (2024) 005, section 3: https://arxiv.org/abs/2311.13359
BEAD_EPS_R = 9.23
INPUT_POWER_W = 1.0
EPSILON_0 = 8.85418782e-12
MIRROR_Z_M = 3.249
DISK_THICKNESS_M = 1e-3
# Empty surface-to-surface gaps, ordered from the mirror towards the antenna.
GAP_WIDTHS_M = (7.803e-3, 8.853e-3, 8.855e-3)
SUBTHREAD_0_DISK_OFFSET_M = 3.5e-3
Z_FROM_MIRROR_M = np.array([
    sum(GAP_WIDTHS_M) + 3 * DISK_THICKNESS_M + SUBTHREAD_0_DISK_OFFSET_M,
    sum(GAP_WIDTHS_M[:2]) + 2 * DISK_THICKNESS_M + GAP_WIDTHS_M[2] / 2,
    GAP_WIDTHS_M[0] + DISK_THICKNESS_M + GAP_WIDTHS_M[1] / 2,
    GAP_WIDTHS_M[0] / 2,
])
SUBTHREAD_Z_M = MIRROR_Z_M - Z_FROM_MIRROR_M


@dataclass
class LineData:
    position_m: np.ndarray
    traces: np.ndarray
    length_m: float


@dataclass
class LineResult:
    ET: np.ndarray
    center_m: float
    radii_m: np.ndarray
    electric_field: np.ndarray


def load_run(run_dir: str | Path) -> tuple[np.ndarray, dict[int, LineData]]:
    """Load only 20–22 GHz, then join calibrated positions by trace_index."""
    run_dir = Path(run_dir)
    frequencies, traces = load_raw_traces(run_dir)
    manifest = json.loads((run_dir / "scan_manifest.json").read_text())
    calibration = {int(sub["index"]): sub for sub in manifest["calibration"]["sub_threads"]}
    log = run_dir / "scan_log.jsonl"
    if log.exists():
        records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    else:
        with (run_dir / "scan_log.csv").open(newline="") as stream:
            records = list(csv.DictReader(stream))
        for record in records:
            record["measurement"] = json.loads(record["measurement_json"])

    grouped: dict[int, list[tuple[float, int]]] = {}
    for record in records:
        index = int(record["sub_thread_index"])
        trace_index = int(record["measurement"]["trace_index"])
        if not 0 <= trace_index < len(traces):
            raise ValueError(f"Invalid trace_index {trace_index} in scan log.")
        grouped.setdefault(index, []).append((float(record["position_m"]), trace_index))
    if not grouped:
        raise ValueError("The run contains no logged measurements.")

    lines = {}
    for index, entries in sorted(grouped.items()):
        sub = calibration[index]
        length = float(np.linalg.norm(np.asarray(sub["end_xyz_m"]) - sub["start_xyz_m"]))
        entries.sort()
        positions, indices = zip(*entries)
        lines[index] = LineData(np.asarray(positions), traces[list(indices)], length)
    return frequencies, lines


def field_conversion(frequencies_Hz: np.ndarray) -> np.ndarray:
    """Return C(f) in E=C(f)*sqrt(delta_gamma), matching the legacy convention."""
    frequencies = np.asarray(frequencies_Hz, dtype=float)
    alpha_0 = 4 * np.pi * (BEAD_DIAMETER_M / 2) ** 3 * (BEAD_EPS_R - 1) / (BEAD_EPS_R + 2)
    return np.sqrt(4 * INPUT_POWER_W / (EPSILON_0 * alpha_0 * 1j * 2 * np.pi * frequencies))


def subtract_baseline(position_m: np.ndarray, traces: np.ndarray, length_m: float) -> np.ndarray:
    """Fit each complex baseline to the calibrated 2.5 cm end windows."""
    position = np.asarray(position_m, dtype=float)
    traces = np.asarray(traces, dtype=complex)
    left = position <= REFERENCE_EDGE_M + 1e-12
    right = position >= length_m - REFERENCE_EDGE_M - 1e-12
    if not np.any(left) or not np.any(right):
        raise ValueError("Both calibrated 2.5 cm reference windows need measured points.")
    reference = left | right
    design = np.column_stack((np.ones(position.size), position / length_m))
    coefficients = np.linalg.lstsq(design[reference], traces[reference], rcond=None)[0]
    delta = traces - design @ coefficients
    delta[reference] = 0  # These regions are known to contain no bead signal.
    return delta


def fit_beam_center(position_m: np.ndarray, delta: np.ndarray, length_m: float) -> float:
    """Find one symmetry centre across frequencies, in metres from the thread start.

    Maximize the full-support complex reflection correlation. Zero extension is
    justified by the off-beam ends. Unlike an overlap-only mean residual, this
    objective cannot improve merely by moving the centre to a nearly empty edge.
    """
    interpolate = interp1d(position_m, delta, axis=0, bounds_error=False, fill_value=0.0)
    grid = np.linspace(0, length_m, max(257, len(position_m)))
    values = interpolate(grid)
    energy = float(trapezoid(np.sum(np.abs(values) ** 2, axis=1), grid))
    if not np.isfinite(energy) or energy <= np.finfo(float).tiny:
        raise ValueError("Cannot determine a beam centre from an empty bead signal.")

    def objective(center):
        reflected = interpolate(2 * center - grid)
        overlap = np.sum((values.conj() * reflected).real, axis=1)
        return -float(trapezoid(overlap, grid)) / energy

    candidates = np.linspace(REFERENCE_EDGE_M, length_m - REFERENCE_EDGE_M, 129)
    best = int(np.argmin([objective(center) for center in candidates]))
    bounds = (candidates[max(0, best - 1)], candidates[min(len(candidates) - 1, best + 1)])
    result = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-10})
    if not result.success:
        raise RuntimeError("Beam-centre fit did not converge.")
    return float(result.x)


def analyze_line(position_m, traces, frequencies_Hz, length_m: float) -> LineResult:
    """Restrict to 20–22 GHz, then subtract, centre, fold and integrate."""
    position = np.asarray(position_m, dtype=float)
    traces = np.asarray(traces, dtype=complex)
    frequencies = np.asarray(frequencies_Hz, dtype=float)
    if position.ndim != 1 or frequencies.ndim != 1 or traces.shape != (position.size, frequencies.size):
        raise ValueError("Expected positions (point,), frequencies (frequency,), traces (point, frequency).")
    keep = frequency_mask(frequencies)
    frequencies, traces = frequencies[keep], traces[:, keep]
    if (not np.all(np.isfinite(position)) or not np.all(np.isfinite(traces))
            or not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0)):
        raise ValueError("Positions and traces must be finite; frequencies must be finite and positive.")
    if np.any(np.diff(frequencies) <= 0):
        raise ValueError("Frequencies must be increasing for phase unwrapping.")
    if not np.isfinite(length_m) or length_m <= 2 * REFERENCE_EDGE_M:
        raise ValueError("The sub-thread must be longer than its two reference windows.")

    # Average repeated physical positions before interpolation (e.g. step rounding).
    position, inverse, counts = np.unique(position, return_inverse=True, return_counts=True)
    combined = np.zeros((position.size, frequencies.size), dtype=complex)
    np.add.at(combined, inverse, traces)
    traces = combined / counts[:, None]
    if position.size < 3 or frequencies.size == 0:
        raise ValueError("At least three distinct positions and one frequency are required.")

    delta = subtract_baseline(position, traces, length_m)
    center = fit_beam_center(position, delta, length_m)
    radius = min(center - REFERENCE_EDGE_M, length_m - REFERENCE_EDGE_M - center)
    if radius <= 0:
        raise ValueError("The fitted centre leaves no radial interval between the reference windows.")
    sampled_radii = np.abs(position - center)
    radii = np.unique(np.r_[0.0, sampled_radii[sampled_radii < radius], radius])
    interpolate = interp1d(position, delta, axis=0, bounds_error=False, fill_value=0.0)
    folded = (interpolate(center - radii) + interpolate(center + radii)) / 2
    folded[-1] = 0  # At this radius one half enters its known off-beam region.
    phase = np.unwrap(np.angle(folded), axis=0)
    # Use the centre to align the radial branches across frequency. This fixes
    # artificial frequency-to-frequency sign flips, not the overall line sign.
    phase += (np.unwrap(phase[0]) - phase[0])[None, :]
    field = field_conversion(frequencies)[None, :] * np.sqrt(np.abs(folded)) * np.exp(0.5j * phase)
    integrated = 2 * np.pi * trapezoid(radii[:, None] * field, radii, axis=0)
    return LineResult(integrated, center, radii, field)


def analyze_run(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    """Save integrals and folded radial profiles as arrays without pickle objects."""
    run_dir = Path(run_dir)
    frequencies, lines = load_run(run_dir)
    indices = np.asarray(sorted(lines), dtype=int)
    if np.any((indices < 0) | (indices >= len(SUBTHREAD_Z_M))):
        raise ValueError("The fixed geometry defines sub-thread indices 0 through 3.")
    results = [analyze_line(lines[i].position_m, lines[i].traces, frequencies, lines[i].length_m)
               for i in indices]
    profile_sizes = np.asarray([result.radii_m.size for result in results])
    output_path = Path(output_path) if output_path is not None else analysis_output_path(run_dir)
    if output_path.suffix != ".npz":
        output_path = Path(f"{output_path}.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ET=np.asarray([result.ET for result in results]),
        frequencies_Hz=frequencies,
        z_m=SUBTHREAD_Z_M[indices],
        z_from_mirror_m=Z_FROM_MIRROR_M[indices],
        sub_thread_indices=indices,
        beam_center_m=np.asarray([result.center_m for result in results]),
        integration_radius_m=np.asarray([result.radii_m[-1] for result in results]),
        electric_field=np.concatenate([result.electric_field for result in results]),
        radius_m=np.concatenate([result.radii_m for result in results]),
        plane_index=np.repeat(np.arange(len(results)), profile_sizes),
        # Each last point was forced to zero for integration, rather than measured.
        profile_fit_mask=np.concatenate([np.arange(size) < size - 1 for size in profile_sizes]),
        sampling_kind="radial_from_one_central_diameter",
        electric_field_kind="sqrt_finite_bead_response",
        reference_edge_m=REFERENCE_EDGE_M,
        bead_diameter_m=BEAD_DIAMETER_M,
        bead_eps_r=BEAD_EPS_R,
        input_power_W=INPUT_POWER_W,
        mirror_z_m=MIRROR_Z_M,
        disk_thickness_m=DISK_THICKNESS_M,
        gap_widths_m=np.asarray(GAP_WIDTHS_M),
        source_run=str(run_dir.resolve()),
    )
    return output_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="directory containing the measurement traces, logs and manifest")
    parser.add_argument("-o", "--output", type=Path,
                        help="output NPZ (default: data/outputs/RUN_NAME/integrated_fields.npz)")
    args = parser.parse_args(argv)
    print(f"Integrated fields saved to: {analyze_run(args.run_dir, args.output)}")


if __name__ == "__main__":
    main()
