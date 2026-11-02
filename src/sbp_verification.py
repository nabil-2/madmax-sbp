"""Run SBP verification on full bead-pull grids reduced to a few lines.

The original calculation lives in verification_core.py. This module provides
short notebook functions for selecting scans, running it and saving ET arrays.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import h5py
import numpy as np
from . import verification_core as core

DEFAULT_GAP_Z_SLICE_STARTS = {1: "smallest", 2: "largest", 3: "smallest", 4: 4}

@dataclass(frozen=True)
class GapInfo:
    rank: int
    path: Path
    z_min_m: float
    z_max_m: float
    z_positions_m: tuple[float, ...]
    shape: tuple[int, ...]

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def z_min_mm(self) -> float:
        return self.z_min_m * 1e3

    @property
    def z_max_mm(self) -> float:
        return self.z_max_m * 1e3

def _read_file_z_positions(file_path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    with h5py.File(file_path, "r") as file:
        shape = tuple(file["s11_map"].shape)
        metadata = json.loads(file.attrs["measurement_config"])
    z_pos = np.linspace(metadata["space"][2][0], metadata["space"][2][1], shape[2]) * 1e-3
    return z_pos, shape

def normalize_z_slice_count(z_slice_option) -> int:
    if np.isscalar(z_slice_option):
        z_slice_count = int(z_slice_option)
    else:
        z_slice_count = len(tuple(int(v) for v in z_slice_option))
    if z_slice_count <= 0:
        raise ValueError(f"z_slice_count must be positive, got {z_slice_count}.")
    return z_slice_count

def gap_z_index_order(gap: GapInfo) -> tuple[int, ...]:
    # Starts are expressed in absolute z (z_abs = z_mirror - z_rel). Since gap.z_positions_m
    # is stored ascending in z_rel, ascending z_abs corresponds to descending z_rel-index.
    # Numeric `k` means "k-th smallest z_abs" (0-indexed); selection proceeds by adding the
    # next-larger z_abs slice, except for the "largest" rule which proceeds by descending z_abs.
    n_z = len(gap.z_positions_m)
    z_abs_ascending = tuple(range(n_z - 1, -1, -1))
    start_rule = DEFAULT_GAP_Z_SLICE_STARTS.get(int(gap.rank), "smallest")

    if isinstance(start_rule, str):
        if start_rule == "smallest":
            return z_abs_ascending
        if start_rule == "largest":
            return tuple(range(n_z))
        raise ValueError(f"Unknown start_rule {start_rule!r} for gap {gap.rank}.")

    start_k = int(start_rule)
    if start_k < 0 or start_k >= n_z:
        raise ValueError(
            f"Gap {gap.rank} has {n_z} z-slices, but its start index is configured as {start_k}."
        )
    return z_abs_ascending[start_k:]

def select_gap_z_ixs(gap: GapInfo, z_slice_option) -> tuple[int, ...]:
    z_slice_count = normalize_z_slice_count(z_slice_option)
    order = gap_z_index_order(gap)
    if z_slice_count > len(order):
        raise ValueError(
            f"Gap {gap.rank} only supports {len(order)} z-slices from its configured start, "
            f"but {z_slice_count} were requested."
        )
    return tuple(int(ix) for ix in order[:z_slice_count])

def match_full_to_reduced(ETs_full, z_full, z_reduced, atol=1e-9):
    z_full = np.asarray(z_full).ravel()
    z_reduced = np.asarray(z_reduced).ravel()
    ix = np.empty(z_reduced.size, dtype=int)
    for i, zr in enumerate(z_reduced):
        matches = np.flatnonzero(np.isclose(z_full, zr, atol=atol, rtol=0.0))
        ix[i] = int(matches[0]) if matches.size > 0 else int(np.argmin(np.abs(z_full - zr)))
    return np.asarray(ETs_full)[ix, :], ix

def _metric_percentiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    return float(np.nanmedian(values)), float(np.nanpercentile(values, 90))

def compute_metrics_record(
    line_count,
    z_slice_count,
    gap_count,
    n_reduced_points,
    ETs_reduced,
    ETs_full,
    z_full,
    z_reduced,
    frequencies,
    fit_costs,
    fit_success,
):
    full_overlap, _ = match_full_to_reduced(ETs_full, z_full, z_reduced)
    reduced = np.asarray(ETs_reduced)
    valid = np.isfinite(reduced) & np.isfinite(full_overlap)
    denom = np.linalg.norm(full_overlap[valid]) if np.any(valid) else np.nan
    complex_rmse = np.linalg.norm((reduced - full_overlap)[valid]) / denom if np.isfinite(denom) and denom > 0 else np.nan

    full_abs = np.abs(full_overlap)
    red_abs = np.abs(reduced)
    rel_abs = np.full_like(full_abs, np.nan, dtype=float)
    valid_abs = valid & (full_abs > np.finfo(float).eps)
    rel_abs[valid_abs] = np.abs(red_abs[valid_abs] - full_abs[valid_abs]) / full_abs[valid_abs]
    med_rel, p90_rel = _metric_percentiles(rel_abs)

    freq = np.asarray(frequencies)
    focus = (freq >= 19e9) & (freq <= 20e9)
    if np.any(focus):
        focus_valid = valid[:, focus]
        focus_full = full_overlap[:, focus]
        focus_red = reduced[:, focus]
        focus_denom = np.linalg.norm(focus_full[focus_valid]) if np.any(focus_valid) else np.nan
        complex_rmse_focus = (
            np.linalg.norm((focus_red - focus_full)[focus_valid]) / focus_denom
            if np.isfinite(focus_denom) and focus_denom > 0
            else np.nan
        )
        focus_rel = rel_abs[:, focus]
        med_rel_focus, p90_rel_focus = _metric_percentiles(focus_rel)
    else:
        complex_rmse_focus = np.nan
        med_rel_focus = np.nan
        p90_rel_focus = np.nan

    fit_costs = np.asarray(fit_costs, dtype=float)
    fit_success = np.asarray(fit_success, dtype=bool)
    finite_costs = fit_costs[np.isfinite(fit_costs)]
    return {
        "line_count": int(line_count),
        "z_slice_count": int(z_slice_count),
        "gap_count": int(gap_count),
        "n_reduced_points": int(n_reduced_points),
        "n_reduced_z": int(np.asarray(z_reduced).size),
        "fit_success_fraction": float(np.nanmean(fit_success)) if fit_success.size else np.nan,
        "fit_cost_median": float(np.nanmedian(finite_costs)) if finite_costs.size else np.nan,
        "fit_cost_p90": float(np.nanpercentile(finite_costs, 90)) if finite_costs.size else np.nan,
        "et_complex_nrmse": float(complex_rmse),
        "rel_abs_et_median": float(med_rel),
        "rel_abs_et_p90": float(p90_rel),
        "et_complex_nrmse_19_20ghz": float(complex_rmse_focus),
        "rel_abs_et_median_19_20ghz": float(med_rel_focus),
        "rel_abs_et_p90_19_20ghz": float(p90_rel_focus),
    }


def discover_gaps(directory):
    """Read grid metadata and number gaps by ascending acquisition z (metres)."""
    paths = sorted(Path(directory).glob('*.hdf5'))
    if not paths:
        raise FileNotFoundError(f'No full-grid .hdf5 files in {directory}')
    gaps = []
    for path in paths:
        z, shape = _read_file_z_positions(path)
        gaps.append(GapInfo(0, path, float(z.min()), float(z.max()), tuple(z), shape))
    gaps.sort(key=lambda gap: (gap.z_min_m, gap.z_max_m))
    return [GapInfo(i + 1, g.path, g.z_min_m, g.z_max_m, g.z_positions_m, g.shape)
            for i, g in enumerate(gaps)]


def verify_file(path, *, line_count=3, z_indices=(0,), frequency_indices=None,
                reference_alignment=True, time_gating=True, n_workers=1):
    """Reconstruct one grid from selected lines using the original fit defaults.

    reference_alignment=True reproduces the original full-map-assisted sign
    and complex-amplitude correction. False uses only line data for the fit.
    The full-grid reference always uses every acquired z plane.
    """
    if line_count not in (1, 2, 3, 4):
        raise ValueError('line_count must be 1, 2, 3 or 4.')
    z_indices = tuple(z_indices)
    z, _ = _read_file_z_positions(Path(path))
    if not z_indices or any(int(i) != i or i < 0 or i >= len(z) for i in z_indices):
        raise ValueError('z_indices must select existing grid planes.')
    data = core.compute_reduced_products(
        path, line_count=line_count, z_ixs_used=z_indices,
        frequency_indices=frequency_indices, time_gating=time_gating,
        reference_alignment=reference_alignment,
    )
    fitter = core.BeamFit(data['E_reduced'], data['frequencies'],
                          data['x_reduced'], data['y_reduced'])
    params, costs, success = fitter.fit_all(n_workers=n_workers)
    reduced = core.compute_ET_reduced(
        params, data['x_pos'], data['y_pos'], core.beam_model,
        data['frequencies'], fit_success=success,
    )
    full = core.compute_ET_full(data['E_full'], data['x_pos'], data['y_pos'])
    return dict(data=data, params=params, costs=costs, success=success,
                ETs_reduced=reduced, ETs_full=full,
                z_reduced=data['z_pos'][list(z_indices)], z_full=data['z_pos'],
                frequencies=data['frequencies'], line_count=line_count,
                reference_alignment=reference_alignment)


def verify_scan(gaps, *, line_count=3, z_slice_count=1, gap_count=None,
                frequency_indices=None, reference_alignment=True,
                time_gating=True, n_workers=1):
    """Compare selected gaps/planes against all full grids.

    Pass discover_gaps(directory). Plane choices reproduce the old notebook:
    in absolute z they start smallest/largest/smallest/fifth-smallest for
    gap ranks 1/2/3/4. The input acquisition z runs in the opposite direction.
    Only the first gap_count gaps are reconstructed from limited lines.
    """
    gaps = list(gaps)
    count = len(gaps) if gap_count is None else gap_count
    if not gaps or not isinstance(count, int) or not 1 <= count <= len(gaps):
        raise ValueError('gap_count must select at least one available gap.')
    if not isinstance(z_slice_count, int) or z_slice_count < 1:
        raise ValueError('z_slice_count must be a positive integer.')
    full_fields, full_z, reduced_fields, reduced_z = [], [], [], []
    costs, successes = [], []
    frequencies = None
    example = None
    z_selection = []
    for gap in gaps:
        if gap.rank <= count:
            indices = select_gap_z_ixs(gap, z_slice_count)
            run = verify_file(gap.path, line_count=line_count, z_indices=indices,
                              frequency_indices=frequency_indices,
                              reference_alignment=reference_alignment,
                              time_gating=time_gating, n_workers=n_workers)
            f = run['frequencies']
            reduced_fields.extend(run['ETs_reduced'])
            reduced_z.extend(run['z_reduced'])
            costs.extend(run['costs'].ravel())
            successes.extend(run['success'].ravel())
            z_selection.append(list(indices))
            example = run if example is None else example
            full_fields.extend(run['ETs_full'])
            full_z.extend(run['z_full'])
        else:
            data = core.load_file(gap.path, frequency_indices=frequency_indices,
                                  time_gating=time_gating)
            root = core.compute_sqrt_Dgamma_full(data['s11'], data['x_pos'], data['y_pos'])
            field = core.sqrt_dgamma_to_E(root, data['frequencies'])
            full_fields.extend(core.compute_ET_full(field, data['x_pos'], data['y_pos']))
            full_z.extend(data['z_pos'])
            f = data['frequencies']
        if frequencies is not None and not np.array_equal(frequencies, f):
            raise ValueError('All grids must have identical selected frequencies.')
        frequencies = f
    order_full, order_reduced = np.argsort(full_z), np.argsort(reduced_z)
    full_z, reduced_z = np.asarray(full_z)[order_full], np.asarray(reduced_z)[order_reduced]
    full_fields = np.asarray(full_fields)[order_full]
    reduced_fields = np.asarray(reduced_fields)[order_reduced]
    metrics = compute_metrics_record(
        line_count, z_slice_count, count, example['data']['x_reduced'].size,
        reduced_fields, full_fields, full_z, reduced_z, frequencies, costs, successes,
    )
    return dict(ETs_reduced=reduced_fields, ETs_full=full_fields,
                z_reduced=reduced_z, z_full=full_z, frequencies=frequencies,
                line_count=line_count, z_slice_count=z_slice_count, gap_count=count,
                z_ixs_used=z_selection, metrics=metrics, example=example,
                source_files=[g.path.name for g in gaps],
                reference_alignment=reference_alignment, time_gating=time_gating)


def save_results(result, path):
    """Save the original ET_results keys as numeric NPZ arrays (no pickle)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **{key: result[key] for key in ('ETs_reduced', 'ETs_full',
                    'z_reduced', 'z_full', 'frequencies')},
        line_count=result['line_count'], z_slice_count=result['z_slice_count'],
        gap_count=result['gap_count'], z_ixs_used=np.asarray(result['z_ixs_used']),
        reference_alignment=result['reference_alignment'], time_gating=result['time_gating'],
        source_files=np.asarray(result['source_files']),
    )
    return path if path.suffix == '.npz' else Path(str(path) + '.npz')
