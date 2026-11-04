"""Read saved bead-pull results and create figures without running measurements or fits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from dataclasses import fields as _fitplot_dataclass_fields, replace as _fitplot_replace
from matplotlib.lines import Line2D as _FitPlotLine
from matplotlib.patches import Patch as _FitPlotPatch, Rectangle as _FitPlotRectangle
from matplotlib.colors import LogNorm as _FitPlotLogNorm

from .fit_1D import ModelConfig as _FitPlotConfig, evaluate_model as _fitplot_evaluate

from . import sbp_analysis
from .data_paths import DATA_DIR, OUTPUTS_DIR, REFERENCE_DATA_DIR, analysis_output_path
from .plot_text import render_text
from .analysis_band import require_analysis_band


_THREAD_COLORS = {0: "#3b6fb6", 1: "#db7b32", 2: "#2a9d8f", 3: "#9c4f96"}


@dataclass
class PipelineData:
    run_dir: Path
    output_dir: Path
    manifest: dict
    frequencies_Hz: np.ndarray
    lines: dict
    fields: dict
    fits_1d: dict
    fit_3d: dict | None
    s_parameter: str
    fit_3d_source: Path | None = None
    fit_3d_is_trial: bool = False
    frequency_range_GHz: tuple[float, float] | None = None
    geometry_correction: dict | None = None
    fit_3d_uses_previous_geometry: bool = False

    def frequency_index(self, frequency_GHz):
        """Index of the acquired frequency nearest the requested GHz value."""
        frequency = float(frequency_GHz) * 1e9
        if not np.isfinite(frequency):
            raise ValueError("The requested frequency must be finite.")
        return int(np.argmin(abs(self.frequencies_Hz - frequency)))


def _read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _matches_spatial_input(fields, spatial, previous_geometry=None):
    """Associate saved spatial fits by their actual inputs, not stale source paths."""
    required = {"profile_fit_mask", "radius_m", "plane_index", "electric_field"}
    if not required.issubset(fields):
        return False
    frequency = fields["frequencies_Hz"]
    ids = np.array([np.argmin(abs(frequency - value)) for value in spatial["frequencies_Hz"]])
    if not np.allclose(frequency[ids], spatial["frequencies_Hz"], rtol=0, atol=1):
        return False
    mask = fields["profile_fit_mask"].astype(bool)
    plane = fields["plane_index"][mask]
    positions = np.column_stack((fields["radius_m"][mask], np.zeros(mask.sum()), fields["z_from_mirror_m"][plane]))
    stored_positions = spatial.get("positions_nominal_m", spatial["positions_m"])
    measured = fields["electric_field"][mask][:, ids]
    stored_field = spatial["electric_field_measured"]
    if (positions.shape != stored_positions.shape or measured.shape != stored_field.shape
            or not np.allclose(measured**2, stored_field**2, rtol=1e-6, atol=1e-8)):
        return False
    if np.allclose(positions, stored_positions, rtol=0, atol=1e-8):
        return True
    # A documented coordinate-only correction may retain an older fit of the
    # identical profiles. Never relax arbitrary position or field mismatches.
    if previous_geometry is None or "gap_widths_nominal_m" not in spatial:
        return False
    previous_z = np.asarray(previous_geometry["z_from_mirror_m"], float)
    previous_gaps = np.asarray(previous_geometry["gap_widths_m"], float)
    saved_gaps = np.asarray(spatial["gap_widths_nominal_m"], float)
    if previous_z.shape != fields["z_from_mirror_m"].shape or previous_gaps.shape != saved_gaps.shape:
        return False
    positions[:, 2] = previous_z[plane]
    return (np.allclose(positions, stored_positions, rtol=0, atol=1e-8)
            and np.allclose(previous_gaps, saved_gaps, rtol=0, atol=1e-12))


def _geometry_correction(output_dir, fields):
    """Read an explicit correction only when it describes the current archive."""
    path = output_dir / "geometry_correction.json"
    if not path.is_file():
        return None
    correction = json.loads(path.read_text())
    if correction.get("schema_version") != 1:
        raise ValueError(f"Unsupported geometry correction schema in {path}.")
    for key in ("gap_widths_m", "z_from_mirror_m", "z_m"):
        expected = np.asarray(correction["corrected_geometry"][key], float)
        actual = np.asarray(fields.get(key, []), float)
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=1e-12):
            raise ValueError(f"Geometry correction does not describe the current {key} in {path}.")
    return correction


def load_pipeline(run_dir=None, output_dir=None, fit_3d_path=None, *, frequency_range_GHz=None):
    """Load the latest completed run and existing analysis files; perform no fits."""
    if run_dir is None:
        completed = []
        for manifest_path in (DATA_DIR / "spider_bead_pull_runs").glob("*/scan_manifest.json"):
            info = json.loads(manifest_path.read_text())
            if info.get("status") == "completed":
                completed.append((info.get("started_utc", manifest_path.parent.name), manifest_path.parent))
        if not completed:
            raise FileNotFoundError(f"No completed measurement runs found in {DATA_DIR / 'spider_bead_pull_runs'}.")
        run_dir = max(completed, key=lambda entry: entry[0])[1]
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "scan_manifest.json").read_text())
    output_dir = Path(output_dir).resolve() if output_dir is not None else analysis_output_path(run_dir).parent
    archive_path = output_dir / "integrated_fields.npz"
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"No saved transverse analysis at {archive_path}. Create it first with:\n"
            f"uv run python -m src.sbp_analysis {run_dir} -o {archive_path}")
    fields = _read_npz(archive_path)
    require_analysis_band(fields["frequencies_Hz"], f"Saved reconstruction {archive_path}")
    correction = _geometry_correction(output_dir, fields)
    previous_geometry = correction["previous_geometry"] if correction is not None else None
    frequencies, lines = sbp_analysis.load_run(run_dir)
    if not np.array_equal(frequencies, fields["frequencies_Hz"]):
        raise ValueError("The saved analysis and raw run have different frequency axes.")
    with h5py.File(run_dir / "s11_traces.h5", "r") as handle:
        attrs = handle["s_parameter"].attrs
        name = attrs.get("s_param")
        if name is None:
            vna = manifest.get("metadata", {}).get("config", {}).get("vna", {})
            a, b = attrs.get("port_a", vna.get("port_a")), attrs.get("port_b", vna.get("port_b"))
            name = f"S{a}{b}" if a is not None and b is not None else "S-parameter"
        s_parameter = name.decode() if isinstance(name, bytes) else str(name)
    fits = {}
    for path in sorted(output_dir.glob("integrated_fields_fit_1D*.npz")):
        fit = _read_npz(path)
        require_analysis_band(fit["frequencies_Hz"], f"Saved 1D fit {path}")
        mode = str(fit.get("fit_method", "fixed"))
        fits[mode] = fit
    spatial = None
    spatial_path = None
    is_trial = False
    if fit_3d_path is not None:
        spatial_path = Path(fit_3d_path).resolve()
        spatial = _read_npz(spatial_path)
        if not _matches_spatial_input(fields, spatial, previous_geometry):
            raise ValueError("The selected 3D fit does not match this run's saved spatial field samples.")
    else:
        candidates = sorted(output_dir.glob("integrated_fields_fit_3D*.npz"))
        if candidates:
            preferred = next((path for path in candidates if path.stem.endswith("_fixed")), candidates[0])
            spatial_path = preferred
            spatial = _read_npz(preferred)
            if not _matches_spatial_input(fields, spatial, previous_geometry):
                raise ValueError(f"The 3D fit at {preferred} does not match this run's spatial field samples.")
        else:
            trial = OUTPUTS_DIR / "validation/fit_3D/recorded_fit_3D_smoke.npz"
            if trial.is_file():
                candidate = _read_npz(trial)
                if _matches_spatial_input(fields, candidate, previous_geometry):
                    spatial_path, spatial, is_trial = trial, candidate, True
    if spatial_path is not None:
        is_trial = is_trial or "validation" in spatial_path.parts or "smoke" in spatial_path.stem
    data = PipelineData(run_dir, output_dir, manifest, frequencies, lines, fields, fits, spatial,
                        s_parameter, spatial_path, is_trial, geometry_correction=correction,
                        fit_3d_uses_previous_geometry=(spatial is not None
                                                      and not _matches_spatial_input(fields, spatial)))
    return select_frequency_range(data, frequency_range_GHz) if frequency_range_GHz is not None else data


def select_frequency_range(data, frequency_range_GHz):
    """Return a display-only frequency subset; saved fits and geometry are unchanged.

    Explicit axis mappings prevent geometry vectors being mistaken for frequency
    arrays when their lengths coincide. No acquisition or fitting is performed.
    """
    bounds = np.asarray(frequency_range_GHz, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or not 0 < bounds[0] < bounds[1]:
        raise ValueError("frequency_range_GHz must be two positive increasing finite bounds.")
    low, high = bounds*1e9

    def mask_for(frequency):
        return (frequency >= low) & (frequency <= high)

    def subset(record, axes):
        mask = mask_for(record["frequencies_Hz"])
        if not np.any(mask):
            return None
        result = dict(record)
        for key, axis in {"frequencies_Hz": 0, **axes}.items():
            if key in result:
                result[key] = np.compress(mask, result[key], axis=axis)
        return result

    mask = mask_for(data.frequencies_Hz)
    if not np.any(mask):
        raise ValueError("No acquired frequencies lie in the requested plotting range.")
    fields = subset(data.fields, {"ET": 1, "electric_field": 1})
    lines = {index: _fitplot_replace(line, traces=line.traces[:, mask]) for index, line in data.lines.items()}
    fit_axes = {"ET_measured": 1, "ET_fitted_magnitude": 1, "ET_unperturbed": 1,
                "amplitude": 0, "integral_ET_dz": 0, "beta2": 0, "residual_rms": 0}
    fits = {mode: fit for mode, saved in data.fits_1d.items() if (fit := subset(saved, fit_axes)) is not None}
    spatial_axes = {"electric_field_measured": 1, "electric_field_unperturbed": 1, "predicted_squared": 1,
                    "coefficients": 0, "beta2": 0, "beta2_total": 0, "relative_residual": 0,
                    "incident_power_fraction": 0, "beam_width_seed_m": 0}
    spatial = subset(data.fit_3d, spatial_axes) if data.fit_3d is not None else None
    return _fitplot_replace(data, frequencies_Hz=data.frequencies_Hz[mask], lines=lines, fields=fields,
                            fits_1d=fits, fit_3d=spatial, frequency_range_GHz=tuple(bounds))


def _limit_frequency_axis(axis, data):
    """Trim automatic margins to the selected band without widening a zoom."""
    if data.frequency_range_GHz is not None:
        low, high = axis.get_xlim()
        axis.set_xlim(max(low, data.frequency_range_GHz[0]), min(high, data.frequency_range_GHz[1]))


def _thread_indices(data):
    return [int(index) for index in data.fields["sub_thread_indices"]]


def _plane(data, index):
    return _thread_indices(data).index(index)


def _thread_title(data, index, template="{thread_name} · {z_mm:.3f} mm from mirror", **context):
    entries = data.manifest["calibration"]["sub_threads"]
    sub = next((entry for entry in entries if int(entry["index"]) == index), {})
    label = sub.get("name") or f"Thread {index}"
    z = float(data.fields["z_from_mirror_m"][_plane(data, index)])
    return render_text(template, thread_name=label, thread_index=index, z_m=z, z_mm=z*1e3,
                       s_parameter=data.s_parameter, **context)


def _deduplicated_line(line):
    position, inverse, counts = np.unique(line.position_m, return_inverse=True, return_counts=True)
    traces = np.zeros((position.size, line.traces.shape[1]), dtype=complex)
    np.add.at(traces, inverse, line.traces)
    return position, traces / counts[:, None]


def _reference_profile(data, index, frequency_index):
    line = data.lines[index]
    position, all_traces = _deduplicated_line(line)
    trace = all_traces[:, frequency_index]
    edge = float(data.fields.get("reference_edge_m", sbp_analysis.REFERENCE_EDGE_M))
    reference = (position <= edge + 1e-12) | (position >= line.length_m - edge - 1e-12)
    design = np.column_stack((np.ones(position.size), position / line.length_m))
    baseline = design @ np.linalg.lstsq(design[reference], trace[reference], rcond=None)[0]
    delta = trace - baseline
    delta[reference] = 0
    return position, trace, baseline, delta


def _shade_reference(ax, data, index, coordinate_axis="x"):
    edge = float(data.fields.get("reference_edge_m", sbp_analysis.REFERENCE_EDGE_M))
    length = data.lines[index].length_m
    shade = ax.axvspan if coordinate_axis == "x" else ax.axhspan
    shade(0, 100 * edge, color="0.7", alpha=0.2)
    shade(100 * (length - edge), 100 * length, color="0.7", alpha=0.2)


def _radial_profile(data, index, frequency_index):
    selected = data.fields["plane_index"] == _plane(data, index)
    radius, field = data.fields["radius_m"][selected], data.fields["electric_field"][selected, frequency_index]
    order = np.argsort(radius)
    return radius[order], field[order]


def _masked_phase(values, weak_fraction=1e-3):
    phase = np.unwrap(np.angle(values))
    amplitude = abs(values)
    return np.where(amplitude > weak_fraction * np.max(amplitude, initial=0.0), phase, np.nan)


def _grid(count, columns=2, row_height=3.1):
    rows = max(1, int(np.ceil(count / columns)))
    figure, axes = plt.subplots(rows, columns, figsize=(12, row_height * rows), squeeze=False,
                                constrained_layout=True)
    for ax in axes.ravel()[count:]:
        ax.set_visible(False)
    return figure, axes.ravel()


def plot_measurement_map(
        data, *, title="Raw measurement · {s_parameter}",
        panel_title_template="{thread_name} · {z_mm:.3f} mm from mirror",
        x_label="Frequency [GHz]", y_label="Distance from calibrated start [cm]",
        colorbar_label="{s_parameter} magnitude [dB]"):
    """Acquired magnitudes; panel templates support thread_name/thread_index/z_mm."""
    indices = _thread_indices(data)
    profiles = [_deduplicated_line(data.lines[index]) for index in indices]
    maps = [20 * np.log10(np.maximum(abs(trace), 1e-12)) for _, trace in profiles]
    lower, upper = min(np.min(values) for values in maps), max(np.max(values) for values in maps)
    figure, axes = _grid(len(indices))
    for ax, index, (position, _), values in zip(axes, indices, profiles, maps):
        image = ax.pcolormesh(data.frequencies_Hz / 1e9, position * 100, values,
                              shading="auto", cmap="viridis", vmin=lower, vmax=upper, rasterized=True)
        _shade_reference(ax, data, index, coordinate_axis="y")
        ax.set(title=_thread_title(data, index, panel_title_template), xlabel=x_label, ylabel=y_label)
    figure.colorbar(image, ax=list(axes[:len(indices)]), label=render_text(colorbar_label, s_parameter=data.s_parameter))
    figure.suptitle(render_text(title, s_parameter=data.s_parameter))
    for axis in axes[:len(indices)]:
        _limit_frequency_axis(axis, data)
    return figure


def plot_raw_traces(
        data, frequency_GHz=20.825, *,
        title="Raw traces · {frequency_GHz:.4f} GHz · shaded off-beam references",
        panel_title_template="{thread_name} · {z_mm:.3f} mm from mirror",
        x_label="Distance from calibrated start [cm]", y_label="{s_parameter} [dimensionless]",
        real_label="Real", imaginary_label="Imaginary"):
    """Complex raw traces; frequency_GHz in text is the actual acquired bin."""
    fi = data.frequency_index(frequency_GHz)
    frequency = data.frequencies_Hz[fi]/1e9
    figure, axes = _grid(len(_thread_indices(data)))
    for ax, index in zip(axes, _thread_indices(data)):
        position, traces = _deduplicated_line(data.lines[index])
        ax.plot(position * 100, traces[:, fi].real, label=real_label)
        ax.plot(position * 100, traces[:, fi].imag, label=imaginary_label)
        _shade_reference(ax, data, index)
        ax.set(title=_thread_title(data, index, panel_title_template, frequency_GHz=frequency), xlabel=x_label,
               ylabel=render_text(y_label, s_parameter=data.s_parameter))
        ax.legend()
    figure.suptitle(render_text(title, frequency_GHz=frequency, s_parameter=data.s_parameter))
    return figure


def plot_reference_subtraction(
        data, frequency_GHz=20.825, *,
        title="Off-beam reference subtraction · {frequency_GHz:.4f} GHz",
        panel_title_template="{thread_name} · {z_mm:.3f} mm from mirror · {component}",
        x_label="Distance from calibrated start [cm]", real_y_label="Re {s_parameter}",
        imaginary_y_label="Im {s_parameter}", real_component_label="Real", imaginary_component_label="Imaginary",
        measured_label="Raw {s_parameter}", baseline_label="Complex affine baseline"):
    """Complex affine reference fit; component supplies the editable panel suffix."""
    fi = data.frequency_index(frequency_GHz)
    frequency = data.frequencies_Hz[fi]/1e9
    indices = _thread_indices(data)
    figure, axes = _grid(2 * len(indices), row_height=2.5)
    for row, index in enumerate(indices):
        position, trace, baseline, _ = _reference_profile(data, index, fi)
        for component, name in enumerate((real_component_label, imaginary_component_label)):
            ax = axes[2 * row + component]
            measured = trace.real if component == 0 else trace.imag
            fitted = baseline.real if component == 0 else baseline.imag
            ax.plot(position * 100, measured, ".-", markersize=3,
                    label=render_text(measured_label, s_parameter=data.s_parameter))
            ax.plot(position * 100, fitted, "--", color="0.2", label=baseline_label)
            _shade_reference(ax, data, index)
            ax.set(title=_thread_title(data, index, panel_title_template, component=name, frequency_GHz=frequency), xlabel=x_label,
                   ylabel=render_text(real_y_label if component == 0 else imaginary_y_label, s_parameter=data.s_parameter))
            if row == 0:
                ax.legend()
    figure.suptitle(render_text(title, frequency_GHz=frequency, s_parameter=data.s_parameter))
    return figure


def plot_beam_center(
        data, frequency_GHz=20.825, *, title="Beam-centre symmetry · {frequency_GHz:.4f} GHz",
        panel_title_template="{thread_name} · {z_mm:.3f} mm from mirror · centre {center_cm:.2f} cm",
        x_label="Distance from saved centre [cm]", y_label="ΔΓ component [dimensionless]",
        left_real_label="Left real", left_imaginary_label="Left imaginary",
        right_real_label="Right real", right_imaginary_label="Right imaginary"):
    """Compare halves around the saved centre; templates also support center_cm."""
    fi = data.frequency_index(frequency_GHz)
    frequency = data.frequencies_Hz[fi]/1e9
    figure, axes = _grid(len(_thread_indices(data)))
    for ax, index in zip(axes, _thread_indices(data)):
        position, _, _, delta = _reference_profile(data, index, fi)
        center = float(data.fields["beam_center_m"][_plane(data, index)])
        radius, _ = _radial_profile(data, index, fi)
        left = np.interp(center - radius, position, delta, left=0, right=0)
        right = np.interp(center + radius, position, delta, left=0, right=0)
        for values, style, real, imaginary in ((left, "-", left_real_label, left_imaginary_label),
                                              (right, "--", right_real_label, right_imaginary_label)):
            ax.plot(radius * 100, values.real, style, color="C0", label=real)
            ax.plot(radius * 100, values.imag, style, color="C1", label=imaginary)
        ax.set(title=_thread_title(data, index, panel_title_template, center_cm=center*100, frequency_GHz=frequency),
               xlabel=x_label, ylabel=y_label)
        ax.legend(fontsize="small")
    figure.suptitle(render_text(title, frequency_GHz=frequency))
    return figure


def plot_radial_fields(
        data, frequency_GHz=20.825, weak_fraction=1e-3, *,
        title="Finite-bead effective radial fields · {frequency_GHz:.4f} GHz",
        amplitude_title_template="{thread_name} · {z_mm:.3f} mm from mirror",
        phase_title="Saved radial square-root branch", x_label="Radius [cm]",
        amplitude_y_label="Effective |E| [V/m]", phase_y_label="Effective field phase [rad]",
        endpoint_label="Forced-zero integration endpoint"):
    """Folded effective fields with editable amplitude/phase panel text."""
    fi = data.frequency_index(frequency_GHz)
    frequency = data.frequencies_Hz[fi]/1e9
    indices = _thread_indices(data)
    figure, axes = _grid(2 * len(indices), row_height=2.5)
    for row, index in enumerate(indices):
        radius, field = _radial_profile(data, index, fi)
        color = _THREAD_COLORS.get(index, "C0")
        amplitude_ax, phase_ax = axes[2 * row:2 * row + 2]
        amplitude_ax.plot(radius * 100, abs(field), color=color)
        amplitude_ax.plot(radius[-1] * 100, abs(field[-1]), "o", color="0.5", fillstyle="none", label=endpoint_label)
        phase_ax.plot(radius * 100, _masked_phase(field, weak_fraction), color=color)
        amplitude_ax.set(title=_thread_title(data, index, amplitude_title_template, frequency_GHz=frequency), ylabel=amplitude_y_label)
        phase_ax.set(title=_thread_title(data, index, phase_title, frequency_GHz=frequency), ylabel=phase_y_label)
        for ax in (amplitude_ax, phase_ax):
            ax.set_xlabel(x_label)
        if row == 0:
            amplitude_ax.legend(fontsize="small")
    figure.suptitle(render_text(title, frequency_GHz=frequency))
    return figure


def plot_radial_integration(
        data, frequency_GHz=20.825, *, title="Transverse integration · {frequency_GHz:.4f} GHz",
        integrand_title_template="{thread_name} · {z_mm:.3f} mm from mirror",
        cumulative_title="Cumulative complex transverse integral", x_label="Radius [cm]",
        integrand_y_label="2πr E(r) [V]", cumulative_y_label="Integral [V m]",
        real_label="Real", imaginary_label="Imaginary", magnitude_label="Magnitude"):
    """Complex integrands and cumulative integrals with editable panel/legend labels."""
    fi = data.frequency_index(frequency_GHz)
    frequency = data.frequencies_Hz[fi]/1e9
    indices = _thread_indices(data)
    figure, axes = _grid(2 * len(indices), row_height=2.5)
    for row, index in enumerate(indices):
        radius, field = _radial_profile(data, index, fi)
        integrand = 2 * np.pi * radius * field
        cumulative = cumulative_trapezoid(integrand, radius, initial=0)
        left, right = axes[2 * row:2 * row + 2]
        left.plot(radius * 100, integrand.real, label=real_label)
        left.plot(radius * 100, integrand.imag, label=imaginary_label)
        right.plot(radius * 100, abs(cumulative), color=_THREAD_COLORS.get(index, "C0"), label=magnitude_label)
        right.plot(radius * 100, cumulative.real, "--", label=real_label)
        right.plot(radius * 100, cumulative.imag, ":", label=imaginary_label)
        left.set(title=_thread_title(data, index, integrand_title_template, frequency_GHz=frequency), ylabel=integrand_y_label)
        right.set(title=_thread_title(data, index, cumulative_title, frequency_GHz=frequency), ylabel=cumulative_y_label)
        for ax in (left, right):
            ax.set_xlabel(x_label)
            ax.legend(fontsize="small")
    figure.suptitle(render_text(title, frequency_GHz=frequency))
    return figure


def plot_transverse_maps(
        data, frequency_GHz=20.825, grid_size=181, *,
        title="Radial-symmetry reconstruction · {frequency_GHz:.4f} GHz\nOne central diameter measured per plane",
        panel_title_template="{thread_name} · {z_mm:.3f} mm from mirror",
        x_label="Recentered x [cm]", y_label="Recentered y [cm]", colorbar_label="Effective |E| [V/m]"):
    """Radial-symmetry reconstructions; all figure text is supplied by keywords."""
    fi = data.frequency_index(frequency_GHz)
    frequency = data.frequencies_Hz[fi]/1e9
    indices = _thread_indices(data)
    profiles = [_radial_profile(data, index, fi) for index in indices]
    extent = max(radius[-1] for radius, _ in profiles)
    axis = np.linspace(-extent, extent, grid_size)
    x, y = np.meshgrid(axis, axis)
    radial_grid = np.hypot(x, y)
    maximum = max(np.max(abs(field)) for _, field in profiles)
    figure, axes = _grid(len(indices), row_height=3.5)
    for ax, index, (radius, field) in zip(axes, indices, profiles):
        image_field = np.interp(radial_grid.ravel(), radius, field, left=np.nan, right=np.nan).reshape(x.shape)
        image = ax.imshow(abs(image_field), origin="lower", extent=np.array([-extent, extent, -extent, extent]) * 100,
                           cmap="magma", vmin=0, vmax=maximum, aspect="equal")
        ax.set(title=_thread_title(data, index, panel_title_template, frequency_GHz=frequency), xlabel=x_label, ylabel=y_label)
    figure.colorbar(image, ax=list(axes[:len(indices)]), label=colorbar_label)
    figure.suptitle(render_text(title, frequency_GHz=frequency))
    return figure


def plot_integrated_fields(
        data, *, title="Transversely integrated effective field", phase_title="",
        x_label="Frequency [GHz]", amplitude_y_label="|ET| [V m]", phase_y_label="Saved branch phase [rad]",
        thread_label_template="{thread_name} · {z_mm:.3f} mm from mirror"):
    """Saved transverse integrals; thread_label_template controls the legend."""
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for plane, index in enumerate(_thread_indices(data)):
        values = data.fields["ET"][plane]
        color = _THREAD_COLORS.get(index, "C0")
        axes[0].plot(data.frequencies_Hz / 1e9, abs(values), color=color,
                     label=_thread_title(data, index, thread_label_template))
        axes[1].plot(data.frequencies_Hz / 1e9, _masked_phase(values), color=color)
    axes[0].set(ylabel=amplitude_y_label, title=title)
    axes[0].legend(fontsize="small")
    axes[1].set(xlabel=x_label, ylabel=phase_y_label, title=phase_title)
    _limit_frequency_axis(axes[0], data)
    return figure


def configure_style():
    """Readable notebook figures and lightweight, publication-ready exports."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "figure.titlesize": 14, "figure.dpi": 110,
        "savefig.dpi": 160, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.16, "grid.linewidth": 0.6,
        "lines.linewidth": 1.6, "legend.framealpha": 0.95,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def present(figure, stem, output_dir=None):
    """Display one figure once; optionally export PNG/PDF under the run's outputs."""
    from IPython.display import display
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            figure.savefig(destination / f"{stem}.{extension}", bbox_inches="tight")
    display(figure)
    plt.close(figure)


def summary(data, *, compact=False):
    """A compact notebook table describing the selected run and saved results."""
    import pandas as pd
    from IPython.display import Markdown, display
    rows = []
    for index in _thread_indices(data):
        plane = _plane(data, index)
        rows.append({"Sub-thread": index,
                     "z from mirror [mm]": data.fields["z_from_mirror_m"][plane] * 1e3,
                     "Raw positions": len(data.lines[index].position_m),
                     "Line length [cm]": data.lines[index].length_m * 100,
                     "Beam centre from start [cm]": data.fields["beam_center_m"][plane] * 100,
                     "Integration radius [cm]": data.fields["integration_radius_m"][plane] * 100})
    try:
        output_label = data.output_dir.relative_to(DATA_DIR.parent)
    except ValueError:
        output_label = data.output_dir
    display(Markdown(f"**Run:** `{data.run_dir.name}` · **{data.s_parameter}** · "
                     f"{len(data.frequencies_Hz):,} frequencies, "
                     f"{data.frequencies_Hz.min()/1e9:g}–{data.frequencies_Hz.max()/1e9:g} GHz.  \n"
                     f"**Saved 1D cases:** {', '.join(data.fits_1d) or 'none'}.  \n"
                     f"**Outputs:** `{output_label}`"))
    display(pd.DataFrame(rows).set_index("Sub-thread").round(4))
    if compact:
        return
    if data.geometry_correction is not None:
        display(Markdown(f"**Geometry correction:** {data.geometry_correction['description']}"))
    if data.fit_3d is not None:
        result = data.fit_3d
        label = "validation trial" if data.fit_3d_is_trial else str(result.get("fit_method", "saved fit"))
        residual = result["relative_residual"]
        display(Markdown(f"**3D {label}:** {len(result['frequencies_Hz'])} saved frequencies, "
                         f"weighted complex squared-field residual "
                         f"{residual.min():.1%}–{residual.max():.1%}. "
                         "The 3D figures below retain that fit-quality information."))
        if data.fit_3d_uses_previous_geometry:
            display(Markdown("This saved 3D result uses the **geometry before the correction**. "
                             "Its profile values match this run; its old plane coordinates are retained."))
    else:
        display(Markdown("**3D:** no matching saved result. Its optional plots will show a loading note."))


def plot_transverse_coherence(
        data, *, title=r"Transverse phase coherence: $|\int E\,dA| / \int |E|\,dA$",
        x_label="Frequency [GHz]", y_label="Coherent / amplitude-only integral",
        thread_label_template="{thread_name} · {z_mm:.3f} mm from mirror"):
    """Radial phase cancellation, with editable axis and per-plane legend text."""
    figure, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for plane, index in enumerate(_thread_indices(data)):
        selected = data.fields["plane_index"] == plane
        radius = data.fields["radius_m"][selected]
        field = data.fields["electric_field"][selected]
        incoherent = 2*np.pi*trapezoid(radius[:, None]*abs(field), radius, axis=0)
        ratio = np.divide(abs(data.fields["ET"][plane]), incoherent,
                          out=np.full_like(incoherent, np.nan), where=incoherent > 0)
        ax.plot(data.frequencies_Hz/1e9, ratio, color=_THREAD_COLORS.get(index, "C0"),
                label=_thread_title(data, index, thread_label_template))
    ax.set(xlabel=x_label, ylabel=y_label, ylim=(0, 1.04), title=title)
    ax.legend(fontsize="small", ncol=2)
    _limit_frequency_axis(ax, data)
    return figure


def plot_1d_residual_map(
        data, *, title="1D spatial residuals · shared colour scale",
        x_label="Frequency [GHz]", y_label="Sub-thread",
        colorbar_label="(Model − measured magnitude) / measured norm [%]",
        fixed_panel_title="Fixed", gaps_panel_title="Gaps", positions_panel_title="Positions", both_panel_title="Both",
        thread_tick_template="{thread_index}", missing_title="No saved 1D fits",
        missing_message="Run src.fit_1D on integrated_fields.npz first."):
    """Signed magnitude residuals, including editable case names and thread ticks."""
    if not data.fits_1d:
        return _message_figure(missing_title, missing_message)
    modes = list(data.fits_1d)
    panel_titles = {"fixed": fixed_panel_title, "gaps": gaps_panel_title,
                    "positions": positions_panel_title, "both": both_panel_title}
    residuals = []
    for mode in modes:
        fit = data.fits_1d[mode]
        observed = abs(fit["ET_measured"])
        norm = np.linalg.norm(observed, axis=0)
        residuals.append(np.divide(fit["ET_fitted_magnitude"] - observed, norm,
                                   out=np.zeros_like(observed), where=norm > 0))
    maximum = max(max(float(np.max(abs(values))) for values in residuals), 1e-12)
    figure, axes = _grid(len(modes))
    for ax, mode, residual in zip(axes, modes, residuals):
        fit = data.fits_1d[mode]
        image = ax.pcolormesh(fit["frequencies_Hz"]/1e9, np.arange(residual.shape[0]),
                              residual*100, shading="nearest", cmap="RdBu_r",
                              vmin=-maximum*100, vmax=maximum*100, rasterized=True)
        ax.set(title=render_text(panel_titles.get(mode, "{geometry}"), geometry=mode), xlabel=x_label, ylabel=y_label)
        ax.set_yticks(np.arange(residual.shape[0]),
                      [render_text(thread_tick_template, thread_index=index) for index in _thread_indices(data)])
    figure.colorbar(image, ax=list(axes[:len(modes)]), label=colorbar_label)
    figure.suptitle(title)
    for axis in axes[:len(modes)]:
        _limit_frequency_axis(axis, data)
    return figure


def plot_beam_reference(
        data, frequency_GHz=20.825, *,
        title="Beam-width initialization · reference z is not registered to the booster",
        spectrum_title="Measured S31 beam-width fits",
        planes_title="Reference scan planes · {frequency_GHz:.3f} GHz",
        frequency_x_label="Reference frequency [GHz]", position_x_label="Reference scan z [cm]",
        y_label="1/e field-amplitude radius [cm]",
        reference_label_template="Reference z={z_cm:.0f} cm", mean_label="Mean used for initial guess"):
    """Reference beam radii; plane title uses the actual selected reference bin."""
    import pandas as pd
    path = REFERENCE_DATA_DIR / "beam_measurements/beam_fit_S31.csv"
    frame = pd.read_csv(path, comment="#", header=None,
                        names=["frequency_hz", "z_m", "amplitude", "waist_m", "x0_m", "y0_m"])
    if data.frequency_range_GHz is not None:
        low, high = np.asarray(data.frequency_range_GHz)*1e9
        frame = frame[frame.frequency_hz.between(low, high)]
        if frame.empty:
            raise ValueError("No beam-reference frequencies lie in the requested plotting range.")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for z, rows in frame.groupby("z_m"):
        axes[0].plot(rows.frequency_hz/1e9, rows.waist_m*100, alpha=0.7,
                     label=render_text(reference_label_template, z_cm=z*100, z_m=z))
    means = frame.groupby("frequency_hz").waist_m.mean()
    axes[0].plot(means.index/1e9, means.values*100, color="black", lw=2.4, label=mean_label)
    requested = data.frequencies_Hz[data.frequency_index(frequency_GHz)]
    chosen = min(means.index, key=lambda value: abs(value-requested))
    rows = frame[frame.frequency_hz == chosen].sort_values("z_m")
    axes[0].axvline(chosen/1e9, color="0.4", ls=":")
    axes[1].plot(rows.z_m*100, rows.waist_m*100, "o-", color="C0")
    axes[0].set(xlabel=frequency_x_label, ylabel=y_label, title=render_text(spectrum_title, frequency_GHz=chosen/1e9))
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set(xlabel=position_x_label, ylabel=y_label, title=render_text(planes_title, frequency_GHz=chosen/1e9))
    figure.suptitle(render_text(title, frequency_GHz=chosen/1e9))
    _limit_frequency_axis(axes[0], data)
    return figure


def _message_figure(title, message):
    figure, ax = plt.subplots(figsize=(12, 2.4), constrained_layout=True)
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, wrap=True)
    return figure


def plot_3d_profiles(
    data, frequency_GHz=20.825, *,
    title="{result_label} · saved frequency {frequency_GHz:.4f} GHz · weighted Q residual {relative_residual:.1%}",
    panel_title_template="Sub-thread {thread_index} · z={z_mm:.3f} mm",
    x_label="Radius from beam axis [cm]",
    magnitude_y_label="Local field magnitude [V/m]",
    phase_y_label="arg(Q measured / Q model) [rad]",
    measured_curve_label="Measured effective field",
    bead_curve_label="Fitted bead response",
    unperturbed_curve_label="Unperturbed model",
    trial_label="3D validation trial",
    fit_label="3D spatial fit",
    missing_title="3D spatial fit",
    missing_message="No matching saved 3D result. Set FIT_3D_PATH in the configuration cell.",
):
    """Saved local fields and complex-Q phase residuals, without rerunning Julia.

    Text templates can use frequency_GHz, relative_residual and result_label;
    panel text additionally provides plane_index, thread_index, z_m and z_mm.
    """
    result = data.fit_3d
    if result is None:
        return _message_figure(render_text(missing_title), render_text(missing_message))
    fi = int(np.argmin(abs(result["frequencies_Hz"]-frequency_GHz*1e9)))
    frequency = result["frequencies_Hz"][fi]
    context = dict(frequency_GHz=frequency/1e9, relative_residual=result["relative_residual"][fi])
    context["result_label"] = render_text(trial_label if data.fit_3d_is_trial else fit_label, **context)
    planes = np.unique(result["plane_index"])
    figure, axes = plt.subplots(2, len(planes), figsize=(3.8*len(planes), 6.7),
                                squeeze=False, constrained_layout=True)
    for col, plane in enumerate(planes):
        selected = np.flatnonzero(result["plane_index"] == plane)
        radius = np.hypot(*result["positions_m"][selected, :2].T)
        order = np.argsort(radius)
        selected, radius = selected[order], radius[order]
        measured = result["electric_field_measured"][selected, fi]
        squared = result["predicted_squared"][selected, fi]
        unperturbed = result["electric_field_unperturbed"][selected, fi]
        index = _thread_indices(data)[int(plane)] if int(plane) < len(_thread_indices(data)) else int(plane)
        z = result["positions_m"][selected[0], 2]
        panel_context = dict(context, plane_index=int(plane), thread_index=index, z_m=z, z_mm=z*1e3)
        upper, lower = axes[:, col]
        upper.plot(radius*100, abs(measured), "o", markersize=3.5, color="#df8533", label=render_text(measured_curve_label, **panel_context))
        upper.plot(radius*100, np.sqrt(abs(squared)), "--", color="#b650a0", label=render_text(bead_curve_label, **panel_context))
        upper.plot(radius*100, abs(unperturbed), "-.", color="#238b58", label=render_text(unperturbed_curve_label, **panel_context))
        threshold = max(np.max(abs(measured)**2), np.max(abs(squared)), 1e-30)*1e-3
        valid = (abs(measured)**2 > threshold) & (abs(squared) > threshold)
        phase_error = np.angle(measured**2*np.conj(squared))
        lower.plot(radius[valid]*100, phase_error[valid], ".", color="C0", markersize=4)
        lower.axhline(0, color="0.35", lw=1)
        upper.set(title=render_text(panel_title_template, **panel_context), ylabel=render_text(magnitude_y_label, **panel_context))
        lower.set(xlabel=render_text(x_label, **panel_context), ylabel=render_text(phase_y_label, **panel_context), ylim=(-np.pi, np.pi))
        if col == 0:
            upper.legend(fontsize=8)
    figure.suptitle(render_text(title, **context))
    return figure


def plot_3d_summary(
    data, *,
    title="{result_label} · {n_frequencies} saved frequencies · markers show evaluated points",
    x_label="Saved frequency [GHz]",
    boost_y_label="Power boost β²",
    residual_y_label="Weighted complex squared-field residual [%]",
    incident_power_y_label="Fitted incident power / reference input power",
    mode_y_label="Incoming mode index",
    colorbar_label="|c|²",
    boost_title="Saved 3D model values",
    quality_title="3D fit quality",
    incident_power_title="Incoming-mode normalization",
    coefficient_title="Coefficient magnitudes squared",
    receive_curve_label="Receive-coupled β²",
    total_curve_label="Total emitted β²",
    input_power_curve_label="Input-power normalization",
    frequency_tick_template="{frequency_GHz:.3f}",
    mode_tick_template="{mode_index}",
    trial_label="3D validation trial",
    fit_label="3D fit",
    missing_title="3D fit summary",
    missing_message="No matching saved 3D result is available.",
):
    """Distinct 3D received/emitted boosts and fit diagnostics at saved frequencies.

    Text templates can use n_frequencies and result_label. Manual frequency
    ticks also provide frequency_GHz, and mode ticks provide mode_index.
    """
    result = data.fit_3d
    if result is None:
        return _message_figure(render_text(missing_title), render_text(missing_message))
    frequency = result["frequencies_Hz"]/1e9
    context = dict(n_frequencies=len(frequency))
    context["result_label"] = render_text(trial_label if data.fit_3d_is_trial else fit_label, **context)
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    axes[0, 0].plot(frequency, result["beta2"], "o", color="C0", label=render_text(receive_curve_label, **context))
    axes[0, 0].plot(frequency, result["beta2_total"], "s", color="C1", label=render_text(total_curve_label, **context))
    if np.all(result["beta2"] > 0) and np.all(result["beta2_total"] > 0):
        axes[0, 0].set_yscale("log")
    axes[0, 0].set(ylabel=render_text(boost_y_label, **context), title=render_text(boost_title, **context))
    axes[0, 0].legend(fontsize="small")
    axes[0, 1].plot(frequency, result["relative_residual"]*100, "o", color="#b73838")
    axes[0, 1].set(ylabel=render_text(residual_y_label, **context), title=render_text(quality_title, **context))
    axes[0, 1].set_ylim(0, max(100, float(np.max(result["relative_residual"])*110)))
    axes[1, 0].plot(frequency, result["incident_power_fraction"], "o", color="C2")
    axes[1, 0].axhline(1, color="0.45", ls="--", label=render_text(input_power_curve_label, **context))
    axes[1, 0].set(ylabel=render_text(incident_power_y_label, **context), title=render_text(incident_power_title, **context))
    axes[1, 0].legend(fontsize="small")
    coefficients = abs(result["coefficients"])**2
    image = axes[1, 1].imshow(coefficients.T, origin="lower", aspect="auto", interpolation="nearest", cmap="viridis")
    axes[1, 1].set_xticks(np.arange(len(frequency)),
                         [render_text(frequency_tick_template, frequency_GHz=value, **context) for value in frequency])
    axes[1, 1].set_yticks(np.arange(coefficients.shape[1]),
                         [render_text(mode_tick_template, mode_index=index, **context) for index in range(1, coefficients.shape[1]+1)])
    axes[1, 1].set(ylabel=render_text(mode_y_label, **context), title=render_text(coefficient_title, **context))
    figure.colorbar(image, ax=axes[1, 1], label=render_text(colorbar_label, **context))
    for ax in axes.ravel():
        ax.set_xlabel(render_text(x_label, **context))
    figure.suptitle(render_text(title, **context))
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        _limit_frequency_axis(axis, data)
    return figure


def interactive_explorer(
        data, frequency_GHz=20.825, geometry="both", *, plot_functions=None,
        frequency_control_label="GHz", geometry_control_label="1D case", view_control_label="View",
        diagnostic_view_label="1D field + setup", radial_view_label="Radial field",
        reference_view_label="Reference subtraction", center_view_label="Beam-centre symmetry",
        transverse_view_label="Transverse reconstruction", spatial_view_label="3D saved profiles",
        fixed_geometry_label="fixed", gaps_geometry_label="gaps",
        positions_geometry_label="positions", both_geometry_label="both"):
    """Live views; plot_functions accepts the same configured plot callables as static cells.

    Keys are diagnostic, radial, reference, center, transverse and spatial.
    Configure a callable with functools.partial to share its named text arguments
    with the static plot and keep frequency-dependent templates live.
    """
    import ipywidgets as widgets
    from IPython.display import display
    if not data.fits_1d:
        raise ValueError("The interactive explorer needs at least one saved 1D fit.")
    available = list(data.fits_1d)
    geometry = geometry if geometry in available else available[0]
    grid = data.frequencies_Hz/1e9
    initial = float(grid[data.frequency_index(frequency_GHz)])
    frequency = widgets.FloatSlider(value=initial, min=float(grid.min()), max=float(grid.max()),
                                    step=float(np.min(np.diff(grid))) if len(grid)>1 else 0.001,
                                    description=frequency_control_label, readout_format=".4f", continuous_update=False,
                                    layout=widgets.Layout(width="65%"))
    geometry_labels = {"fixed": fixed_geometry_label, "gaps": gaps_geometry_label,
                       "positions": positions_geometry_label, "both": both_geometry_label}
    mode = widgets.Dropdown(options=[(geometry_labels.get(value, value), value) for value in available],
                            value=geometry, description=geometry_control_label)
    view_labels = {"diagnostic": diagnostic_view_label, "radial": radial_view_label,
                   "reference": reference_view_label, "center": center_view_label,
                   "transverse": transverse_view_label, "spatial": spatial_view_label}
    functions = {"diagnostic": plot_1d_diagnostic, "radial": plot_radial_fields,
                 "reference": plot_reference_subtraction, "center": plot_beam_center,
                 "transverse": plot_transverse_maps, "spatial": plot_3d_profiles}
    if plot_functions is not None:
        unknown = set(plot_functions)-set(functions)
        if unknown:
            raise ValueError(f"Unknown interactive view keys: {sorted(unknown)}")
        functions.update(plot_functions)
    view = widgets.Dropdown(options=[(label, key) for key, label in view_labels.items()],
                             description=view_control_label, layout=widgets.Layout(width="55%"))

    def draw(frequency, geometry, view):
        function = functions[view]
        figure = function(data, frequency, geometry) if view == "diagnostic" else function(data, frequency)
        display(figure)
        plt.close(figure)

    output = widgets.interactive_output(draw, {"frequency": frequency, "geometry": mode, "view": view})
    return widgets.VBox([widgets.HBox([view, mode]), frequency, output])


# Longitudinal fit and setup views. These functions only read saved results.
_FIT_GEOMETRY_COLORS = {
    "fixed": "#3b6fb6", "gaps": "#2a9d8f", "positions": "#d99222", "both": "#9c4f96",
}
_FIT_THREAD_COLORS = {0: "#3b6fb6", 1: "#db7b32", 2: "#2a9d8f", 3: "#9c4f96"}
_FIT_GEOMETRY_LABELS = {
    "fixed": "Fixed geometry", "gaps": "Fit gaps", "positions": "Fit plane positions",
    "both": "Fit gaps + planes",
}
_FIT_CURVE_COLORS = {"bead": "#bb4cc8", "true": "#27965b", "measured": "#e88a45"}


def _fitplot_geometry_label(geometry, geometry_labels):
    return render_text(geometry_labels.get(geometry, geometry), geometry=geometry)


def _fitplot_config(saved, nominal=False):
    values = {}
    for item in _fitplot_dataclass_fields(_FitPlotConfig):
        if item.name in saved:
            array = np.asarray(saved[item.name])
            values[item.name] = tuple(array.astype(float)) if item.name == "gap_widths_m" else array.item()
    config = _FitPlotConfig(**values)
    if nominal and "gap_widths_nominal_m" in saved:
        config = _fitplot_replace(config, gap_widths_m=tuple(np.asarray(saved["gap_widths_nominal_m"], float)))
    return config


def _fitplot_result(data, geometry):
    if geometry not in data.fits_1d:
        available = ", ".join(data.fits_1d) or "none"
        raise ValueError(f"No saved 1D fit for {geometry!r}; available fits: {available}.")
    return data.fits_1d[geometry]


def _fitplot_disks(config):
    edge = 0.0
    disks = []
    for gap in config.gap_widths_m:
        start = edge + gap
        edge = start + config.disk_thickness_m
        disks.append((start, edge))
    return np.asarray(disks, dtype=float).reshape(-1, 2)


def _fitplot_thread_ids(data, count):
    ids = np.asarray(data.fields.get("sub_thread_indices", np.arange(count)), dtype=int)
    return ids if ids.shape == (count,) else np.arange(count)


def _fitplot_frequency(saved, requested_GHz, frequency_label_template,
                        nearest_frequency_label_template):
    frequencies = np.asarray(saved["frequencies_Hz"], float)
    requested = float(requested_GHz)*1e9
    if not np.isfinite(requested):
        raise ValueError("frequency_GHz must be finite.")
    index = int(np.argmin(abs(frequencies-requested)))
    actual = frequencies[index]
    template = nearest_frequency_label_template if abs(actual-requested) > 1.0 else frequency_label_template
    label = render_text(template, frequency_GHz=actual*1e-9,
                        requested_frequency_GHz=float(requested_GHz))
    return index, actual, label


def _fitplot_z_limit(config, z):
    disks = _fitplot_disks(config)
    end = disks[-1, 1] if len(disks) else 0.0
    return max(end+0.012, float(np.max(z))+0.006)


def _fitplot_style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(True, color="#e5e9ef", linewidth=0.6, alpha=0.75)
    axis.set_axisbelow(True)


def _fitplot_setup_bands(axis, config, nominal=None, *, labels=False,
                         disk_label_template=None):
    axis.axvspan(-1.4, 0, color="#69737e", alpha=0.55, linewidth=0, zorder=0)
    for i, (start, end) in enumerate(_fitplot_disks(config)):
        axis.axvspan(start*1e3, end*1e3, color="#b4bec8", alpha=0.35, linewidth=0, zorder=0)
        if labels and disk_label_template is not None:
            axis.text((start+end)*500, 1.015, render_text(disk_label_template, disk_index=i+1),
                      transform=axis.get_xaxis_transform(),
                      ha="center", va="bottom", fontsize=8, color="#687381")
    if nominal is not None and not np.allclose(config.gap_widths_m, nominal.gap_widths_m, atol=1e-12, rtol=0):
        for start, end in _fitplot_disks(nominal):
            axis.axvspan(start*1e3, end*1e3, facecolor="none", edgecolor="#707c89",
                         linestyle=":", linewidth=1.1, zorder=1)


def plot_setup(
    data, *,
    title="Measurement geometry · one central diameter per z",
    x_label="z from mirror [mm]",
    mirror_label="Mirror",
    disk_label_template="Disk {disk_index}",
    thread_label_template="T{thread_index}",
    thread_position_template="{z_mm:.3f}",
    gap_width_template="{gap_mm:.3f} mm",
    antenna_label_template="Antenna: {antenna_distance_m:.3f} m from mirror  →",
    antenna_unknown_label="Antenna lies towards +z  →",
    table_headers=("Plane", "z [mm]", "Line [cm]", "R int. [cm]"),
    table_thread_template="T{thread_index}",
    table_position_template="{z_mm:.3f}",
    table_length_template="{length_cm:.2f}",
    table_radius_template="{radius_cm:.2f}",
    missing_value_label="—",
    properties_template=("{disk_count} disks · {disk_thickness_mm:g} mm thick · εᵣ = {disk_eps_r:g}\n"
                         "Disk radius {disk_radius_cm:g} cm\n"
                         "Bead Ø {bead_diameter_mm:g} mm · εᵣ = {bead_eps_r:g}\n"
                         "Saved input-power normalization: {input_power_W:g} W"),
    footnote=("Surface-to-surface gaps; +z points towards the antenna. Antenna offset is not drawn to scale. "
              "R int. is the radial integration limit."),
):
    """Nominal physical setup, calibrated scan lengths, and integration radii."""
    # This is the measured setup, not the geometry of a previously saved fit.
    source = data.fields
    config = _fitplot_config(source, nominal=True)
    z = np.asarray(source.get("z_nominal_m", data.fields["z_from_mirror_m"]), float)
    ids = _fitplot_thread_ids(data, len(z))
    disks = _fitplot_disks(config)
    xmax = _fitplot_z_limit(config, z)*1e3 + 3
    fig, (axis, details) = plt.subplots(1, 2, figsize=(13, 4.9),
                                      gridspec_kw={"width_ratios": [2.15, 1]})
    axis.add_patch(_FitPlotRectangle((-1.4, -0.78), 1.4, 1.56, facecolor="#69737e"))
    axis.text(-0.7, 0.94, render_text(mirror_label), ha="center", fontsize=10, color="#48535f")
    for i, (start, end) in enumerate(disks):
        axis.add_patch(_FitPlotRectangle((start*1e3, -0.78), (end-start)*1e3, 1.56,
                                        facecolor="#c4ccd5", edgecolor="#8996a5", linewidth=1))
        axis.text((start+end)*500, 0.94, render_text(disk_label_template, disk_index=i+1),
                  ha="center", fontsize=9, color="#48535f")
    axis.annotate("", xy=(xmax, 0), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": "#52616f", "lw": 1.5})
    for row, (index, position) in enumerate(zip(ids, z)):
        color = _FIT_THREAD_COLORS.get(int(index), "#566779")
        axis.vlines(position*1e3, -0.47, 0.47, color=color, linewidth=2.5)
        axis.scatter([position*1e3], [0], s=38, color=color, edgecolors="white", zorder=4)
        axis.text(position*1e3, 0.54, render_text(thread_label_template, thread_index=int(index)),
                  ha="center", fontsize=10, fontweight="bold", color=color)
        axis.text(position*1e3, -0.55, render_text(thread_position_template, z_mm=position*1e3, thread_index=int(index)),
                  ha="center", fontsize=8, color=color)
    left = 0.0
    for i, gap in enumerate(config.gap_widths_m):
        right = left+gap
        axis.annotate("", xy=(right*1e3, -1.0), xytext=(left*1e3, -1.0),
                      arrowprops={"arrowstyle": "<->", "color": "#647181", "lw": 0.9})
        axis.text((left+right)*500, -1.07, render_text(gap_width_template, gap_index=i+1, gap_mm=gap*1e3),
                  ha="center", va="top", fontsize=9)
        left = right+config.disk_thickness_m
    antenna_distance = data.fields.get("mirror_z_m")
    antenna = (render_text(antenna_label_template, antenna_distance_m=float(np.asarray(antenna_distance)))
               if antenna_distance is not None else render_text(antenna_unknown_label))
    axis.text(0.99, 1.14, antenna, transform=axis.transAxes, ha="right", fontsize=10, color="#48535f")
    axis.set(xlim=(-2.5, xmax), ylim=(-1.35, 1.14), yticks=[], xlabel=render_text(x_label))
    axis.spines[["left", "right", "top"]].set_visible(False)
    axis.tick_params(axis="y", left=False)
    details.axis("off")
    radii = np.asarray(data.fields.get("integration_radius_m", np.full(len(z), np.nan)), float)
    rows = []
    for row, index in enumerate(ids):
        line = data.lines.get(int(index))
        length = getattr(line, "length_m", np.nan)
        radius = radii[row] if radii.shape == (len(z),) else np.nan
        rows.append([
            render_text(table_thread_template, thread_index=int(index)),
            render_text(table_position_template, z_mm=z[row]*1e3, thread_index=int(index)),
            render_text(table_length_template, length_cm=length*100, thread_index=int(index)) if np.isfinite(length) else render_text(missing_value_label),
            render_text(table_radius_template, radius_cm=radius*100, thread_index=int(index)) if np.isfinite(radius) else render_text(missing_value_label),
        ])
    table = details.table(cellText=rows, colLabels=[render_text(header) for header in table_headers],
                          cellLoc="center", bbox=[0.0, 0.36, 1, 0.52])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#e2e7ed")
        if row == 0:
            cell.set_facecolor("#eef2f6")
            cell.set_text_props(weight="bold", color="#394957")
        elif column == 0:
            cell.set_text_props(color=_FIT_THREAD_COLORS.get(int(ids[row-1]), "#566779"), weight="bold")
    details.text(0, 0.23, render_text(
                 properties_template, disk_count=len(disks), disk_thickness_mm=config.disk_thickness_m*1e3,
                 disk_eps_r=config.disk_eps_r, disk_radius_cm=config.disk_radius_m*100,
                 bead_diameter_mm=config.bead_diameter_m*1e3, bead_eps_r=config.bead_eps_r,
                 input_power_W=config.input_power_W),
                 transform=details.transAxes, fontsize=10, va="top", linespacing=1.7)
    fig.suptitle(render_text(title), x=0.06, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.06, 0.035, render_text(footnote),
             fontsize=9, color="#637180")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.19, top=0.78, wspace=0.22)
    return fig


def _fitplot_draw_field(
    axis, data, geometry, frequency_GHz, *, compact=False, z_max_m=None,
    title_template, y_label, true_field_label, bead_model_label,
    nominal_position_label, measured_field_label, thread_label_template,
    residual_annotation_template, geometry_labels, frequency_label_template,
    nearest_frequency_label_template, disk_label_template=None,
):
    saved = _fitplot_result(data, geometry)
    index, frequency, frequency_label = _fitplot_frequency(
        saved, frequency_GHz, frequency_label_template, nearest_frequency_label_template)
    config = _fitplot_config(saved)
    nominal_config = _fitplot_config(saved, nominal=True)
    z = np.asarray(saved["z_from_mirror_m"], float)
    z_nominal = np.asarray(saved.get("z_nominal_m", z), float)
    ids = _fitplot_thread_ids(data, len(z))
    amplitude = np.asarray(saved["amplitude"])[index]
    sampled_z = np.linspace(0, _fitplot_z_limit(config, z) if z_max_m is None else z_max_m, 1300)
    model = _fitplot_evaluate([frequency], sampled_z, config)
    bead_curve = model.bead_magnitude[:, 0]*abs(amplitude)
    bead_radius = config.bead_diameter_m/2
    valid_bead_center = sampled_z >= bead_radius
    for start, end in _fitplot_disks(config):
        valid_bead_center &= (sampled_z <= start-bead_radius) | (sampled_z >= end+bead_radius)
    bead_curve[~valid_bead_center] = np.nan
    _fitplot_style_axis(axis)
    _fitplot_setup_bands(axis, config, nominal_config, labels=not compact,
                         disk_label_template=disk_label_template)
    axis.plot(sampled_z*1e3, abs(model.field[:, 0]*amplitude), color=_FIT_CURVE_COLORS["true"],
              linewidth=1.65, label=render_text(true_field_label), zorder=2)
    axis.plot(sampled_z*1e3, bead_curve, "--",
              color=_FIT_CURVE_COLORS["bead"], linewidth=1.65, label=render_text(bead_model_label), zorder=3)
    measured = abs(np.asarray(saved["ET_measured"])[:, index])
    prediction = np.asarray(saved["ET_fitted_magnitude"])[:, index]
    moved = abs(z-z_nominal) > 1e-10
    if np.any(moved):
        axis.scatter(z_nominal[moved]*1e3, measured[moved], s=43, facecolors="white", edgecolors="#778594",
                     linewidth=1.05, label=render_text(nominal_position_label), zorder=4)
        for old, new, y in zip(z_nominal[moved], z[moved], measured[moved]):
            axis.plot([old*1e3, new*1e3], [y, y], color="#8b96a3", linewidth=0.9, zorder=3)
    axis.scatter(z*1e3, measured, s=43, facecolor=_FIT_CURVE_COLORS["measured"], edgecolor="#20262d",
                 linewidth=0.9, label=render_text(measured_field_label), zorder=5)
    for thread, position, value in zip(ids, z, measured):
        axis.annotate(render_text(thread_label_template, thread_index=int(thread), z_mm=position*1e3),
                      (position*1e3, value), xytext=(5, 6), textcoords="offset points",
                      color=_FIT_THREAD_COLORS.get(int(thread), "#566779"), fontsize=8.5, fontweight="bold")
    denominator = np.linalg.norm(measured)
    relative = np.linalg.norm(prediction-measured)/denominator if denominator > 0 else np.nan
    geometry_label = _fitplot_geometry_label(geometry, geometry_labels)
    axis.set_title(render_text(title_template, geometry_label=geometry_label, geometry=geometry,
                              frequency_label=frequency_label, frequency_GHz=frequency*1e-9,
                              requested_frequency_GHz=float(frequency_GHz)),
                   loc="left", fontsize=11 if compact else 13,
                   color=_FIT_GEOMETRY_COLORS.get(geometry, "#394957"), pad=18 if not compact else 10)
    axis.text(0.985, 0.97, render_text(residual_annotation_template, residual_percent=relative*100,
                                    geometry_label=geometry_label, frequency_GHz=frequency*1e-9),
              transform=axis.transAxes,
              ha="right", va="top", fontsize=8.5, color="#485563",
              bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2.5})
    top = max(float(np.max(abs(model.field[:, 0]*amplitude))), float(np.max(measured)),
              float(np.max(bead_curve[np.isfinite(bead_curve)], initial=0)), 1e-15)
    axis.set(xlim=(-1.4, sampled_z[-1]*1e3), ylim=(0, top*1.22), ylabel=render_text(y_label))
    axis.legend(loc="upper center", bbox_to_anchor=(0.48, 1),
                fontsize=7.8 if compact else 9, framealpha=0.92)
    return saved, z, ids, measured, prediction, denominator


def plot_1d_diagnostic(
    data, frequency_GHz=20.825, geometry="both", *,
    title_template="{geometry_label} · {frequency_label}",
    x_label="z from mirror [mm]",
    y_label=r"$|E_T|$ [V m]",
    residual_y_label="Residual /\n‖data‖ [%]",
    true_field_label="Deconvolved field",
    bead_model_label="Finite-bead model",
    nominal_position_label="Nominal z",
    measured_field_label="Measured |ET|",
    disk_label_template="D{disk_index}",
    thread_label_template="T{thread_index}",
    residual_annotation_template="Relative residual {residual_percent:.1f}%",
    geometry_labels=_FIT_GEOMETRY_LABELS,
    frequency_label_template="{frequency_GHz:.6g} GHz",
    nearest_frequency_label_template="{frequency_GHz:.6g} GHz (nearest to {requested_frequency_GHz:.6g} GHz)",
    footnote=("Residual = finite-bead model − measured magnitude. Bead curves require the whole bead in vacuum; "
              "dotted disk outlines mark nominal geometry."),
):
    """Representative fit, setup and per-plane residuals at a saved frequency."""
    fig, (axis, residual_axis) = plt.subplots(2, 1, figsize=(12.5, 7.0), sharex=True,
                                             gridspec_kw={"height_ratios": [3.6, 1]})
    saved, z, ids, measured, prediction, denominator = _fitplot_draw_field(
        axis, data, geometry, frequency_GHz, title_template=title_template, y_label=y_label,
        true_field_label=true_field_label, bead_model_label=bead_model_label,
        nominal_position_label=nominal_position_label, measured_field_label=measured_field_label,
        thread_label_template=thread_label_template, disk_label_template=disk_label_template,
        residual_annotation_template=residual_annotation_template, geometry_labels=geometry_labels,
        frequency_label_template=frequency_label_template,
        nearest_frequency_label_template=nearest_frequency_label_template)
    _fitplot_style_axis(residual_axis)
    _fitplot_setup_bands(residual_axis, _fitplot_config(saved), _fitplot_config(saved, nominal=True))
    residual_axis.axhline(0, color="#566373", linewidth=0.9)
    residual = 100*(prediction-measured)/denominator if denominator > 0 else np.zeros_like(measured)
    for index, position, value in zip(ids, z, residual):
        color = _FIT_THREAD_COLORS.get(int(index), "#566779")
        residual_axis.vlines(position*1e3, 0, value, color=color, linewidth=1.2)
        residual_axis.scatter([position*1e3], [value], s=36, color=color, edgecolor="white", zorder=4)
    limit = max(1.0, np.max(abs(residual))*1.3)
    residual_axis.set(ylim=(-limit, limit), xlabel=render_text(x_label), ylabel=render_text(residual_y_label))
    fig.text(0.075, 0.025, render_text(footnote),
             fontsize=9, color="#637180")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.9, hspace=0.14)
    return fig


def plot_1d_comparison(
    data, frequency_GHz=20.825, *,
    title="Longitudinal reconstruction · compare geometry assumptions",
    panel_title_template="{geometry_label} · {frequency_label}",
    x_label="z from mirror [mm]",
    y_label=r"$|E_T|$ [V m]",
    true_field_label="Deconvolved field",
    bead_model_label="Finite-bead model",
    nominal_position_label="Nominal z",
    measured_field_label="Measured |ET|",
    thread_label_template="T{thread_index}",
    residual_annotation_template="Relative residual {residual_percent:.1f}%",
    geometry_labels=_FIT_GEOMETRY_LABELS,
    frequency_label_template="{frequency_GHz:.6g} GHz",
    nearest_frequency_label_template="{frequency_GHz:.6g} GHz (nearest to {requested_frequency_GHz:.6g} GHz)",
    missing_fit_template="{geometry_label}\nNo saved result",
    footnote=("Each panel uses its own saved geometry and amplitude. Filled measurements use fitted z; "
              "hollow points and dotted disk outlines retain nominal locations."),
):
    """Four saved geometry fits at the nearest actual frequency in each result."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.0), sharex=True)
    limits = [_fitplot_z_limit(_fitplot_config(saved), saved["z_from_mirror_m"])
              for mode, saved in data.fits_1d.items() if mode in _FIT_GEOMETRY_COLORS]
    common_z_max = max(limits, default=0.04)
    for axis, geometry in zip(axes.flat, _FIT_GEOMETRY_COLORS):
        if geometry in data.fits_1d:
            _fitplot_draw_field(
                axis, data, geometry, frequency_GHz, compact=True, z_max_m=common_z_max,
                title_template=panel_title_template, y_label=y_label,
                true_field_label=true_field_label, bead_model_label=bead_model_label,
                nominal_position_label=nominal_position_label, measured_field_label=measured_field_label,
                thread_label_template=thread_label_template,
                residual_annotation_template=residual_annotation_template, geometry_labels=geometry_labels,
                frequency_label_template=frequency_label_template,
                nearest_frequency_label_template=nearest_frequency_label_template)
        else:
            axis.text(0.5, 0.5, render_text(missing_fit_template,
                                          geometry_label=_fitplot_geometry_label(geometry, geometry_labels), geometry=geometry),
                      ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
    for axis in axes[-1]:
        axis.set_xlabel(render_text(x_label))
    fig.suptitle(render_text(title), x=0.065, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.065, 0.025, render_text(footnote),
             fontsize=9, color="#637180")
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.1, top=0.9, hspace=0.3, wspace=0.18)
    return fig


def plot_fit_spectra(
    data, frequency_range_GHz=None, logarithmic=False, *,
    title="Boost spectrum and fit quality",
    x_label="Frequency [GHz]",
    boost_y_label=r"Reconstructed boost power $\beta^2$",
    residual_y_label="Relative residual [%]",
    geometry_labels=_FIT_GEOMETRY_LABELS,
    footnote=("Each fit has one geometry shared across the spectrum. "
              "Relative residual = ‖model − |ET|‖₂ / ‖|ET|‖₂ across measurement planes."),
):
    """Boost power and dimensionless magnitude-fit residual for all saved 1D fits."""
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.3]})
    for geometry, color in _FIT_GEOMETRY_COLORS.items():
        if geometry not in data.fits_1d:
            continue
        saved = data.fits_1d[geometry]
        frequency = np.asarray(saved["frequencies_Hz"])*1e-9
        measured = abs(np.asarray(saved["ET_measured"]))
        norm = np.linalg.norm(measured, axis=0)
        residual = np.divide(np.linalg.norm(np.asarray(saved["ET_fitted_magnitude"])-measured, axis=0), norm,
                             out=np.full_like(norm, np.nan), where=norm > 0)
        axes[0].plot(frequency, saved["beta2"], color=color, linewidth=1.35,
                     label=_fitplot_geometry_label(geometry, geometry_labels), alpha=0.93)
        axes[1].plot(frequency, residual*100, color=color, linewidth=1.0, alpha=0.9)
    for axis in axes:
        _fitplot_style_axis(axis)
    axes[0].set_ylabel(render_text(boost_y_label))
    if logarithmic:
        axes[0].set_yscale("log")
    else:
        axes[0].set_ylim(bottom=0)
    axes[1].set(ylabel=render_text(residual_y_label), xlabel=render_text(x_label), ylim=(0, 100))
    if frequency_range_GHz is not None:
        axes[1].set_xlim(*frequency_range_GHz)
    axes[0].legend(loc="upper left", ncol=2, fontsize=10, framealpha=0.92)
    axes[0].set_title(render_text(title), loc="left", fontsize=15, fontweight="bold", pad=13)
    fig.text(0.08, 0.025, render_text(footnote),
             fontsize=9, color="#637180")
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.13, top=0.91, hspace=0.15)
    _limit_frequency_axis(axes[0], data)
    return fig


def plot_geometry_shifts(
    data, *,
    title="Geometry adjustments shared across all frequencies",
    gap_panel_title="Surface-to-surface gap shifts",
    position_panel_title="Measurement-plane shifts",
    y_label="Shift from nominal [µm]",
    gap_tick_template="Gap {gap_index}",
    thread_tick_template="T{thread_index}",
    value_annotation_template="{shift_um:.0f}",
    zero_shift_label="0",
    geometry_labels=_FIT_GEOMETRY_LABELS,
    bound_label="Allowed shift bounds",
    footnote=("The dashed limits constrain optimization; they are not uncertainties. "
              "Positive shifts increase a gap or move a plane away from the mirror."),
):
    """Shared geometry shifts; dashed limits are optimization bounds, not errors."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    offsets = np.linspace(-0.30, 0.30, 4)
    limits = [[], []]
    labels = [[], []]
    for offset, (geometry, color) in zip(offsets, _FIT_GEOMETRY_COLORS.items()):
        if geometry not in data.fits_1d:
            continue
        saved = data.fits_1d[geometry]
        values = [np.asarray(saved["gap_widths_m"])-np.asarray(saved["gap_widths_nominal_m"]),
                  np.asarray(saved["z_from_mirror_m"])-np.asarray(saved["z_nominal_m"])]
        ids = _fitplot_thread_ids(data, len(values[1]))
        labels = [[render_text(gap_tick_template, gap_index=i+1) for i in range(len(values[0]))],
                  [render_text(thread_tick_template, thread_index=int(i)) for i in ids]]
        for panel, (axis, shifts) in enumerate(zip(axes, values)):
            bars = axis.bar(np.arange(len(shifts))+offset, shifts*1e6, width=0.18,
                            color=color, alpha=0.92, label=_fitplot_geometry_label(geometry, geometry_labels), zorder=3)
            axis.bar_label(bars, labels=[render_text(value_annotation_template, shift_um=value,
                                                    geometry_label=_fitplot_geometry_label(geometry, geometry_labels), geometry=geometry)
                                         if abs(value) >= 0.5 else render_text(zero_shift_label) for value in shifts*1e6],
                           fontsize=7.5, padding=3, color="#4a5662")
            key = "gap_bound_m" if panel == 0 else "position_bound_m"
            if key in saved:
                limits[panel].append(float(np.asarray(saved[key]))*1e6)
    for panel, axis in enumerate(axes):
        _fitplot_style_axis(axis)
        axis.axhline(0, color="#697585", linewidth=0.8)
        for bound in sorted(set(limits[panel])):
            axis.axhline(bound, color="#87929f", linestyle="--", linewidth=0.9)
            axis.axhline(-bound, color="#87929f", linestyle="--", linewidth=0.9)
        bound = max(limits[panel], default=500)
        axis.set(xticks=np.arange(len(labels[panel])), xticklabels=labels[panel],
                 ylabel=render_text(y_label), ylim=(-1.32*bound, 1.32*bound))
        axis.set_title(render_text(gap_panel_title if panel == 0 else position_panel_title), loc="left", fontsize=12)
    handles = [_FitPlotPatch(facecolor=color, label=_fitplot_geometry_label(mode, geometry_labels)) for mode, color in _FIT_GEOMETRY_COLORS.items() if mode in data.fits_1d]
    handles.append(_FitPlotLine([], [], color="#87929f", linestyle="--", label=render_text(bound_label)))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.92), ncol=5, frameon=False, fontsize=9)
    fig.suptitle(render_text(title), x=0.065, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.065, 0.025, render_text(footnote),
             fontsize=9, color="#637180")
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.17, top=0.76, wspace=0.22)
    return fig


def plot_1d_field_map(
    data, geometry="both", frequency_range_GHz=None, logarithmic=False, *,
    title_template="Deconvolved transverse field integral · {geometry_label}",
    x_label="Frequency [GHz]",
    y_label="z from mirror [mm]",
    colorbar_label=r"$|E_T(z,f)|$ [V m]",
    logarithmic_colorbar_suffix=" · logarithmic color scale",
    thread_label_template="T{thread_index}",
    geometry_labels=_FIT_GEOMETRY_LABELS,
    footnote=("White bands outline fitted disks; dotted outlines/lines retain nominal geometry. "
              "Colored dashed lines locate fitted measurement planes."),
):
    """True reconstructed transverse integral on a dense (z, saved-frequency) grid."""
    saved = _fitplot_result(data, geometry)
    config = _fitplot_config(saved)
    nominal_config = _fitplot_config(saved, nominal=True)
    frequencies = np.asarray(saved["frequencies_Hz"], float)
    select = np.ones(len(frequencies), dtype=bool)
    if frequency_range_GHz is not None:
        lower, upper = frequency_range_GHz
        select &= (frequencies >= lower*1e9) & (frequencies <= upper*1e9)
    if np.count_nonzero(select) < 2:
        raise ValueError("The field map needs at least two saved frequencies in the requested range.")
    frequencies = frequencies[select]
    z_fit = np.asarray(saved["z_from_mirror_m"], float)
    z_nominal = np.asarray(saved.get("z_nominal_m", z_fit), float)
    z = np.linspace(0, _fitplot_z_limit(config, z_fit), 320)
    magnitude = abs(_fitplot_evaluate(frequencies, z, config).field*np.asarray(saved["amplitude"])[select])
    norm = None
    if logarithmic:
        positive = magnitude[magnitude > 0]
        if positive.size:
            norm = _FitPlotLogNorm(vmin=max(float(positive.min()), float(positive.max())*1e-5), vmax=float(positive.max()))
    fig, axis = plt.subplots(figsize=(12.5, 6.3))
    image = axis.pcolormesh(frequencies*1e-9, z*1e3, magnitude, shading="auto", cmap="magma", norm=norm, rasterized=True)
    for start, end in _fitplot_disks(config):
        axis.axhspan(start*1e3, end*1e3, facecolor="none", edgecolor="white", linewidth=1.1)
    if not np.allclose(config.gap_widths_m, nominal_config.gap_widths_m, atol=1e-12, rtol=0):
        for start, end in _fitplot_disks(nominal_config):
            axis.axhspan(start*1e3, end*1e3, facecolor="none", edgecolor="#b8c5d3", linestyle=":", linewidth=1.0)
    ids = _fitplot_thread_ids(data, len(z_fit))
    for thread, position, nominal in zip(ids, z_fit, z_nominal):
        color = _FIT_THREAD_COLORS.get(int(thread), "#dce6ef")
        axis.axhline(position*1e3, color=color, linewidth=1.25, linestyle="--")
        if abs(position-nominal) > 1e-10:
            axis.axhline(nominal*1e3, color="#c5ccd5", linewidth=0.8, linestyle=":")
        axis.text(1.008, position*1e3, render_text(thread_label_template, thread_index=int(thread), z_mm=position*1e3),
                  transform=axis.get_yaxis_transform(),
                  ha="left", va="center", color=color, fontsize=9, fontweight="bold")
    axis.axhline(0, color="#69737e", linewidth=4)
    axis.set(xlabel=render_text(x_label), ylabel=render_text(y_label))
    axis.set_title(render_text(title_template, geometry_label=_fitplot_geometry_label(geometry, geometry_labels), geometry=geometry),
                   loc="left", fontsize=14, fontweight="bold", pad=13)
    colorbar = fig.colorbar(image, ax=axis, pad=0.075, fraction=0.035)
    colorbar.set_label(render_text(colorbar_label)+(render_text(logarithmic_colorbar_suffix) if logarithmic else ""))
    fig.text(0.075, 0.025, render_text(footnote),
             fontsize=9, color="#637180")
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.14, top=0.9)
    _limit_frequency_axis(axis, data)
    return fig


def plot_1d_refit_comparison(
    data, previous_dir, *,
    title="1D refits · corrected nominal geometry versus previous fits",
    panel_title_template="{geometry_label} · {quantity_label}",
    x_label="Frequency [GHz]",
    boost_y_label="Power boost β²",
    residual_y_label="Relative magnitude residual [%]",
    current_label="Refit · {current_gap_mm:.3f} mm nominal mirror gap",
    previous_label="Previous · {previous_gap_mm:.3f} mm nominal mirror gap",
    boost_panel_label="Power boost",
    residual_panel_label="Magnitude residual",
    geometry_labels=_FIT_GEOMETRY_LABELS,
):
    """Compare saved current/previous fits on common bins in the 20–22 GHz band.

    A narrower display range is respected. No frequencies are interpolated and
    no model is fitted or reevaluated. Label templates accept geometry,
    geometry_label, quantity_label, current_gap_mm and previous_gap_mm.
    """
    previous_dir = Path(previous_dir)
    lower, upper = 20.0, 22.0
    if data.frequency_range_GHz is not None:
        lower = max(lower, data.frequency_range_GHz[0])
        upper = min(upper, data.frequency_range_GHz[1])
    if lower >= upper:
        raise ValueError("The refit comparison needs a display range overlapping 20–22 GHz.")
    for mode in _FIT_GEOMETRY_COLORS:
        if mode not in data.fits_1d:
            raise ValueError(f"No current saved 1D fit for {mode!r}.")
    previous_fits = {mode: _read_npz(previous_dir / f"integrated_fields_fit_1D_{mode}.npz")
                     for mode in _FIT_GEOMETRY_COLORS}
    figure, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True, constrained_layout=True)
    title_context = {}
    for row, (mode, color) in enumerate(_FIT_GEOMETRY_COLORS.items()):
        previous = previous_fits[mode]
        current = data.fits_1d[mode]
        frequency = np.asarray(current["frequencies_Hz"])
        selected = np.flatnonzero((frequency >= lower * 1e9) & (frequency <= upper * 1e9))
        old_frequency = np.asarray(previous["frequencies_Hz"])
        old_indices = np.array([np.argmin(abs(old_frequency - value)) for value in frequency[selected]], dtype=int)
        matched = np.isclose(frequency[selected], old_frequency[old_indices], rtol=0, atol=1)
        selected, old_indices = selected[matched], old_indices[matched]
        if not selected.size:
            plt.close(figure)
            raise ValueError(f"The {mode!r} fits have no common acquired bins in {lower:g}–{upper:g} GHz.")
        context = dict(geometry=mode, geometry_label=_fitplot_geometry_label(mode, geometry_labels),
                       current_gap_mm=float(current["gap_widths_nominal_m"][0]) * 1e3,
                       previous_gap_mm=float(previous["gap_widths_nominal_m"][0]) * 1e3)
        if not title_context:
            title_context = context
        for saved, bins, label, style in ((current, selected, current_label, "-"),
                                         (previous, old_indices, previous_label, "--")):
            observed = abs(np.asarray(saved["ET_measured"])[:, bins])
            prediction = np.asarray(saved["ET_fitted_magnitude"])[:, bins]
            denominator = np.linalg.norm(observed, axis=0)
            relative = np.divide(100 * np.linalg.norm(prediction - observed, axis=0), denominator,
                                 out=np.full(denominator.shape, np.nan), where=denominator > 0)
            plotted_frequency = frequency[selected] / 1e9
            axes[row, 0].plot(plotted_frequency, np.asarray(saved["beta2"])[bins], style,
                              color=color, label=render_text(label, **context), alpha=1 if style == "-" else 0.7)
            axes[row, 1].plot(plotted_frequency, relative, style, color=color,
                              label=render_text(label, **context), alpha=1 if style == "-" else 0.7)
        for column, (quantity, ylabel) in enumerate(((boost_panel_label, boost_y_label),
                                                    (residual_panel_label, residual_y_label))):
            axis = axes[row, column]
            _fitplot_style_axis(axis)
            axis.set(title=render_text(panel_title_template, quantity_label=render_text(quantity, **context), **context),
                     xlabel=render_text(x_label, **context), ylabel=render_text(ylabel, **context),
                     xlim=(lower, upper), ylim=(0, None))
            if row == 0:
                axis.legend(fontsize=8)
    figure.suptitle(render_text(title, **title_context))
    return figure
