"""Read MADBead's saved full-scan results and compare them with the spider scan.

The JLD2 file stores Julia Measurements and DataFrames through HDF5 object
references. This reader extracts numeric values only; it never loads Julia code,
repeats the Monte Carlo fit, or treats the transverse integral as a local field.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from .data_paths import DATA_DIR, OUTPUTS_DIR
from .plot_text import render_text


@dataclass
class FullBeadData:
    source: Path
    frequencies_Hz: np.ndarray
    z_absolute_m: np.ndarray
    ET: np.ndarray
    beta2: np.ndarray
    beta2_error: np.ndarray
    parameters: dict[str, np.ndarray]
    parameter_errors: dict[str, np.ndarray]
    reflection: np.ndarray
    n_mc: int

    def parameters_at(self, index):
        return {name: float(values[index]) for name, values in self.parameters.items() if name != "f"}

    def z_from_mirror(self, index):
        return self.parameters["z_m"][index] - self.z_absolute_m


def _measurement(handle, value):
    """Extract a numeric value and saved error, resolving only required objects."""
    if isinstance(value, h5py.Reference):
        value = handle[value][()]
    names = getattr(getattr(value, "dtype", None), "names", None) or ()
    if "val" in names:
        return float(value["val"]), float(value["err"])
    if "re" in names and "im" in names:
        real, real_error = _measurement(handle, value["re"])
        imag, imag_error = _measurement(handle, value["im"])
        return real + 1j*imag, real_error + 1j*imag_error
    return float(value), 0.0


def _measurement_array(handle, values):
    entries = [_measurement(handle, item) for item in values.ravel()]
    nominal, errors = np.asarray(entries).T
    return nominal.reshape(values.shape), errors.reshape(values.shape)


def _read_full_archive(path):
    with h5py.File(path, "r") as handle:
        table = handle["p_final_ν"][()]
        names = handle[table["colindex"]["names"]][()]
        columns = handle[table["columns"]][()]
        parameters, errors = {}, {}
        for name, reference in zip(names, columns):
            name = name.decode() if isinstance(name, bytes) else str(name)
            values = handle[reference][()]
            parameters[name], errors[name] = _measurement_array(handle, values)
        frequencies = parameters["f"]
        # A Julia matrix (frequency, z) appears to h5py as (z, frequency).
        electric, _ = _measurement_array(handle, handle["∫dA_E"][()])
        z_absolute = handle["z_pos"][()]
        if electric.shape != (len(z_absolute), len(frequencies)):
            raise ValueError("Full-bead transverse integral does not have the expected HDF5 (z, frequency) axes.")
        boost, boost_error = _measurement_array(handle, handle["boostfactor"][()])
        reflection, _ = _measurement_array(handle, handle["Γ"][()])
        # Independently verify the serialized StepRangeLen rather than guessing
        # that the file's '21_0' name represents a single frequency.
        frequency_range = handle["ν"][()]
        names = frequency_range.dtype.names or ()
        if {"ref", "step", "len", "offset"}.issubset(names):
            ref = float(frequency_range["ref"]["hi"] + frequency_range["ref"]["lo"])
            step = float(frequency_range["step"]["hi"] + frequency_range["step"]["lo"])
            axis = ref + (np.arange(int(frequency_range["len"])) + 1 - int(frequency_range["offset"]))*step
            if not np.array_equal(frequencies, axis):
                raise ValueError("The full-bead parameter and acquisition frequency axes disagree.")
        n_mc = int(handle["N_mc"][()])
    return FullBeadData(Path(path), frequencies, z_absolute, electric, boost, boost_error,
                        parameters, errors, reflection, n_mc)


def load_full_bead(path=DATA_DIR / "full_bead_pull/result_bf_21_0.jld2", *, cache_dir=OUTPUTS_DIR / "full_bead_pull"):
    """Read saved results, caching numeric arrays under data/outputs by source hash.

    Set cache_dir=None to disable the cache. The original JLD2 is always read-only.
    The cache contains nominal arrays and saved errors, not executable objects.
    """
    path = Path(path).resolve()
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = Path(cache_dir)/f"{path.stem}_comparison.npz" if cache_dir is not None else None
    if cache is not None and cache.is_file():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved["source_sha256"]) == fingerprint and int(saved["reader_version"]) == 1:
                parameters = {key.removeprefix("parameter__"): saved[key] for key in saved.files if key.startswith("parameter__")}
                errors = {key.removeprefix("error__"): saved[key] for key in saved.files if key.startswith("error__")}
                return FullBeadData(path, saved["frequencies_Hz"], saved["z_absolute_m"], saved["ET"],
                                    saved["beta2"], saved["beta2_error"], parameters, errors,
                                    saved["reflection"], int(saved["n_mc"]))
    full = _read_full_archive(path)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        values = dict(source_sha256=fingerprint, reader_version=1, frequencies_Hz=full.frequencies_Hz,
                      z_absolute_m=full.z_absolute_m, ET=full.ET, beta2=full.beta2, beta2_error=full.beta2_error,
                      reflection=full.reflection, n_mc=full.n_mc)
        values.update({f"parameter__{key}": value for key, value in full.parameters.items()})
        values.update({f"error__{key}": value for key, value in full.parameter_errors.items()})
        np.savez_compressed(cache, **values)
    return full


def field_reference(full, frequency_GHz=20.825, z_m=None):
    """Full-scan measured integrals and saved mean-parameter model for plotting."""
    frequency = frequency_GHz * 1e9
    k = int(np.argmin(abs(full.frequencies_Hz - frequency)))
    if abs(full.frequencies_Hz[k] - frequency) > 1:
        raise ValueError("The field comparison requires an acquired full-scan frequency.")
    z = np.linspace(0, 0.05, 1001) if z_m is None else np.asarray(z_m, float)
    parameters = full.parameters_at(k)
    model = evaluate_full_model(frequency, z, parameters)
    edges = disk_edges(parameters)
    bead = _mask_bead_centres(model.bead_magnitude[:, 0], z, edges, parameters["r_b"])
    return dict(
        frequency_hz=np.array([frequency]), z_m=z,
        field_Vm=model.field.T, bead_field_Vm=bead[None],
        measured_z_m=full.z_from_mirror(k), measured_field_Vm=full.ET[:, k],
        disk_edges_m=edges,
    )


def common_frequency_indices(data, full, geometry="both"):
    """Match actual acquired bins within one Hz; never interpolate boost values."""
    frequency = np.asarray(_plots._fitplot_result(data, geometry)["frequencies_Hz"])
    full_frequency = full.frequencies_Hz
    right = np.searchsorted(full_frequency, frequency).clip(0, len(full_frequency)-1)
    left = (right-1).clip(0)
    nearest = np.where(abs(full_frequency[left]-frequency) <= abs(full_frequency[right]-frequency), left, right)
    selected = abs(full_frequency[nearest]-frequency) <= 1.0
    if not np.any(selected):
        raise ValueError("The spider and full-bead results have no matching saved frequency bins.")
    return np.flatnonzero(selected), nearest[selected]


def matched_frequency(data, full, frequency_GHz=21.0, geometry="both"):
    """Indices of a common saved frequency, nearest the requested GHz value."""
    requested = float(frequency_GHz)*1e9
    if not np.isfinite(requested):
        raise ValueError("frequency_GHz must be finite.")
    spider, reference = common_frequency_indices(data, full, geometry)
    frequency = _plots._fitplot_result(data, geometry)["frequencies_Hz"]
    index = int(np.argmin(abs(frequency[spider]-requested)))
    return int(spider[index]), int(reference[index])


# Plotting uses saved means and retains each archive's absolute normalization.
from . import pipeline_plots as _plots
from .full_bead_model import disk_edges, evaluate_full_model

_FULL_COLOR = "#222222"


def comparison_summary(data, full, frequency_GHz=21.0, *, compact=False):
    """Report provenance, matching bins and as-saved metrics without rescaling."""
    import pandas as pd
    from IPython.display import Markdown, display
    si, fi = matched_frequency(data, full, frequency_GHz)
    spider_indices, full_indices = common_frequency_indices(data, full)
    frequency = data.fits_1d["both"]["frequencies_Hz"][si]
    rows = []
    for mode, fit in data.fits_1d.items():
        selected = (fit["frequencies_Hz"] >= full.frequencies_Hz.min()) & (fit["frequencies_Hz"] <= full.frequencies_Hz.max())
        bins = np.flatnonzero(selected)
        peak = bins[np.argmax(fit["beta2"][bins])]
        matches = np.flatnonzero(abs(fit["frequencies_Hz"]-frequency) <= 1)
        residual, boost = np.nan, np.nan
        if len(matches):
            index = int(matches[0])
            measured = abs(fit["ET_measured"][:, index])
            residual = np.linalg.norm(fit["ET_fitted_magnitude"][:, index]-measured)/np.linalg.norm(measured)
            boost = fit["beta2"][index]
        rows.append({"Result": f"Spider 1D · {mode}", "Saved β² at selected f": boost,
                     "Peak f in overlap [GHz]": fit["frequencies_Hz"][peak]/1e9,
                     "Saved peak β²": fit["beta2"][peak], "Magnitude residual [%]": residual*100,
                     "Measured z planes": len(data.fields["ET"])})
    predicted = evaluate_full_model(frequency, full.z_from_mirror(fi), full.parameters_at(fi)).bead_magnitude[:, 0]
    observed = abs(full.ET[:, fi])
    valid = np.isfinite(predicted)
    residual = np.linalg.norm(predicted[valid]-observed[valid])/np.linalg.norm(observed[valid])
    peak = int(np.argmax(full.beta2))
    rows.append({"Result": "Full bead pull · saved MC result", "Saved β² at selected f": full.beta2[fi],
                 "Peak f in overlap [GHz]": full.frequencies_Hz[peak]/1e9, "Saved peak β²": full.beta2[peak],
                 "Magnitude residual [%]": residual*100, "Measured z planes": full.ET.shape[0]})
    if compact:
        display(Markdown(f"**Reference:** {full.frequencies_Hz.min()/1e9:g}–{full.frequencies_Hz.max()/1e9:g} GHz · "
                         f"comparison at {frequency/1e9:.4f} GHz. Values retain saved normalizations."))
        display(pd.DataFrame(rows).set_index("Result").round(4))
        return
    display(Markdown(f"**Full-bead source:** `{full.source.name}` · {len(full.frequencies_Hz):,} frequencies "
                     f"from {full.frequencies_Hz.min()/1e9:g} to {full.frequencies_Hz.max()/1e9:g} GHz · "
                     f"{len(full.z_absolute_m)} longitudinal samples · {full.n_mc} saved Monte Carlo draws.  \n"
                     f"**Selected common frequency:** {frequency/1e9:.4f} GHz · "
                     f"**Common acquired bins:** {len(spider_indices)}. No frequency interpolation.  \n"
                     "Absolute values below retain their saved normalizations. Full-pull residuals describe "
                     "the model at mean parameters, not the mean residual of the Monte Carlo ensemble."))
    display(pd.DataFrame(rows).set_index("Result").round(4))
    if data.fit_3d is not None:
        low, high = data.fit_3d["frequencies_Hz"][[0, -1]]/1e9
        display(Markdown(f"**3D:** the saved trial has {len(data.fit_3d['frequencies_Hz'])} points "
                         f"at {low:g}–{high:g} GHz; it has no result at 21 GHz. Its receive-coupled boost "
                         "is shown at those actual frequencies, with its squared-field residual. "
                         "The full-pull archive has no local E(x,y,z) map for a local-field overlay."))


def reference_peak_frequency(data, full):
    """Full-pull boost maximum evaluated on the common acquired frequency grid."""
    _, indices = common_frequency_indices(data, full)
    return float(full.frequencies_Hz[indices[np.argmax(full.beta2[indices])]]/1e9)


def plot_boost_comparison(
    data, full, *,
    title="Spider and full bead pull · saved normalizations",
    x_label="Frequency [GHz]",
    y_label="Boost factor $\\beta^2$",
    full_curve_label="Full bead pull",
    spider_curve_label="Spider bead pull",
):
    """Full scan versus the 1D 'both' fit, styled like the 3D comparison.

    The spider band, when present, is the existing pipeline's saved MC spread.
    """
    fit = _plots._fitplot_result(data, "both")
    select = (fit["frequencies_Hz"] >= 20e9) & (fit["frequencies_Hz"] <= 22e9)
    frequency, values = fit["frequencies_Hz"][select]/1e9, fit["beta2"][select]
    spider_error = np.asarray(fit.get("beta2_error", np.zeros_like(fit["beta2"])))[select]
    figure, axis = plt.subplots(figsize=(6.5, 4.0), layout="constrained")
    axis.fill_between(full.frequencies_Hz/1e9, np.maximum(0, full.beta2-full.beta2_error),
                      full.beta2+full.beta2_error, color="#42484D", alpha=0.16, lw=0)
    axis.plot(full.frequencies_Hz/1e9, full.beta2, color="#42484D", ls="--", lw=1.6,
              label=render_text(full_curve_label))
    axis.plot(frequency, values, color="#AA3377", lw=1.8, label=render_text(spider_curve_label))
    if "beta2_error" in fit:
        axis.fill_between(frequency, np.maximum(0, values-spider_error), values+spider_error,
                          color="#AA3377", alpha=0.24, lw=0)
    axis.set(xlabel=render_text(x_label), ylabel=render_text(y_label), title=render_text(title),
             ylim=(0, 1.08*max(np.max(full.beta2+full.beta2_error), np.max(values+spider_error))), xlim=(20, 22))
    axis.legend(fontsize=8.5, loc="best")
    axis.grid(alpha=0.15)
    return figure


def _mask_bead_centres(values, z, edges, radius):
    values = np.array(values, copy=True)
    valid = z >= radius
    for start, end in edges:
        valid &= (z <= start-radius) | (z >= end+radius)
    values[~valid] = np.nan
    return values


def _profile_comparison(data, full, frequency_GHz, geometry):
    si, fi = matched_frequency(data, full, frequency_GHz, geometry)
    fit = _plots._fitplot_result(data, geometry)
    frequency = fit["frequencies_Hz"][si]
    config = _plots._fitplot_config(fit)
    parameters = full.parameters_at(fi)
    reference_disks = disk_edges(parameters)
    spider_disks = _plots._fitplot_disks(config)
    maximum = max(reference_disks[-1, 1], spider_disks[-1, 1], full.z_from_mirror(fi).max(),
                  np.max(fit["z_from_mirror_m"]))+0.008
    z = np.linspace(0, maximum, 1400)
    reference = evaluate_full_model(frequency, z, parameters)
    spider = _plots._fitplot_evaluate_saved(fit, [frequency], z)
    reference_bead = _mask_bead_centres(reference.bead_magnitude[:, 0], z, reference_disks, parameters["r_b"])
    spider_bead = _mask_bead_centres(spider.bead_magnitude[:, 0], z, spider_disks, config.bead_diameter_m/2)
    return si, fi, frequency, fit, config, parameters, z, reference, spider, reference_bead, spider_bead


def plot_field_comparison(
    data, full, frequency_GHz=21.0, geometry="both", *,
    title="Full versus spider bead pull · common saved frequency {frequency_GHz:.4f} GHz\nGrey disks: full fit · dotted disks: spider fit · saved absolute values",
    x_label="Distance outward from each fitted mirror [mm]",
    y_label="|ET| [V m] · as saved",
    effective_field_title="Finite-bead effective field",
    deconvolved_field_title="Deconvolved field",
    full_bead_label="Full · bead model at mean parameters",
    spider_bead_label_template="Spider {geometry} · bead model",
    full_measured_label_template="Full · {n_full} measured integrals",
    spider_measured_label_template="Spider · {n_spider} measured integrals",
    full_field_label="Full · model at mean parameters",
    spider_field_label_template="Spider {geometry} · model",
):
    """Compare measured and fitted ET in their saved absolute normalization."""
    si, fi, frequency, fit, config, params, z, reference, spider, full_bead, spider_bead = _profile_comparison(data, full, frequency_GHz, geometry)
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    color = _plots._FIT_GEOMETRY_COLORS.get(geometry, "C0")
    full_measured, spider_measured = abs(full.ET[:, fi]), abs(fit["ET_measured"][:, si])
    full_true = abs(reference.field[:, 0])
    spider_true = abs(spider.field[:, 0])
    context = dict(frequency_GHz=frequency/1e9, geometry=geometry,
                   n_full=len(full_measured), n_spider=len(spider_measured))
    axis = axes[0]
    axis.plot(z*1e3, full_bead, "--", color=_FULL_COLOR, label=render_text(full_bead_label, **context))
    axis.plot(z*1e3, spider_bead, "--", color=color, label=render_text(spider_bead_label_template, **context))
    axis.scatter(full.z_from_mirror(fi)*1e3, full_measured, facecolors="white", edgecolors=_FULL_COLOR,
                 s=23, zorder=5, label=render_text(full_measured_label_template, **context))
    axis.scatter(fit["z_from_mirror_m"]*1e3, spider_measured, color=color, edgecolors="white", s=48,
                 zorder=6, label=render_text(spider_measured_label_template, **context))
    axis.set(ylabel=render_text(y_label, **context), title=render_text(effective_field_title, **context))
    lower = axes[1]
    lower.plot(z*1e3, full_true, color=_FULL_COLOR, label=render_text(full_field_label, **context))
    lower.plot(z*1e3, spider_true, color=color, label=render_text(spider_field_label_template, **context))
    lower.set(xlabel=render_text(x_label, **context), ylabel=render_text(y_label, **context),
              title=render_text(deconvolved_field_title, **context))
    for axis in axes.ravel():
        for start, end in disk_edges(params):
            axis.axvspan(start*1e3, end*1e3, color="0.5", alpha=0.13, zorder=0)
        for start, end in _plots._fitplot_disks(config):
            axis.axvspan(start*1e3, end*1e3, facecolor="none", edgecolor=color, ls=":", lw=1, zorder=0)
        axis.set_xlim(0, z[-1]*1e3)
        axis.set_ylim(bottom=0)
    axes[0].legend(fontsize=7.5)
    axes[1].legend(fontsize=8)
    figure.suptitle(render_text(title, **context))
    return figure


def plot_geometry_comparison(
    data, full, frequency_GHz=21.0, geometry="both", *,
    title="Geometry and sampling · {frequency_GHz:.4f} GHz\nFull absolute mirror coordinate {full_mirror_z_m:.6f} m; spider nominal mirror coordinate {spider_mirror_z_m:.3f} m",
    x_label="Distance outward from each fitted mirror [mm]",
    full_panel_title_template="Full bead pull · fitted geometry · {n_positions} sampled z positions",
    spider_panel_title_template="Spider · {geometry} fitted geometry · {n_positions} sampled z positions",
    disk_label_template="D{disk_index}",
    thread_label_template="T{thread_index}",
):
    """Sampling positions and independent fitted setups in a common mirror frame."""
    si, fi = matched_frequency(data, full, frequency_GHz, geometry)
    fit = _plots._fitplot_result(data, geometry)
    params = full.parameters_at(fi)
    setups = [(full_panel_title_template, disk_edges(params), full.z_from_mirror(fi), _FULL_COLOR),
              (spider_panel_title_template, _plots._fitplot_disks(_plots._fitplot_config(fit)),
               fit["z_from_mirror_m"], _plots._FIT_GEOMETRY_COLORS.get(geometry, "C0"))]
    context = dict(frequency_GHz=fit["frequencies_Hz"][si]/1e9, geometry=geometry,
                   full_mirror_z_m=params["z_m"], spider_mirror_z_m=float(data.fields["mirror_z_m"]))
    figure, axes = plt.subplots(2, 1, figsize=(12.5, 5.7), sharex=True, constrained_layout=True)
    for axis, (label, edges, positions, color) in zip(axes, setups):
        axis.axvline(0, color="0.3", lw=5)
        axis.axhline(0, color="0.7", lw=1)
        for i, (start, end) in enumerate(edges):
            axis.axvspan(start*1e3, end*1e3, color=color, alpha=0.18)
            axis.text((start+end)*500, 0.55, render_text(disk_label_template, disk_index=i+1, **context), ha="center", fontsize=9)
        axis.scatter(positions*1e3, np.zeros(len(positions)), color=color, s=35 if len(positions) == 4 else 18, zorder=5)
        if len(positions) == 4:
            for index, z in zip(data.fields["sub_thread_indices"], positions):
                axis.text(z*1e3, 0.15, render_text(thread_label_template, thread_index=index, **context), color=color, ha="center", fontsize=9)
        axis.set(title=render_text(label, n_positions=len(positions), **context), yticks=[], ylim=(-0.5, 0.9))
        axis.spines[["left", "right", "top"]].set_visible(False)
    axes[-1].set(xlabel=render_text(x_label, **context), xlim=(-0.5, 39))
    figure.suptitle(render_text(title, **context))
    return figure


def plot_gap_comparison(
    data, full, *,
    title="Surface-to-surface gaps · full per-frequency geometry versus shared spider geometry",
    x_label="Frequency [GHz]",
    y_label_template="Gap {gap_index} [mm]",
    full_curve_label="Full · fitted per frequency",
    spider_curve_label_template="Spider · {mode}",
):
    """Per-frequency full-pull fitted gaps versus shared spider fitted gaps."""
    figure, axes = plt.subplots(3, 1, figsize=(12.5, 8), sharex=True, constrained_layout=True)
    frequency = full.frequencies_Hz/1e9
    for i, axis in enumerate(axes):
        axis.plot(frequency, full.parameters[f"d_v_{i+1}"]*1e3, color=_FULL_COLOR, lw=2, label=render_text(full_curve_label, gap_index=i+1))
        for mode, color in _plots._FIT_GEOMETRY_COLORS.items():
            if mode in data.fits_1d:
                axis.axhline(data.fits_1d[mode]["gap_widths_m"][i]*1e3, color=color, ls="--", label=render_text(spider_curve_label_template, mode=mode, gap_index=i+1))
        axis.set(ylabel=render_text(y_label_template, gap_index=i+1), xlim=(frequency[0], frequency[-1]))
    axes[0].legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel(render_text(x_label))
    figure.suptitle(render_text(title))
    return figure


def _common_model_grid(data, full, geometry, points=240):
    spider_indices, full_indices = common_frequency_indices(data, full, geometry)
    fit = _plots._fitplot_result(data, geometry)
    frequency = fit["frequencies_Hz"][spider_indices]
    z = np.linspace(0, 0.04, points)
    reference = np.column_stack([evaluate_full_model(value, z, full.parameters_at(fi)).field[:, 0]
                                 for value, fi in zip(frequency, full_indices)])
    spider = _plots._fitplot_evaluate_saved(fit, frequency, z).field
    return frequency, z, reference, spider


def plot_field_map_comparison(
    data, full, geometry="both", *,
    title="Longitudinal field magnitudes · shared colour scale; saved normalizations",
    x_label="Common acquired frequency [GHz]",
    y_label="Distance outward from mirror [mm]",
    colorbar_label="Deconvolved |ET| [V m] · as saved",
    full_panel_title="Full · model at mean parameters",
    spider_panel_title_template="Spider · {geometry}",
):
    """Dense models on exactly matched frequencies, preserving per-disk properties."""
    frequency, z, reference, spider = _common_model_grid(data, full, geometry)
    maps = [abs(reference), abs(spider)]
    maximum = max(value.max() for value in maps)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True, constrained_layout=True)
    context = dict(geometry=geometry)
    for axis, values, label in zip(axes, maps, [full_panel_title, spider_panel_title_template]):
        image = axis.pcolormesh(frequency/1e9, z*1e3, values, shading="auto", cmap="magma", vmin=0, vmax=maximum, rasterized=True)
        axis.set(title=render_text(label, **context), xlabel=render_text(x_label, **context))
    axes[0].set_ylabel(render_text(y_label, **context))
    figure.colorbar(image, ax=list(axes), label=render_text(colorbar_label, **context))
    figure.suptitle(render_text(title, **context))
    return figure


def plot_fit_quality_comparison(
    data, full, *,
    title="‖model − measured |ET|‖₂ / ‖measured |ET|‖₂ · each scan at its own samples",
    trial_title="3D trial uses a different observable · no matching full-pull spatial data",
    x_label="Frequency [GHz]",
    y_label="Magnitude residual [%]",
    trial_y_label="Weighted complex-Q residual [%]",
    full_curve_label_template="Full · {n_full} planes, mean-parameter model",
    spider_curve_label_template="Spider {mode} · {n_spider} planes",
    trial_curve_label="Spider 3D trial · saved frequencies",
):
    """Magnitude residuals use the same formula, but each scan has its own z set."""
    spider_indices, full_indices = common_frequency_indices(data, full)
    frequency = data.fits_1d["both"]["frequencies_Hz"][spider_indices]
    residuals = []
    for value, fi in zip(frequency, full_indices):
        predicted = evaluate_full_model(value, full.z_from_mirror(fi), full.parameters_at(fi)).bead_magnitude[:, 0]
        observed = abs(full.ET[:, fi])
        finite = np.isfinite(predicted)
        residuals.append(np.linalg.norm(predicted[finite]-observed[finite])/np.linalg.norm(observed[finite]))
    figure, axes = plt.subplots(2, 1, figsize=(12.5, 7), sharex=True, constrained_layout=True)
    context = dict(n_full=full.ET.shape[0])
    axes[0].plot(frequency/1e9, np.array(residuals)*100, color=_FULL_COLOR, lw=2,
                 label=render_text(full_curve_label_template, **context))
    for mode, color in _plots._FIT_GEOMETRY_COLORS.items():
        if mode not in data.fits_1d:
            continue
        fit = data.fits_1d[mode]
        indices, _ = common_frequency_indices(data, full, mode)
        observed = abs(fit["ET_measured"][:, indices])
        residual = np.linalg.norm(fit["ET_fitted_magnitude"][:, indices]-observed, axis=0)/np.linalg.norm(observed, axis=0)
        axes[0].plot(fit["frequencies_Hz"][indices]/1e9, residual*100, color=color,
                     label=render_text(spider_curve_label_template, mode=mode, n_spider=observed.shape[0], **context))
    axes[0].set(ylabel=render_text(y_label, **context), title=render_text(title, **context), ylim=(0, None))
    axes[0].legend(fontsize=8, ncol=2)
    if data.fit_3d is not None:
        fit = data.fit_3d
        axes[1].plot(fit["frequencies_Hz"]/1e9, fit["relative_residual"]*100, "D", color="#c54036", label=render_text(trial_curve_label, **context))
        axes[1].legend(fontsize=9)
    axes[1].set(xlabel=render_text(x_label, **context), ylabel=render_text(trial_y_label, **context), ylim=(0, 105),
                title=render_text(trial_title, **context),
                xlim=(frequency[0]/1e9, frequency[-1]/1e9))
    return figure


def _boost_normalization_audit(handle, values, frequencies_Hz, *, reference_radius_m=0.15,
                               reference_input_power_W=1.0):
    """Extract quadratic/inverse factors from the saved boost derivative chain.

    This archive's Measurement derivative map contains two nonzero independent
    inputs: one enters as a square and the other as an inverse. Their physical
    variable names are not serialized. We identify them by derivatives 2*b/x
    and -b/y, rather than assuming that y is input power or a beam correction.
    The quadratic input agrees with the saved MC mean absolute overlap, but
    this audit does not rerun MC or rescale any measured/saved values.
    """
    frequency = np.asarray(frequencies_Hz, dtype=float)
    raw = np.asarray(values).ravel()
    if raw.size != frequency.size:
        return {"available": False, "reason": "Boost/frequency lengths differ."}
    rows = []
    for item in raw:
        try:
            record = handle[item][()] if isinstance(item, h5py.Reference) else item
            boost = float(record["val"])
            node = record["der"]
            candidates = []
            for _ in range(32):
                names = node.dtype.names or ()
                if not {"key", "value"}.issubset(names):
                    break
                key = node["key"]
                value, error, derivative = float(key["1"]), float(key["2"]), float(node["value"])
                if value != 0 and derivative != 0:
                    candidates.append((value, error, derivative))
                if "parent" not in names or not node["parent"] or derivative == 0:
                    break
                node = handle[node["parent"]][()]
            quadratic = [entry for entry in candidates
                         if np.isclose(entry[2]*entry[0], 2*boost, rtol=1e-9, atol=0)]
            inverse = [entry for entry in candidates
                       if np.isclose(entry[2]*entry[0], -boost, rtol=1e-9, atol=0)]
            if len(quadratic) != 1 or len(inverse) != 1 or boost <= 0:
                return {"available": False, "reason": "Derivative keys do not identify unique positive factors."}
            rows.append((boost, quadratic[0][0], quadratic[0][1], inverse[0][0], inverse[0][1]))
        except (KeyError, TypeError, ValueError, AttributeError):
            return {"available": False, "reason": "Saved Measurements derivative map is unavailable."}
    boost, quadratic, quadratic_error, inverse, inverse_error = np.asarray(rows).T
    reference = (8.85418782e-12*(2*np.pi*frequency)**2*quadratic**2
                 / (8*299792458.0*np.pi*reference_radius_m**2*reference_input_power_W))
    factor = boost/reference
    return {
        "available": True,
        "quadratic_input": quadratic,
        "quadratic_input_error": quadratic_error,
        "inverse_input": inverse,
        "inverse_input_error": inverse_error,
        "reference_radius_m": float(reference_radius_m),
        "reference_input_power_W": float(reference_input_power_W),
        "reference_beta2_from_quadratic_input": reference,
        "saved_to_reference_factor": factor,
        "factor_times_inverse_input": factor*inverse,
        "effective_input_power_W_for_reference_area": reference_input_power_W/factor,
    }


def normalization_summary(full, *, compact=False):
    """Expose the saved normalization difference without inventing its meaning."""
    from IPython.display import Markdown, display
    with h5py.File(full.source, "r") as handle:
        audit = _boost_normalization_audit(handle, handle["boostfactor"][()], full.frequencies_Hz)
    if not audit["available"]:
        if compact:
            display(Markdown("**Normalization:** unavailable; values are shown as saved."))
            return
        display(Markdown("**Normalization:** the archive does not expose enough bookkeeping to compare its absolute normalization. "
                         "The comparison retains the saved absolute values."))
        return
    ratio = audit["saved_to_reference_factor"]
    inverse = audit["inverse_input"]
    if compact:
        display(Markdown(f"**Normalization:** full-reference β² carries a factor {ratio.min():.6g}–{ratio.max():.6g} "
                         "relative to the spider convention. Physical correction factors are unknown; no rescaling is applied."))
        return
    display(Markdown(
        "**Normalization audit:** relative to the spider formula evaluated with the full scan's saved "
        "Monte Carlo mean overlap, a **1 W** reference and a **150 mm** disk radius, the full archive's "
        f"boost carries a factor **{ratio.min():.8g}–{ratio.max():.8g}**. Its saved uncertainty bookkeeping "
        f"contains an inverse factor **{inverse.min():.6g}–{inverse.max():.6g}**; "
        f"the remaining multiplier is **{np.median(audit['factor_times_inverse_input']):.6g}**. "
        "The file does not name their physical origin. No power or coupling correction has been guessed or applied. "
        "All comparison plots show **as-saved** absolute values. "
        "The full-pull input-power and correction metadata are needed before interpreting an absolute amplitude difference."))
