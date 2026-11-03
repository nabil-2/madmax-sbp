"""Verification summary figures with text controls for the notebook."""

import matplotlib.pyplot as plt
import numpy as np
from .verification_core import beam_model
from .sbp_verification import match_full_to_reduced
from .plot_text import render_text
from .pipeline_plots import present


def plot_fields(
    run,
    *,
    z_index=0,
    frequency_index=0,
    title_template="{frequency_GHz:.3f} GHz · acquisition z={z_mm:.1f} mm",
    full_title="Full bead pull",
    fit_title="Limited-line fit",
    residual_title="Complex residual / full-grid RMS",
    x_label="x [mm]",
    y_label="y [mm]",
    field_colorbar_label=r"$|E_{\mathrm{conv}}|$ [V/m]",
    residual_colorbar_label="%",
):
    """Full-grid field magnitude, fitted field and complex residual."""
    data = run["data"]
    original_z = data["z_ixs_used"][z_index]
    full = data["E_full"][:, :, original_z, frequency_index]
    fit = beam_model(
        data["x_plt"], data["y_plt"], *run["params"][z_index, frequency_index]
    )
    maximum = max(np.max(abs(full)), np.max(abs(fit)))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
    x, y = data["x_pos"] * 1e3, data["y_pos"] * 1e3
    for ax, field, title in zip(axes[:2], (full, fit), (full_title, fit_title)):
        image = ax.pcolormesh(x, y, abs(field).T, shading="auto", vmin=0, vmax=maximum)
        ax.set_title(title)
    fig.colorbar(image, ax=list(axes[:2]), label=field_colorbar_label)
    axes[0].plot(
        data["x_reduced"] * 1e3, data["y_reduced"] * 1e3, ".", color="white", ms=2
    )
    residual = abs(full - fit) / max(np.linalg.norm(full) / np.sqrt(full.size), 1e-30)
    image = axes[2].pcolormesh(x, y, residual.T * 100, shading="auto", cmap="magma")
    axes[2].set_title(residual_title)
    fig.colorbar(image, ax=axes[2], label=residual_colorbar_label)
    for ax in axes:
        ax.set(xlabel=x_label, ylabel=y_label, aspect="equal")
    fig.suptitle(
        render_text(
            title_template,
            frequency_GHz=data["frequencies"][frequency_index] * 1e-9,
            z_mm=data["z_pos"][original_z] * 1e3,
        )
    )
    return fig


def plot_integrals(
    result,
    *,
    frequency_index=0,
    title_template="{frequency_GHz:.3f} GHz",
    x_label="Acquisition z [mm]",
    magnitude_y_label=r"$|E_T|$ [V m]",
    real_y_label=r"$\mathrm{Re}(E_T)$ [V m]",
    imaginary_y_label=r"$\mathrm{Im}(E_T)$ [V m]",
    full_label="Full bead pull",
    reduced_label="Limited lines",
):
    """Complex transverse integrals along the acquired z coordinate."""
    full = result["ETs_full"][:, frequency_index]
    reduced = result["ETs_reduced"][:, frequency_index]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), layout="constrained")
    for ax, transform, label in zip(
        axes,
        (np.abs, np.real, np.imag),
        (magnitude_y_label, real_y_label, imaginary_y_label),
    ):
        ax.plot(result["z_full"] * 1e3, transform(full), "o", label=full_label)
        ax.plot(
            result["z_reduced"] * 1e3,
            transform(reduced),
            "x",
            ms=7,
            label=reduced_label,
        )
        ax.set(xlabel=x_label, ylabel=label)
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle(
        render_text(
            title_template, frequency_GHz=result["frequencies"][frequency_index] * 1e-9
        )
    )
    return fig


def plot_integral_error(
    result,
    *,
    title="Transverse-integral reconstruction error",
    x_label="Frequency [GHz]",
    y_label="Complex ET error [%]",
):
    """Relative complex ET error over matched planes, per frequency."""
    full, _ = match_full_to_reduced(
        result["ETs_full"], result["z_full"], result["z_reduced"]
    )
    denominator = np.linalg.norm(full, axis=0)
    error = np.full(denominator.shape, np.nan)
    np.divide(
        np.linalg.norm(result["ETs_reduced"] - full, axis=0),
        denominator,
        out=error,
        where=denominator > 0,
    )
    fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    ax.plot(result["frequencies"] * 1e-9, error * 100, ".-")
    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    ax.grid(alpha=0.2)
    return fig
