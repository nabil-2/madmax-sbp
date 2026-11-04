"""Plot measured inputs and saved 3D-fit diagnostics without fitting anything."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, Normalize
from matplotlib.patches import Circle
from .analysis_3d_plots import entry, at_frequency, kappa, COLOR, THREAD_COLORS
from .plot_text import render_text


def _indices(result, thread):
    indices = np.flatnonzero(result["thread_index"] == thread)
    return indices[np.argsort(result["signed_line_m"][indices])]


def _four_panels(title):
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    figure.suptitle(title)
    return figure, axes


def incident_beam_figure(
    beam,
    *,
    title="Incident beam supplied to the 3D model · {frequency_GHz:.4f} GHz",
    x_label="x [mm]",
    y_label="y [mm]",
    colorbar_label="$|E_{in}|$ [V/m]",
    cut_x_label="Distance from beam peak [mm]",
    cut_y_label="$|E_{in}|$ [V/m]",
    x_cut_label="x cut through peak",
    y_cut_label="y cut through peak",
):
    x, y = beam["x_m"] * 1000, beam["y_m"] * 1000
    field = abs(beam["field_V_per_m"])
    i, j = np.unravel_index(np.argmax(field), field.shape)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    image = axes[0].pcolormesh(
        x, y, field.T, shading="auto", cmap="viridis", rasterized=True
    )
    axes[0].add_patch(
        Circle(
            (0, 0),
            beam["disk_radius_m"] * 1000,
            fill=False,
            edgecolor="white",
            ls="--",
            lw=1,
        )
    )
    axes[0].set(
        xlabel=render_text(x_label), ylabel=render_text(y_label), aspect="equal"
    )
    figure.colorbar(image, ax=axes[0], label=render_text(colorbar_label))
    axes[1].plot(x - x[i], field[:, j], label=render_text(x_cut_label))
    axes[1].plot(y - y[j], field[i], label=render_text(y_cut_label))
    axes[1].set(
        xlabel=render_text(cut_x_label), ylabel=render_text(cut_y_label), ylim=(0, None)
    )
    axes[1].legend()
    figure.suptitle(render_text(title, frequency_GHz=beam["frequency_hz"][0] / 1e9))
    return entry(
        figure,
        "incident_beam_profile",
        "Prescribed incident beam",
        "Actual six-profile beam mixture at the model's outer reference face, normalized to 1 W over the whole plane. Phase is assumed flat; the original scan z origin is not registered to the mirror. The circle marks the 150 mm aperture, not an input-power renormalization.",
    )


def profile_quality_figure(
    profiles,
    *,
    title="Preliminary line fits and frequency selection",
    x_label="Frequency [GHz]",
    center_y_label="Gaussian centre [mm]",
    width_y_label="Field 1/e radius [mm]",
    thread_label_template="T{thread_index}",
    selection_label="Geometry bins",
    rejected_label="Rejected",
    accepted_label="Accepted",
):
    f = profiles["frequency_hz"] / 1e9
    parameters = profiles["gaussian_parameters"]
    figure, axes = plt.subplots(
        3, 1, figsize=(10, 8), sharex=True, layout="constrained"
    )
    for thread, color in enumerate(THREAD_COLORS):
        label = render_text(thread_label_template, thread_index=thread)
        axes[0].plot(f, parameters[thread, :, 0] * 100, color=color, label=label)
        axes[1].plot(f, parameters[thread, :, 1] * 100, color=color, label=label)
    quality = np.vstack((profiles["profile_quality"].T, profiles["shared_fit_mask"]))
    image = axes[2].pcolormesh(
        f,
        np.arange(5),
        quality,
        shading="nearest",
        cmap=ListedColormap(["#e4e8eb", "#228833"]),
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    axes[0].set(ylabel=render_text(center_y_label))
    axes[1].set(ylabel=render_text(width_y_label))
    axes[2].set(
        xlabel=render_text(x_label),
        yticks=np.arange(5),
        yticklabels=[
            render_text(thread_label_template, thread_index=t) for t in range(4)
        ]
        + [render_text(selection_label)],
    )
    bar = figure.colorbar(image, ax=axes[2], ticks=[0, 1], pad=0.02)
    bar.ax.set_yticklabels([render_text(rejected_label), render_text(accepted_label)])
    axes[0].legend(ncol=4)
    axes[2].set_xlim(20, 22)
    figure.suptitle(render_text(title))
    return entry(
        figure,
        "profile_quality",
        "Frequency selection",
        "Saved preliminary Gaussian line fits are diagnostics only. Acceptance on T1–T3 selects geometry-fit frequencies; T0 quality does not select frequencies, but T0 still enters the 3D fit at those frequencies.",
    )


def raw_fit_figure(
    result,
    frequency_GHz=20.825,
    *,
    title="Measured reflection, model and background · {frequency_GHz:.4f} GHz",
    x_label="Signed transverse position [mm]",
    real_y_label=r"Re $\Gamma$",
    imaginary_y_label=r"Im $\Gamma$",
    thread_label_template="T{thread_index}",
    measured_label="Measured",
    model_label="Fitted model",
    background_label="Fitted background",
):
    k = at_frequency(result, frequency_GHz)
    figure, axes = plt.subplots(
        4, 2, figsize=(11, 10), sharex=True, layout="constrained"
    )
    for thread in range(4):
        ids = _indices(result, thread)
        x = result["signed_line_m"][ids] * 1000
        for column, component in enumerate((np.real, np.imag)):
            axis = axes[thread, column]
            axis.plot(
                x,
                component(result["observed_S22"][k, ids]),
                ".",
                color="#333333",
                ms=3,
                label=render_text(measured_label),
            )
            axis.plot(
                x,
                component(result["predicted_S22"][k, ids]),
                color=COLOR,
                label=render_text(model_label),
            )
            axis.plot(
                x,
                component(result["fitted_background"][k, ids]),
                "--",
                color="#77838b",
                label=render_text(background_label),
            )
            axis.set(
                ylabel=render_text(real_y_label if column == 0 else imaginary_y_label),
                title=render_text(thread_label_template, thread_index=thread),
            )
    axes[0, 0].legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel(render_text(x_label))
    figure.suptitle(render_text(title, frequency_GHz=result["frequency_hz"][k] / 1e9))
    return entry(
        figure,
        "raw_reflection_fit",
        "Reflection before background subtraction",
        "Measured complex reflection, the complete saved prediction, and its fitted affine background on the actual sloped bead paths. No additional subtraction or fit is applied.",
    )


def complex_plane_figure(
    result,
    frequency_GHz=20.825,
    *,
    title="Complex reflection along each chord · {frequency_GHz:.4f} GHz",
    x_label=r"Re $\Delta\Gamma$",
    y_label=r"Im $\Delta\Gamma$",
    colorbar_label="Signed transverse position [mm]",
    thread_label_template="T{thread_index}",
    measured_label="Measured",
    model_label="Fitted model",
):
    k = at_frequency(result, frequency_GHz)
    figure, axes = _four_panels(
        render_text(title, frequency_GHz=result["frequency_hz"][k] / 1e9)
    )
    limit = np.max(abs(result["signed_line_m"])) * 1000
    for thread, axis in enumerate(axes.flat):
        ids = _indices(result, thread)
        observed = result["observed_S22"][k, ids] - result["fitted_background"][k, ids]
        predicted = (
            result["predicted_S22"][k, ids] - result["fitted_background"][k, ids]
        )
        points = axis.scatter(
            observed.real,
            observed.imag,
            c=result["signed_line_m"][ids] * 1000,
            cmap="viridis",
            norm=Normalize(-limit, limit),
            s=16,
            label=render_text(measured_label),
        )
        axis.plot(
            predicted.real, predicted.imag, color=COLOR, label=render_text(model_label)
        )
        axis.set(
            xlabel=render_text(x_label),
            ylabel=render_text(y_label),
            title=render_text(thread_label_template, thread_index=thread),
        )
        xvalues = np.r_[observed.real, predicted.real]
        yvalues = np.r_[observed.imag, predicted.imag]
        span = 1.12 * max(np.ptp(xvalues), np.ptp(yvalues), 1e-12)
        center_x = (xvalues.min() + xvalues.max()) / 2
        center_y = (yvalues.min() + yvalues.max()) / 2
        axis.set(
            xlim=(center_x - span / 2, center_x + span / 2),
            ylim=(center_y - span / 2, center_y + span / 2),
        )
        axis.set_aspect("equal", adjustable="box")
    axes[0, 0].legend(fontsize=8)
    figure.colorbar(points, ax=axes, label=render_text(colorbar_label), shrink=0.8)
    return entry(
        figure,
        "complex_plane_fit",
        "Complex-plane fit",
        "Data and prediction after subtracting the same saved background. Colours show bead position; the curve is the complex model trajectory in absolute reflection units.",
    )


def directional_field_figure(
    result,
    frequency_GHz=20.825,
    *,
    title="Model electric fields on the actual bead paths · {frequency_GHz:.4f} GHz",
    x_label="Signed transverse position [mm]",
    y_label="$|E|$ [V/m]",
    thread_label_template="T{thread_index}",
    incoming_label="Incoming",
    outgoing_label="Outgoing",
    total_label="Coherent sum",
):
    k = at_frequency(result, frequency_GHz)
    waves = result["amplitude"][k] * result["directional_waves_V_per_m"][k]
    figure, axes = _four_panels(
        render_text(title, frequency_GHz=result["frequency_hz"][k] / 1e9)
    )
    for thread, axis in enumerate(axes.flat):
        ids = _indices(result, thread)
        x = result["signed_line_m"][ids] * 1000
        axis.plot(x, abs(waves[ids, 0]), label=render_text(incoming_label))
        axis.plot(x, abs(waves[ids, 1]), label=render_text(outgoing_label))
        axis.plot(
            x, abs(waves[ids].sum(axis=-1)), color=COLOR, label=render_text(total_label)
        )
        axis.set(
            xlabel=render_text(x_label),
            ylabel=render_text(y_label),
            ylim=(0, None),
            title=render_text(thread_label_template, thread_index=thread),
        )
    axes[0, 0].legend(fontsize=8)
    return entry(
        figure,
        "directional_fields",
        "Directional model electric fields",
        "Saved incoming and outgoing fields, multiplied once by the fitted amplitude. Their coherent sum is the model electric field without the bead at each sample's actual fitted z.",
    )


def _path_heatmap(
    result,
    values,
    *,
    title,
    stem,
    colorbar_label,
    x_label,
    y_label,
    thread_label_template,
    logarithmic=False,
):
    figure, axes = _four_panels(render_text(title))
    positive = values[np.isfinite(values) & (values > 0)]
    vmax = float(np.max(positive)) if len(positive) else 1.0
    norm = (
        LogNorm(max(float(np.percentile(positive, 2)), vmax * 1e-5), vmax)
        if logarithmic and len(positive)
        else Normalize(0, vmax)
    )
    for thread, axis in enumerate(axes.flat):
        ids = _indices(result, thread)
        image = axis.pcolormesh(
            result["frequency_hz"] / 1e9,
            result["signed_line_m"][ids] * 1000,
            values[:, ids].T,
            shading="auto",
            cmap="viridis",
            norm=norm,
            rasterized=True,
        )
        axis.set(
            xlabel=render_text(x_label),
            ylabel=render_text(y_label),
            xlim=(20, 22),
            title=render_text(thread_label_template, thread_index=thread),
        )
    figure.colorbar(image, ax=axes, label=render_text(colorbar_label), shrink=0.8)
    return entry(
        figure,
        stem,
        render_text(title),
        "The four panels share an absolute colour scale and use the actual sampled sloped paths. No shape normalization is applied.",
    )


def field_spectrum_figure(
    result,
    *,
    title="Model electric field along all four sloped paths",
    x_label="Frequency [GHz]",
    y_label="Signed transverse position [mm]",
    colorbar_label="$|E|$ [V/m]",
    thread_label_template="T{thread_index}",
):
    field = result["amplitude"][:, None] * abs(
        result["directional_waves_V_per_m"].sum(axis=-1)
    )
    return _path_heatmap(
        result,
        field,
        title=title,
        stem="electric_field_spectrum",
        colorbar_label=colorbar_label,
        x_label=x_label,
        y_label=y_label,
        thread_label_template=thread_label_template,
    )


def residual_map_figure(
    result,
    *,
    title="Absolute complex-reflection residuals",
    x_label="Frequency [GHz]",
    y_label="Signed transverse position [mm]",
    colorbar_label=r"$|\widehat\Gamma-\Gamma|$ · logarithmic colour scale",
    thread_label_template="T{thread_index}",
):
    residual = abs(result["predicted_S22"] - result["observed_S22"])
    return _path_heatmap(
        result,
        residual,
        title=title,
        stem="absolute_residual_maps",
        colorbar_label=colorbar_label,
        x_label=x_label,
        y_label=y_label,
        thread_label_template=thread_label_template,
        logarithmic=True,
    )


def effective_spectrum_figure(
    result,
    *,
    title="Finite-bead effective fields: measurement and model",
    x_label="Frequency [GHz]",
    y_label="Signed transverse position [mm]",
    colorbar_label="Effective |E| [V/m]",
    measured_label="Measured",
    model_label="Model",
    thread_label_template="T{thread_index}",
):
    factor = abs(kappa(result["frequency_hz"]))[:, None]
    measured = np.sqrt(
        abs(result["observed_S22"] - result["fitted_background"]) / factor
    )
    fitted = np.sqrt(
        abs(result["predicted_S22"] - result["fitted_background"]) / factor
    )
    vmax = max(measured.max(), fitted.max())
    figure, axes = plt.subplots(
        4, 2, figsize=(11, 11), sharex=True, layout="constrained"
    )
    for thread in range(4):
        ids = _indices(result, thread)
        for column, (values, label) in enumerate(
            ((measured, measured_label), (fitted, model_label))
        ):
            axis = axes[thread, column]
            image = axis.pcolormesh(
                result["frequency_hz"] / 1e9,
                result["signed_line_m"][ids] * 1000,
                values[:, ids].T,
                shading="auto",
                cmap="viridis",
                vmin=0,
                vmax=vmax,
                rasterized=True,
            )
            axis.set(
                title=render_text(thread_label_template, thread_index=thread)
                + " · "
                + render_text(label),
                ylabel=render_text(y_label),
                xlim=(20, 22),
            )
    for axis in axes[-1]:
        axis.set_xlabel(render_text(x_label))
    figure.colorbar(image, ax=axes, label=render_text(colorbar_label), shrink=0.8)
    figure.suptitle(render_text(title))
    return entry(
        figure,
        "finite_bead_field_spectra",
        "Measured and model finite-bead fields",
        "Magnitude-only field estimates from the saved background-subtracted reflection and prediction, using the same bead calibration. These include finite-bead response; the separate electric-field map uses the model field without the bead.",
    )


def signal_strength_figure(
    result,
    *,
    title="Absolute signal and residual strengths",
    x_label="Frequency [GHz]",
    signal_y_label=r"RMS $|\Delta\Gamma|$",
    residual_y_label=r"RMS $|\widehat\Gamma-\Gamma|$",
    thread_label_template="T{thread_index}",
):
    figure, axes = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, layout="constrained"
    )
    for thread, color in enumerate(THREAD_COLORS):
        ids = _indices(result, thread)
        observed = result["observed_S22"][:, ids] - result["fitted_background"][:, ids]
        residual = result["predicted_S22"][:, ids] - result["observed_S22"][:, ids]
        for axis, values in zip(axes, (observed, residual)):
            axis.semilogy(
                result["frequency_hz"] / 1e9,
                np.maximum(np.sqrt(np.mean(abs(values) ** 2, axis=1)), 1e-15),
                color=color,
                label=render_text(thread_label_template, thread_index=thread),
            )
    axes[0].set_ylabel(render_text(signal_y_label))
    axes[1].set(
        xlabel=render_text(x_label), ylabel=render_text(residual_y_label), xlim=(20, 22)
    )
    axes[0].legend(ncol=4)
    figure.suptitle(render_text(title))
    return entry(
        figure,
        "signal_and_residual_strength",
        "Signal and residual strengths",
        "Absolute RMS reflection signal after the fitted background and absolute RMS model error. The same units make the different thread strengths visible without normalizing their shapes.",
    )


def scale_background_figure(
    result,
    *,
    title="Fitted common scale and affine backgrounds",
    x_label="Frequency [GHz]",
    amplitude_y_label="Field-amplitude scale",
    phase_y_label="Common scale phase [rad]",
    intercept_y_label=r"Background $|\Gamma(s=0)|$",
    slope_y_label=r"Background $|\Gamma(B)-\Gamma(A)|$",
    thread_label_template="T{thread_index}",
):
    figure, axes = plt.subplots(
        2, 2, figsize=(11, 7), sharex=True, layout="constrained"
    )
    f = result["frequency_hz"] / 1e9
    axes[0, 0].plot(f, result["amplitude"], color=COLOR)
    axes[0, 1].plot(f, np.unwrap(np.angle(result["quadratic_scale"])), color=COLOR)
    for thread, color in enumerate(THREAD_COLORS):
        ids = _indices(result, thread)
        x = result["signed_line_m"][ids]
        background = result["fitted_background"][:, ids]
        center = np.array(
            [
                np.interp(0, x, row.real) + 1j * np.interp(0, x, row.imag)
                for row in background
            ]
        )
        label = render_text(thread_label_template, thread_index=thread)
        axes[1, 0].plot(f, abs(center), color=color, label=label)
        axes[1, 1].plot(
            f, abs(background[:, -1] - background[:, 0]), color=color, label=label
        )
    for axis, label in zip(
        axes.flat, (amplitude_y_label, phase_y_label, intercept_y_label, slope_y_label)
    ):
        axis.set(xlabel=render_text(x_label), ylabel=render_text(label), xlim=(20, 22))
    axes[1, 0].legend(ncol=4, fontsize=8)
    figure.suptitle(render_text(title))
    return entry(
        figure,
        "scale_and_backgrounds",
        "Fitted scale and backgrounds",
        "The saved common complex coefficient supplies the field scale and phase. Lower panels show each fitted affine background at the scan centre and its change across the retained chord; no new parameters are fitted.",
    )
