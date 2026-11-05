"""Presentation figures for the saved Gaussian frequency-selection diagnostic.

Reconstruct the existing fitted profiles; no measurement or fit is modified.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .analysis_3d_plots import entry

THRESHOLD = 0.20
ACCEPTED_COLOR = "#18804A"
REJECTED_COLOR = "#B33B40"
REAL_COLOR = "#0072B2"
IMAGINARY_COLOR = "#D55E00"
THREAD_COLORS = ("#0072B2", "#D55E00", "#009E73")


def _profile_prediction(position_m, parameters, coefficients):
    u = np.asarray(position_m) / 0.1
    center, width, curvature = np.asarray(parameters).T
    offset, slope, amplitude = np.asarray(coefficients).T
    gaussian = np.exp(
        (-2 / width**2 + 1j * curvature)[..., None]
        * (u - center[..., None])**2
    )
    return offset[..., None] + slope[..., None]*u + amplitude[..., None]*gaussian


def selection_diagnostics(profiles):
    """Recover RMS/|A| and bound hits from the saved raw-data Gaussian fits."""
    nfrequency = len(profiles["frequency_hz"])
    q = np.empty((nfrequency, 4))
    rms = np.empty_like(q)
    at_bound = np.empty_like(q, dtype=bool)
    for thread in range(4):
        take = (profiles["thread_index"] == thread) & ~profiles["transient_trace_mask"]
        p = profiles["gaussian_parameters"][thread]
        c = profiles["gaussian_coefficients"][thread]
        prediction = _profile_prediction(profiles["signed_line_m"][take], p, c)
        rms[:, thread] = np.sqrt(np.mean(
            abs(prediction - profiles["observed_S22"][take].T)**2, axis=1
        ))
        q[:, thread] = rms[:, thread] / np.maximum(abs(c[:, 2]), 1e-10)
        at_bound[:, thread] = np.any(
            (p - [-1.1, 0.2, -8] < 1e-3)
            | (np.array([1.1, 2.2, 8]) - p < 1e-3), axis=1
        )
    accepted = np.asarray(profiles["profile_quality"], dtype=bool)
    selected = np.asarray(profiles["shared_fit_mask"], dtype=bool)
    np.testing.assert_array_equal(selected, accepted[:, 1:].all(axis=1))
    if np.any(accepted & ((q >= THRESHOLD) | at_bound)):
        raise ValueError("Saved acceptance disagrees with the reconstructed quality diagnostic")
    # Optimizer success was not stored separately. A rejected profile below
    # threshold and away from bounds necessarily failed an additional fit check.
    other_failure = ~accepted & (q < THRESHOLD) & ~at_bound
    return dict(q=q, rms=rms, at_bound=at_bound, other_failure=other_failure,
                accepted=accepted, selected=selected)


def _example(profiles, metrics, thread, frequency_GHz, accepted):
    frequency = profiles["frequency_hz"]
    k = int(np.argmin(abs(frequency - frequency_GHz*1e9)))
    if abs(frequency[k] - frequency_GHz*1e9) > 1:
        raise ValueError("The example frequency must be a measured bin")
    if bool(metrics["selected"][k]) != accepted:
        raise ValueError("The example frequency has the wrong selection status")
    if bool(metrics["accepted"][k, thread]) != accepted:
        raise ValueError("Choose a thread that illustrates the frequency's acceptance/rejection")
    take = np.flatnonzero(
        (profiles["thread_index"] == thread) & ~profiles["transient_trace_mask"]
    )
    take = take[np.argsort(profiles["signed_line_m"][take])]
    x = profiles["signed_line_m"][take]
    dense = np.linspace(x.min(), x.max(), 501)
    curve = _profile_prediction(dense, profiles["gaussian_parameters"][thread, k],
                                profiles["gaussian_coefficients"][thread, k])
    return dict(
        frequency_GHz=frequency[k]/1e9, thread=thread, accepted=accepted,
        position_m=x, observed=profiles["observed_S22"][take, k],
        dense_position_m=dense, prediction=curve, q=metrics["q"][k, thread],
        at_bound=metrics["at_bound"][k, thread],
    )


def _draw_example(axis, sample, *, compact=False):
    for component, color, marker in (("real", REAL_COLOR, "o"),
                                      ("imag", IMAGINARY_COLOR, "s")):
        axis.plot(sample["position_m"]*1000, getattr(sample["observed"], component),
                  ls="none", marker=marker, ms=2.5 if compact else 3.6,
                  color=color, alpha=.78)
        axis.plot(sample["dense_position_m"]*1000, getattr(sample["prediction"], component),
                  color=color, lw=1.5 if compact else 2.0)
    status = "Accepted" if sample["accepted"] else "Rejected"
    color = ACCEPTED_COLOR if sample["accepted"] else REJECTED_COLOR
    axis.set_title(
        f"{status}: {sample['frequency_GHz']:.3f} GHz · T{sample['thread']}",
        color=color, fontsize=9.5 if compact else 12,
    )
    relation = "<" if sample["q"] < THRESHOLD else r"\geq"
    axis.text(.035, .045, rf"$q={sample['q']:.3f}\ {relation}\ 0.20$",
              transform=axis.transAxes, fontsize=8 if compact else 10,
              color=color, bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=2))
    axis.set_xlabel("Bead position [mm]", fontsize=8 if compact else 11)
    axis.set_ylabel(r"Raw complex reflection $\Gamma$", fontsize=8 if compact else 11)
    axis.tick_params(labelsize=7.5 if compact else 9)
    axis.margins(x=.02, y=.16)
    axis.grid(alpha=.15)


def _example_handles():
    return [
        Line2D([], [], color=REAL_COLOR, marker="o", ls="none", ms=4,
               label=r"Measured Re $\Gamma$"),
        Line2D([], [], color=REAL_COLOR, lw=2, label=r"Gaussian model: Re $\Gamma$"),
        Line2D([], [], color=IMAGINARY_COLOR, marker="s", ls="none", ms=4,
               label=r"Measured Im $\Gamma$"),
        Line2D([], [], color=IMAGINARY_COLOR, lw=2, label=r"Gaussian model: Im $\Gamma$"),
    ]


def gaussian_fit_examples(
    profiles, metrics=None, *, thread=3,
    accepted_frequency_GHz=20.825, rejected_frequency_GHz=21.010,
    title="Gaussian frequency-selection diagnostic: accepted and rejected examples",
):
    metrics = selection_diagnostics(profiles) if metrics is None else metrics
    accepted = _example(profiles, metrics, thread, accepted_frequency_GHz, True)
    rejected = _example(profiles, metrics, thread, rejected_frequency_GHz, False)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    figure.subplots_adjust(top=.72, bottom=.22, left=.075, right=.98, wspace=.25)
    for axis, sample in zip(axes, (accepted, rejected)):
        _draw_example(axis, sample)
    figure.suptitle(title, y=.98, fontsize=13)
    figure.legend(handles=_example_handles(), ncol=2, loc="upper center",
                  bbox_to_anchor=(.5, .92), frameon=False, fontsize=9)
    excluded = np.count_nonzero(
        (profiles["thread_index"] == thread) & profiles["transient_trace_mask"]
    )
    measurement_note = (f"All T{thread} measurements are shown." if excluded == 0
                        else f"{excluded} flagged T{thread} measurements are excluded.")
    figure.text(.5, .075,
        "Raw measured values and the saved complex Gaussian-profile model, including its offset and slope.\n"
        r"$q=\mathrm{RMS}(\Gamma-\widehat\Gamma)/|A|$; the acceptance threshold is 0.20. "
        + measurement_note, ha="center", va="center", fontsize=9)
    return entry(
        figure, "gaussian_accepted_rejected", "Accepted and rejected Gaussian-profile fits",
        "The same thread is shown at an accepted and a rejected frequency. Real and imaginary raw reflection samples are plotted without baseline subtraction or shape normalization. Curves reconstruct the saved preliminary Gaussian model, including its own affine nuisance terms. The quality metric uses complex residuals. These diagnostic nuisance terms do not determine the physical endpoint background used by the 3D fit.",
    )


def frequency_selection_overview(
    profiles, metrics=None, *, thread=3,
    accepted_frequency_GHz=20.825, rejected_frequency_GHz=21.010,
    title="Selecting frequencies for the 3D geometry fit",
):
    metrics = selection_diagnostics(profiles) if metrics is None else metrics
    samples = (
        _example(profiles, metrics, thread, accepted_frequency_GHz, True),
        _example(profiles, metrics, thread, rejected_frequency_GHz, False),
    )
    f = np.asarray(profiles["frequency_hz"])/1e9
    figure, quality_axis = plt.subplots(figsize=(10.5, 6.4))
    figure.subplots_adjust(left=.09, right=.98, bottom=.21, top=.80)
    for j, color in zip((1, 2, 3), THREAD_COLORS):
        quality_axis.plot(f, metrics["q"][:, j], color=color, lw=1.15, label=f"T{j}")
    quality_axis.axhline(THRESHOLD, color="#333333", ls="--", lw=1.4,
                         label="Acceptance threshold: 0.20")
    quality_axis.set(yscale="log", ylabel=r"Gaussian fit quality $q=\mathrm{RMS}/|A|$",
                     xlabel="Frequency [GHz]", xlim=(f[0], f[-1]))
    quality_axis.grid(alpha=.16, which="both")
    quality_axis.legend(fontsize=8.5, ncol=2, loc="upper left")
    for frequency in f[metrics["selected"]]:
        quality_axis.axvspan(frequency-.0025, frequency+.0025,
                             color=ACCEPTED_COLOR, alpha=.055, lw=0, zorder=0)
    for sample, color in zip(samples, (ACCEPTED_COLOR, REJECTED_COLOR)):
        quality_axis.axvline(sample["frequency_GHz"], color=color, ls=":", lw=1.2)
    figure.suptitle(title, y=.975, fontsize=15)
    figure.text(.09, .875,
        f"{int(metrics['selected'].sum())} of {len(f)} measured bins selected\n"
        "all T1–T3 profiles must converge, stay away from parameter bounds and satisfy q < 0.20",
        fontsize=10.5)
    figure.legend(handles=[
        Patch(facecolor=ACCEPTED_COLOR, alpha=.2, label="Selected frequency bins"),
    ], loc="lower left", bbox_to_anchor=(.08, .065), frameon=False, fontsize=9)
    figure.text(.09, .035,
        "T0 does not determine frequency acceptance. All four threads, T0–T3, enter the geometry fit\n"
        "at the selected frequencies.",
        fontsize=9.5)
    return entry(
        figure, "geometry_frequency_selection", "Geometry-fit frequency selection",
        "The curves reconstruct the normalized complex residuals of the saved Gaussian profile fits. The stored acceptance mask also includes convergence and parameter-bound checks. A bin enters the geometry fit only if T1, T2 and T3 all pass. T0 still enters the physical geometry objective. Accepted/rejected raw-reflection examples appear in the separate figure below.",
    )
