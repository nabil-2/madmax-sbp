"""Original full-grid/reduced-line diagnostics, with notebook-editable text.

Data selection and rendering follow adaptive_bead_pull.py. Numerical products
come from the verification runner; no measurement is refitted while plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from .verification_core import beam_model
from .plot_text import render_text


def plot_point_selection(
    run,
    *,
    z_index=0,
    frequency_index=0,
    show_point_ix=False,
    show_line_ix=True,
    title_template="Grid and selected lines · z={z_m:g} m · {frequency_GHz:.3f} GHz",
    x_label="x [m]",
    y_label="y [m]",
    colorbar_label="|S11|",
    grid_label="Grid points",
    lines_label="Selection lines",
    selected_label="Selected points",
    line_label_template="Line {line_index}",
):

    data = run["data"]
    s11 = data["s11"]
    frequencies = data["frequencies"]
    ixs = data["all_ixs"]
    lines = data["lines"]
    x_pos = data["x_pos"]
    y_pos = data["y_pos"]
    x_pts = data["x_pts"]
    y_pts = data["y_pts"]
    z_pos = data["z_pos"]
    z_ix = int(data["z_ixs_used"][z_index])
    f_ix = frequency_index
    fig, ax = plt.subplots(figsize=(10, 5))
    cf = ax.contourf(
        x_pos, y_pos, np.abs(s11[:, :, z_ix, f_ix]).T, levels=20, cmap="viridis"
    )
    fig.colorbar(cf, ax=ax, label=colorbar_label)

    ax.scatter(
        x_pts,
        y_pts,
        s=10,
        c="grey",
        alpha=0.6,
        linewidths=0,
        label=grid_label,
        zorder=3,
    )
    x_line = np.linspace(x_pos.min(), x_pos.max(), 400)
    for i, (a, b) in enumerate(lines):
        y_line = a * x_line + b
        mask = (y_line >= y_pos.min()) & (y_line <= y_pos.max())
        ax.plot(
            x_line[mask],
            y_line[mask],
            color="orange",
            linewidth=1,
            zorder=5,
            label=lines_label if i == 0 else None,
        )
        if show_line_ix and np.any(mask):
            x_seg = x_line[mask]
            y_seg = y_line[mask]
            mid = len(x_seg) // 2
            ax.text(
                x_seg[mid],
                y_seg[mid] - 0.025 * np.ptp(y_pos),
                render_text(line_label_template, line_index=i),
                ha="center",
                va="top",
                fontsize=8,
                color="orange",
                zorder=6,
                clip_on=True,
            )

    if show_point_ix:
        ixs_arr = np.asarray(ixs)
        sel_idx = (
            np.flatnonzero(ixs_arr) if ixs_arr.dtype == bool else ixs_arr.astype(int)
        )
        for idx in sel_idx:
            ax.text(
                x_pts[idx],
                y_pts[idx] - 2.0,
                str(idx),
                ha="center",
                va="top",
                fontsize=7,
                color="red",
                zorder=6,
                clip_on=True,
            )

    ax.scatter(
        x_pts[ixs],
        y_pts[ixs],
        s=20,
        c="red",
        alpha=0.9,
        linewidths=0,
        label=selected_label,
        zorder=4,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(
        render_text(
            title_template, z_m=z_pos[z_ix], frequency_GHz=frequencies[f_ix] * 1e-9
        )
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig, ax


def plot_sqrt_Dgamma_line(
    run,
    *,
    line_index=0,
    z_index=0,
    frequency_index=0,
    n_baseline_margin=2,
    title_template="Line {line_index} · y={slope:.3f}x + {intercept:.3f} m · z={z_m:g} m · {frequency_GHz:.3f} GHz",
    x_label="Signed line position [m]",
    reflection_y_label="|S11|",
    field_y_label="$|\\sqrt{\\Delta\\Gamma}|$ ∝ |E|",
    measured_label="Measured |S11|",
    baseline_label="Fitted baseline magnitude",
    reference_label="Reference samples",
    root_label="$|\\sqrt{\\Delta\\Gamma}|$",
):
    data = run["data"]
    params = run["params"]
    f_ix = frequency_index
    z_ix = z_index
    frequencies = data["frequencies"]
    z_pos = data["z_pos"][list(data["z_ixs_used"])]
    x_plt = data["x_plt"]
    y_plt = data["y_plt"]

    lines = data["lines"]
    idxs_sorted_lines = data["idxs_sorted_lines"]
    s_sorted_lines = data["s_sorted_lines"]
    baseline_lines = data["baseline_lines"]
    sqrt_Dgamma_lines = data["sqrt_Dgamma_lines"]
    s11_flat = data["s11_z_flat"]
    if not (0 <= line_index < len(lines)):
        raise ValueError(
            f"Line index i={line_index} out of range (0..{len(lines) - 1})."
        )
    if z_ix < 0 or z_ix >= s11_flat.shape[1]:
        raise ValueError(f"z_ix out of range (0..{s11_flat.shape[1] - 1}).")
    if f_ix < 0 or f_ix >= s11_flat.shape[2]:
        raise ValueError(f"f_ix out of range (0..{s11_flat.shape[2] - 1}).")
    if (
        idxs_sorted_lines[line_index] is None
        or s_sorted_lines[line_index] is None
        or baseline_lines[line_index] is None
        or sqrt_Dgamma_lines[line_index] is None
    ):
        raise ValueError("Precomputed data for this line is missing.")

    idxs_sorted = idxs_sorted_lines[line_index]
    s_sorted = s_sorted_lines[line_index]
    baseline = baseline_lines[line_index][:, z_ix, f_ix]
    sqrt_dg = sqrt_Dgamma_lines[line_index][:, z_ix, f_ix]
    sqrt_dg_abs = np.abs(sqrt_dg)

    m_points = idxs_sorted.size
    n = min(n_baseline_margin, m_points // 2)
    if n == 0:
        raise ValueError("Baseline margin results in empty baseline set.")
    base_idx = np.r_[np.arange(n), np.arange(m_points - n, m_points)]
    y_line = s11_flat[idxs_sorted, z_ix, f_ix]

    a, b = lines[line_index]
    u = np.array([1.0, a], dtype=float)
    u /= np.linalg.norm(u)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax0.plot(s_sorted, np.abs(y_line), "o", ms=4, alpha=0.7, label=measured_label)
    ax0.plot(
        s_sorted, np.abs(baseline), "-", lw=2, color="tab:orange", label=baseline_label
    )
    ax0.plot(
        s_sorted[base_idx],
        np.abs(y_line[base_idx]),
        "o",
        ms=6,
        color="red",
        label=reference_label,
    )
    ax0.set_ylabel(reflection_y_label)
    ax0.legend(loc="best")
    ax0.set_title(
        render_text(
            title_template,
            line_index=line_index,
            slope=a,
            intercept=b,
            z_m=z_pos[z_ix],
            frequency_GHz=frequencies[f_ix] * 1e-9,
        )
    )
    ax0.grid()

    ax1.plot(s_sorted, sqrt_dg_abs, "o-", ms=4, label=root_label)
    ax1.plot(
        s_sorted[base_idx],
        sqrt_dg_abs[base_idx],
        "o",
        ms=6,
        color="red",
        label=reference_label,
    )
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(field_y_label)
    ax1.legend(loc="best")
    ax1.grid()
    fig.tight_layout()
    return fig, (ax0, ax1)


def plot_fitted_efield(
    run,
    *,
    z_index=0,
    frequency_index=0,
    overlay_points=True,
    title_template="Fitted field · z={z_m:g} m · {frequency_GHz:.3f} GHz · cost={cost:.3e}",
    fitted_real_title="Fitted Re(E)",
    fitted_imaginary_title="Fitted Im(E)",
    data_magnitude_title="Full-grid |E|",
    data_real_title="Full-grid Re(E)",
    data_imaginary_title="Full-grid Im(E)",
    residual_title="Magnitude residual (data − fit)",
    x_label="x [m]",
    y_label="y [m]",
    magnitude_colorbar_label="|Econv| [V/m]",
    real_colorbar_label="Re(Econv) [V/m]",
    imaginary_colorbar_label="Im(Econv) [V/m]",
    residual_colorbar_label="Residual / data [%]",
    samples_label="Retained samples",
):
    data = run["data"]
    params = run["params"]
    f_ix = frequency_index
    z_ix = z_index
    frequencies = data["frequencies"]
    z_pos = data["z_pos"][list(data["z_ixs_used"])]
    x_plt = data["x_plt"]
    y_plt = data["y_plt"]

    x_reduced = data["x_reduced"]
    y_reduced = data["y_reduced"]
    e_field_reduced = data["E_reduced"]
    e_field_full = data["E_full"][:, :, list(data["z_ixs_used"]), :]
    beam_model_func = beam_model
    costs = run["costs"]
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    p = params[z_ix, f_ix]
    x0, y0, A, s = p[:4]
    kappa = float(p[4]) if p.size >= 5 else 0.0
    phi0 = float(p[5]) if p.size >= 6 else 0.0
    w = p[6:] if p.size > 6 else (1.0, 0.0, 0.0, 0.0)

    e_field_fit = beam_model_func(x_plt, y_plt, x0, y0, A, s, kappa, phi0, *w)
    e_fit_mag = np.abs(e_field_fit)
    e_fit_real = np.real(e_field_fit)
    e_fit_imag = np.imag(e_field_fit)

    e_field_data = e_field_full[:, :, z_ix, f_ix]
    e_data_mag = np.abs(e_field_data)
    e_data_real = np.real(e_field_data)
    e_data_imag = np.imag(e_field_data)

    overlay_mag = overlay_real = overlay_imag = None
    if overlay_points and e_field_reduced is not None:
        e_field_samples = np.asarray(e_field_reduced[:, z_ix, f_ix])
        overlay_mag = np.abs(e_field_samples)
        overlay_real = np.real(e_field_samples)
        overlay_imag = np.imag(e_field_samples)

    def _finite_limits(values, symmetric=False, percent=95):
        values = np.concatenate([np.asarray(v).ravel() for v in values])
        values = values[np.isfinite(values)]
        if values.size == 0:
            return (-1.0, 1.0) if symmetric else (0.0, 1.0)
        if symmetric:
            lim = np.nanpercentile(np.abs(values), percent)
            if not np.isfinite(lim) or lim == 0:
                lim = np.nanmax(np.abs(values))
            if not np.isfinite(lim) or lim == 0:
                lim = 1.0
            return -lim, lim
        lo = np.nanpercentile(values, 5)
        hi = np.nanpercentile(values, percent)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = np.nanmin(values), np.nanmax(values)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = 0.0, 1.0
        return lo, hi

    mag_vals = [e_fit_mag, e_data_mag]
    real_vals = [e_fit_real, e_data_real]
    imag_vals = [e_fit_imag, e_data_imag]
    if overlay_mag is not None:
        mag_vals.append(overlay_mag)
        real_vals.append(overlay_real)
        imag_vals.append(overlay_imag)
    mag_vmin, mag_vmax = _finite_limits(mag_vals, symmetric=False)
    real_vmin, real_vmax = _finite_limits(real_vals, symmetric=True)
    imag_vmin, imag_vmax = _finite_limits(imag_vals, symmetric=True)

    denom = np.where(e_data_mag > np.finfo(float).eps, e_data_mag, np.nan)
    rel_residual = 100.0 * (e_data_mag - e_fit_mag) / denom
    rmaxp = np.nanpercentile(np.abs(rel_residual), 95)
    if not np.isfinite(rmaxp) or rmaxp == 0:
        rmaxp = np.nanmax(np.abs(rel_residual))
    if not np.isfinite(rmaxp) or rmaxp == 0:
        rmaxp = 1.0

    fig_top, ax = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=False)
    fig_top.subplots_adjust(
        left=0.06, right=0.97, bottom=0.08, top=0.93, wspace=0.35, hspace=0
    )
    ax00, ax01, ax02 = ax[0, 0], ax[0, 1], ax[0, 2]
    ax10, ax11, ax12 = ax[1, 0], ax[1, 1], ax[1, 2]

    c0_fit = ax00.contourf(
        x_plt, y_plt, e_fit_mag, levels=30, cmap="viridis", vmin=mag_vmin, vmax=mag_vmax
    )
    c0_data = ax10.contourf(
        x_plt,
        y_plt,
        e_data_mag,
        levels=30,
        cmap="viridis",
        vmin=mag_vmin,
        vmax=mag_vmax,
    )
    cax00 = make_axes_locatable(ax00).append_axes("right", size="4%", pad=0.08)
    cax10 = make_axes_locatable(ax10).append_axes("right", size="4%", pad=0.08)
    fig_top.colorbar(c0_fit, cax=cax00, label=magnitude_colorbar_label)
    fig_top.colorbar(c0_data, cax=cax10, label=magnitude_colorbar_label)
    if overlay_mag is not None and x_reduced is not None and y_reduced is not None:
        ax00.scatter(
            x_reduced,
            y_reduced,
            c=overlay_mag,
            cmap="viridis",
            vmin=mag_vmin,
            vmax=mag_vmax,
            s=25,
            edgecolors="k",
            linewidths=0.2,
            alpha=0.9,
            label=samples_label,
        )
        ax00.legend(loc="upper right")

    c1_fit = ax01.contourf(
        x_plt,
        y_plt,
        e_fit_real,
        levels=30,
        cmap="RdBu_r",
        vmin=real_vmin,
        vmax=real_vmax,
    )
    c1_data = ax11.contourf(
        x_plt,
        y_plt,
        e_data_real,
        levels=30,
        cmap="RdBu_r",
        vmin=real_vmin,
        vmax=real_vmax,
    )
    cax01 = make_axes_locatable(ax01).append_axes("right", size="4%", pad=0.08)
    cax11 = make_axes_locatable(ax11).append_axes("right", size="4%", pad=0.08)
    fig_top.colorbar(c1_fit, cax=cax01, label=real_colorbar_label)
    fig_top.colorbar(c1_data, cax=cax11, label=real_colorbar_label)
    if overlay_real is not None and x_reduced is not None and y_reduced is not None:
        ax01.scatter(
            x_reduced,
            y_reduced,
            c=overlay_real,
            cmap="RdBu_r",
            vmin=real_vmin,
            vmax=real_vmax,
            s=25,
            edgecolors="k",
            linewidths=0.2,
            alpha=0.9,
            label=samples_label,
        )
        ax01.legend(loc="upper right")

    c2_fit = ax02.contourf(
        x_plt,
        y_plt,
        e_fit_imag,
        levels=30,
        cmap="RdBu_r",
        vmin=imag_vmin,
        vmax=imag_vmax,
    )
    c2_data = ax12.contourf(
        x_plt,
        y_plt,
        e_data_imag,
        levels=30,
        cmap="RdBu_r",
        vmin=imag_vmin,
        vmax=imag_vmax,
    )
    cax02 = make_axes_locatable(ax02).append_axes("right", size="4%", pad=0.08)
    cax12 = make_axes_locatable(ax12).append_axes("right", size="4%", pad=0.08)
    fig_top.colorbar(c2_fit, cax=cax02, label=imaginary_colorbar_label)
    fig_top.colorbar(c2_data, cax=cax12, label=imaginary_colorbar_label)
    if overlay_imag is not None and x_reduced is not None and y_reduced is not None:
        ax02.scatter(
            x_reduced,
            y_reduced,
            c=overlay_imag,
            cmap="RdBu_r",
            vmin=imag_vmin,
            vmax=imag_vmax,
            s=25,
            edgecolors="k",
            linewidths=0.2,
            alpha=0.9,
            label=samples_label,
        )
        ax02.legend(loc="upper right")

    title_fit = render_text(
        title_template,
        z_m=z_pos[z_ix],
        frequency_GHz=frequencies[f_ix] * 1e-9,
        cost=float(costs[z_ix, f_ix]),
    )
    ax00.set_title(title_fit)
    ax01.set_title(fitted_real_title)
    ax02.set_title(fitted_imaginary_title)
    ax10.set_title(data_magnitude_title)
    ax11.set_title(data_real_title)
    ax12.set_title(data_imaginary_title)
    for axi in (ax00, ax01, ax02, ax10, ax11, ax12):
        axi.set_xlabel(x_label)
        axi.set_ylabel(y_label)
        axi.set_aspect("equal", adjustable="box")
        axi.grid(True, alpha=0.2)

    fig_res = plt.figure(figsize=(10, 5), constrained_layout=True)
    axr = fig_res.add_subplot(111)
    cfr = axr.contourf(
        x_plt, y_plt, rel_residual, levels=30, cmap="coolwarm", vmin=-rmaxp, vmax=rmaxp
    )
    fig_res.colorbar(cfr, ax=axr, label=residual_colorbar_label)
    axr.set_title(residual_title)
    axr.set_xlabel(x_label)
    axr.set_ylabel(y_label)
    axr.set_aspect("equal", adjustable="box")
    axr.grid(True, alpha=0.2)
    return fig_top, (ax00, ax01, ax02, ax10, ax11, ax12), fig_res, axr


def plot_sqrt_dgamma_before_after_correction(
    run,
    *,
    z_index=0,
    frequency_index=0,
    colors=("tab:orange", "tab:blue"),
    labels=("Before alignment", "After alignment"),
    title_template="Full-reference alignment · z={z_m:g} m · {frequency_GHz:.3f} GHz",
    component_labels=(
        "$|\\sqrt{\\Delta\\Gamma}|$",
        "$\\mathrm{Re}(\\sqrt{\\Delta\\Gamma})$",
        "$\\mathrm{Im}(\\sqrt{\\Delta\\Gamma})$",
    ),
    x_label_template="Full {component}",
    y_label_template="Reduced {component}",
):

    d = run["data"]
    sqrt_dgamma_full = d["sqrt_dgamma_full"]
    sqrt_dgamma_reduced_before = d["sqrt_Dgamma_reduced"]
    sqrt_dgamma_reduced_after = d["sqrt_Dgamma_reduced_corr"]
    all_ixs = d["all_ixs"]
    reduced_z_ix = z_index
    full_z_ix = int(d["z_ixs_used"][z_index])
    f_ix = frequency_index
    if full_z_ix is None:
        full_z_ix = reduced_z_ix
    full_line = np.asarray(sqrt_dgamma_full).reshape(
        -1, sqrt_dgamma_full.shape[2], sqrt_dgamma_full.shape[3]
    )[all_ixs, full_z_ix, f_ix]
    red_before = np.asarray(sqrt_dgamma_reduced_before)[:, reduced_z_ix, f_ix]
    red_after = np.asarray(sqrt_dgamma_reduced_after)[:, reduced_z_ix, f_ix]
    valid = np.isfinite(full_line) & np.isfinite(red_before) & np.isfinite(red_after)
    full_line = full_line[valid]
    red_before = red_before[valid]
    red_after = red_after[valid]
    if full_line.size == 0:
        raise ValueError("No valid overlapping line-point samples for plotting.")

    def _lims(xv, yv1, yv2):
        lo = np.nanmin(np.r_[xv, yv1, yv2])
        hi = np.nanmax(np.r_[xv, yv1, yv2])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = -1.0, 1.0
        pad = 0.05 * (hi - lo)
        return lo - pad, hi + pad

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    comps = [
        (np.abs(full_line), np.abs(red_before), np.abs(red_after), component_labels[0]),
        (
            np.real(full_line),
            np.real(red_before),
            np.real(red_after),
            component_labels[1],
        ),
        (
            np.imag(full_line),
            np.imag(red_before),
            np.imag(red_after),
            component_labels[2],
        ),
    ]
    for ax, (xv, yb, ya, name) in zip(axes, comps):
        ax.scatter(
            xv,
            yb,
            s=22,
            label=labels[0],
            marker="o",
            edgecolors=colors[0],
            linewidths=0.8,
            facecolors="none",
        )
        ax.scatter(
            xv,
            ya,
            s=22,
            label=labels[1],
            marker="o",
            edgecolors=colors[1],
            linewidths=0.8,
            facecolors="none",
        )
        lo, hi = _lims(xv, yb, ya)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(render_text(x_label_template, component=name))
        ax.set_ylabel(render_text(y_label_template, component=name))
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    fig.suptitle(
        render_text(
            title_template,
            z_m=d["z_pos"][full_z_ix],
            frequency_GHz=d["frequencies"][f_ix] * 1e-9,
        ),
        y=1.03,
    )
    return fig, axes


def plot_ET_complex_plane(
    result,
    *,
    frequency_index=0,
    title_template="Transverse integrals · {frequency_GHz:.3f} GHz",
    x_label="$\\mathrm{Re}(E_T)$ [V m]",
    y_label="$\\mathrm{Im}(E_T)$ [V m]",
    full_label="Full grid at retained z",
    reduced_label="Limited lines",
    annotation_template="z={z_mm:.2f} mm",
):

    ETs_full = result["ETs_full"]
    ETs_reduced = result["ETs_reduced"]
    z_full = result["z_full"]
    z_reduced = result["z_reduced"]
    f_ix = frequency_index
    z_decimals = 2
    ETs_full = np.asarray(ETs_full)
    ETs_reduced = np.asarray(ETs_reduced)
    z_full = np.asarray(z_full).ravel()
    z_reduced = np.asarray(z_reduced).ravel()
    ix_z_full_matched = [int(np.argmin(np.abs(z_full - zr))) for zr in z_reduced]
    full_vals = ETs_full[ix_z_full_matched, f_ix]
    reduced_vals = ETs_reduced[:, f_ix]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        np.real(full_vals),
        np.imag(full_vals),
        label=full_label,
        marker="o",
        s=100,
        alpha=0.7,
    )
    ax.scatter(
        np.real(reduced_vals),
        np.imag(reduced_vals),
        label=reduced_label,
        marker="x",
        s=100,
        alpha=0.8,
    )
    for i, zr in enumerate(z_reduced):
        z_mm = zr * 1e3
        z_label = render_text(annotation_template, z_mm=z_mm)
        ax.annotate(
            z_label,
            (np.real(full_vals[i]), np.imag(full_vals[i])),
            xytext=(5, 5),
            textcoords="offset points",
            color="tab:blue",
        )
        ax.annotate(
            z_label,
            (np.real(reduced_vals[i]), np.imag(reduced_vals[i])),
            xytext=(5, -15),
            textcoords="offset points",
            color="tab:orange",
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(
        render_text(title_template, frequency_GHz=result["frequencies"][f_ix] * 1e-9)
    )
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_abs_ET_over_z(
    result,
    *,
    frequency_index=0,
    atol=1e-09,
    title_template="Matched transverse integrals · {frequency_GHz:.3f} GHz",
    x_label="Acquisition z [mm]",
    y_label="$|E_T|$ [V m]",
    full_label="Full grid",
    reduced_label="Limited lines",
    annotation_template="{z_mm:.2f} mm",
):

    ETs_full = result["ETs_full"]
    ETs_reduced = result["ETs_reduced"]
    z_full = result["z_full"]
    z_reduced = result["z_reduced"]
    f_ix = frequency_index
    z_decimals = 2
    ETs_full = np.asarray(ETs_full)
    ETs_reduced = np.asarray(ETs_reduced)
    z_full = np.asarray(z_full).ravel()
    z_reduced = np.asarray(z_reduced).ravel()

    pairs = []
    for i_red, zr in enumerate(z_reduced):
        is_match = np.isclose(z_full, zr, atol=atol, rtol=0.0)
        if np.any(is_match):
            i_full = int(np.flatnonzero(is_match)[0])
            pairs.append((i_full, i_red))
    if len(pairs) == 0:
        raise ValueError("No overlapping z values found between z_full and z_reduced.")

    full_idx = np.array([p[0] for p in pairs], dtype=int)
    red_idx = np.array([p[1] for p in pairs], dtype=int)
    z_plot = z_reduced[red_idx]
    full_abs = np.abs(ETs_full[full_idx, f_ix])
    red_abs = np.abs(ETs_reduced[red_idx, f_ix])
    order = np.argsort(z_plot)
    z_plot = z_plot[order]
    full_abs = full_abs[order]
    red_abs = red_abs[order]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(z_plot * 1e3, full_abs, "o-", label=full_label, alpha=0.8)
    ax.plot(z_plot * 1e3, red_abs, "x--", label=reduced_label, alpha=0.8)
    if z_decimals is not None:
        for z_mm, y in zip(z_plot * 1e3, red_abs):
            ax.annotate(
                render_text(annotation_template, z_mm=z_mm),
                (z_mm, y),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="tab:orange",
            )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(
        render_text(title_template, frequency_GHz=result["frequencies"][f_ix] * 1e-9)
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_abs_ET_grid_over_frequency(
    result,
    *,
    atol=1e-09,
    max_et_for_scaling=500.0,
    colors=("tab:blue", "tab:red"),
    colors_real=("tab:green", "tab:orange"),
    colors_imag=("tab:purple", "tab:brown"),
    x_label="Frequency [GHz]",
    magnitude_y_label="$|E_T|$ [V m]",
    real_y_label="$\\mathrm{Re}(E_T)$ [V m]",
    imaginary_y_label="$\\mathrm{Im}(E_T)$ [V m]",
    magnitude_title="Transverse-integral magnitude",
    real_title="Real transverse integrals",
    imaginary_title="Imaginary transverse integrals",
    panel_title_template="Acquisition z={z_mm:.2f} mm",
    full_label="Full grid",
    reduced_label="Limited lines",
):

    ETs_full = result["ETs_full"]
    ETs_reduced = result["ETs_reduced"]
    z_full = result["z_full"]
    z_reduced = result["z_reduced"]
    frequencies = result["frequencies"]
    reduced_linewidth = full_linewidth = 1.0
    reduced_alpha = 0.75
    full_alpha = 0.65
    reduced_linestyle = "-"
    full_linestyle = (0, (4, 2))
    return_all = True
    ETs_full = np.asarray(ETs_full)
    ETs_reduced = np.asarray(ETs_reduced)
    z_full = np.asarray(z_full).ravel()
    z_reduced = np.asarray(z_reduced).ravel()
    frequencies = np.asarray(frequencies, dtype=float).ravel()
    n = z_reduced.size
    if n == 0:
        raise ValueError("z_reduced is empty; nothing to plot.")

    ix_z_full_matched = np.empty(n, dtype=int)
    for i, zr in enumerate(z_reduced):
        matches = np.flatnonzero(np.isclose(z_full, zr, atol=atol, rtol=0.0))
        ix_z_full_matched[i] = (
            int(matches[0]) if matches.size > 0 else int(np.argmin(np.abs(z_full - zr)))
        )

    freq_ghz = frequencies * 1e-9
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))

    def _component_ylim(y_reduced, y_full, symmetric):
        vals = np.r_[
            y_reduced[
                np.isfinite(y_reduced) & (np.abs(y_reduced) <= max_et_for_scaling)
            ],
            y_full[np.isfinite(y_full) & (np.abs(y_full) <= max_et_for_scaling)],
        ]
        if vals.size == 0:
            lim = max_et_for_scaling
        elif symmetric:
            lim = float(np.nanmax(np.abs(vals)))
            lim = max(lim, np.finfo(float).eps)
        else:
            lim = float(np.nanmax(vals))
            lim = max(lim, np.finfo(float).eps)
        if symmetric:
            return -1.05 * lim, 1.05 * lim
        return 0.0, 1.05 * lim

    def _plot_grid(component_fn, ylabel, title, palette, symmetric=False):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(12, 3.6 * n_rows),
            sharex=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).ravel()
        for i in range(n):
            ax = axes[i]
            y_reduced = component_fn(ETs_reduced[i, :])
            y_full = component_fn(ETs_full[ix_z_full_matched[i], :])
            reduced_line = ax.plot(
                freq_ghz,
                y_reduced,
                color=palette[0],
                lw=reduced_linewidth,
                alpha=reduced_alpha,
                linestyle=reduced_linestyle,
                zorder=2,
                label=reduced_label,
            )[0]
            full_line = ax.plot(
                freq_ghz,
                y_full,
                color=palette[1],
                lw=full_linewidth,
                alpha=full_alpha,
                linestyle=full_linestyle,
                zorder=3,
                label=full_label,
            )[0]
            ax.set_ylim(*_component_ylim(y_reduced, y_full, symmetric=symmetric))
            ax.set_title(render_text(panel_title_template, z_mm=z_reduced[i] * 1e3))
            ax.set_xlabel(x_label)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(handles=[reduced_line, full_line], loc="best")
        for j in range(n, axes.size):
            axes[j].set_visible(False)
        fig.suptitle(title, y=1.02)
        return fig, axes[:n]

    fig_abs, axes_abs = _plot_grid(
        lambda x: np.abs(x),
        magnitude_y_label,
        magnitude_title,
        colors,
        symmetric=False,
    )
    fig_real, axes_real = _plot_grid(
        lambda x: np.real(x),
        real_y_label,
        real_title,
        colors_real,
        symmetric=True,
    )
    fig_imag, axes_imag = _plot_grid(
        lambda x: np.imag(x),
        imaginary_y_label,
        imaginary_title,
        colors_imag,
        symmetric=True,
    )
    if return_all:
        return (
            (fig_abs, axes_abs),
            (fig_real, axes_real),
            (fig_imag, axes_imag),
            ix_z_full_matched,
        )
    return fig_abs, axes_abs, ix_z_full_matched


def plot_beam_model_components(
    run,
    *,
    z_index=0,
    frequency_index=0,
    magnitude_title="Fitted Gaussian magnitude",
    real_title="Fitted Gaussian real part",
    imaginary_title="Fitted Gaussian imaginary part",
    x_label="x [m]",
    y_label="y [m]",
    magnitude_colorbar_label="|Econv| [V/m]",
    real_colorbar_label="Re(Econv) [V/m]",
    imaginary_colorbar_label="Im(Econv) [V/m]",
):
    data = run["data"]
    params = run["params"]
    f_ix = frequency_index
    z_ix = z_index
    frequencies = data["frequencies"]
    z_pos = data["z_pos"][list(data["z_ixs_used"])]
    x_plt = data["x_plt"]
    y_plt = data["y_plt"]
    init_params = np.asarray(params[z_ix, f_ix, :], dtype=float)
    x0, y0, A, s, kappa, phi0 = init_params[:6]
    field = beam_model(x_plt, y_plt, x0, y0, A, s, kappa, phi0, 1.0, 0.0, 0.0, 0.0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].contourf(x_plt, y_plt, np.abs(field), levels=30, cmap="viridis")
    axes[0].set_title(magnitude_title)
    fig.colorbar(im0, ax=axes[0], label=magnitude_colorbar_label)
    im1 = axes[1].contourf(x_plt, y_plt, np.real(field), levels=30, cmap="RdBu_r")
    axes[1].set_title(real_title)
    fig.colorbar(im1, ax=axes[1], label=real_colorbar_label)
    im2 = axes[2].contourf(x_plt, y_plt, np.imag(field), levels=30, cmap="RdBu_r")
    axes[2].set_title(imaginary_title)
    fig.colorbar(im2, ax=axes[2], label=imaginary_colorbar_label)
    for ax in axes:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig, axes


def create_hg_sanity_figures(
    *,
    magnitude_title="Hermite–Gaussian mode check · magnitude",
    real_title="Hermite–Gaussian mode check · real part",
    imaginary_title="Hermite–Gaussian mode check · imaginary part",
    x_label="x [m]",
    y_label="y [m]",
    mode_labels=("HG00", "HG10", "HG01", "HG11"),
):
    x_lin = np.linspace(-0.15, 0.15, 200)
    y_lin = np.linspace(-0.15, 0.15, 200)
    X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")
    modes = [
        (mode_labels[0], (1.0, 0.0, 0.0, 0.0)),
        (mode_labels[1], (0.0, 0.0, 1.0, 0.0)),
        (mode_labels[2], (0.0, 1.0, 0.0, 0.0)),
        (mode_labels[3], (0.0, 0.0, 0.0, 1.0)),
    ]
    fields = [
        (name, beam_model(X, Y, 0.0, 0.0, 1.0, 0.05, 400.0, 0.0, *w))
        for name, w in modes
    ]
    figures = []
    for component, title, cmap in [
        (lambda z: np.abs(z), magnitude_title, "viridis"),
        (lambda z: np.real(z), real_title, "coolwarm"),
        (lambda z: np.imag(z), imaginary_title, "coolwarm"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        for ax, (name, field) in zip(axes.ravel(), fields):
            value = component(field)
            if cmap == "coolwarm":
                vmax = np.nanmax(np.abs(value))
                vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
                im = ax.contourf(
                    X, Y, value, levels=40, cmap=cmap, vmin=-vmax, vmax=vmax
                )
            else:
                im = ax.contourf(X, Y, value, levels=40, cmap=cmap)
            ax.set_title(name)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_aspect("equal", adjustable="box")
            fig.colorbar(im, ax=ax)
        fig.suptitle(title)
        figures.append(fig)
    return figures
