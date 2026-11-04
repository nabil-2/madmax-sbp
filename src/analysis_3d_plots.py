"""The original full-tilt report figures, displayed directly in notebooks.

Numerical results are read-only; plotting performs no fit or rescaling.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from .receive_field import NOM_DISKS_MM
from .analysis_3d import NOM_Z_MM
from .plot_text import render_text

TITLE = "Disk tilts + sloped threads"
CASE = "disk_tilts_threads"
FIT_THREADS = (0, 1, 2, 3)
COLOR = "#AA3377"
GREY = "#687782"
THREAD_COLORS = ["#CC6677", "#4477AA", "#228833", "#AA4499"]
NOMINAL_DISKS = FIT_DISKS_MM = NOM_DISKS_MM
NOMINAL_THREADS = FIT_THREADS_MM = NOM_Z_MM
FREQUENCIES = (20.795, 20.825)
RADIUS_MM = 150.0
SCOPE = "all four complex S22 chords, T0–T3"
SCOPE_DESCRIPTION = "All four measured chords enter the common-scale fit. The horizontal sections are illustrative; the likelihood uses each point on its actual sloping path."
UNCERTAINTY = "Shading is the conditional pointwise 16–84% residual-bootstrap spread, with geometry and incident beam fixed; it excludes geometry, calibration, material, beam and model uncertainties."


def load(path):
    """Load numeric NPZ arrays without pickle."""
    with np.load(path, allow_pickle=False) as archive:
        result = dict(archive)
    if "observed_S22" in result and (
        "reflection_preprocessing" not in result
        or result["reflection_preprocessing"].item() != "none"
    ):
        raise ValueError("Saved 3D results use the removed template correction; regenerate results from measured S22.")
    return result


def disk_shifts_mm(faces):
    return np.asarray(faces) - NOMINAL_DISKS


def entry(figure, stem, title, caption):
    return dict(figure=figure, stem=stem, title=title, caption=caption)


def present(entries, output_dir=None, *, captions=False):
    """Display the original figure(s), optionally save PNG/PDF and captions."""
    from IPython.display import display, Markdown

    if isinstance(entries, dict):
        entries = [entries]
    for item in entries:
        fig = item["figure"]
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            for extension in ("png", "pdf"):
                fig.savefig(
                    out / (item["stem"] + "." + extension), dpi=160, bbox_inches="tight"
                )
        display(fig)
        if captions:
            display(Markdown(item["caption"]))
        plt.close(fig)


def at_frequency(a, ghz):
    hits = np.flatnonzero(abs(a["frequency_hz"] - ghz * 1e9) < 1.0)
    if len(hits) != 1:
        raise ValueError(f"Expected one saved frequency at {ghz:.3f} GHz")
    return int(hits[0])


def kappa(frequency_hz):
    """Same 1-W, radius=1.465-mm, epsilon=9.23 calibration as fit_endpoints."""
    alpha = 4 * np.pi * 0.001465**3 * (9.23 - 1) / (9.23 + 2)
    return 1j * 2 * np.pi * np.asarray(frequency_hz) * 8.8541878128e-12 * alpha / 4


def boost_figures(
    a,
    bands,
    *,
    title_template="3D · {case_title_lower}{zoom_suffix}",
    x_label="Frequency [GHz]",
    y_label="Boost factor $\\beta^2$",
):
    entries = []
    f, beta = a["frequency_hz"] / 1e9, a["beta2_conditional"]
    for stem, limits, suffix in [
        ("boost", (20.0, 22.0), ""),
        ("boost_zoom", (20.7, 21.05), " · resonance"),
    ]:
        fig, ax = plt.subplots(figsize=(6.5, 3.7), layout="constrained")
        ax.fill_between(
            f,
            bands["quantiles"][:, 1],
            bands["quantiles"][:, 3],
            color=COLOR,
            alpha=0.23,
            lw=0,
        )
        ax.plot(f, beta, color=COLOR, lw=1.7)
        visible = (f >= limits[0]) & (f <= limits[1])
        ymax = 1.08 * np.max(np.r_[beta[visible], bands["quantiles"][visible, 3]])
        ax.set(
            xlim=limits,
            ylim=(0, ymax),
            xlabel=x_label,
            ylabel=y_label,
            title=render_text(
                title_template,
                case_title=TITLE,
                case_title_lower=TITLE.lower(),
                zoom_suffix=suffix,
            ),
        )
        ax.grid(alpha=0.15)
        entries.append(
            entry(
                fig,
                stem,
                "Conditional boost" + suffix,
                UNCERTAINTY
                + f" One common complex scale uses {SCOPE}. "
                + f"{int(np.sum(a['eligible']))} bins pass the measurement profile-quality selection; other frequencies are diagnostics. "
                + ("The saved geometry is held fixed while scale and backgrounds are refitted to measured S22."
                   if "geometry_refitted" in a and not bool(a["geometry_refitted"]) else ""),
            )
        )
    return entries


def profile_figures(
    a,
    *,
    title_template="{case_title} · {frequency_GHz:.3f} GHz",
    real_component_label="Real ΔS22",
    imaginary_component_label="Imaginary ΔS22",
    x_label="Signed transverse line position [mm]",
    thread_label_template="T{thread_index}{held_out}\nz: {z_start_mm:.2f}→{z_end_mm:.2f} mm",
    measured_label="Measured",
    model_label="Fitted model",
    held_out_label=" · held out",
):
    profiles, limits = {}, np.zeros((4, 2))
    for f in FREQUENCIES:
        k = at_frequency(a, f)
        y = a["observed_S22"][k] - a["fitted_background"][k]
        p = a["predicted_S22"][k] - a["fitted_background"][k]
        profiles[f] = y, p
        for t in range(4):
            take = a["thread_index"] == t
            for c, fn in enumerate((np.real, np.imag)):
                limits[t, c] = max(
                    limits[t, c], np.max(abs(fn(y[take]))), np.max(abs(fn(p[take])))
                )
    entries = []
    for f, (y, p) in profiles.items():
        fig, axs = plt.subplots(
            4, 2, figsize=(6.5, 8.7), layout="constrained", sharex=True
        )
        for t in range(4):
            take = a["thread_index"] == t
            x = a["signed_line_m"][take] * 1000
            order = np.argsort(x)
            for c, fn in enumerate((np.real, np.imag)):
                ax = axs[t, c]
                ax.plot(
                    x[order],
                    fn(y[take])[order],
                    ".",
                    color="#222222",
                    ms=3.3,
                    label=measured_label,
                )
                ax.plot(
                    x[order],
                    fn(p[take])[order],
                    color=COLOR,
                    lw=1.45,
                    label=model_label,
                )
                ax.axhline(0, color="#888888", lw=0.5)
                ax.set_ylim(
                    -1.1 * max(limits[t, c], 1e-8), 1.1 * max(limits[t, c], 1e-8)
                )
                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.grid(alpha=0.14)
                if t == 0:
                    ax.set_title((real_component_label, imaginary_component_label)[c])
                if t == 3:
                    ax.set_xlabel(x_label)
            za, zb = a["thread_endpoint_z_mm"][t]
            axs[t, 0].set_ylabel(
                render_text(
                    thread_label_template,
                    thread_index=t,
                    held_out=held_out_label if t not in FIT_THREADS else "",
                    z_start_mm=za,
                    z_end_mm=zb,
                    frequency_GHz=f,
                )
            )
        axs[0, 0].legend(fontsize=8, loc="best")
        fig.suptitle(
            render_text(title_template, case_title=TITLE, frequency_GHz=f), fontsize=11
        )
        entries.append(
            entry(
                fig,
                "raw_profiles_" + f"{f:.3f}".replace(".", "p"),
                f"Complex profiles · {f:.3f} GHz",
                "Measured S22 and the saved model after subtracting the same fitted affine background, in dimensionless complex reflection units. "
                f"{SCOPE_DESCRIPTION} Each sample is evaluated at its saved z; labels give A→B endpoint heights at minimum/maximum retained signed scan coordinate. "
                "Vertical ranges differ by component and thread but are matched between the two displayed frequencies.",
            )
        )
    return entries


def effective_field_figures(
    a,
    *,
    title_template="{case_title} · finite-bead proxy · {frequency_GHz:.3f} GHz",
    x_label="Signed transverse line position [mm]",
    y_label_template="T{thread_index}{held_out} · proxy [V/m]",
    measured_label="Measured finite-bead proxy",
    model_label="Fitted finite-bead proxy",
    held_out_label=" · held out",
):
    entries, profiles = [], {}
    maxima = np.zeros(4)
    for f in FREQUENCIES:
        k = at_frequency(a, f)
        denom = abs(kappa(a["frequency_hz"][k]))
        y = np.sqrt(abs(a["observed_S22"][k] - a["fitted_background"][k]) / denom)
        p = np.sqrt(abs(a["quadratic_scale"][k] * a["template_S22"][k]) / denom)
        profiles[f] = y, p
        for t in range(4):
            use = a["thread_index"] == t
            maxima[t] = max(maxima[t], np.max(y[use]), np.max(p[use]))
    for f, (y, p) in profiles.items():
        fig, axs = plt.subplots(
            4, 1, figsize=(6.5, 7.6), layout="constrained", sharex=True
        )
        for t, ax in enumerate(axs):
            use = a["thread_index"] == t
            x = a["signed_line_m"][use] * 1000
            order = np.argsort(x)
            ax.plot(
                x[order],
                y[use][order],
                ".",
                ms=3.6,
                color="#222222",
                label=measured_label,
            )
            ax.plot(
                x[order],
                p[use][order],
                lw=1.5,
                color=COLOR,
                label=model_label,
            )
            ax.set(
                ylabel=render_text(
                    y_label_template,
                    thread_index=t,
                    held_out=held_out_label if t not in FIT_THREADS else "",
                    frequency_GHz=f,
                ),
                ylim=(0, 1.09 * maxima[t]),
            )
            ax.grid(alpha=0.15)
            ax.yaxis.set_major_locator(MaxNLocator(4))
        axs[0].legend(fontsize=8, loc="best")
        axs[-1].set_xlabel(x_label)
        fig.suptitle(
            render_text(title_template, case_title=TITLE, frequency_GHz=f),
            fontsize=10.5,
        )
        entries.append(
            entry(
                fig,
                "effective_fields_" + f"{f:.3f}".replace(".", "p"),
                f"Finite-bead field proxies · {f:.3f} GHz",
                "Points are √(|S22−background|/|κ|); the curve is √(|c·template|/|κ|)=√|cQ| with the saved finite-bead response. "
                "κ uses the 1-W normalization, bead radius 1.465 mm and relative permittivity 9.23. This is a magnitude-only finite-bead proxy, not a measurement of the unperturbed vector electric field. "
                "The background and common scale are those of the complex fit; no further fit, Gaussian extraction or field deconvolution is performed. "
                + SCOPE_DESCRIPTION,
            )
        )
    return entries, profiles


def geometry_figures(
    record,
    *,
    endpoint_positions_title="Endpoint heights and central disk positions",
    endpoint_shifts_title="Changes from corrected nominal; search ranges",
    projection_title_template="{coordinate}–z projection",
    disk_shifts_title="Shifts from corrected nominal",
    disk_slopes_title="Disk surface slopes",
    view_title="Fitted disk surfaces and all four straight paths",
    z_label="Distance from mirror z [mm]",
    transverse_label_template="{coordinate} [mm]",
    endpoint_shift_y_label="Endpoint Δz [mm]",
    disk_shift_y_label="Central near-face Δz [mm]",
    disk_slope_y_label="Disk slope [mrad]",
    x_label="x [mm]",
    y_label="y [mm]",
    view_z_label="z from mirror [mm]",
    disk_label_template="D{disk_index}",
    thread_label_template="T{thread_index}",
    position_annotation_template="{z_mm:.3f}",
    shift_annotation_template="{shift_mm:+.3f}",
    nominal_disk_band_label="Corrected nominal disk",
    fitted_disk_band_label="Fitted disk centre",
    nominal_thread_marker_label="Corrected nominal thread",
    nominal_disk_label="Nominal disk",
    fitted_disk_label="Fitted disk",
    nominal_thread_label="Nominal thread",
    fitted_thread_label="Fitted thread",
    endpoint_a_label="Endpoint A (min signed s)",
    endpoint_b_label="Endpoint B (max signed s)",
    endpoint_shift_label_template="Endpoint {endpoint}",
    search_range_label="Search range",
    slope_x_label="dz/dx",
    slope_y_label="dz/dy",
):
    endpoint = np.asarray(record["thread_endpoint_z_mm"])
    xy = np.asarray(record["thread_endpoint_xy_m"]) * 1000
    slopes = np.asarray(record["disk_slopes_mrad"])
    faces = np.asarray(record["disk_nearface_z_mm"])
    fig, axs = plt.subplots(
        2,
        1,
        figsize=(6.5, 5.8),
        layout="constrained",
        gridspec_kw={"height_ratios": [1.25, 1]},
    )
    ax = axs[0]
    for j, z in enumerate(NOMINAL_DISKS):
        ax.axvspan(
            z,
            z + 1,
            color=GREY,
            alpha=0.2,
            label=nominal_disk_band_label if j == 0 else None,
        )
        ax.axvspan(
            faces[j],
            faces[j] + 1,
            color=COLOR,
            alpha=0.23,
            label=fitted_disk_band_label if j == 0 else None,
        )
        ax.text(
            faces[j] + 0.5,
            3.65,
            render_text(disk_label_template, disk_index=j + 1),
            ha="center",
            color=COLOR,
            fontsize=8,
        )
    for t in range(4):
        y = 3 - t
        ax.plot(
            NOMINAL_THREADS[t],
            y,
            "D",
            mfc="white",
            mec=GREY,
            ms=5,
            label=nominal_thread_marker_label if t == 0 else None,
        )
        ax.plot(endpoint[t], [y, y], color=THREAD_COLORS[t], lw=2)
        ax.plot(
            endpoint[t, 0],
            y,
            "<",
            color=THREAD_COLORS[t],
            ms=6,
            label=endpoint_a_label if t == 0 else None,
        )
        ax.plot(
            endpoint[t, 1],
            y,
            ">",
            color=THREAD_COLORS[t],
            ms=6,
            label=endpoint_b_label if t == 0 else None,
        )
        for col, offset in ((0, -0.15), (1, 0.15)):
            ax.annotate(
                render_text(
                    position_annotation_template,
                    z_mm=endpoint[t, col],
                    thread_index=t,
                    endpoint="AB"[col],
                ),
                (endpoint[t, col], y),
                xytext=(0, 11 if col else -15),
                textcoords="offset points",
                ha="center",
                color=THREAD_COLORS[t],
                fontsize=8,
            )
    ax.axvline(0, color="#30363B", lw=3)
    ax.set(
        xlim=(-0.5, 37),
        ylim=(-0.55, 4.1),
        yticks=np.arange(4),
        yticklabels=[
            render_text(thread_label_template, thread_index=t) for t in (3, 2, 1, 0)
        ],
        xlabel=z_label,
        title=endpoint_positions_title,
    )
    ax.legend(
        ncol=2,
        fontsize=7.6,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.33),
        frameon=False,
    )
    ax.grid(axis="x", alpha=0.12)
    shifts = endpoint - NOMINAL_THREADS[:, None]
    ax = axs[1]
    x = np.arange(4)
    for col, marker, offset, label in [(0, "<", -0.1, "A"), (1, ">", 0.1, "B")]:
        ax.plot(
            x + offset,
            shifts[:, col],
            marker,
            color=COLOR,
            ms=7,
            label=render_text(endpoint_shift_label_template, endpoint=label),
        )
        for xx, yy in zip(x + offset, shifts[:, col]):
            ax.annotate(
                render_text(shift_annotation_template, shift_mm=yy, endpoint="AB"[col]),
                (xx, yy),
                xytext=(0, 7 if col else -14),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    bound = (
        float(record["fit_details"].get("endpoint_adjustment_bound_mm", 2.0))
        if len(FIT_THREADS) == 4
        else 0.0
    )
    ax.axhline(0, color=GREY, lw=0.7)
    lower = FIT_THREADS_MM - NOMINAL_THREADS - bound
    upper = FIT_THREADS_MM - NOMINAL_THREADS + bound
    for t in range(4):
        ax.vlines(
            t,
            lower[t],
            upper[t],
            color=GREY,
            lw=0.7,
            ls=":",
            label=search_range_label if t == 0 else None,
        )
        ax.hlines([lower[t], upper[t]], t - 0.28, t + 0.28, color=GREY, lw=0.7, ls=":")
    ax.set(
        xticks=x,
        xticklabels=[
            render_text(thread_label_template, thread_index=t) for t in range(4)
        ],
        ylim=(
            min(lower.min(), shifts.min()) - 0.7,
            max(upper.max(), shifts.max()) + 0.7,
        ),
        ylabel=endpoint_shift_y_label,
        title=endpoint_shifts_title,
    )
    ax.grid(axis="y", alpha=0.15)
    ax.legend(
        fontsize=7.8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
    )
    first = entry(
        fig,
        "endpoint_geometry",
        "Thread endpoint heights",
        "The upper panel shows endpoint z coordinates along a common mirror-distance axis; row spacing is only for readability. Disk bands show central near-face positions and 1 mm thickness, not their tilted extent. "
        "A and B are the minimum and maximum retained signed scan-coordinate locations, not measured mechanical supports. They remain at their recorded x/y positions. "
        "Each straight path interpolates between its own endpoint pair. The lower panel shows shifts from the corrected nominal and the search ranges in those coordinates. "
        + SCOPE_DESCRIPTION,
    )
    split_geometry = CASE == "disk_tilts_threads"
    # This projection plot uses the requested T0/T1 colour swap.
    projection_colors = (
        [THREAD_COLORS[t] for t in (1, 0, 2, 3)] if split_geometry else THREAD_COLORS
    )
    if split_geometry:
        fig, projection_axes = plt.subplots(
            1, 2, figsize=(8, 4.2), layout="constrained"
        )
    else:
        fig, axs = plt.subplots(
            2,
            2,
            figsize=(6.5, 6.4),
            layout="constrained",
            gridspec_kw={"height_ratios": [1.4, 1]},
        )
        projection_axes = axs[0]
    cross = np.array([-RADIUS_MM, RADIUS_MM])
    for col, name in enumerate(("x", "y")):
        ax = projection_axes[col]
        for j in range(3):
            if split_geometry:
                nominal = NOMINAL_DISKS[j]
                nominal_pts = np.array(
                    [
                        [nominal, cross[0]],
                        [nominal, cross[1]],
                        [nominal + 1, cross[1]],
                        [nominal + 1, cross[0]],
                    ]
                )
                ax.add_patch(
                    Polygon(
                        nominal_pts, fc="#E4E8EB", ec=GREY, lw=1.1, ls="--", zorder=1
                    )
                )
            near = faces[j] + slopes[j, col] * cross / 1000
            pts = np.array(
                [
                    [near[0], -150],
                    [near[1], 150],
                    [near[1] + 1, 150],
                    [near[0] + 1, -150],
                ]
            )
            ax.add_patch(Polygon(pts, fc=COLOR, ec=COLOR, alpha=0.35))
            if split_geometry:
                ax.text(
                    NOMINAL_DISKS[j] + 0.5,
                    RADIUS_MM + 9,
                    render_text(disk_label_template, disk_index=j + 1),
                    ha="center",
                    fontsize=8,
                    color=GREY,
                )
            else:
                ax.axvline(NOMINAL_DISKS[j], color=GREY, lw=0.6, ls=":")
        for t in range(4):
            if split_geometry:
                ax.plot(
                    [NOMINAL_THREADS[t]] * 2,
                    xy[t, :, col],
                    color=projection_colors[t],
                    lw=1.3,
                    ls="--",
                    zorder=3,
                )
            ax.plot(
                endpoint[t],
                xy[t, :, col],
                color=projection_colors[t],
                lw=1.4,
                label=render_text(thread_label_template, thread_index=t),
            )
            ax.plot(
                endpoint[t, 0], xy[t, 0, col], "<", color=projection_colors[t], ms=4
            )
            ax.plot(
                endpoint[t, 1], xy[t, 1, col], ">", color=projection_colors[t], ms=4
            )
        ax.axvline(0, color="#30363B", lw=3)
        ax.set(
            xlim=(-0.5, 37),
            ylim=(-215, 215),
            xlabel=z_label,
            ylabel=render_text(transverse_label_template, coordinate=name),
            title=render_text(projection_title_template, coordinate=name),
        )
        ax.grid(alpha=0.13)
    if split_geometry:
        handles = [
            Patch(fc="#E4E8EB", ec=GREY, ls="--", label=nominal_disk_label),
            Patch(fc=COLOR, ec=COLOR, alpha=0.35, label=fitted_disk_label),
            Line2D([], [], color=GREY, ls="--", lw=1.3, label=nominal_thread_label),
            Line2D([], [], color=GREY, lw=1.4, label=fitted_thread_label),
        ]
        handles += [
            Line2D(
                [],
                [],
                color=projection_colors[t],
                lw=2,
                label=render_text(thread_label_template, thread_index=t),
            )
            for t in range(4)
        ]
        # Column-wise legend filling pairs each geometric style with a thread colour.
        order = [0, 4, 1, 5, 2, 6, 3, 7]
        fig.legend(
            handles=[handles[i] for i in order],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=4,
            fontsize=8,
            frameon=False,
        )
        second = entry(
            fig,
            "disk_surfaces_and_paths",
            "Nominal and fitted disk surfaces and thread paths",
            "Grey dashed outlines show the corrected nominal disks, perpendicular to z; coloured dashed lines show nominal thread heights at the recorded x/y endpoints. "
            "Filled disks and solid paths show the fitted geometry. Disk cross-sections are taken at the other transverse coordinate equal to zero; thread paths are orthogonal projections. "
            "A projected crossing does not establish physical overlap. The axial and transverse scales differ; disk thickness is 1 mm.",
        )
        fig, adjustment_axes = plt.subplots(
            1, 2, figsize=(7.4, 3.2), layout="constrained"
        )
    else:
        axs[0, 0].legend(ncol=2, fontsize=8, loc="upper left")
        adjustment_axes = axs[1]
    x = np.arange(3)
    adjustment_axes[0].bar(x, disk_shifts_mm(faces), color=COLOR, width=0.55)
    adjustment_axes[0].set(
        xticks=x,
        xticklabels=[render_text(disk_label_template, disk_index=j) for j in (1, 2, 3)],
        ylabel=disk_shift_y_label,
        title=disk_shifts_title,
    )
    for col, offset, label, color in (
        (0, -0.17, slope_x_label, COLOR),
        (1, 0.17, slope_y_label, "#DDAA33"),
    ):
        adjustment_axes[1].bar(
            x + offset, slopes[:, col], width=0.32, color=color, label=label
        )
    adjustment_axes[1].set(
        xticks=x,
        xticklabels=[render_text(disk_label_template, disk_index=j) for j in (1, 2, 3)],
        ylabel=disk_slope_y_label,
        title=disk_slopes_title,
    )
    adjustment_axes[1].legend(fontsize=8)
    for ax in adjustment_axes:
        ax.axhline(0, color=GREY, lw=0.6)
        ax.grid(axis="y", alpha=0.15)
    if split_geometry:
        adjustments = entry(
            fig,
            "disk_shifts_and_slopes",
            "Fitted disk shifts and surface slopes",
            "Left: fitted central near-face z minus the corrected nominal position. Right: fitted dz/dx and dz/dy in mrad. Both disk faces share the same slope and have 1 mm vertical separation.",
        )
    else:
        second = entry(
            fig,
            "disk_surfaces_and_paths",
            "Disk surfaces and projected sloped paths",
            "Each top panel shows the fitted disk cross-section at the other transverse coordinate equal to zero, together with the orthogonal projection of every actual sloped chord. "
            "Projected line–disk intersections do not establish physical overlap: the chord can have a nonzero other transverse coordinate. Both axial and transverse scales are deliberately different. "
            "The saved constraint uses normal distance to infinite face planes along the path clipped to the expanded disk footprint; this conservative plane margin is distinct from distance to the physical finite disk surface or edge. Disk faces are parallel with 1 mm vertical separation.",
        )
    fig = plt.figure(figsize=(6.5, 5.6), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    radial = np.linspace(0, RADIUS_MM, 16)
    phi = np.linspace(0, 2 * np.pi, 65)
    xx, yy = radial[:, None] * np.cos(phi), radial[:, None] * np.sin(phi)
    ax.plot_surface(
        xx, yy, np.zeros_like(xx), color="#89949E", alpha=0.22, linewidth=0, shade=False
    )
    for j in range(3):
        zz = faces[j] + (slopes[j, 0] * xx + slopes[j, 1] * yy) / 1000
        ax.plot_surface(xx, yy, zz, color=COLOR, alpha=0.22, linewidth=0, shade=False)
        ax.plot(xx[-1], yy[-1], zz[-1], color=COLOR, lw=0.8)
        ax.plot(xx[-1], yy[-1], zz[-1] + 1, color=COLOR, lw=0.6, alpha=0.65)
    for t in range(4):
        ax.plot(
            xy[t, :, 0],
            xy[t, :, 1],
            endpoint[t],
            color=THREAD_COLORS[t],
            lw=2,
            label=render_text(thread_label_template, thread_index=t),
        )
        ax.scatter(xy[t, :, 0], xy[t, :, 1], endpoint[t], color=THREAD_COLORS[t], s=15)
    ax.set(
        xlim=(-210, 210),
        ylim=(-210, 210),
        zlim=(0, 37),
        xlabel=x_label,
        ylabel=y_label,
        zlabel=view_z_label,
        title=view_title,
    )
    ax.set_box_aspect((1, 1, 1.15))
    ax.view_init(elev=23, azim=-57)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        frameon=False,
    )
    third = entry(
        fig,
        "geometry_3d",
        "Oblique view of the fitted disk surfaces and paths",
        "Axial distances are strongly exaggerated relative to transverse distances so the fitted slopes and endpoint heights can be seen. "
        "Translucent surfaces show near faces; paired rim curves indicate the 1 mm vertical disk thickness. The grey plane is the mirror. "
        "Coloured straight segments connect the actual retained x/y endpoints at their fitted z coordinates. These recorded segment endpoints are not measured support anchors.",
    )
    return (
        [first, second, adjustments, third]
        if split_geometry
        else [first, second, third]
    )


def field_map_figures(
    maps,
    a,
    record,
    *,
    title_template="{case_title} · horizontal sections · {frequency_GHz:.3f} GHz",
    panel_title_template="T{thread_index}{held_out}\nz = {z_mm:.4f} mm",
    x_label="x [mm]",
    y_label="y [mm]",
    colorbar_label="$|E|$ [V/m]",
    held_out_label=" · held out",
):
    x, y = maps["x_m"] * 1000, maps["y_m"] * 1000
    z = maps.get("thread_midpoint_z_mm", maps.get("thread_z_mm"))
    np.testing.assert_allclose(
        z, np.asarray(record["thread_endpoint_z_mm"]).mean(axis=1), atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        maps["thread_endpoint_z_mm"], a["thread_endpoint_z_mm"], atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        maps["thread_endpoint_xy_m"], a["thread_endpoint_xy_m"], atol=1e-14, rtol=0
    )
    np.testing.assert_allclose(
        maps["disk_slopes"] * 1000, record["disk_slopes_mrad"], atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        maps["disk_nearface_z_m"] * 1000,
        record["disk_nearface_z_mm"],
        atol=1e-10,
        rtol=0,
    )
    unperturbed = CASE == "disk_tilts_threads"
    field_key = (
        "unperturbed_magnitude_V_per_m" if unperturbed else "field_proxy_V_per_m"
    )
    field = maps[field_key]
    if field.shape != maps["support"].shape or field.shape[1] != 4:
        raise ValueError("Maps must contain all four thread midpoint planes")
    window = (abs(x[:, None]) <= 205) & (abs(y[None, :]) <= 205)
    vmax = max(
        np.nanmax(
            np.where(
                maps["support"][at_frequency(maps, f)],
                field[at_frequency(maps, f)],
                np.nan,
            )[:, window]
        )
        for f in FREQUENCIES
    )
    entries = []
    for f in FREQUENCIES:
        k = at_frequency(maps, f)
        fig, axs = plt.subplots(
            2, 2, figsize=(6.5, 6.4), layout="constrained", sharex=True, sharey=True
        )
        for t, ax in enumerate(axs.flat):
            plane = np.where(maps["support"][k, t], field[k, t], np.nan)
            display = ax.pcolormesh(
                x,
                y,
                plane.T,
                shading="auto",
                cmap="viridis",
                vmin=0,
                vmax=vmax,
                rasterized=True,
            )
            xy = a["xy_m"][a["thread_index"] == t] * 1000
            ax.plot(xy[:, 0], xy[:, 1], color="#333333", lw=2.1, ls=":")
            ax.plot(xy[:, 0], xy[:, 1], color="white", lw=1.1, ls=":")
            ax.add_patch(
                Circle((0, 0), RADIUS_MM, fc="none", ec="white", lw=0.8, ls="--")
            )
            ax.set(
                xlim=(-205, 205),
                ylim=(-205, 205),
                aspect="equal",
                title=render_text(
                    panel_title_template,
                    thread_index=t,
                    held_out=held_out_label if t not in FIT_THREADS else "",
                    z_mm=z[t],
                    frequency_GHz=f,
                ),
            )
            if t >= 2:
                ax.set_xlabel(x_label)
            if t % 2 == 0:
                ax.set_ylabel(y_label)
        colour_label = colorbar_label
        fig.colorbar(display, ax=axs, shrink=0.84, label=colour_label)
        fig.suptitle(
            render_text(title_template, case_title=TITLE, frequency_GHz=f),
            fontsize=10.5,
        )
        if unperturbed:
            caption = (
                "Colours show the fitted model's electric-field magnitude without the bead: the magnitude of the summed incoming and outgoing fields, including the fitted amplitude scale. "
                "Each panel is a horizontal section at its thread's fitted midpoint z. "
                + SCOPE_DESCRIPTION
                + " "
                "The black/white dotted overlay marks one projected x/y path; a sloped thread generally leaves this plane. "
                "The existing aperture and conservative clearance display mask is retained; blank regions do not mean zero field. "
                "Dashed circles mark the 150 mm disk radius. All four threads and both frequencies share one colour scale."
            )
            figure = entry(
                fig,
                "transverse_fields_" + f"{f:.3f}".replace(".", "p"),
                f"Electric field magnitude · {f:.3f} GHz",
                caption,
            )
            figure.update(
                source_array=field_key,
                quantity="sqrt(abs(c)) * abs(E_in + E_out)",
                units="V/m",
                colour_limits_V_per_m=[0.0, float(vmax)],
                finite_bead_response_applied=False,
            )
            entries.append(figure)
        else:
            entries.append(
                entry(
                    fig,
                    "transverse_fields_" + f"{f:.3f}".replace(".", "p"),
                    f"Illustrative midpoint planes · {f:.3f} GHz",
                    "These are horizontal model cross-sections at each saved endpoint-pair midpoint z. "
                    + SCOPE_DESCRIPTION
                    + " "
                    "White dotted lines show only projected x/y footprints; most points on a sloped chord lie outside this displayed plane. "
                    "The proxy includes the saved common amplitude √|c|. Blank pixels fail the physical-air/bead-clearance support mask. Dashed circles show the 150 mm aperture. Both frequencies share one colour scale.",
                )
            )
    return entries


def reference_figure(
    a,
    bands,
    reference,
    notice,
    *,
    title="Spider and full bead pull · saved normalizations",
    x_label="Frequency [GHz]",
    y_label="Boost factor $\\beta^2$",
    full_curve_label="Full bead pull",
    spider_curve_label="Spider bead pull",
):
    fig, ax = plt.subplots(figsize=(6.5, 4.0), layout="constrained")
    f = a["frequency_hz"] / 1e9
    fr = reference["frequency_hz"] / 1e9
    yr, er = reference["beta2_saved"], reference["beta2_saved_error"]
    ax.fill_between(
        fr, np.maximum(0, yr - er), yr + er, color="#42484D", alpha=0.16, lw=0
    )
    ax.fill_between(
        f,
        bands["quantiles"][:, 1],
        bands["quantiles"][:, 3],
        color=COLOR,
        alpha=0.24,
        lw=0,
    )
    ax.plot(fr, yr, color="#42484D", ls="--", lw=1.6, label=full_curve_label)
    ax.plot(f, a["beta2_conditional"], color=COLOR, lw=1.8, label=spider_curve_label)
    ax.set(
        xlim=(20, 22),
        ylim=(
            0,
            1.08
            * max(
                np.max(yr + er),
                np.max(bands["quantiles"][:, 3]),
                np.max(a["beta2_conditional"]),
            ),
        ),
        xlabel=x_label,
        ylabel=y_label,
        title=title,
    )
    ax.legend(fontsize=8.5, loc="best")
    ax.grid(alpha=0.15)
    return entry(
        fig,
        "boost_full_reference",
        "Full bead pull and spider boost",
        notice
        + " The grey band is the saved full-scan ±1σ spread; the coloured band is the conditional spider 16–84% residual-resampling spread. "
        "Shared axes cover 20–22 GHz; the full reference has native support only from 20.5 to 21.5 GHz. No rescaling, extrapolation or fitted frequency alignment is applied.",
    )


def residual_figure(
    a,
    control=None,
    *,
    title_template="{case_title} · fit quality",
    x_label="Frequency [GHz]",
    combined_y_label="Combined residual [%]",
    individual_y_label="Individual residual [%]",
    objective_label_template="Fitted objective · {scope}",
    fit_bins_label_template="{fit_bin_count} quality-selected bins",
    control_label="New fixed-thread tilts · same all-four objective",
    thread_label_template="T{thread_index}{held_out}",
    held_out_label=" · held out",
):
    fig, axs = plt.subplots(2, 1, figsize=(6.5, 5.8), layout="constrained", sharex=True)
    f = a["frequency_hz"] / 1e9
    use = a["eligible"].astype(bool)
    axs[0].plot(
        f,
        100 * a["relative_residual"],
        color=COLOR,
        label=render_text(objective_label_template, scope=SCOPE),
    )
    axs[0].plot(
        f[use],
        100 * a["relative_residual"][use],
        ".",
        color=COLOR,
        ms=4,
        label=render_text(fit_bins_label_template, fit_bin_count=int(use.sum())),
    )
    if control is not None:
        axs[0].plot(
            control["frequency_hz"] / 1e9,
            100 * control["all_thread_residual"],
            color=GREY,
            ls="--",
            label=control_label,
        )
    for t in range(4):
        axs[1].plot(
            f,
            100 * a["relative_residual_by_thread"][:, t],
            lw=1.2,
            color=THREAD_COLORS[t],
            label=render_text(
                thread_label_template,
                thread_index=t,
                held_out=held_out_label if t not in FIT_THREADS else "",
            ),
        )
    axs[0].set(
        ylim=(0, 105),
        title=render_text(title_template, case_title=TITLE),
        ylabel=combined_y_label,
    )
    axs[1].set(
        yscale="log",
        ylim=(1, 1e4),
        ylabel=individual_y_label,
        xlabel=x_label,
    )
    for ax in axs:
        ax.set_xlim(20, 22)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=7.8, ncol=2)
    return entry(
        fig,
        "objective_comparison",
        "Combined and individual-thread residuals",
        f"The upper curve uses {SCOPE}; dots mark the {int(use.sum())} profile-quality-selected bins. The lower panel uses a logarithmic vertical scale and the same fitted common coefficient on every thread. "
        + (
            "T0 is a held-out prediction and does not determine the common coefficient. "
            if len(FIT_THREADS) == 3
            else ("The dashed control uses the new fixed-thread tilt geometry, with its common coefficient reprofiled on the same four chords; geometry is not refitted for this diagnostic. "
                  if control is not None else "")
        )
        + "Equal sample weights within a frequency give stronger signals more influence. A small pooled residual does not imply agreement on each weak thread.",
    )


def reference_field_figure(
    maps,
    reference,
    *,
    frequency_GHz=20.825,
    title_template="Model field integrals · {frequency_GHz:.3f} GHz",
    x_label="Distance from mirror z [mm]",
    y_label="Unperturbed $|E_T|$ [V m]",
    full_curve_label="Full bead pull · reconstructed 1D model",
    spider_curve_label="Spider · 3D horizontal model sections",
    thread_label_template="T{thread_index}",
):
    f = frequency_GHz
    k = at_frequency(maps, f)
    kr = at_frequency(reference, f)
    fig, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained")
    ax.plot(
        reference["z_m"] * 1000,
        abs(reference["field_Vm"][kr]),
        color="#42484D",
        ls="--",
        lw=1.5,
        label=full_curve_label,
    )
    valid = maps["coherent_integral_full_aperture_in_air"][k].astype(bool)
    z = maps["thread_midpoint_z_mm"][1:]
    values = abs(maps["coherent_transverse_integral_Vm"][k])
    ax.plot(
        z[valid],
        values[valid],
        "o",
        color=COLOR,
        ms=6,
        label=spider_curve_label,
    )
    for t in range(3):
        if valid[t]:
            ax.annotate(
                render_text(thread_label_template, thread_index=t + 1),
                (z[t], values[t]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set(
        xlim=(0, 50),
        ylim=(0, None),
        xlabel=x_label,
        ylabel=y_label,
        title=render_text(title_template, frequency_GHz=f),
    )
    ax.grid(alpha=0.15)
    ax.legend(fontsize=8)
    return entry(
        fig,
        "fields_full_reference_" + f"{f:.3f}".replace(".", "p"),
        "Full-reference and spider model fields",
        "Both series are model predictions of the unperturbed coherent transverse field integral, not raw finite-bead measurements. The full curve uses its saved 1D mean parameters; spider points integrate the fitted 3D field at horizontal midpoint sections with the saved common amplitude. Invalid full-aperture-in-air sections are omitted. No phase comparison is made because the global phase reference is unregistered. Aperture and absolute-calibration differences remain; a horizontal section is not a sloped sampling path.",
    )
