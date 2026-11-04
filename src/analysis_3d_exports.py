"""Conditional bands and model electric-field maps; no optimization."""

from pathlib import Path
import json
import numpy as np
from .analysis_3d import NOM_Z_MM
from .analysis_3d_inputs import load_measurement

Q = np.array([0.025, 0.16, 0.5, 0.84, 0.975])


def reprofile_saved_geometry(saved, measurement_path):
    """Use measured S22 with saved physical response templates, without a refit.

    Recompute the common complex scale, affine backgrounds and dependent result
    arrays. Geometry, directional waves and geometry-only boost stay fixed. The
    historical geometry-fit mask is kept separately from current profile quality.
    """
    measurement = load_measurement(measurement_path)
    indices = np.searchsorted(measurement["frequency_hz"], saved["frequency_hz"])
    np.testing.assert_array_equal(
        measurement["frequency_hz"][indices], saved["frequency_hz"]
    )
    rows = np.searchsorted(measurement["trace_index"], saved["trace_index"])
    np.testing.assert_array_equal(
        measurement["trace_index"][rows], saved["trace_index"]
    )
    np.testing.assert_array_equal(
        measurement["thread_index"][rows], saved["thread_index"]
    )
    np.testing.assert_allclose(
        measurement["signed_line_m"][rows], saved["signed_line_m"], atol=1e-14
    )
    np.testing.assert_allclose(
        measurement["transverse_xyz_log_m"][rows, :2], saved["xy_m"], atol=1e-14
    )
    if np.any(measurement["transient_trace_mask"][rows]):
        raise ValueError("Saved templates include excluded transition samples.")
    observed = measurement["observed_S22"][rows][:, indices].T
    projected = np.empty_like(observed)
    templates = np.empty_like(observed)
    designs = []
    for thread in range(4):
        take = saved["thread_index"] == thread
        X = np.column_stack([np.ones(take.sum()), saved["signed_line_m"][take] / 0.1])
        inverse = np.linalg.pinv(X)
        projection = np.eye(take.sum()) - X @ inverse
        projected[:, take] = observed[:, take] @ projection
        templates[:, take] = saved["template_S22"][:, take] @ projection
        designs.append((take, X, inverse))
    coefficient = np.sum(templates.conj() * projected, axis=1) / np.sum(
        abs(templates) ** 2, axis=1
    )
    projected_error = coefficient[:, None] * templates - projected
    background = np.empty_like(observed)
    rms, relative, energy = [], [], []
    for take, X, inverse in designs:
        error = (
            observed[:, take] - coefficient[:, None] * saved["template_S22"][:, take]
        )
        background[:, take] = error @ inverse.T @ X.T
        residual = error - background[:, take]
        rms.append(np.sqrt(np.mean(abs(residual) ** 2, axis=1)))
        relative.append(
            np.linalg.norm(residual, axis=1)
            / np.linalg.norm(projected[:, take], axis=1)
        )
        energy.append(np.sum(abs(projected[:, take]) ** 2, axis=1))
    interior = saved["thread_index"] != 0
    ip, iy = templates[:, interior], projected[:, interior]
    interior_coefficient = np.sum(ip.conj() * iy, axis=1) / np.sum(abs(ip) ** 2, axis=1)
    energy = np.asarray(energy).T
    relative_all = np.linalg.norm(projected_error, axis=1) / np.linalg.norm(
        projected, axis=1
    )
    result = dict(saved)
    result.update(
        reflection_preprocessing="none",
        geometry_refitted=False,
        geometry_fit_reflection_preprocessing=saved.get(
            "geometry_fit_reflection_preprocessing", "additive_template"
        ),
        geometry_fit_mask=saved.get("geometry_fit_mask", saved["eligible"]),
        evaluation_method="Common scale and affine backgrounds refitted to measured S22 at saved geometry.",
        eligible=measurement["shared_fit_mask"][indices],
        observed_S22=observed,
        fitted_background=background,
        predicted_S22=coefficient[:, None] * saved["template_S22"] + background,
        quadratic_scale=coefficient,
        amplitude=np.sqrt(abs(coefficient)),
        beta2_conditional=abs(coefficient) * saved["beta2_received_geometry_only"],
        relative_residual=relative_all,
        all_thread_residual=relative_all,
        raw_rms_by_thread=np.asarray(rms).T,
        relative_residual_by_thread=np.asarray(relative).T,
        projected_signal_energy_fraction_by_thread=energy
        / energy.sum(axis=1, keepdims=True),
        interior_shared_scale_residual=np.linalg.norm(
            coefficient[:, None] * ip - iy, axis=1
        )
        / np.linalg.norm(iy, axis=1),
        interior_reprofiled_scale_residual=np.linalg.norm(
            interior_coefficient[:, None] * ip - iy, axis=1
        )
        / np.linalg.norm(iy, axis=1),
        interior_quadratic_scale=interior_coefficient,
    )
    return result


def conditional_bands(a, fit_threads, draws=1000):
    """Exact conditional common-scale update, using only fitted threads."""
    rng = np.random.default_rng(20261025)
    projections = []
    residuals = []
    indices = []
    for t in fit_threads:
        mask = a["thread_index"] == t
        u = a["signed_line_m"][mask] / 0.1
        X = np.column_stack([np.ones(len(u)), u])
        P = np.eye(len(u)) - X @ np.linalg.pinv(X)
        projections.append(a["template_S22"][:, mask] @ P)
        residual = a["observed_S22"][:, mask] - a["predicted_S22"][:, mask]
        residuals.append(residual - residual.mean(axis=1, keepdims=True))
        start = rng.integers(0, len(u), (draws, int(np.ceil(len(u) / 8))))
        indices.append(
            ((start[:, :, None] + np.arange(8)) % len(u)).reshape(draws, -1)[
                :, : len(u)
            ]
        )
    denominator = sum(np.sum(abs(p) ** 2, axis=1) for p in projections)
    if np.any(denominator <= 0):
        raise ValueError("Zero common-scale denominator")
    delta = np.zeros((draws, len(a["frequency_hz"])), complex)
    for f in range(len(a["frequency_hz"])):
        for t in range(len(fit_threads)):
            delta[:, f] += (
                residuals[t][f][indices[t]] @ projections[t][f].conj() / denominator[f]
            )
    beta = (
        abs(a["quadratic_scale"][None, :] + delta)
        * a["beta2_received_geometry_only"][None, :]
    )
    return np.quantile(beta, Q, axis=0).T


def midpoint_maps(p, q, folder, frequencies_hz=(20.795e9, 20.825e9, 21e9)):
    """Horizontal illustrative sections; likelihood uses actual pointwise z."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    r = p.evaluate(q)
    m = p.model
    s = r["solution"]
    shape = (len(m.x), len(m.y))
    frequencies = np.asarray(frequencies_hz)
    indices = np.searchsorted(m.f, frequencies)
    np.testing.assert_array_equal(m.f[indices], frequencies)
    ends = NOM_Z_MM[:, None] + q[9:].reshape(4, 2)
    midpoint = ends.mean(axis=1) * 0.001
    disks = s["disks_m"]
    slopes = q[3:9].reshape(3, 2) * 0.001
    proxy = np.zeros((len(indices), 4, *shape))
    unperturbed = proxy.copy()
    support = np.zeros_like(proxy, bool)
    integrated = np.zeros((len(indices), 3), complex)
    valid = np.ones((len(indices), 3), bool)
    xx, yy = m.X, m.Y
    aperture = xx**2 + yy**2 <= m.radius**2
    rb = 0.001465
    for j, k in enumerate(indices):
        for t, z in enumerate(midpoint):
            if t == 0:
                incoming = s["input_field"][k].reshape(shape, order="F")
                outgoing = (
                    m.B @ np.linalg.solve(s["P"][k, 7], s["reflected"][k])
                ).reshape(shape, order="F")
                kz = p.exterior.kz[k]
                prop = p.exterior.propagating[k]
                d = z - disks[2] - 0.001
                if d < 0:
                    raise ValueError(
                        "Exterior midpoint below horizontal front reference"
                    )
                incoming = np.fft.ifft2(
                    np.where(prop, np.fft.fft2(incoming), 0.0)
                    * np.exp(1j * kz.real * d)
                )
                outgoing = np.fft.ifft2(np.fft.fft2(outgoing) * np.exp(-1j * kz * d))
                waves = m.scale * np.stack([incoming, outgoing], axis=-1)
                clearance = (
                    z - disks[2] - 0.001 - slopes[2, 0] * xx - slopes[2, 1] * yy
                ) / np.sqrt(1 + sum(slopes[2] ** 2))
                ok = (xx**2 + yy**2 > (m.radius + rb) ** 2) | (clearance >= rb)
            else:
                high = 3 - t
                low = high - 1
                region = 7 - 2 * t
                dr = disks[high] - z
                dl = z - (disks[low] + 0.001 if low >= 0 else 0.0)
                cin = m.air(dr)[k] @ s["trace"][k, region, 0]
                cout = m.air(dl)[k] @ np.linalg.solve(
                    s["P"][k, region], s["trace"][k, region, 1]
                )
                waves = (m.scale * (m.B @ np.stack([cin, cout], axis=-1))).reshape(
                    (*shape, 2), order="F"
                )
                right = (
                    disks[high] + slopes[high, 0] * xx + slopes[high, 1] * yy - z
                ) / np.sqrt(1 + sum(slopes[high] ** 2))
                left = (
                    np.full(shape, z)
                    if low < 0
                    else (
                        z
                        - disks[low]
                        - 0.001
                        - slopes[low, 0] * xx
                        - slopes[low, 1] * yy
                    )
                    / np.sqrt(1 + sum(slopes[low] ** 2))
                )
                ok = aperture & (right >= rb) & (left >= rb)
                if np.any(aperture & ((right <= 0) | (left <= 0))):
                    valid[j, t - 1] = False
            em = waves[..., 0].conj()
            ep = waves[..., 1].conj()
            quadratic = m.de[k] * (em**2 + ep**2) + 2 * m.dc[k] * em * ep
            scale = np.sqrt(abs(r["coefficient"][k]))
            proxy[j, t] = scale * np.sqrt(abs(quadratic))
            unperturbed[j, t] = scale * abs(em + ep)
            support[j, t] = ok
            if t > 0:
                integrated[j, t - 1] = (
                    scale * np.sum((em + ep)[aperture]) * m.area
                    if valid[j, t - 1]
                    else complex(np.nan, np.nan)
                )
    proxy[~support] = np.nan
    unperturbed[~support] = np.nan
    t0 = (
        np.concatenate([np.full(len(line["u"]), line["thread"]) for line in p.lines])
        == 0
    )
    np.savez_compressed(
        folder / "field_maps.npz",
        frequency_hz=frequencies,
        x_m=m.x,
        y_m=m.y,
        thread_z_mm=midpoint * 1000,
        thread_midpoint_z_mm=midpoint * 1000,
        field_proxy_V_per_m=proxy,
        unperturbed_magnitude_V_per_m=unperturbed,
        support=support,
        coherent_transverse_integral_Vm=integrated,
        coherent_integral_full_aperture_in_air=valid,
        thread_endpoint_z_mm=ends,
        thread_endpoint_xy_m=np.array([l["endpoint_xy"] for l in p.lines]),
        disk_slopes=slopes,
        disk_nearface_z_m=disks,
        thread0_signed_line_m=p.lines[0]["u"],
        thread0_xy_m=p.lines[0]["xy"],
        thread0_observed_S22=p.lines[0]["y"][indices],
        thread0_predicted_S22=r["predicted"][indices][:, t0],
        thread0_fitted_background=r["background"][indices][:, t0],
    )
    meta = {
        "scope": "Illustrative horizontal sections at each straight thread midpoint z. Sloped-thread likelihood/chord profiles use point-specific z, not these sections.",
        "support": "Interior disk aperture and full bead clearance from actual tilted face planes; exterior points in expanded disk footprint require bead clearance. This is conservative at finite disk rims.",
        "coherent_integral": "Unperturbed complex transverse integral at the illustrative horizontal section; valid only when the entire disk aperture lies in air. It is not a measurement integral along a sloped path.",
        "reference_thread_z_mm_T0_T1_T2_T3": NOM_Z_MM.tolist(),
        "actual_thread_midpoint_z_mm": (midpoint * 1000).tolist(),
        "fit_threads": list(p.fit_threads),
        "fitted_common_scale_used": True,
    }
    (folder / "field_maps.json").write_text(json.dumps(meta, indent=2) + "\n")


def longitudinal_field_integrals(
    saved, bank_path, beam_path, frequency_GHz=20.825, z_m=None
):
    """Evaluate the saved 3D geometry and amplitude on horizontal air sections.

    This is a field export, not a fit. Both complex transverse integrals use the
    same 150 mm circular aperture. The finite-bead root uses the receive model's
    existing branch nearest its field without the bead. Sections intersecting a
    tilted disk are omitted; the bead response also requires full bead clearance.
    """
    from scipy.interpolate import BSpline
    from .tilted_disks import TiltedReceiveModel

    model = TiltedReceiveModel(bank_path, beam_path=beam_path)
    matches = np.flatnonzero(abs(model.f - frequency_GHz * 1e9) < 1)
    if len(matches) != 1:
        raise ValueError("The selected field frequency must be present in the bank.")
    k = int(matches[0])
    sk = np.flatnonzero(abs(saved["frequency_hz"] - model.f[k]) < 1)
    if len(sk) != 1:
        raise ValueError("The field export needs a saved amplitude at this frequency.")
    amplitude = np.sqrt(abs(saved["quadratic_scale"][int(sk[0])]))
    q = saved["parameters"]
    from .receive_field import NOM_DISKS_MM

    disks = NOM_DISKS_MM + q[:3]
    slopes = q[3:9].reshape(3, 2) * 0.001
    solution = model.solve_tilted(disks, slopes)
    disks = np.asarray(disks) * 0.001
    midpoint = saved["thread_endpoint_z_mm"].mean(axis=1) * 0.001
    z = (
        np.unique(np.r_[np.linspace(0, 0.05, 1001), midpoint])
        if z_m is None
        else np.asarray(z_m, float)
    )
    if z.ndim != 1 or not np.all(np.isfinite(z)) or np.any(z < 0):
        raise ValueError("Section coordinates must be a finite vector with z >= 0.")
    integrals = np.full(len(z), complex(np.nan, np.nan))
    bead_integrals = integrals.copy()
    valid = np.zeros(len(z), bool)
    bead_valid = valid.copy()
    extent = model.radius * np.linalg.norm(slopes, axis=1)
    bead_clearance = 0.001465 * np.sqrt(1 + np.sum(slopes**2, axis=1))
    aperture = model.aperture.reshape((len(model.x), len(model.y)), order="F")
    basis = model.Binside
    coefficients = model.Pspline.c[..., k]
    incident = solution["input_field"][k].reshape(aperture.shape, order="F")
    reflected = (
        model.B @ np.linalg.solve(solution["P"][k, 7], solution["reflected"][k])
    ).reshape(aperture.shape, order="F")
    kz2 = (
        (2 * np.pi * model.f[k] / 299792458.0) ** 2
        - model.arr["kx_fft_per_m"][:, None] ** 2
        - model.arr["ky_fft_per_m"][None, :] ** 2
    )
    kz = np.sqrt(kz2.astype(complex)).conj()
    incoming_fft = np.where(kz2 >= 0, np.fft.fft2(incident), 0)
    outgoing_fft = np.fft.fft2(reflected)

    for gap in range(4):
        low = 0.0 if gap == 0 else disks[gap - 1] + 0.001 + extent[gap - 1]
        high = disks[gap] - extent[gap] if gap < 3 else np.inf
        select = np.flatnonzero((z >= low) & (z <= high))
        if not len(select):
            continue
        zz = z[select]
        bead_low = 0.001465 if gap == 0 else low + bead_clearance[gap - 1]
        bead_high = high - bead_clearance[gap] if gap < 3 else np.inf
        valid[select] = True
        bead_valid[select] = (zz >= bead_low) & (zz <= bead_high)
        if gap < 3:
            region = 2 * gap + 1
            inward = solution["trace"][k, region, 0]
            outward = np.linalg.solve(
                solution["P"][k, region], solution["trace"][k, region, 1]
            )
            forward_spline = BSpline(
                model.Pspline.t,
                coefficients @ inward,
                model.Pspline.k,
                extrapolate=False,
            )
            backward_spline = BSpline(
                model.Pspline.t,
                coefficients @ outward,
                model.Pspline.k,
                extrapolate=False,
            )
            lower_face = 0.0 if gap == 0 else disks[gap - 1] + 0.001
            inward_fields = forward_spline(disks[gap] - zz) @ basis.T
            outward_fields = backward_spline(zz - lower_face) @ basis.T
        else:
            # Same FFT continuation as midpoint_maps, restricted to the aperture.
            inward_fields, outward_fields = [], []
            for position in zz:
                distance = position - disks[-1] - 0.001
                inward_fields.append(
                    np.fft.ifft2(incoming_fft * np.exp(1j * kz.real * distance))[
                        aperture
                    ]
                )
                outward_fields.append(
                    np.fft.ifft2(outgoing_fft * np.exp(-1j * kz * distance))[aperture]
                )
            inward_fields, outward_fields = np.asarray(inward_fields), np.asarray(
                outward_fields
            )
        waves = model.scale * np.stack((inward_fields, outward_fields), axis=-1)
        root, _, field = model.finite_bead_field(waves[None], frequency_index=k)
        integrals[select] = amplitude * field[0].sum(axis=-1) * model.area
        good = bead_valid[select]
        bead_integrals[select[good]] = (
            amplitude * root[0, good].sum(axis=-1) * model.area
        )
    return dict(
        frequency_hz=np.array([model.f[k]]),
        z_m=z,
        field_Vm=integrals[None],
        bead_field_Vm=bead_integrals[None],
        full_aperture_in_air=valid[None],
        full_bead_clearance=bead_valid[None],
        disk_nearface_z_m=disks,
        disk_slopes=slopes,
        thread_midpoint_z_mm=midpoint * 1000,
        disk_radius_m=model.radius,
    )


def measured_transverse_integrals(run_dir, frequency_GHz=20.825):
    """Same raw-data radial reconstruction as 1D, for comparison diagnostics.

    The full 20–22 GHz band determines the centres and branches. This returns
    four measurement-derived values only; they never enter the 3D likelihood.
    """
    from .sbp_analysis import load_run, analyze_line

    frequencies, lines = load_run(run_dir)
    k = int(np.argmin(abs(frequencies - frequency_GHz * 1e9)))
    if abs(frequencies[k] - frequency_GHz * 1e9) > 1:
        raise ValueError("The spider comparison requires an acquired frequency.")
    return np.array(
        [
            abs(
                analyze_line(
                    line.position_m, line.traces, frequencies, line.length_m
                ).ET[k]
            )
            for _, line in sorted(lines.items())
        ]
    )
