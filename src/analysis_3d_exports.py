"""Conditional bands and model electric-field maps; no optimization."""

from pathlib import Path
import json
import numpy as np
from .analysis_3d import NOM_Z_MM

Q = np.array([0.025, 0.16, 0.5, 0.84, 0.975])


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
