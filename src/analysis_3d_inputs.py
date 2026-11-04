"""Prepare measured complex S22 and the external measured magnitude beam.

The Gaussian line fits only select geometry-fit frequencies and provide
transverse diagnostics. They are never rotated into the 3D likelihood.
Measured S22 is retained unchanged within the analysis frequency band.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import iv
from numpy.polynomial.legendre import leggauss
from .analysis_band import frequency_mask, load_raw_traces, require_analysis_band

EPS0 = 8.8541878128e-12
TRANSIENT_IDS = [73, 74, 192, 193, 194, 195]


def projection(u, y, p):
    """Variable projection of complex baseline and amplitude (p real)."""
    center, width, curvature = p
    g = np.exp((-2 / width**2 + 1j * curvature) * (u - center) ** 2)
    design = np.column_stack([np.ones(len(u)), u, g])
    coeff = np.linalg.lstsq(design, y, rcond=None)[0]
    return design @ coeff, coeff


def fit_profile(u, y, start=None):
    """u=s/(0.1 m); field Gaussian 1/e radius = 0.1*width m."""
    y0 = np.mean(y)
    yy = y - y0
    scale = max(np.sqrt(np.mean(abs(yy) ** 2)), 1e-8)

    def resid(p):
        r = (projection(u, yy, p)[0] - yy) / scale
        return np.r_[r.real, r.imag]

    p0 = np.array([0.15, 0.9, 0.0]) if start is None else start
    fit = least_squares(
        resid,
        p0,
        bounds=([-1.1, 0.2, -8], [1.1, 2.2, 8]),
        max_nfev=70,
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
    )
    pred, coeff = projection(u, yy, fit.x)
    # One independent default restart prevents frequency continuation trapping.
    if start is not None and np.linalg.norm(fit.fun) > 0.2:
        alt = least_squares(
            resid,
            [0.15, 0.9, 0],
            bounds=([-1.1, 0.2, -8], [1.1, 2.2, 8]),
            max_nfev=50,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        if np.linalg.norm(alt.fun) < np.linalg.norm(fit.fun):
            fit = alt
            pred, coeff = projection(u, yy, fit.x)
    coeff[0] += y0
    pred += y0
    rms = np.sqrt(np.mean(abs(pred - y) ** 2))
    # Residual based quality measure includes model discrepancy, not just VNA noise.
    fraction = rms / max(abs(coeff[-1]), 1e-10)
    at_bound = bool(
        np.any(fit.x - [-1.1, 0.2, -8] < 1e-3)
        or np.any(np.array([1.1, 2.2, 8]) - fit.x < 1e-3)
    )
    return fit.x, coeff, rms, fraction, fit.success and not at_bound


def transverse_integral(
    amplitude, p, frequency, radius=0.15, bead_radius=0.001465, bead_epsilon=9.23
):
    """Conditional circular symmetric integral, including center offset.

    sqrt(DeltaGamma/K) gives Econv, not deconvolved E. A separate standing
    wave model using BOTH Mie coefficients is needed to recover E.
    """
    alpha = 4 * np.pi * bead_radius**3 * (bead_epsilon - 1) / (bead_epsilon + 2)
    kappa = 1j * 2 * np.pi * frequency * EPS0 * alpha / 4  # Pin=1 W
    center, width, curvature = p
    q = (1 / width**2 - 0.5j * curvature) / 0.1**2
    center *= 0.1
    nodes, weights = leggauss(64)
    r = (nodes + 1) * radius / 2
    area_integral = (
        2
        * np.pi
        * np.exp(-q * center**2)
        * np.sum(
            weights * radius / 2 * r * np.exp(-q * r**2) * iv(0, 2 * q * r * center)
        )
    )
    return np.sqrt(amplitude / kappa) * area_integral


def prepare_measurement(run_dir, output_file):
    """Read measured S22 and calculate profile diagnostics and frequency quality.

    Only 20–22 GHz enters any calculation. No additive template is subtracted
    from the measured reflection. Six recorded transition samples are omitted
    from profile fitting and the later 3D likelihood. Complex affine backgrounds
    remain parameters of the downstream fits.
    """
    run_dir = Path(run_dir)
    frequency, raw = load_raw_traces(run_dir)
    log = pd.read_csv(run_dir / "scan_log.csv")
    ids = np.array([json.loads(v)["trace_index"] for v in log.measurement_json])
    log = log.assign(trace_index=ids).sort_values("trace_index").reset_index(drop=True)
    np.testing.assert_array_equal(log.trace_index, np.arange(len(raw)))
    thread = log.sub_thread_index.to_numpy(dtype=int)
    xyz = log[["position_x_m", "position_y_m", "position_z_m"]].to_numpy()
    coord = log.position_m.to_numpy().copy()
    for t in range(4):
        q = np.flatnonzero(thread == t)
        coord[q] -= 0.5 * (coord[q[0]] + coord[q[-1]])
    transient = np.isin(log.trace_index, TRANSIENT_IDS)
    nf = len(frequency)
    params = np.empty((4, nf, 3))
    fits = np.empty((4, nf, 3), complex)
    quality = np.empty((nf, 4), bool)
    et = np.empty((nf, 4))
    for t in range(4):
        q = np.flatnonzero((thread == t) & ~transient)
        u = coord[q] / 0.1
        previous = None
        for k, f in enumerate(frequency):
            p, c, rms, fraction, ok = fit_profile(u, raw[q, k], previous)
            previous = p if ok else None
            params[t, k] = p
            fits[t, k] = c
            quality[k, t] = ok and fraction < 0.20
            et[k, t] = abs(transverse_integral(c[2], p, f))
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        frequency_hz=frequency,
        ET_magnitude_Vm=et,
        profile_quality=quality,
        shared_fit_mask=quality[:, 1:].all(1),
        gaussian_parameters=params,
        gaussian_coefficients=fits,
        observed_S22=raw,
        reflection_preprocessing="none",
        trace_index=log.trace_index.to_numpy(),
        thread_index=thread,
        transverse_xyz_log_m=xyz,
        signed_line_m=coord,
        transient_trace_mask=transient,
    )
    return output_file


def load_measurement(path):
    """Load uncorrected measurements; reject old template-corrected products."""
    with np.load(path, allow_pickle=False) as archive:
        if "observed_S22" not in archive or "reflection_preprocessing" not in archive:
            raise ValueError(
                "This 3D measurement input predates removal of the additive template "
                "correction. Regenerate it with prepare_measurement(run_dir, output_file)."
            )
        if archive["reflection_preprocessing"].item() != "none":
            raise ValueError("3D analysis requires measured S22 without template correction.")
        return dict(archive)


def prepare_beam(csv_file, output_file, frequencies_hz):
    """Interpolate the six measured Gaussian magnitude profiles, without phase.

    Samples outside 20–22 GHz are discarded before interpolation.
    CSV columns: frequency_Hz, scan_z_m, amplitude, field_1/e_radius_m, x0_m, y0_m.
    Scan z has no registered mapping to the spider mirror. The receive model
    centres each fitted profile on the disk axis, combines them at its outer
    reference face and normalizes their whole-plane power to 1 W. Stored fit
    parameters retain the measured centres. No free-space transport from the
    scan z is inferred.
    """
    table = np.loadtxt(csv_file, delimiter=",", comments="#")
    table = table[frequency_mask(table[:, 0])]
    f = np.asarray(frequencies_hz)
    require_analysis_band(f, "3D incident-beam frequencies")
    z = np.unique(table[:, 1])
    for zz in z:
        rows = table[table[:, 1] == zz]
        if f.min() < rows[:, 0].min() or f.max() > rows[:, 0].max():
            raise ValueError("Requested frequencies exceed measured beam range")
    p = np.asarray(
        [
            [
                [np.interp(ff, b[:, 0], b[:, j]) for j in range(2, 6)]
                for zz in z
                for b in [table[table[:, 1] == zz]]
            ]
            for ff in f
        ]
    )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, frequencies_hz=f, scan_z_m=z, slice_parameters=p)
    return output_file
