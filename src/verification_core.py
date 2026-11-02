"""Numerical routines for full-grid versus limited-line SBP verification.

Extracted from spider-bead-pull-ai/src/adaptive_bead_pull.py. The numerical
routines retain the original defaults and array ordering: S11 and full fields
(x, y, z, frequency); reduced fields (point, z, frequency); ET (z, frequency).
All coordinates are in metres and frequencies in Hz.
"""
from pathlib import Path
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import h5py
from scipy.integrate import trapezoid as scipy_trapezoid
from scipy.optimize import minimize
from tqdm.auto import tqdm

DEFAULT_LINES = ((0.15, 0.080), (-0.15, 0.040), (0.15, 0.), (-0.15, -0.040))
DEFAULT_N_BASELINE_MARGIN = 2
DEFAULT_T_WIDTH = 180e-9
DEFAULT_T_OFFSET = 55e-9
DEFAULT_F_IX = 250
TRAPEZOID = getattr(np, "trapezoid", None) or getattr(np, "trapz", None) or scipy_trapezoid

def apply_time_gating(
    s11,
    frequencies,
    x_pos,
    y_pos,
    z_pos,
    t_width=DEFAULT_T_WIDTH,
    t_offset=DEFAULT_T_OFFSET,
    c_const=299792458.0,
    edge_fraction=0.2,
):
    s11 = np.asarray(s11)
    frequencies = np.asarray(frequencies, dtype=float).ravel()
    x_pos = np.asarray(x_pos, dtype=float).ravel()
    y_pos = np.asarray(y_pos, dtype=float).ravel()
    z_pos = np.asarray(z_pos, dtype=float).ravel()

    Nx, Ny, Nz, Nf = s11.shape
    if frequencies.size != Nf:
        raise ValueError("frequencies length must match last axis of s11")
    if Nf < 2:
        raise ValueError("Need at least 2 frequency points for time-domain transform")

    df = float(np.mean(np.diff(frequencies)))
    if not np.isfinite(df) or df <= 0:
        raise ValueError("frequencies must be strictly increasing and finite")

    taxis = np.fft.fftshift(np.fft.fftfreq(Nf, d=df))
    s11_t = np.fft.fftshift(np.fft.ifft(s11, axis=-1), axes=-1)

    mid_x = Nx // 2
    mid_y = Ny // 2
    t_bead_z = np.empty(Nz, dtype=float)
    for z in range(Nz):
        idx = np.argmax(np.abs(s11_t[mid_x, mid_y, z, :]))
        t_bead_z[z] = taxis[idx] + float(t_offset)

    if Nz >= 2:
        p = np.polyfit(z_pos, t_bead_z, 1)
        t_bead_fit_z = np.polyval(p, z_pos)
    else:
        t_bead_fit_z = np.full_like(z_pos, t_bead_z[0], dtype=float)

    r_axis = np.maximum((t_bead_fit_z * c_const / 2.0) ** 2, 0.0)
    t_bead_xyz = (
        2.0
        / c_const
        * np.sqrt(
            r_axis[None, None, :]
            + x_pos[:, None, None] ** 2
            + y_pos[None, :, None] ** 2
        )
    )

    t = taxis[None, None, None, :]
    center = t_bead_xyz[..., None]
    start = center - t_width / 2.0
    end = center + t_width / 2.0
    edge = max(float(t_width) * float(edge_fraction), np.finfo(float).eps)

    in_core = (t >= start) & (t <= end)
    in_rise = (t >= (start - edge)) & (t < start)
    in_fall = (t > end) & (t <= (end + edge))

    rise = 0.5 * (1.0 - np.cos(np.pi * (t - (start - edge)) / edge))
    fall = 0.5 * (1.0 + np.cos(np.pi * (t - end) / edge))
    gate = np.where(in_core, 1.0, np.where(in_rise, rise, np.where(in_fall, fall, 0.0)))

    s11_t_gated = s11_t * gate
    s11_gated = np.fft.fft(np.fft.ifftshift(s11_t_gated, axes=-1), axis=-1)
    return s11_gated, gate, taxis, t_bead_xyz


def _unwrap_with_period(phi, axis, period):
    try:
        return np.unwrap(phi, axis=axis, period=period)
    except TypeError:
        scale = 2.0 * np.pi / float(period)
        return np.unwrap(phi * scale, axis=axis) / scale


def calculate_sqrt_dgamma(delta_gamma, unwrap_xy=True):
    delta_gamma = np.asarray(delta_gamma)
    sqrt_abs = np.sqrt(np.abs(delta_gamma))
    arg_wrapped = np.angle(np.sqrt(delta_gamma))

    if not unwrap_xy:
        return sqrt_abs * np.exp(1j * arg_wrapped)

    arg_unwrapped = np.array(arg_wrapped, copy=True)
    if arg_unwrapped.ndim >= 4:
        arg_unwrapped = _unwrap_with_period(arg_unwrapped, axis=0, period=np.pi)
        arg_unwrapped = _unwrap_with_period(arg_unwrapped, axis=1, period=np.pi)
    elif arg_unwrapped.ndim == 3:
        arg_unwrapped = _unwrap_with_period(arg_unwrapped, axis=0, period=np.pi)
    elif arg_unwrapped.ndim == 2:
        arg_unwrapped = _unwrap_with_period(arg_unwrapped, axis=0, period=np.pi)
        arg_unwrapped = _unwrap_with_period(arg_unwrapped, axis=1, period=np.pi)
    elif arg_unwrapped.ndim == 1:
        arg_unwrapped = _unwrap_with_period(arg_unwrapped, axis=0, period=np.pi)
    return sqrt_abs * np.exp(1j * arg_unwrapped)


def load_file(
    file_path,
    z_ixs_used=(0,),
    time_gating=True,
    t_width=DEFAULT_T_WIDTH,
    t_offset=DEFAULT_T_OFFSET,
    frequency_indices=None,
):
    file_path = Path(file_path)
    z_ixs_used = tuple(int(v) for v in z_ixs_used)
    with h5py.File(file_path, "r") as file:
        s11 = np.array(file["s11_map"])
        freq_info = json.loads(file.attrs["vna_config"])
        metadata = json.loads(file.attrs["measurement_config"])

    frequencies = np.linspace(freq_info["start"], freq_info["stop"], freq_info["points"])
    x_pos, y_pos, z_pos = [
        np.linspace(metadata["space"][ax][0], metadata["space"][ax][1], s11.shape[ax]) * 1e-3
        for ax in range(3)
    ]

    if time_gating:
        s11, _, _, _ = apply_time_gating(
            s11,
            frequencies,
            x_pos,
            y_pos,
            z_pos,
            t_width=t_width,
            t_offset=t_offset,
        )

    if frequency_indices is not None:
        frequency_indices = np.asarray(frequency_indices, dtype=int)
        s11 = s11[..., frequency_indices]
        frequencies = frequencies[frequency_indices]

    x_plt, y_plt = np.meshgrid(x_pos, y_pos, indexing="ij")
    x_pts, y_pts = x_plt.flatten(), y_plt.flatten()
    s11_z = s11[:, :, z_ixs_used, :]
    s11_flat = s11.reshape(-1, s11.shape[2], s11.shape[3])
    s11_z_flat = s11_z.reshape(-1, s11_z.shape[2], s11_z.shape[3])

    return {
        "s11": s11,
        "s11_flat": s11_flat,
        "s11_z": s11_z,
        "s11_z_flat": s11_z_flat,
        "frequencies": frequencies,
        "x_pos": x_pos,
        "x_plt": x_plt,
        "x_pts": x_pts,
        "y_pos": y_pos,
        "y_plt": y_plt,
        "y_pts": y_pts,
        "z_pos": z_pos,
        "z_ixs_used": z_ixs_used,
    }


def compute_sqrt_Dgamma_full(s11, x_pos, y_pos, z_pos=None, n_baseline_margin=DEFAULT_N_BASELINE_MARGIN):
    Nx, Ny, Nz, Nf = s11.shape
    n = min(int(n_baseline_margin), Nx // 2)
    if n <= 0:
        raise ValueError("n_baseline_margin must be >= 1 and less than half the number of x points.")

    baseline_ix = np.r_[np.arange(n), np.arange(Nx - n, Nx)]
    m = baseline_ix.size
    s_base = x_pos[baseline_ix]
    sum_s = s_base.sum()
    sum_s2 = (s_base**2).sum()
    den = m * sum_s2 - sum_s**2
    delta_gamma_full = np.empty(s11.shape, dtype=np.complex128)

    for j in range(Ny):
        y_base = s11[baseline_ix, j, :, :]
        sum_y = y_base.sum(axis=0)
        sum_sy = (s_base[:, None, None] * y_base).sum(axis=0)
        if den == 0:
            beta0 = y_base.mean(axis=0)
            beta1 = np.zeros_like(beta0)
        else:
            beta0 = (sum_s2 * sum_y - sum_s * sum_sy) / den
            beta1 = (m * sum_sy - sum_s * sum_y) / den
        base_row = beta0[None, :, :] + beta1[None, :, :] * x_pos[:, None, None]
        delta_gamma_full[:, j, :, :] = s11[:, j, :, :] - base_row
    return calculate_sqrt_dgamma(delta_gamma_full, unwrap_xy=True)


def sqrt_dgamma_to_E(sqrt_dgamma, frequencies):
    eps_b = 9.23
    delta_eps = eps_b - 1.0
    r_b = (2.93e-3) / 2.0
    V_b = 4.0 / 3.0 * np.pi * r_b**3
    alpha_0 = 3.0 * delta_eps * V_b / (eps_b + 2.0)
    P_in = 1.0
    epsilon0 = 8.85418782e-12
    nu_hz = np.asarray(frequencies, dtype=float).ravel()
    conv = np.sqrt(4.0 * P_in / (epsilon0 * alpha_0 * 1j * 2.0 * np.pi * nu_hz))
    conv_shape = (1,) * (sqrt_dgamma.ndim - 1) + (conv.size,)
    return sqrt_dgamma * conv.reshape(conv_shape)


def compute_ET_full(field_full, x_pos, y_pos, frequencies=None):
    return np.nansum(field_full, axis=(0, 1)) * np.mean(np.diff(x_pos)) * np.mean(np.diff(y_pos))


def reduce_points_to_lines(s11_z_flat, x_pts, y_pts, n_lines=4, lines=DEFAULT_LINES, tol=3e-3):
    lines_all = list(lines)
    n_lines = int(n_lines)
    if n_lines <= 0:
        lines = []
    elif n_lines == 1 and len(lines_all) >= 3:
        # For a single-line reduced set, use the central line rather than the outermost one.
        lines = [lines_all[2]]
    elif n_lines == 2 and len(lines_all) >= 3:
        # For two lines, use the two middle lines (smaller |y|) instead of the outer lines.
        lines = [lines_all[1], lines_all[2]]
    else:
        lines = list(lines_all[:n_lines])
    if not lines:
        raise ValueError("n_lines must select at least one line.")
    line_arr = np.asarray(lines, dtype=float)
    a = line_arr[:, 0]
    b = line_arr[:, 1]
    residuals = np.abs(y_pts[:, None] - (x_pts[:, None] * a[None, :] + b[None, :]))
    per_line_masks = residuals <= float(tol)
    all_ixs = per_line_masks.any(axis=1)
    per_line_ixs = per_line_masks.T
    s11_reduced = s11_z_flat[all_ixs, :, :]
    x_reduced = x_pts[all_ixs]
    y_reduced = y_pts[all_ixs]
    return s11_reduced, x_reduced, y_reduced, lines, all_ixs, per_line_ixs


def compute_sqrt_Dgamma_lines(s11_z_flat, x_pts, y_pts, lines, per_line_ixs, all_ixs, n_baseline_margin):
    sqrt_Dgamma = np.empty(s11_z_flat.shape, dtype=np.complex128)
    sqrt_Dgamma_lines = [None] * len(lines)
    baseline_lines = [None] * len(lines)
    s_sorted_lines = [None] * len(lines)
    idxs_sorted_lines = [None] * len(lines)

    for line_index, (a, b) in enumerate(lines):
        mask = per_line_ixs[line_index]
        idxs = np.flatnonzero(mask)
        if idxs.size < 2:
            continue
        u = np.array([1.0, a], dtype=float)
        u /= np.linalg.norm(u)
        s_all = u[0] * x_pts[idxs] + u[1] * y_pts[idxs]
        order = np.argsort(s_all)
        idxs_sorted = idxs[order]
        s_all_sorted = s_all[order]
        m_points = idxs_sorted.size
        n = min(n_baseline_margin, m_points // 2)
        if n == 0:
            continue

        s_base = np.concatenate([s_all_sorted[:n], s_all_sorted[-n:]])
        y_base = s11_z_flat[np.concatenate([idxs_sorted[:n], idxs_sorted[-n:]]), :, :]
        m = s_base.size
        sum_s = s_base.sum()
        sum_s2 = (s_base**2).sum()
        sum_y = y_base.sum(axis=0)
        sum_sy = (s_base[:, None, None] * y_base).sum(axis=0)
        den = m * sum_s2 - sum_s**2
        if den == 0:
            beta0 = y_base.mean(axis=0)
            beta1 = np.zeros_like(beta0)
        else:
            beta0 = (sum_s2 * sum_y - sum_s * sum_sy) / den
            beta1 = (m * sum_sy - sum_s * sum_y) / den
        base = beta0[None, :, :] + beta1[None, :, :] * s_all_sorted[:, None, None]
        delta_gamma_line = s11_z_flat[idxs_sorted, :, :] - base
        sqrt_Dgamma_line = calculate_sqrt_dgamma(delta_gamma_line, unwrap_xy=True)
        sqrt_Dgamma[idxs_sorted, :, :] = sqrt_Dgamma_line
        sqrt_Dgamma_lines[line_index] = sqrt_Dgamma_line
        baseline_lines[line_index] = base
        s_sorted_lines[line_index] = s_all_sorted
        idxs_sorted_lines[line_index] = idxs_sorted

    sqrt_Dgamma_reduced = sqrt_Dgamma[all_ixs]
    return sqrt_Dgamma_reduced, sqrt_Dgamma_lines, baseline_lines, s_sorted_lines, idxs_sorted_lines


def correct_sqrt_dgamma_reduced_with_full(
    sqrt_dgamma_full,
    sqrt_Dgamma_lines,
    idxs_sorted_lines,
    all_ixs,
    z_ixs_used,
):
    full_flat = np.asarray(sqrt_dgamma_full).reshape(-1, sqrt_dgamma_full.shape[2], sqrt_dgamma_full.shape[3])
    first_line = next((arr for arr in sqrt_Dgamma_lines if arr is not None), None)
    if first_line is None:
        raise ValueError("No line data available in sqrt_Dgamma_lines.")
    Zred, F = first_line.shape[1], first_line.shape[2]
    z_map = np.asarray(z_ixs_used, dtype=int).ravel()
    if z_map.size != Zred:
        raise ValueError(f"z_ixs_used has length {z_map.size}, expected {Zred}.")

    sqrt_Dgamma_lines_corr = [None] * len(sqrt_Dgamma_lines)
    for line_ix, idxs in enumerate(idxs_sorted_lines):
        if idxs is None or sqrt_Dgamma_lines[line_ix] is None:
            continue
        idxs = np.asarray(idxs, dtype=int)
        full_line = full_flat[idxs[:, None], z_map[None, :], :]
        red_line = np.asarray(sqrt_Dgamma_lines[line_ix])
        corr_line = np.array(red_line, copy=True)
        for z_r in range(Zred):
            for f_ix in range(F):
                fvals = full_line[:, z_r, f_ix]
                rvals = red_line[:, z_r, f_ix]
                valid = np.isfinite(fvals) & np.isfinite(rvals)
                if not np.any(valid):
                    continue
                sign = np.where(np.real(rvals[valid] * np.conj(fvals[valid])) >= 0.0, 1.0, -1.0)
                r_signed = np.array(rvals, copy=True)
                r_signed[valid] = rvals[valid] * sign
                den = np.vdot(r_signed[valid], r_signed[valid]).real
                alpha = np.vdot(r_signed[valid], fvals[valid]) / den if den > np.finfo(float).eps else 1.0 + 0.0j
                corr_line[:, z_r, f_ix] = alpha * r_signed
        sqrt_Dgamma_lines_corr[line_ix] = corr_line

    n_points = full_flat.shape[0]
    sqrt_Dgamma_corr_pts = np.full((n_points, Zred, F), np.nan + 0j, dtype=np.complex128)
    for line_ix, idxs in enumerate(idxs_sorted_lines):
        if idxs is None or sqrt_Dgamma_lines_corr[line_ix] is None:
            continue
        sqrt_Dgamma_corr_pts[np.asarray(idxs, dtype=int), :, :] = sqrt_Dgamma_lines_corr[line_ix]
    sqrt_Dgamma_reduced_corr = sqrt_Dgamma_corr_pts[all_ixs, :, :]
    return sqrt_Dgamma_reduced_corr, sqrt_Dgamma_lines_corr


def hermite_0(_, __):
    return 1.0


def hermite_1(x, s):
    return 2.0 * x / s


def hermite_2(x, s):
    return 4.0 * (x / s) ** 2 - 2.0


H_POLYN = (hermite_0, hermite_1, hermite_2)


def beam_model(x, y, x0, y0, A, s, kappa, phi0, *weights):
    degree = 1
    s_eff = max(abs(float(s)), 1e-12)
    kappa = float(kappa)
    phi0 = float(phi0)
    x_shift = x - x0
    y_shift = y - y0
    envelope = np.exp(-(x_shift**2 + y_shift**2) / (s_eff**2))

    n_modes = (degree + 1) ** 2
    if len(weights) == 0:
        w = np.zeros(n_modes, dtype=float)
        w[0] = 1.0
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size < n_modes:
            w = np.pad(w, (0, n_modes - w.size), mode="constant")
        elif w.size > n_modes:
            w = w[:n_modes]
        norm = np.linalg.norm(w)
        if not np.isfinite(norm) or norm <= 0:
            w = np.zeros(n_modes, dtype=float)
            w[0] = 1.0
        else:
            w = w / norm

    hg_sum = np.zeros_like(x_shift, dtype=float)
    k = 0
    for i in range(degree + 1):
        hx = H_POLYN[i](x_shift, s_eff)
        for j in range(degree + 1):
            hy = H_POLYN[j](y_shift, s_eff)
            hg_sum += w[k] * hx * hy
            k += 1
    phase = np.exp(1j * (kappa * (x_shift**2 + y_shift**2) + phi0))
    return A * envelope * hg_sum * phase


class BeamFit:
    def __init__(
        self,
        sqrt_Dgamma_reduced,
        frequencies,
        x_reduced,
        y_reduced,
        kappa_max=800.0,
        kappa_reg=5e-4,
        center_reg=3e-3,
        s_bounds=(8e-3, 0.12),
        alpha_max=1e4,
        model_norm_floor=1e-9,
        alpha_reg=1e-4,
        s_reg=1e-4,
    ):
        self.sqrt_Dgamma_reduced = sqrt_Dgamma_reduced
        self.frequencies = frequencies
        self.x_reduced = np.asarray(x_reduced, dtype=float).ravel()
        self.y_reduced = np.asarray(y_reduced, dtype=float).ravel()
        self.kappa_max = float(kappa_max)
        self.kappa_reg = float(kappa_reg)
        self.center_reg = float(center_reg)
        self.alpha_max = float(alpha_max)
        self.model_norm_floor = float(model_norm_floor)
        self.alpha_reg = float(alpha_reg)
        self.s_reg = float(s_reg)

        self.x_min = float(np.nanmin(self.x_reduced))
        self.x_max = float(np.nanmax(self.x_reduced))
        self.y_min = float(np.nanmin(self.y_reduced))
        self.y_max = float(np.nanmax(self.y_reduced))
        self.x_center = 0.5 * (self.x_min + self.x_max)
        self.y_center = 0.5 * (self.y_min + self.y_max)
        self.x_scale = max(0.5 * (self.x_max - self.x_min), 1e-12)
        self.y_scale = max(0.5 * (self.y_max - self.y_min), 1e-12)

        if s_bounds is None:
            dx = np.diff(np.unique(np.sort(self.x_reduced)))
            dy = np.diff(np.unique(np.sort(self.y_reduced)))
            dx_min = float(np.nanmin(dx)) if dx.size else (self.x_max - self.x_min) / 20.0
            dy_min = float(np.nanmin(dy)) if dy.size else (self.y_max - self.y_min) / 20.0
            s_min = max(0.5 * min(abs(dx_min), abs(dy_min)), 1e-6)
            s_max = max(0.75 * np.hypot(self.x_max - self.x_min, self.y_max - self.y_min), s_min * 1.1)
        else:
            s_min, s_max = map(float, s_bounds)
            if not np.isfinite(s_min) or not np.isfinite(s_max) or s_min <= 0 or s_max <= s_min:
                raise ValueError("Invalid s_bounds. Expected (s_min, s_max) with 0 < s_min < s_max.")
        self.s_min = float(s_min)
        self.s_max = float(s_max)

    def _build_bounds(self, n_params):
        bounds = [
            (self.x_min, self.x_max),
            (self.y_min, self.y_max),
            (0.0, np.inf),
            (self.s_min, self.s_max),
        ]
        if n_params >= 5:
            bounds.append((-self.kappa_max, self.kappa_max))
        if n_params >= 6:
            bounds.append((-np.pi, np.pi))
        if n_params > 6:
            bounds.extend([(-np.inf, np.inf)] * (n_params - 6))
        return bounds

    @staticmethod
    def _best_complex_scale(model, data, alpha_max=1e4):
        den = np.vdot(model, model).real
        if (not np.isfinite(den)) or (den <= np.finfo(float).eps):
            return 1.0 + 0.0j
        alpha = np.vdot(model, data) / den
        if not np.isfinite(alpha.real) or not np.isfinite(alpha.imag):
            return 1.0 + 0.0j
        aabs = np.abs(alpha)
        amax = float(max(alpha_max, 1.0))
        if aabs > amax:
            alpha = alpha * (amax / aabs)
        return alpha

    @staticmethod
    def _wrap_phase_pi(phi):
        return (phi + np.pi) % (2.0 * np.pi) - np.pi

    def _absorb_alpha_into_params(self, fit_param, z_ix, f_ix):
        p = np.asarray(fit_param, dtype=float).copy()
        x0, y0, A, s = p[:4]
        kappa = float(p[4]) if p.size >= 5 else 0.0
        phi0 = float(p[5]) if p.size >= 6 else 0.0
        model_base = beam_model(self.x_reduced, self.y_reduced, x0, y0, A, s, kappa, phi0, 1.0, 0.0, 0.0, 0.0)
        data = self.sqrt_Dgamma_reduced[:, z_ix, f_ix]
        mask = np.isfinite(data) & np.isfinite(model_base)
        alpha = self._best_complex_scale(model_base[mask], data[mask], alpha_max=self.alpha_max) if np.any(mask) else 1.0 + 0.0j
        p[2] = p[2] * np.abs(alpha)
        if p.size >= 6:
            p[5] = self._wrap_phase_pi(p[5] + float(np.angle(alpha)))
        return p, alpha

    def cost(self, fit_param, z_ix, f_ix):
        x0, y0, A, s = fit_param[:4]
        kappa = float(fit_param[4]) if len(fit_param) >= 5 else 0.0
        phi0 = float(fit_param[5]) if len(fit_param) >= 6 else 0.0
        model_base = beam_model(self.x_reduced, self.y_reduced, x0, y0, A, s, kappa, phi0, 1.0, 0.0, 0.0, 0.0)
        data = self.sqrt_Dgamma_reduced[:, z_ix, f_ix]
        mask = np.isfinite(data) & np.isfinite(model_base)
        if not np.any(mask):
            return np.inf

        model_rms = np.sqrt(np.nanmean(np.abs(model_base[mask]) ** 2))
        if (not np.isfinite(model_rms)) or (model_rms < self.model_norm_floor):
            return np.inf
        alpha = self._best_complex_scale(model_base[mask], data[mask], alpha_max=self.alpha_max)
        model_output = alpha * model_base
        mse = np.nanmean(np.abs(data - model_output) ** 2)
        reg_kappa = self.kappa_reg * (kappa / max(self.kappa_max, 1e-12)) ** 2
        reg_center = self.center_reg * (
            ((x0 - self.x_center) / self.x_scale) ** 2
            + ((y0 - self.y_center) / self.y_scale) ** 2
        )
        s_mid = 0.5 * (self.s_min + self.s_max)
        s_span = max(0.5 * (self.s_max - self.s_min), 1e-12)
        reg_s = self.s_reg * ((s - s_mid) / s_span) ** 2
        reg_alpha = self.alpha_reg * (np.abs(alpha) / max(self.alpha_max, 1.0)) ** 2
        return float(mse + reg_kappa + reg_center + reg_s + reg_alpha)

    def fit_slice(self, start, z_ix, f_ix, disp=False):
        bounds = self._build_bounds(len(start))
        start = np.asarray(start, dtype=float).copy()
        start[0] = np.clip(start[0], self.x_min, self.x_max)
        start[1] = np.clip(start[1], self.y_min, self.y_max)
        start[3] = np.clip(start[3], self.s_min, self.s_max)
        if start.size >= 5:
            start[4] = np.clip(start[4], -self.kappa_max, self.kappa_max)
        if start.size >= 6:
            start[5] = np.clip(start[5], -np.pi, np.pi)
        res = minimize(
            self.cost,
            start,
            method="L-BFGS-B",
            args=(z_ix, f_ix),
            bounds=bounds,
            options={"maxiter": 500},
        )
        res.x_raw = np.asarray(res.x, dtype=float).copy()
        p_absorbed, alpha = self._absorb_alpha_into_params(res.x, z_ix, f_ix)
        res.alpha = alpha
        res.x = p_absorbed
        if disp:
            print(f"z={z_ix}, f={f_ix}, fit={res.x}, cost={res.fun}, success={res.success}, alpha={res.alpha}")
        return res

    def fit_all(self, start=None, disp=False, n_workers=None, backend="thread"):
        if start is None:
            s0 = float(np.clip(0.05, self.s_min, self.s_max))
            start = np.array([self.x_center, self.y_center, 4e3, s0, 0.0, -np.pi / 2.0], dtype=float)
        else:
            start = np.asarray(start, dtype=float)
        Z = self.sqrt_Dgamma_reduced.shape[1]
        F = self.sqrt_Dgamma_reduced.shape[2]
        self.params = np.full((Z, F, start.size), np.nan)
        self.costs = np.full((Z, F), np.nan)
        self.success = np.zeros((Z, F), dtype=bool)
        n_tasks = Z * F
        if n_workers is None:
            n_workers = max(1, (os.cpu_count() or 2) - 1)
        n_workers = int(max(1, min(n_workers, n_tasks)))
        if backend not in ("thread", "sequential"):
            raise ValueError("backend must be one of {'thread', 'sequential'}")

        print(f"Using backend='{backend}' with n_workers={n_workers} for fitting {n_tasks} tasks.")
        progress_bar = tqdm(total=n_tasks, desc="E-field fit", unit="fit")
        if backend == "thread" and n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(self.fit_slice, start, z, f, False): (z, f)
                    for z in range(Z)
                    for f in range(F)
                }
                for future in as_completed(futures):
                    z, f = futures[future]
                    res = future.result()
                    self.params[z, f, :] = res.x
                    self.costs[z, f] = res.fun
                    self.success[z, f] = res.success
                    progress_bar.set_postfix(z=z, f=f, cost=float(res.fun))
                    progress_bar.update(1)
        else:
            for z in range(Z):
                for f in range(F):
                    res = self.fit_slice(start, z, f, False)
                    self.params[z, f, :] = res.x
                    self.costs[z, f] = res.fun
                    self.success[z, f] = res.success
                    progress_bar.set_postfix(z=z, f=f, cost=float(res.fun))
                    progress_bar.update(1)
        progress_bar.close()
        if disp and F > DEFAULT_F_IX:
            for z in range(Z):
                tqdm.write(
                    f"z={z}, f={self.frequencies[DEFAULT_F_IX] * 1e-9:.2f} GHz, "
                    f"fit={self.params[z, DEFAULT_F_IX, :]}, "
                    f"cost={self.costs[z, DEFAULT_F_IX]}, success={self.success[z, DEFAULT_F_IX]}"
                )
        return self.params, self.costs, self.success


def compute_ET_reduced(
    params,
    x_pos,
    y_pos,
    beam_model_func,
    frequencies,
    fit_costs=None,
    fit_success=None,
    cost_quantile=None,
    max_abs_A=None,
    max_abs_kappa=None,
):
    n1 = 50
    n2 = 50
    x_lin = np.linspace(np.min(x_pos), np.max(x_pos), n1)
    y_lin = np.linspace(np.min(y_pos), np.max(y_pos), n2)
    X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")
    Z, Nf = params.shape[0], params.shape[1]
    ET_reduced = np.full((Z, Nf), np.nan + 0j, dtype=np.complex128)
    good_mask = np.ones((Z, Nf), dtype=bool)
    if fit_success is not None:
        fit_success = np.asarray(fit_success, dtype=bool)
        if fit_success.shape == (Z, Nf):
            good_mask &= fit_success
    if (fit_costs is not None) and (cost_quantile is not None):
        fit_costs = np.asarray(fit_costs, dtype=float)
        if fit_costs.shape == (Z, Nf):
            finite_costs = fit_costs[np.isfinite(fit_costs)]
            if finite_costs.size > 0:
                cost_thr = np.nanquantile(finite_costs, float(cost_quantile))
                good_mask &= np.isfinite(fit_costs) & (fit_costs <= cost_thr)

    for f_ix in range(Nf):
        for z_ix in range(Z):
            if not good_mask[z_ix, f_ix]:
                continue
            p = params[z_ix, f_ix, :]
            if p.size < 4 or not np.all(np.isfinite(p[:4])):
                continue
            x0, y0, A, s = map(float, p[:4])
            kappa = float(p[4]) if p.size >= 5 and np.isfinite(p[4]) else 0.0
            phi0 = float(p[5]) if p.size >= 6 and np.isfinite(p[5]) else 0.0
            if (s <= 0.0) or (not np.isfinite(s)):
                continue
            if (max_abs_A is not None) and (np.abs(A) > float(max_abs_A)):
                continue
            if (max_abs_kappa is not None) and (np.abs(kappa) > float(max_abs_kappa)):
                continue
            E_field = beam_model_func(X, Y, x0, y0, A, s, kappa, phi0, 1.0, 0.0, 0.0, 0.0)
            if not np.all(np.isfinite(E_field)):
                continue
            ET_reduced[z_ix, f_ix] = TRAPEZOID(TRAPEZOID(E_field, y_lin, axis=1), x_lin, axis=0)
    return ET_reduced


def compute_reduced_products(
    file_path,
    line_count,
    z_ixs_used,
    n_baseline_margin=DEFAULT_N_BASELINE_MARGIN,
    time_gating=True,
    t_width=DEFAULT_T_WIDTH,
    t_offset=DEFAULT_T_OFFSET,
    frequency_indices=None,
    reference_alignment=True,
):
    data = load_file(
        file_path,
        z_ixs_used=z_ixs_used,
        time_gating=time_gating,
        t_width=t_width,
        t_offset=t_offset,
        frequency_indices=frequency_indices,
    )
    s11_reduced, x_reduced, y_reduced, lines, all_ixs, per_line_ixs = reduce_points_to_lines(
        data["s11_z_flat"], data["x_pts"], data["y_pts"], n_lines=line_count
    )
    sqrt_Dgamma_reduced, sqrt_Dgamma_lines, baseline_lines, s_sorted_lines, idxs_sorted_lines = compute_sqrt_Dgamma_lines(
        data["s11_z_flat"],
        data["x_pts"],
        data["y_pts"],
        lines,
        per_line_ixs,
        all_ixs,
        n_baseline_margin,
    )
    sqrt_dgamma_full = compute_sqrt_Dgamma_full(data["s11"], data["x_pos"], data["y_pos"], data["z_pos"], n_baseline_margin)
    if reference_alignment:
        sqrt_Dgamma_reduced_corr, sqrt_Dgamma_lines_corr = correct_sqrt_dgamma_reduced_with_full(
            sqrt_dgamma_full, sqrt_Dgamma_lines, idxs_sorted_lines,
            all_ixs, data["z_ixs_used"],
        )
    else:
        sqrt_Dgamma_reduced_corr = sqrt_Dgamma_reduced
        sqrt_Dgamma_lines_corr = sqrt_Dgamma_lines
    E_reduced = sqrt_dgamma_to_E(sqrt_Dgamma_reduced_corr, data["frequencies"])
    E_full = sqrt_dgamma_to_E(sqrt_dgamma_full, data["frequencies"])
    data.update(
        {
            "s11_reduced": s11_reduced,
            "x_reduced": x_reduced,
            "y_reduced": y_reduced,
            "lines": lines,
            "all_ixs": all_ixs,
            "per_line_ixs": per_line_ixs,
            "sqrt_Dgamma_reduced": sqrt_Dgamma_reduced,
            "sqrt_Dgamma_lines": sqrt_Dgamma_lines,
            "sqrt_Dgamma_lines_corr": sqrt_Dgamma_lines_corr,
            "baseline_lines": baseline_lines,
            "s_sorted_lines": s_sorted_lines,
            "idxs_sorted_lines": idxs_sorted_lines,
            "sqrt_dgamma_full": sqrt_dgamma_full,
            "sqrt_Dgamma_reduced_corr": sqrt_Dgamma_reduced_corr,
            "E_reduced": E_reduced,
            "E_full": E_full,
        }
    )
    return data
