"""Fit disk shifts/tilts and straight thread endpoint heights to complex S22.

Geometry is shared across selected frequencies. One complex response scale per
frequency and a complex affine background per thread are eliminated analytically.
No radial rotation or Gaussian field is fed to the receive-field model.
"""

from pathlib import Path
import json
import numpy as np
from scipy.interpolate import BSpline, RegularGridInterpolator
from scipy.optimize import minimize, LinearConstraint
from .receive_field import NOM_DISKS_MM, NOM_THREADS_MM
from .tilted_disks import TiltedReceiveModel
from .exterior_field import ExteriorLineEvaluator

NOM_Z_MM = np.r_[32.011, NOM_THREADS_MM]
NAMES = (
    [f"D{j}_dz_mm" for j in (1, 2, 3)]
    + [f"D{j}_slope_{a}_mrad" for j in (1, 2, 3) for a in ("x", "y")]
    + [f"T{j}_{a}_dz_mm" for j in range(4) for a in ("A", "B")]
)


def clipped_segment(xy, radius):
    """Fractions delimiting the entire line inside an expanded disk footprint."""
    a, b = xy
    v = b - a
    roots = np.roots([v @ v, 2 * a @ v, a @ a - radius**2])
    roots = np.sort(roots[np.isreal(roots)].real)
    if len(roots) != 2 or roots[1] < 0 or roots[0] > 1:
        return []
    return [max(0.0, roots[0]), min(1.0, roots[1])]


class EndpointFitProblem:
    def __init__(
        self,
        bank,
        measurement_path,
        beam_path,
        fit_threads=(0, 1, 2, 3),
        tilt_bound=5.0,
        endpoint_bound=2.0,
        exterior_nodes=25,
        clearance_margin_mm=0.0,
    ):
        self.bank = Path(bank)
        self.model = TiltedReceiveModel(bank, beam_path=beam_path)
        self.f = self.model.f
        self.fit_threads = tuple(fit_threads)
        self.tilt_bound = tilt_bound
        self.endpoint_bound = endpoint_bound
        self.clearance_margin_mm = float(clearance_margin_mm)
        data = np.load(measurement_path)
        ix = np.searchsorted(data["frequency_hz"], self.f)
        np.testing.assert_array_equal(data["frequency_hz"][ix], self.f)
        self.eligible = data["shared_fit_mask"][ix]
        m = self.model
        interpolation = RegularGridInterpolator(
            (m.x, m.y), m.B.reshape((len(m.x), len(m.y), m.n), order="F")
        )
        self.lines = []
        for t in range(4):
            take = (data["thread_index"] == t) & ~data["transient_trace_mask"]
            xy = data["transverse_xyz_log_m"][take, :2]
            u = data["signed_line_m"][take]
            endpoints = xy[[np.argmin(u), np.argmax(u)]]
            fraction = (u - u.min()) / (u.max() - u.min())
            weights = np.column_stack([1 - fraction, fraction])
            np.testing.assert_allclose(weights @ endpoints, xy, atol=1e-14)
            y = data["corrected_S22"][take][:, ix].T
            X = np.column_stack([np.ones(len(u)), u / 0.1])
            inv = np.linalg.pinv(X)
            projection = np.eye(len(u)) - X @ inv
            self.lines.append(
                dict(
                    thread=t,
                    xy=xy,
                    u=u,
                    weights=weights,
                    endpoint_xy=endpoints,
                    basis=interpolation(xy),
                    y=y,
                    yp=y @ projection,
                    nuisance=X,
                    inverse=inv,
                    projection=projection,
                    trace_index=data["trace_index"][take],
                )
            )
        self.alpha = 4 * np.pi * 0.001465**3 * (9.23 - 1) / (9.23 + 2)
        self.kappa = 1j * 2 * np.pi * self.f * 8.8541878128e-12 * self.alpha / 4
        self.exterior = ExteriorLineEvaluator(
            m, self.lines[0]["xy"], node_count=exterior_nodes
        )
        self._geometry_constraints()
        self.calls = 0
        self.last_q = None
        self.last_result = None
        self.disk_q = None
        self.disk_cache = None

    def _geometry_constraints(self):
        rows = []
        lower = []
        meta = []
        # Conservative normal clearance factor makes these inequalities linear.
        # A straight line against a plane has affine clearance, so the two
        # clipped segment endpoints guarantee the entire continuous scan path.
        rb = (1.465 + self.clearance_margin_mm) * np.sqrt(
            1 + 2 * (self.tilt_bound * 0.001) ** 2
        )
        above = {1: 2, 2: 1, 3: 0}
        below = {0: 2, 1: 1, 2: 0}
        for t, line in enumerate(self.lines):
            for fraction in clipped_segment(line["endpoint_xy"], 0.15 + 0.001465):
                weight = np.array([1 - fraction, fraction])
                x, y = weight @ line["endpoint_xy"]
                for sign, j in [(1, above[t])] if t in above else []:
                    r = np.zeros(17)
                    r[j] = sign
                    r[3 + 2 * j : 5 + 2 * j] = sign * np.array([x, y])
                    r[9 + 2 * t : 11 + 2 * t] = -sign * weight
                    rows.append(r)
                    lower.append(rb - (NOM_DISKS_MM[j] - NOM_Z_MM[t]))
                    meta.append((t, j, "above", fraction))
                for sign, j in [(-1, below[t])] if t in below else []:
                    r = np.zeros(17)
                    r[j] = sign
                    r[3 + 2 * j : 5 + 2 * j] = sign * np.array([x, y])
                    r[9 + 2 * t : 11 + 2 * t] = -sign * weight
                    rows.append(r)
                    lower.append(rb - (NOM_Z_MM[t] - NOM_DISKS_MM[j] - 1))
                    meta.append((t, j, "below", fraction))
        self.clear_A = np.asarray(rows)
        self.clear_lower = np.asarray(lower)
        self.clear_meta = meta
        self.clear_rb = rb
        self.gap_A = np.zeros((3, 17))
        self.gap_A[:, :3] = [[1, 0, 0], [-1, 1, 0], [0, -1, 1]]
        # Interior projected propagation and exterior FFT continuation are only
        # used at positive distances from their horizontal reference surfaces.
        # These numerical-domain constraints supplement physical tilted clearance.
        rows = []
        lo = []
        hi = []
        for t in range(4):
            for endpoint in range(2):
                if t == 0:
                    row = np.zeros(17)
                    row[2] = -1
                    row[9 + endpoint] = 1
                    nominal = NOM_Z_MM[0] - NOM_DISKS_MM[2] - 1
                    rows.append(row)
                    lo.append(0.5 - nominal)
                    hi.append(15.0 - nominal)
                else:
                    j = 3 - t
                    row = np.zeros(17)
                    row[j] = 1
                    row[9 + 2 * t + endpoint] = -1
                    nominal = NOM_DISKS_MM[j] - NOM_Z_MM[t]
                    rows.append(row)
                    lo.append(0.05 - nominal)
                    hi.append(14.95 - nominal)
                    row = np.zeros(17)
                    row[9 + 2 * t + endpoint] = 1
                    nominal = NOM_Z_MM[t]
                    if j > 0:
                        row[j - 1] = -1
                        nominal -= NOM_DISKS_MM[j - 1] + 1
                    rows.append(row)
                    lo.append(
                        (1.465 + self.clearance_margin_mm if t == 3 else 0.05) - nominal
                    )
                    hi.append(14.95 - nominal)
        self.domain_A = np.asarray(rows)
        self.domain_lo = np.asarray(lo)
        self.domain_hi = np.asarray(hi)

    def constraints(self):
        return [
            LinearConstraint(self.clear_A, self.clear_lower, np.inf),
            LinearConstraint(self.gap_A, -2.0, 2.0),
            LinearConstraint(self.domain_A, self.domain_lo, self.domain_hi),
        ]

    def bounds(self):
        return (
            [(-2, 2), (-4, 4), (-6, 6)]
            + [(-self.tilt_bound, self.tilt_bound)] * 6
            + [(-self.endpoint_bound, self.endpoint_bound)] * 8
        )

    def feasible(self, q, tol=1e-6):
        bounds = np.asarray(self.bounds())
        return bool(
            np.min(self.clear_A @ q - self.clear_lower) >= -tol
            and np.max(abs(self.gap_A @ q)) <= 2 + tol
            and np.all(self.domain_A @ q >= self.domain_lo - tol)
            and np.all(self.domain_A @ q <= self.domain_hi + tol)
            and np.all(q >= bounds[:, 0] - tol)
            and np.all(q <= bounds[:, 1] + tol)
        )

    def clearance(self, q):
        vertical = self.clear_A @ q - self.clear_lower + self.clear_rb
        slopes = q[3:9].reshape(3, 2) * 0.001
        norms = np.array(
            [np.sqrt(1 + sum(slopes[j] ** 2)) for t, j, side, a in self.clear_meta]
        )
        return float(min(np.min(vertical / norms), np.min(NOM_Z_MM[3] + q[15:17])))

    def _disk_solution(self, q):
        if self.disk_q is not None and np.array_equal(q[:9], self.disk_q):
            return self.disk_cache
        m = self.model
        sol = m.solve_tilted(NOM_DISKS_MM + q[:3], q[3:9].reshape(3, 2) * 0.001)
        vectors = []
        for region in (5, 3, 1):
            vectors.extend(
                [
                    sol["trace"][:, region, 0],
                    np.linalg.solve(sol["P"][:, region], sol["trace"][:, region, 1]),
                ]
            )
        v = np.stack(vectors, axis=-1)  # frequency, mode, six waves
        # Linearity: contract each cubic propagation coefficient with the
        # boundary waves before interpolation. Exactly the original bank
        # spline, with no diagonal-mode approximation or new z interpolation.
        original = m.Pspline.c
        c = np.empty((len(original), len(self.f), m.n, 6), complex)
        for k in range(len(self.f)):
            c[:, k] = np.einsum(
                "aij,jd->aid", original[:, :, :, k], v[k], optimize=True
            )
        splines = [
            BSpline(m.Pspline.t, c[:, :, :, j], m.Pspline.k, extrapolate=False)
            for j in range(6)
        ]
        self.exterior.prepare(sol)
        exterior_state = sol
        self.disk_q = q[:9].copy()
        self.disk_cache = (sol, splines, exterior_state)
        return self.disk_cache

    def evaluate(self, q):
        q = np.asarray(q)
        if self.last_q is not None and np.array_equal(q, self.last_q):
            return self.last_result
        m = self.model
        sol, splines, exterior_state = self._disk_solution(q)
        endpoints = NOM_Z_MM[:, None] + q[9:].reshape(4, 2)
        waves = []
        templates = []
        projected = []
        zs = []
        for t, line in enumerate(self.lines):
            z = (line["weights"] @ endpoints[t]) * 0.001
            zs.append(z)
            if t == 0:
                wave = self.exterior.waves(
                    exterior_state, z - sol["disks_m"][2] - 0.001
                )
            else:
                j = t - 1
                high = 3 - t
                low = high - 1
                dr = sol["disks_m"][high] - z
                dl = z - (sol["disks_m"][low] + 0.001 if low >= 0 else 0.0)
                wave = m.scale * np.stack(
                    [
                        np.einsum(
                            "pm,pfm->fp",
                            line["basis"],
                            splines[2 * j](dr),
                            optimize=True,
                        ),
                        np.einsum(
                            "pm,pfm->fp",
                            line["basis"],
                            splines[2 * j + 1](dl),
                            optimize=True,
                        ),
                    ],
                    axis=-1,
                )
            _, Q, _ = m.finite_bead_field(wave)
            template = np.conj(self.kappa[:, None] * Q)
            waves.append(wave)
            templates.append(template)
            projected.append(template @ line["projection"])
        pp = np.concatenate([projected[t] for t in self.fit_threads], axis=1)
        yp = np.concatenate([self.lines[t]["yp"] for t in self.fit_threads], axis=1)
        norm = np.linalg.norm(yp, axis=1)
        coef = np.sum(pp.conj() * yp, axis=1) / np.sum(abs(pp) ** 2, axis=1)
        residual = (coef[:, None] * pp - yp) / norm[:, None]
        relative = np.linalg.norm(residual, axis=1)
        all_pp = np.concatenate(projected, axis=1)
        all_yp = np.concatenate([l["yp"] for l in self.lines], axis=1)
        interior_pp = np.concatenate(projected[1:], axis=1)
        interior_yp = np.concatenate([l["yp"] for l in self.lines[1:]], axis=1)
        interior_c = np.sum(interior_pp.conj() * interior_yp, axis=1) / np.sum(
            abs(interior_pp) ** 2, axis=1
        )
        interior_shared = np.linalg.norm(
            coef[:, None] * interior_pp - interior_yp, axis=1
        ) / np.linalg.norm(interior_yp, axis=1)
        interior_reprofiled = np.linalg.norm(
            interior_c[:, None] * interior_pp - interior_yp, axis=1
        ) / np.linalg.norm(interior_yp, axis=1)
        all_relative = np.linalg.norm(
            coef[:, None] * all_pp - all_yp, axis=1
        ) / np.linalg.norm(all_yp, axis=1)
        predictions = []
        backgrounds = []
        rms = []
        relative_by_thread = []
        for line, template, pt in zip(self.lines, templates, projected):
            signal = coef[:, None] * template
            background = (line["y"] - signal) @ line["inverse"].T @ line["nuisance"].T
            pred = signal + background
            predictions.append(pred)
            backgrounds.append(background)
            rms.append(np.sqrt(np.mean(abs(pred - line["y"]) ** 2, axis=1)))
            relative_by_thread.append(
                np.linalg.norm(coef[:, None] * pt - line["yp"], axis=1)
                / np.linalg.norm(line["yp"], axis=1)
            )
        result = dict(
            solution=sol,
            coefficient=coef,
            relative_residual=relative,
            pooled=float(np.sqrt(np.mean(relative[self.eligible] ** 2))),
            beta2=abs(coef) * sol["beta_received"],
            predicted=np.concatenate(predictions, axis=1),
            background=np.concatenate(backgrounds, axis=1),
            template=np.concatenate(templates, axis=1),
            raw_rms_by_thread=np.asarray(rms).T,
            relative_by_thread=np.asarray(relative_by_thread).T,
            residual_vectors=residual,
            all_relative=all_relative,
            interior_shared=interior_shared,
            interior_reprofiled=interior_reprofiled,
            interior_coefficient=interior_c,
            waves=np.concatenate(waves, axis=1),
            z_m=np.concatenate(zs),
        )
        self.calls += 1
        self.last_q = q.copy()
        self.last_result = result
        return result

    def loss(self, q):
        try:
            value = self.evaluate(q)["pooled"] ** 2
            return value if np.isfinite(value) else 1e3
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return 1e3


def summarize(p, q, details):
    r = p.evaluate(q)
    s = r["solution"]
    slopes = q[3:9].reshape(3, 2)
    ends = NOM_Z_MM[:, None] + q[9:].reshape(4, 2)
    energy = np.asarray([np.sum(abs(line["yp"]) ** 2, axis=1) for line in p.lines]).T
    energy_fraction = energy / energy.sum(axis=1, keepdims=True)
    local = {}
    for frequency in (20.795, 20.820, 20.825, 21.0):
        hits = np.flatnonzero(abs(p.f - frequency * 1e9) < 1)
        if not len(hits):
            continue
        k = hits[0]
        c = r["coefficient"][k]
        local[f"{frequency:.3f}"] = dict(
            frequency_hz=float(p.f[k]),
            beta2_conditional=float(r["beta2"][k]),
            relative_raw_residual=float(r["relative_residual"][k]),
            all_thread_residual=float(r["all_relative"][k]),
            interior_shared_scale_residual=float(r["interior_shared"][k]),
            interior_reprofiled_scale_residual=float(r["interior_reprofiled"][k]),
            relative_residual_by_thread=r["relative_by_thread"][k].tolist(),
            projected_signal_energy_fraction_by_thread=energy_fraction[k].tolist(),
            quadratic_scale_complex=[float(c.real), float(c.imag)],
            incident_power_scale=float(abs(c)),
            beta2_received_geometry_only=float(s["beta_received"][k]),
            beta2_total_geometry_only=float(s["beta_total"][k]),
        )
    pooled = lambda a: float(np.sqrt(np.mean(a[p.eligible] ** 2)))
    rec = dict(
        family="disk_tilts_threads",
        parameter_order=NAMES,
        parameters=q.tolist(),
        fit_observable="Complex corrected S22 on each fitted physical scan chord, with pointwise z along the straight thread.",
        fit_threads=list(p.fit_threads),
        held_out_threads=[t for t in range(4) if t not in p.fit_threads],
        geometry_frequency_count=int(sum(p.eligible)),
        disk_nearface_z_mm=(NOM_DISKS_MM + q[:3]).tolist(),
        disk_shifts_mm=q[:3].tolist(),
        disk_slopes_mrad=slopes.tolist(),
        disk_tilt_magnitude_mrad=(
            1000 * np.arctan(np.linalg.norm(slopes, axis=1) * 0.001)
        ).tolist(),
        gaps_mm=np.r_[
            NOM_DISKS_MM[0] + q[0], np.diff(NOM_DISKS_MM + q[:3]) - 1
        ].tolist(),
        thread_endpoint_z_mm=ends.tolist(),
        thread_endpoint_shifts_mm=q[9:].reshape(4, 2).tolist(),
        thread_endpoint_xy_m=[line["endpoint_xy"].tolist() for line in p.lines],
        thread_midpoint_z_mm=ends.mean(axis=1).tolist(),
        thread_end_to_end_delta_z_mm=np.diff(ends, axis=1).ravel().tolist(),
        thread_slope_mrad=[
            float(
                np.diff(ends[t])[0]
                / np.linalg.norm(line["endpoint_xy"][1] - line["endpoint_xy"][0])
            )
            for t, line in enumerate(p.lines)
        ],
        endpoint_definition="A=min and B=max retained signed scan coordinate; actual recorded XY endpoints, not measured mechanical supports.",
        thread_shape="Straight linear interpolation between two independently fitted endpoint z positions; no sag or xy motion.",
        pooled_relative_rms=r["pooled"],
        pooled_all_thread_residual=pooled(r["all_relative"]),
        pooled_interior_shared_scale_residual=pooled(r["interior_shared"]),
        pooled_interior_reprofiled_scale_residual=pooled(r["interior_reprofiled"]),
        residual_weighting="Equal weight per selected frequency after normalizing by the combined projected measured signal norm; within each frequency raw complex samples have equal absolute weight, so strong threads dominate. Per-thread relative residuals and measured energy fractions are also provided.",
        minimum_continuous_bead_center_to_surface_distance_mm=p.clearance(q),
        minimum_continuous_bead_surface_clearance_mm=p.clearance(q) - 1.465,
        minimum_conservative_plane_clearance_margin_mm=p.clearance(q) - 1.465,
        clearance_definition="Conservative normal distance to infinite neighboring face planes over the continuous chord clipped at radius R+bead_radius, minus bead radius. This is a lower-bound safety margin, not the true Euclidean gap to a finite disk edge. The mirror distance is exact.",
        required_bead_surface_clearance_margin_mm=p.clearance_margin_mm,
        surface_RMS_displacement_mm=float(
            np.sqrt(np.mean(q[:3] ** 2 + 0.15**2 / 4 * np.sum(slopes**2, axis=1)))
        ),
        fit_details=details,
        local=local,
        interface_projection_scheme="scattering",
        fixed_beam="Measured OMC six-profile mixture, nominal center, width, zero phase curvature; whole-plane incident-power normalization.",
        uncertainty_scope="Conditional complex-residual bootstrap with geometry and beam fixed; excludes geometry, beam, calibration and model uncertainty.",
        interior_propagation="Exact contraction and evaluation of the original cubic matrix-bank spline at each bead z.",
        exterior_propagation="Angular spectrum of the full prescribed incident beam and outgoing modal field; positive heights, incident evanescent components removed.",
        feasibility=p.feasible(q),
        source_file="src/analysis_3d.py",
    )
    arr = dict(
        frequency_hz=p.f,
        eligible=p.eligible,
        parameters=q,
        observed_S22=np.concatenate([l["y"] for l in p.lines], axis=1),
        trace_index=np.concatenate([l["trace_index"] for l in p.lines]),
        thread_index=np.concatenate(
            [np.full(len(l["u"]), l["thread"]) for l in p.lines]
        ),
        signed_line_m=np.concatenate([l["u"] for l in p.lines]),
        xy_m=np.concatenate([l["xy"] for l in p.lines]),
        z_m=r["z_m"],
        directional_waves_V_per_m=r["waves"],
        thread_endpoint_z_mm=ends,
        thread_endpoint_xy_m=np.asarray([l["endpoint_xy"] for l in p.lines]),
        predicted_S22=r["predicted"],
        fitted_background=r["background"],
        template_S22=r["template"],
        relative_residual=r["relative_residual"],
        quadratic_scale=r["coefficient"],
        amplitude=np.sqrt(abs(r["coefficient"])),
        beta2_conditional=r["beta2"],
        beta2_received_geometry_only=s["beta_received"],
        beta2_total_geometry_only=s["beta_total"],
        raw_rms_by_thread=r["raw_rms_by_thread"],
        relative_residual_by_thread=r["relative_by_thread"],
        projected_signal_energy_fraction_by_thread=energy_fraction,
        all_thread_residual=r["all_relative"],
        interior_shared_scale_residual=r["interior_shared"],
        interior_reprofiled_scale_residual=r["interior_reprofiled"],
        interior_quadratic_scale=r["interior_coefficient"],
        mirror_termination_residual=s["mirror_termination_residual"],
        reflected_power_fraction=np.einsum(
            "fi,ij,fj->f", s["reflected"].conj(), p.model.gram, s["reflected"]
        ).real,
    )
    return rec, arr


def fit_full_tilt(problem, initial=None, *, maxiter=180):
    """One constrained SLSQP start; use several explicit starts for a search.

    The default starts from the nominal parallel geometry. A supplied 17-vector
    uses mm for disk/endpoint shifts and mrad for disk slopes, in NAMES order.
    """
    initial = np.zeros(17) if initial is None else np.asarray(initial, dtype=float)
    if initial.shape != (17,) or not problem.feasible(initial):
        raise ValueError("Initial geometry must be a feasible 17-vector")
    if not np.any(problem.eligible):
        raise ValueError("No eligible geometry-fit frequencies in the bank")
    if tuple(problem.fit_threads) != (0, 1, 2, 3):
        raise ValueError(
            "This full-tilt fitting wrapper requires T0–T3; held-out endpoints are unidentifiable"
        )
    return minimize(
        problem.loss,
        initial,
        method="SLSQP",
        bounds=problem.bounds(),
        constraints=problem.constraints(),
        options={"ftol": 1e-8, "maxiter": maxiter, "eps": 1e-4},
    )


def save_result(problem, parameters, output_dir, details=None):
    """Save the same JSON/NPZ quantities as the original full-tilt analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record, arrays = summarize(problem, np.asarray(parameters, float), details or {})
    (output_dir / "disk_tilts_threads.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    np.savez_compressed(output_dir / "disk_tilts_threads.npz", **arrays)
    return record, arrays
