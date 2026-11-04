"""Tilted scalar modal interfaces, projected in scattering form.

Measured face heights and fitted rigid slopes shift each interface phase.
This is a scalar diffraction model, not a vector Maxwell solver.
"""

import math
import numpy as np
from .receive_field import ReceiveModel, C
from .disk_surfaces import DiskSurfaces
from .surface_moments import load_surface_moments, TOTAL_ORDER


class TiltedReceiveModel(ReceiveModel):
    def __init__(self, bank_path, order=36, projection_scheme="scattering", measured_surfaces=True, **kwargs):
        super().__init__(bank_path, **kwargs)
        self.order = order
        self.projection_scheme = projection_scheme
        self._prepare_moments()
        self.surfaces = DiskSurfaces() if measured_surfaces else None
        if self.surfaces is not None:
            self.surface_powers, self.surface_moments, self.surface_sources = load_surface_moments(self, self.surfaces)
            xy = np.column_stack([self.inside_x, self.inside_y])
            self.surface_height_max = [
                np.max(abs(self.surfaces.evaluate(face // 2, face % 2, xy)))
                for face in range(6)
            ]
        self._fixed_field = self.beam_field()
        self._fixed_incident = np.einsum(
            "pm,fp->fm", self.B.conj(), self._fixed_field, optimize=True
        )
        self._fixed_reciprocal = self._fixed_field @ self.B

    def _prepare_moments(self):
        # Exact sampled-grid moments of exp(i*k*(sx*x+sy*y)). Projection
        # uses the dual basis so that multiplication by 1 is the identity.
        x = self.inside_x / self.radius
        y = self.inside_y / self.radius
        B = self.Binside
        dual = np.linalg.solve(self.gram, B.conj().T)
        pairs = [
            (p, q) for n in range(self.order + 1) for p in range(n + 1) for q in [n - p]
        ]
        powers_x = np.array(
            [x**p / float(math.factorial(p)) for p in range(self.order + 1)]
        )
        powers_y = np.array(
            [y**q / float(math.factorial(q)) for q in range(self.order + 1)]
        )
        moments = []
        source = []
        uniform = self.uniform[self.aperture]
        for p, q in pairs:
            weight = powers_x[p] * powers_y[q]
            moments.append((dual * weight[None, :]) @ B)
            # Retain the original source's B^H projection convention.
            source.append(B.conj().T @ (weight * uniform))
        self.powers = np.asarray(pairs)
        self.moments = np.asarray(moments)
        self.moments[0] = np.eye(self.n)
        self.source_moments = np.asarray(source)
        self.source_moments[0] = self.axion_initial

    def _degree(self, k, slope, face=None):
        k = np.asarray(k)
        height = 0 if face is None or self.surfaces is None else self.surface_height_max[face]
        phase = float(np.max(abs(k))) * (self.radius * np.linalg.norm(slope) + height)
        degree = min(self.order, max(12, int(np.ceil(3 * phase + 16))))
        if face is not None and self.surfaces is not None and degree > TOTAL_ORDER:
            raise ValueError("Measured-face phase exceeds its validated expansion range")
        return degree

    def _grouped_series(self, slope):
        # Group the identical bivariate polynomial by total degree once per
        # disk geometry, then evaluate its short univariate series at every
        # frequency/material phase. No approximation is added by grouping.
        key = tuple(slope)
        if not hasattr(self, "_series_cache"):
            self._series_cache = {}
        if key not in self._series_cache:
            matrices = []
            sources = []
            for degree in range(self.order + 1):
                lo = degree * (degree + 1) // 2
                hi = (degree + 1) * (degree + 2) // 2
                pq = self.powers[lo:hi]
                weight = slope[0] ** pq[:, 0] * slope[1] ** pq[:, 1]
                matrices.append(
                    (weight @ self.moments[lo:hi].reshape(degree + 1, -1)).reshape(
                        self.n, self.n
                    )
                )
                sources.append(weight @ self.source_moments[lo:hi])
            self._series_cache[key] = (np.asarray(matrices), np.asarray(sources))
        return self._series_cache[key]

    def _face_series(self, slope, face):
        if face is None or self.surfaces is None:
            return self._grouped_series(slope)
        key = (face, *slope)
        if not hasattr(self, "_surface_series_cache"):
            self._surface_series_cache = {}
        if key not in self._surface_series_cache:
            matrices, sources = self._grouped_series(slope)
            matrices = matrices[:TOTAL_ORDER + 1].copy()
            sources = sources[:TOTAL_ORDER + 1].copy()
            degrees = self.surface_powers.sum(axis=1)
            for n in range(1, TOTAL_ORDER + 1):
                take = np.flatnonzero(degrees == n)
                p, q, _ = self.surface_powers[take].T
                weight = slope[0]**p * slope[1]**q
                lo, hi = take[0], take[-1] + 1
                matrices[n] += (weight @ self.surface_moments[face][lo:hi].reshape(len(take), -1)).reshape(self.n, self.n)
                sources[n] += weight @ self.surface_sources[face][lo:hi]
            self._surface_series_cache[key] = matrices, sources
        return self._surface_series_cache[key]

    def phase_matrix(self, k, slope, face=None):
        """Project exp(+i*k*h), including the measured height of this face."""
        if not np.any(slope) and (face is None or self.surfaces is None):
            return np.broadcast_to(np.eye(self.n), np.shape(k) + (self.n, self.n))
        degree = self._degree(k, slope, face)
        matrices = self._face_series(slope, face)[0][: degree + 1]
        weights = (1j * np.asarray(k)[..., None] * self.radius) ** np.arange(degree + 1)
        return (weights @ matrices.reshape(degree + 1, -1)).reshape(
            np.shape(k) + (self.n, self.n)
        )

    def phase_source(self, k, slope, face=None):
        degree = self._degree(k, slope, face)
        source = self._face_series(slope, face)[1][: degree + 1]
        weights = (1j * np.asarray(k)[..., None] * self.radius) ** np.arange(degree + 1)
        return weights @ source

    def direct_phase_matrix(self, k, slope, face=None):
        """Independent direct-grid check of the polynomial evaluation."""
        h = slope[0] * self.inside_x + slope[1] * self.inside_y
        if face is not None and self.surfaces is not None:
            h += self.surfaces.evaluate(face // 2, face % 2, np.column_stack([self.inside_x, self.inside_y]))
        phase = np.exp(1j * k * h)
        return np.linalg.solve(
            self.gram, self.Binside.conj().T @ (phase[:, None] * self.Binside)
        )

    def solve_tilted(
        self,
        disks_mm,
        slopes,
        curvature=0.0,
        width_scale=1.0,
        center_scale=0.0,
        center_shift_m=(0.0, 0.0),
    ):
        """slopes[disk,(x,y)] are dz/dx and dz/dy, dimensionless.

        The physical interface scattering blocks are transported to their
        horizontal reference planes, projected into the modal basis, then
        converted to transfer form. Projecting transfer blocks directly can
        violate passivity after mode truncation and is diagnostic-only.
        Surface direction phases are exp(+ik*h), exp(-ik*h); source phases
        are shifted consistently.
        Both faces share the fitted rigid slope; their measured departures
        change the local thickness around the nominal 1 mm reference value.
        """
        slopes = np.asarray(slopes, float).reshape(3, 2)
        self._series_cache = {}
        self._surface_series_cache = {}
        if (
            curvature == 0
            and width_scale == 1
            and center_scale == 0
            and tuple(center_shift_m) == (0.0, 0.0)
        ):
            field, incident, reciprocal = (
                self._fixed_field,
                self._fixed_incident,
                self._fixed_reciprocal,
            )
        else:
            field = self.beam_field(
                curvature, width_scale, center_scale, "omc", center_shift_m
            )
            incident = np.einsum("pm,fp->fm", self.B.conj(), field, optimize=True)
            reciprocal = field @ self.B
        P, eps, gaps = self.propagation_matrices(disks_mm)
        nf, n = len(self.f), self.n
        T = np.broadcast_to(np.eye(2 * n, dtype=complex), (nf, 2 * n, 2 * n)).copy()
        source = np.zeros((nf, n), complex)
        refr = np.sqrt(eps)
        k0 = 2 * np.pi * self.f / C
        interfaces = []
        phase_cache = {}
        for r in range(1, 8):
            slope = np.zeros(2) if r == 1 else slopes[(r - 2) // 2]
            face = None if r == 1 else r - 2
            has_height = np.any(slope) or (face is not None and self.surfaces is not None)
            kl, kr = k0 * refr[r - 1], k0 * refr[r]
            a = (refr[r - 1] + refr[r]) / (2 * refr[r - 1])
            b = (refr[r - 1] - refr[r]) / (2 * refr[r - 1])

            def phase(factor):
                key = (face, complex(factor))
                if key not in phase_cache:
                    phase_cache[key] = self.phase_matrix(k0 * factor, slope, face)
                return phase_cache[key]

            factor = (1 / eps[r - 1] - 1 / eps[r]) / 2
            if not has_height:
                sm = sp = np.broadcast_to(self.axion_initial, (nf, n))
            else:
                sm, sp = self.phase_source(-kl, slope, face), self.phase_source(kl, slope, face)
            if getattr(
                self, "projection_scheme", "scattering"
            ) == "scattering" and has_height:
                # Outputs (inward left, outward right), inputs (outward left,
                # inward right). The lossless physical two-port is unitary
                # in the admittance-weighted norm; projection is contractive.
                S11 = (b / a) * phase(-2 * refr[r - 1])
                S22 = (-b / a) * phase(2 * refr[r])
                S12 = ((a * a - b * b) / a) * phase(refr[r] - refr[r - 1])
                S21 = (1 / a) * phase(refr[r] - refr[r - 1])
                identity = np.broadcast_to(np.eye(n), (nf, n, n))
                F11 = np.linalg.solve(S21, identity)
                F10 = -F11 @ S22
                F01 = S11 @ F11
                F00 = S12 + S11 @ F10
                source_left = (1 - b / a) * sm
                source_right = (-1 / a) * self.phase_source(kr, slope, face)
                sp = -np.einsum("fij,fj->fi", F11, source_right)
                sm = source_left + np.einsum("fij,fj->fi", S11, sp)
            else:
                F00 = a * phase(refr[r] - refr[r - 1])
                F01 = b * phase(-refr[r] - refr[r - 1])
                F10 = b * phase(refr[r] + refr[r - 1])
                F11 = a * phase(refr[r - 1] - refr[r])
            interfaces.append((F00, F01, F10, F11))
            source += factor * (
                np.einsum("fij,fj->fi", T[:, n:, :n], sm)
                + np.einsum("fij,fj->fi", T[:, n:, n:], sp)
            )
            left = T[:, :, :n] @ F00 + T[:, :, n:] @ F10
            right = T[:, :, :n] @ F01 + T[:, :, n:] @ F11
            T[:, :, :n] = left @ P[:, r]
            T[:, :, n:] = np.linalg.solve(
                P[:, r].swapaxes(-1, -2), right.swapaxes(-1, -2)
            ).swapaxes(-1, -2)
        emitted = -np.linalg.solve(T[:, n:, n:], source)
        reflected = -np.linalg.solve(
            T[:, n:, n:], np.einsum("fij,fj->fi", T[:, n:, :n], incident)
        )
        trace = np.zeros((nf, 8, 2, n), complex)
        trace[:, 7, 0], trace[:, 7, 1] = incident, reflected
        for r in range(7, 0, -1):
            inward = np.einsum("fij,fj->fi", P[:, r], trace[:, r, 0])
            outward = np.linalg.solve(P[:, r], trace[:, r, 1])
            F00, F01, F10, F11 = interfaces[r - 1]
            trace[:, r - 1, 0] = np.einsum("fij,fj->fi", F00, inward) + np.einsum(
                "fij,fj->fi", F01, outward
            )
            trace[:, r - 1, 1] = np.einsum("fij,fj->fi", F10, inward) + np.einsum(
                "fij,fj->fi", F11, outward
            )
        overlap = np.sum(reciprocal * emitted, axis=1)
        return dict(
            P=P,
            trace=trace,
            emitted=emitted,
            reflected=reflected,
            incident=incident,
            input_field=field,
            beta_received=abs(overlap) ** 2 * self.aperture_area_ratio,
            beta_total=np.real(
                np.einsum("fi,ij,fj->f", emitted.conj(), self.gram, emitted)
            )
            * self.aperture_area_ratio,
            gaps=gaps,
            disks_m=np.asarray(disks_mm) * 1e-3,
            slopes=slopes,
            measured_disk_surfaces=self.surfaces is not None,
            mirror_termination_residual=np.linalg.norm(trace[:, 0, 1], axis=1)
            / np.linalg.norm(incident, axis=1),
        )
