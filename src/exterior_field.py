"""Exterior receive fields at a different z for every retained T0 sample.

The reference is the horizontal virtual outer face used by TiltedReceiveModel.
Outgoing fields propagate forward; the full prescribed incident beam is
continued backward using propagating components only. FFT-grid sampling uses
the same bilinear XY operator as the existing T0 holdout implementation.

Distance interpolation is Chebyshev polynomial interpolation of the complex
directional fields, before the quadratic finite-bead operator is applied.
No model parameters, modal weights, or independent T0 signal scale are fitted
here. Increasing node_count is a numerical convergence check.
"""

import numpy as np

C = 299792458.0


class BilinearChord:
    """Precomputed equivalent of RegularGridInterpolator((x,y), field)(xy)."""

    def __init__(self, x, y, xy):
        x, y, xy = np.asarray(x), np.asarray(y), np.asarray(xy)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (sample,2)")
        if np.any(xy < [x[0], y[0]]) or np.any(xy > [x[-1], y[-1]]):
            raise ValueError("Chord point outside the FFT spatial grid")
        self.xy = xy.copy()
        self.i = np.clip(np.searchsorted(x, xy[:, 0]) - 1, 0, len(x) - 2)
        self.j = np.clip(np.searchsorted(y, xy[:, 1]) - 1, 0, len(y) - 2)
        tx = (xy[:, 0] - x[self.i]) / (x[self.i + 1] - x[self.i])
        ty = (xy[:, 1] - y[self.j]) / (y[self.j + 1] - y[self.j])
        self.weights = np.array(
            [(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty]
        )

    def __call__(self, field):
        i, j, w = self.i, self.j, self.weights
        return (
            w[0] * field[..., i, j]
            + w[1] * field[..., i + 1, j]
            + w[2] * field[..., i, j + 1]
            + w[3] * field[..., i + 1, j + 1]
        )


class ExteriorLineEvaluator:
    """Reusable variable-height T0 evaluator.

    evaluator = ExteriorLineEvaluator(model, line_xy, node_count=17)
    waves = evaluator.waves(solution, distances_m)  # (f,point,2), V/m

    distances_m is z_bead - (central D3 near-face z + 1 mm). It has one
    element per retained line point. Direction order is inward, outward, as
    required by model.finite_bead_field. Thread-only parameter changes reuse
    prepared node fields when the same solution object is passed.
    """

    def __init__(
        self,
        model,
        xy,
        node_count=17,
        distance_bounds_m=(0.0005, 0.015),
        cache_propagators=True,
    ):
        self.model = model
        self.sample = BilinearChord(model.x, model.y, xy)
        self.node_count = int(node_count)
        if self.node_count < 3:
            raise ValueError("At least three Chebyshev nodes are required")
        self.minimum_distance_m, self.maximum_distance_m = map(float, distance_bounds_m)
        if not 0 <= self.minimum_distance_m < self.maximum_distance_m:
            raise ValueError("Require 0 <= minimum_distance < maximum_distance")
        j = np.arange(self.node_count)
        self.nodes_m = (self.minimum_distance_m + self.maximum_distance_m) / 2 - (
            self.maximum_distance_m - self.minimum_distance_m
        ) / 2 * np.cos(np.pi * j / (self.node_count - 1))
        self.barycentric = (-1.0) ** j
        self.barycentric[[0, -1]] *= 0.5
        q = (
            (2 * np.pi * model.f[:, None, None] / C) ** 2
            - model.arr["kx_fft_per_m"][None, :, None] ** 2
            - model.arr["ky_fft_per_m"][None, None, :] ** 2
        )
        self.propagating = q >= 0
        self.kz = np.sqrt(q.astype(complex)).conj()
        self.outgoing_propagators = None
        if cache_propagators:
            self.outgoing_propagators = np.exp(
                -1j * self.nodes_m[:, None, None, None] * self.kz[None]
            )
        self.incoming_values = None
        self.outgoing_values = None
        self._incoming_field = None
        self._prepared_solution = None

    def interpolation_weights(self, distances_m, derivative=False):
        d = np.asarray(distances_m, dtype=float)
        if d.shape != (len(self.sample.xy),):
            raise ValueError("One distance is required for each chord sample")
        tol = 2e-14
        if np.any(d < self.minimum_distance_m - tol) or np.any(
            d > self.maximum_distance_m + tol
        ):
            raise ValueError(
                "Exterior distance is outside the validated interpolation interval"
            )
        d = np.clip(d, self.minimum_distance_m, self.maximum_distance_m)
        delta = d[None, :] - self.nodes_m[:, None]
        near = abs(delta) < 2e-17
        safe = np.where(near, 1.0, delta)
        raw = self.barycentric[:, None] / safe
        weights = raw / raw.sum(axis=0)
        hits = np.flatnonzero(np.any(near, axis=0))
        if not derivative:
            for p in hits:
                weights[:, p] = 0.0
                weights[np.flatnonzero(near[:, p])[0], p] = 1.0
            return weights
        # d L_j(x)/dx = L_j [sum_k L_k/(x-x_k) - 1/(x-x_j)].
        slope = weights * ((weights / safe).sum(axis=0)[None, :] - 1 / safe)
        for p in hits:
            k = np.flatnonzero(near[:, p])[0]
            others = np.arange(self.node_count) != k
            slope[others, p] = self.barycentric[others] / (
                self.barycentric[k] * (self.nodes_m[k] - self.nodes_m[others])
            )
            slope[k, p] = -slope[others, p].sum()
        return slope

    def _outgoing_factor(self, node):
        if self.outgoing_propagators is not None:
            return self.outgoing_propagators[node]
        return np.exp(-1j * self.kz * self.nodes_m[node])

    def _node_values(self, field, outward):
        transformed = np.fft.fft2(field, axes=(-2, -1))
        if not outward:
            transformed = np.where(self.propagating, transformed, 0.0)
        values = []
        for j, d in enumerate(self.nodes_m):
            if outward:
                factor = self._outgoing_factor(j)
            else:
                # Use real kz on propagating components. No exp(+kappa*d)
                # is ever formed, even before the evanescent mask is applied.
                factor = np.exp(1j * self.kz.real * d)
            propagated = np.fft.ifft2(transformed * factor, axes=(-2, -1))
            values.append(self.sample(propagated))
        return np.asarray(values)  # (node,frequency,sample)

    def prepare(self, solution):
        if solution is self._prepared_solution:
            return
        m = self.model
        shape = (len(m.f), len(m.x), len(m.y))
        input_field = solution["input_field"]
        if input_field is not self._incoming_field:
            front = input_field.reshape(shape, order="F")
            self.incoming_values = self._node_values(front, outward=False)
            self._incoming_field = input_field
        outward_coefficients = np.linalg.solve(
            solution["P"][:, 7], solution["reflected"]
        )
        outgoing_front = np.einsum(
            "pm,fm->fp", m.B, outward_coefficients, optimize=True
        ).reshape(shape, order="F")
        self.outgoing_values = self._node_values(outgoing_front, outward=True)
        self._prepared_solution = solution

    def waves(self, solution, distances_m, *, derivative=False, scaled=True):
        """Return directional fields, or their derivatives with respect to z."""
        self.prepare(solution)
        weights = self.interpolation_weights(distances_m, derivative=derivative)
        incoming = np.einsum("np,nfp->fp", weights, self.incoming_values, optimize=True)
        outgoing = np.einsum("np,nfp->fp", weights, self.outgoing_values, optimize=True)
        waves = np.stack([incoming, outgoing], axis=-1)
        return waves * self.model.scale if scaled else waves

    def direct_waves(self, solution, distances_m, *, scaled=True):
        """Slow independent-height FFT reference for final numerical checks."""
        self.prepare(solution)
        self.interpolation_weights(distances_m)  # validates bounds and shape
        m = self.model
        shape = (len(m.f), len(m.x), len(m.y))
        input_front = solution["input_field"].reshape(shape, order="F")
        outgoing_coefficients = np.linalg.solve(
            solution["P"][:, 7], solution["reflected"]
        )
        output_front = np.einsum(
            "pm,fm->fp", m.B, outgoing_coefficients, optimize=True
        ).reshape(shape, order="F")
        fin = np.where(self.propagating, np.fft.fft2(input_front, axes=(-2, -1)), 0.0)
        fout = np.fft.fft2(output_front, axes=(-2, -1))
        output = np.empty((len(m.f), len(self.sample.xy), 2), complex)
        for p, d in enumerate(distances_m):
            inward = np.fft.ifft2(fin * np.exp(1j * self.kz.real * d), axes=(-2, -1))
            outward = np.fft.ifft2(fout * np.exp(-1j * self.kz * d), axes=(-2, -1))
            output[:, p, 0] = self.sample(inward)[:, p]
            output[:, p, 1] = self.sample(outward)[:, p]
        return output * m.scale if scaled else output
