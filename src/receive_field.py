"""Scalar modal receive fields from a BoostFractor propagation bank.

Bank matrices keep the original sampled-aperture and phasor conventions.
The incident magnitude profile is normalized over the entire transverse plane.
"""

from pathlib import Path
import hashlib
import tomllib
import numpy as np
from scipy.interpolate import make_interp_spline
from .bead_response import bead_factors
from .analysis_band import require_analysis_band

C = 299792458.0
Z0 = 376.730313668
NOM_DISKS_MM = np.array([7.803, 17.656, 27.511])
NOM_THREADS_MM = np.array([23.0835, 13.2295, 3.9015])


def read_bank(path, verify=True):
    path = Path(path)
    metadata = tomllib.loads((path / "manifest.toml").read_text())
    arrays = {}
    for name, spec in metadata["arrays"].items():
        file = path / spec["file"]
        if verify:
            assert hashlib.sha256(file.read_bytes()).hexdigest() == spec["sha256"], name
        arrays[name] = np.fromfile(
            file, dtype=np.dtype(spec["dtype"]).newbyteorder("<")
        ).reshape(spec["shape"], order="F")
    return metadata, arrays


class ReceiveModel:
    def __init__(self, bank_path, beam_path, verify=True):
        self.meta, self.arr = read_bank(bank_path, verify)
        a = self.arr
        self.f = a["frequencies_hz"]
        require_analysis_band(self.f, "3D propagation bank")
        self.ds = a["distances_m"]
        self.n = len(a["kt_per_m"])
        self.gram = a["gram"]
        self.x = a["x_m"]
        self.y = a["y_m"]
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.area = self.dx * self.dy
        self.radius = self.meta["normalization"]["disk_radius_m"]
        self.B = a["basis_fields"].reshape((-1, self.n), order="F")
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")
        self.aperture = (self.X**2 + self.Y**2 <= self.radius**2).ravel(order="F")
        self.Binside = self.B[self.aperture]
        self.inside_x = self.X.ravel(order="F")[self.aperture]
        self.inside_y = self.Y.ravel(order="F")[self.aperture]
        self.uniform = np.zeros(len(self.aperture))
        self.uniform[self.aperture] = 1 / np.sqrt(sum(self.aperture))
        self.axion_initial = self.B.conj().T @ self.uniform
        self.aperture_area_ratio = (
            sum(self.aperture) * self.area / (np.pi * self.radius**2)
        )
        self.scale = np.sqrt(2 * Z0 / self.area)
        # Cubic interpolation is checked separately against direct Julia values.
        self.interpolation_degree = min(3, len(self.ds) - 1)
        if self.interpolation_degree < 1:
            raise ValueError("A propagation bank needs at least two distances")
        self.Pspline = make_interp_spline(
            self.ds, np.moveaxis(a["air_P"], -1, 0), k=self.interpolation_degree, axis=0
        )
        self.de, self.dc = bead_factors(self.f)
        self.beam = None if beam_path is None else np.load(beam_path)
        if self.beam is not None:
            require_analysis_band(self.beam["frequencies_hz"], "3D incident-beam input")
        self.last = None
        self._illumination_cache = None

    def air(self, distance_m):
        if np.any(
            (np.asarray(distance_m) < self.ds[0] - 1e-13)
            | (np.asarray(distance_m) > self.ds[-1] + 1e-13)
        ):
            raise ValueError(f"Distance {distance_m} outside propagation bank")
        out = self.Pspline(distance_m)
        # Scalar distance: (mode,mode,f) -> (f,mode,mode).
        return np.moveaxis(out, -1, -3)

    def beam_field(
        self,
        curvature=0.0,
        width_scale=1.0,
        center_scale=1.0,
        source="omc",
        center_shift_m=(0.0, 0.0),
    ):
        """Sample the measured six-profile mixture; never normalize its aperture.

        Curvature is phase coefficient at 20.825 GHz in BF exp(-ikd) convention;
        it scales with f. It is a sensitivity/nuisance parameter, not measured phase.
        """
        if source == "checkpoint":
            field = self.arr["check_incident_field"].ravel(order="F")
            return np.broadcast_to(field, (len(self.f), len(field))).copy()
        x = self.X.ravel(order="F")
        y = self.Y.ravel(order="F")
        field = np.zeros((len(self.f), len(x)))
        bx = np.zeros(len(self.f))
        by = np.zeros(len(self.f))
        if source == "omc":
            original = self.beam["slice_parameters"]
            # Parameters: amplitude, local 1/e radius, x0, y0.
            indices = np.searchsorted(self.beam["frequencies_hz"], self.f)
            np.testing.assert_allclose(
                self.beam["frequencies_hz"][indices], self.f, rtol=0, atol=1
            )
            p = original[indices].copy()
            p[:, :, 1] *= width_scale
            p[:, :, 2:] *= center_scale
            p[:, :, 2:] += np.asarray(center_shift_m)[None, None, :]
            bx = p[:, :, 2].mean(1)
            by = p[:, :, 3].mean(1)
            norm2 = np.zeros(len(self.f))
            for i in range(p.shape[1]):
                ai, wi, xi, yi = p[:, i].T
                field += ai[:, None] * np.exp(
                    -((x[None, :] - xi[:, None]) ** 2 + (y[None, :] - yi[:, None]) ** 2)
                    / wi[:, None] ** 2
                )
                for j in range(p.shape[1]):
                    aj, wj, xj, yj = p[:, j].T
                    norm2 += (
                        ai
                        * aj
                        * np.pi
                        * wi**2
                        * wj**2
                        / (wi**2 + wj**2)
                        * np.exp(-((xi - xj) ** 2 + (yi - yj) ** 2) / (wi**2 + wj**2))
                    )
            field *= np.sqrt(self.area / norm2)[:, None]
        elif source == "gaussian":
            w = 0.08 * width_scale
            field[:] = np.sqrt(2 * self.area / (np.pi * w * w)) * np.exp(
                -(x * x + y * y) / (w * w)
            )
        else:
            raise ValueError(source)
        phase = (
            curvature
            * (self.f / 20.825e9)[:, None]
            * ((x[None, :] - bx[:, None]) ** 2 + (y[None, :] - by[:, None]) ** 2)
        )
        return field * np.exp(1j * phase)

    def propagation_matrices(self, disks_mm, epsilon=9.3, loss_tangent=1e-5):
        disks = np.asarray(disks_mm) * 1e-3
        gaps = np.r_[disks[0], np.diff(disks) - 0.001]
        if min(gaps) <= 0:
            raise ValueError("Nonpositive gap")
        eps = np.array(
            [
                1e30,
                1,
                epsilon * (1 - 1j * loss_tangent),
                1,
                epsilon * (1 - 1j * loss_tangent),
                1,
                epsilon * (1 - 1j * loss_tangent),
                1,
            ],
            complex,
        )
        kt = self.arr["kt_per_m"]
        kz = np.sqrt((2 * np.pi * self.f[:, None] / C) ** 2 * eps[2] - kt[None, :] ** 2)
        solid = np.exp(-1j * kz * 0.001)[:, :, None] * self.gram[None, :, :]
        pzero = self.air(0.0)
        matrices = np.stack(
            [
                pzero,
                self.air(gaps[0]),
                solid,
                self.air(gaps[1]),
                solid,
                self.air(gaps[2]),
                solid,
                pzero,
            ],
            axis=1,
        )
        return matrices, eps, gaps

    def finite_bead_field(self, waves):
        # Convert BF exp(-ikd) phasor into the existing validated outgoing-h1
        # Mie convention before combining directional fields.
        em = waves[..., 0].conj()
        ep = waves[..., 1].conj()
        shape = (len(self.f),) + (1,) * (ep.ndim - 1)
        squared = (
            self.de.reshape(shape) * (ep * ep + em * em)
            + 2 * self.dc.reshape(shape) * ep * em
        )
        total = ep + em
        root = np.sqrt(squared)
        # Choose the square-root branch locally nearest the unperturbed total
        # field. This branch choice is audited against the measured line operator.
        root = np.where(np.real(root * total.conj()) < 0, -root, root)
        return root, squared, total
