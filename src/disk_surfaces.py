"""Measured disk faces in the shared mirror-to-antenna coordinate system."""

from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.interpolate import RegularGridInterpolator

SURFACE_DIRECTORY = Path(__file__).resolve().parents[1] / "config/disk_surfaces"


class DiskSurfaces:
    """Fixed bilinear height maps, with shape (disk, near/far, x, y).

    Measured disks 2, 3, 1 occupy booster positions D1, D2, D3. The OMC
    convention gives near-face departure -A and far-face departure +B.
    Heights add to the reference planes; no plane or mean is subtracted.
    """

    def __init__(self, directory=SURFACE_DIRECTORY):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        self.manifest = manifest
        self.step = manifest["grid_step_m"]
        self.axis_index = np.asarray(manifest["axis_index_zero_based"])
        order = manifest["booster_disk_measured_disk_order"]
        maps = []
        for disk in order:
            faces = []
            for side, sign in (("A", -1), ("B", 1)):
                name = f"alignedDisk{disk}_{side}_25e-1mm_concave.txt"
                raw = (directory / name).read_bytes()
                if hashlib.sha256(raw).hexdigest() != manifest["files"][name]["sha256"]:
                    raise ValueError(f"Measured disk surface has changed: {name}")
                faces.append(sign * np.loadtxt(directory / name) * manifest["height_scale_m"])
            maps.append(faces)
        self.heights = np.asarray(maps)
        if self.heights.ndim != 4 or not np.all(np.isfinite(self.heights)):
            raise ValueError("Disk surfaces must be finite rectangular height maps.")
        self.axes = tuple(
            (np.arange(n) - origin) * self.step
            for n, origin in zip(self.heights.shape[-2:], self.axis_index)
        )
        self.interpolators = [
            [RegularGridInterpolator(self.axes, face) for face in disk]
            for disk in self.heights
        ]
        # A Lipschitz bound includes both endpoints of every relevant map cell.
        X, Y = np.meshgrid(*self.axes, indexing="ij")
        use = X**2 + Y**2 <= (0.15 + 0.001465 + 2 * self.step)**2
        self.gradient_bounds = np.empty((3, 2, 2))
        for d in range(3):
            for side in range(2):
                h = self.heights[d, side]
                gx, gy = np.diff(h, axis=0) / self.step, np.diff(h, axis=1) / self.step
                self.gradient_bounds[d, side] = [
                    np.max(abs(gx)[use[:-1] | use[1:]]),
                    np.max(abs(gy)[use[:, :-1] | use[:, 1:]]),
                ]
        self.fingerprint = hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest()

    def evaluate(self, disk, side, xy):
        return self.interpolators[disk][side](np.asarray(xy))

    def on_grid(self, x, y):
        X, Y = np.meshgrid(x, y, indexing="ij")
        xy = np.stack((X, Y), axis=-1)
        return np.asarray([
            [self.evaluate(d, side, xy) for side in range(2)] for d in range(3)
        ])

    def face_positions(self, disks_m, slopes, x, y):
        X, Y = np.meshgrid(x, y, indexing="ij")
        result = self.on_grid(x, y)
        result += np.asarray(disks_m)[:, None, None, None]
        result[:, 1] += 0.001
        result += (
            np.asarray(slopes)[:, 0, None, None] * X
            + np.asarray(slopes)[:, 1, None, None] * Y
        )[:, None]
        return result

    def normal_factors(self, slopes):
        """Conservative graph-distance factors for each measured face."""
        components = abs(np.asarray(slopes))[:, None, :] + self.gradient_bounds
        return np.sqrt(1 + np.sum(components**2, axis=-1))

    def aperture_bounds(self, disks_m, slopes, radius=0.15):
        """Conservative face extrema over the continuous circular aperture.

        Every intersecting bilinear cell is covered by including all vertices
        within one cell diagonal of the rim. A bilinear function plus a plane
        lies within the convex hull of its four corner values.
        """
        X, Y = np.meshgrid(*self.axes, indexing="ij")
        use = X**2 + Y**2 <= (radius + np.sqrt(2)*self.step)**2
        result = np.empty((3, 2, 2))
        for d in range(3):
            plane = disks_m[d] + slopes[d, 0]*X[use] + slopes[d, 1]*Y[use]
            for side in range(2):
                values = plane + side*.001 + self.heights[d, side][use]
                result[d, side] = values.min(), values.max()
        return result

    def chord_segments(self, endpoints, bounds):
        """Split a straight chord at every bilinear-map cell boundary."""
        if not len(bounds):
            return np.empty((0, 3))
        endpoints = np.asarray(endpoints)
        a, direction = endpoints[0], endpoints[1] - endpoints[0]
        fractions = [np.asarray(bounds)]
        for axis, start, delta in zip(self.axes, a, direction):
            if abs(delta) > 1e-15:
                v = (axis - start) / delta
                fractions.append(v[(v > bounds[0]) & (v < bounds[1])])
        edges = np.unique(np.concatenate(fractions))
        return np.stack((edges[:-1], (edges[:-1] + edges[1:]) / 2, edges[1:]), axis=1)

    def chord_bernstein_heights(self, disk, side, endpoints, segments):
        """Quadratic Bernstein coefficients of a bilinear face along each segment.

        Their convex hull bounds the entire continuous curve. Combining these
        constants with the affine fitted path/tilt gives linear constraints.
        """
        endpoints = np.asarray(endpoints)
        xy = endpoints[0] + segments[..., None] * (endpoints[1] - endpoints[0])
        h = self.evaluate(disk, side, xy)
        h[:, 1] = 2 * h[:, 1] - (h[:, 0] + h[:, 2]) / 2
        return h
