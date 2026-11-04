"""Physical face assignment, continuous clearance bounds and phase projections."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.disk_surfaces import DiskSurfaces, SURFACE_DIRECTORY
from src.surface_moments import load_surface_moments
from src.tilted_disks import TiltedReceiveModel


class MeasuredFaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.surfaces = DiskSurfaces()

    def test_assignment_orientation_and_map_origin(self):
        for d, measured in enumerate((2, 3, 1)):
            for side, letter, sign in ((0, "A", -1), (1, "B", 1)):
                raw = np.loadtxt(SURFACE_DIRECTORY / f"alignedDisk{measured}_{letter}_25e-1mm_concave.txt")
                expected = sign * raw[199, 199] * 1e-3
                np.testing.assert_allclose(self.surfaces.evaluate(d, side, [[0, 0]]), expected)
                half_step = sign * (raw[199, 199] + raw[200, 199]) / 2 * 1e-3
                np.testing.assert_allclose(self.surfaces.evaluate(d, side, [[0.00125, 0]]), half_step)

    def test_positive_measured_thickness_and_rigid_motion(self):
        axis = np.linspace(-.15, .15, 121)
        X, Y = np.meshgrid(axis, axis, indexing="ij")
        use = X**2 + Y**2 <= .15**2
        reference = np.array([.008, .018, .028])
        slopes = np.array([[.003, -.002], [-.004, .001], [.005, -.005]])
        faces = self.surfaces.face_positions(reference, slopes, axis, axis)
        shapes = self.surfaces.on_grid(axis, axis)
        np.testing.assert_allclose(faces[:, 1] - faces[:, 0], .001 + shapes[:, 1] - shapes[:, 0])
        self.assertGreater(float(np.min((faces[:, 1] - faces[:, 0])[:, use])), 0)
        np.testing.assert_allclose(faces[:, 0] - shapes[:, 0], reference[:, None, None] + slopes[:, 0, None, None]*X + slopes[:, 1, None, None]*Y)
        bounds = self.surfaces.aperture_bounds(reference, slopes)
        for d in range(3):
            for side in range(2):
                self.assertGreaterEqual(float(faces[d, side][use].min()), bounds[d, side, 0])
                self.assertLessEqual(float(faces[d, side][use].max()), bounds[d, side, 1])

    def test_bernstein_bounds_cover_every_point_between_scan_samples(self):
        endpoints = np.array([[-.14, -.03], [.14, .06]])
        segments = self.surfaces.chord_segments(endpoints, [0, 1])
        for d in range(3):
            for side in range(2):
                coefficients = self.surfaces.chord_bernstein_heights(d, side, endpoints, segments)
                for v, h in zip(segments, coefficients):
                    t = np.linspace(0, 1, 11)
                    fraction = v[0] + (v[2] - v[0])*t
                    xy = endpoints[0] + fraction[:, None]*(endpoints[1]-endpoints[0])
                    actual = self.surfaces.evaluate(d, side, xy)
                    reconstructed = h[0]*(1-t)**2 + 2*h[1]*t*(1-t) + h[2]*t**2
                    np.testing.assert_allclose(actual, reconstructed, rtol=1e-12, atol=1e-15)
                    self.assertGreaterEqual(float(actual.min()), float(h.min())-1e-15)
                    self.assertLessEqual(float(actual.max()), float(h.max())+1e-15)


class CurvedPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        model = TiltedReceiveModel.__new__(TiltedReceiveModel)
        model.radius, model.order, model.n = .15, 36, 3
        axis = np.linspace(-.15, .15, 9)
        X, Y = np.meshgrid(axis, axis, indexing="ij")
        model.aperture = (X**2+Y**2 <= .15**2).ravel(order="F")
        model.inside_x = X.ravel(order="F")[model.aperture]
        model.inside_y = Y.ravel(order="F")[model.aperture]
        rng = np.random.default_rng(813)
        B, _ = np.linalg.qr(rng.normal(size=(model.aperture.sum(), 3))+1j*rng.normal(size=(model.aperture.sum(), 3)))
        model.Binside, model.gram = B, B.conj().T@B
        model.uniform = model.aperture.astype(float)/np.sqrt(model.aperture.sum())
        model.axion_initial = B.conj().T@model.uniform[model.aperture]
        model.meta = {"arrays": {key: {"sha256": "synthetic-test-basis"} for key in ("basis_fields", "gram", "x_m", "y_m")}}
        model._prepare_moments()
        def evaluate(d, side, xy):
            x, y = np.asarray(xy).T
            return (-1 if side == 0 else 1)*(d+1)*2e-5*(.2+(x*x+.6*x*y-.7*y*y)/.15**2)
        model.surfaces = SimpleNamespace(fingerprint="synthetic-curved-faces", evaluate=evaluate)
        model.surface_powers, model.surface_moments, model.surface_sources = load_surface_moments(model, model.surfaces, Path(cls.directory.name))
        xy = np.column_stack((model.inside_x, model.inside_y))
        model.surface_height_max = [max(abs(evaluate(face//2, face%2, xy))) for face in range(6)]
        cls.model = model

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_curved_phase_and_axion_source_match_direct_grid_projection(self):
        model = self.model
        xy = np.column_stack((model.inside_x, model.inside_y))
        for face in range(6):
            for slope in (np.zeros(2), np.array([.005, -.005])):
                for k in (440., 2815.-.01j, -2815.+.01j):
                    actual = model.phase_matrix(k, slope, face)
                    direct = model.direct_phase_matrix(k, slope, face)
                    np.testing.assert_allclose(actual, direct, rtol=2e-10, atol=2e-11)
                    h = xy@slope + model.surfaces.evaluate(face//2, face%2, xy)
                    source = model.Binside.conj().T@(np.exp(1j*k*h)*model.uniform[model.aperture])
                    np.testing.assert_allclose(model.phase_source(k, slope, face), source, rtol=2e-10, atol=2e-11)

    def test_separate_faces_are_not_replaced_by_a_common_rigid_plane(self):
        model = self.model
        near = model.phase_matrix(1800., np.zeros(2), 0)
        far = model.phase_matrix(1800., np.zeros(2), 1)
        self.assertGreater(float(np.max(abs(near-far))), 1e-3)
        self.assertGreater(float(np.max(abs(near-np.eye(model.n)))), 1e-3)


if __name__ == "__main__":
    unittest.main()
