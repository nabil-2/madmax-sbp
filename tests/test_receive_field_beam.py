"""Check the prescribed beam alignment and whole-plane power normalization."""

import unittest

import numpy as np

from src.receive_field import ReceiveModel


class CenteredBeamTests(unittest.TestCase):
    def setUp(self):
        # Beam evaluation needs only the sampling grid and measured fit parameters.
        self.model = ReceiveModel.__new__(ReceiveModel)
        self.model.f = np.array([20.825e9])
        axis = np.linspace(-0.4, 0.4, 161)
        self.model.X, self.model.Y = np.meshgrid(axis, axis, indexing="ij")
        self.model.area = (axis[1] - axis[0]) ** 2
        self.parameters = np.array([
            [1.0, 0.040, 0.020, -0.030],
            [0.9, 0.045, -0.015, 0.025],
            [0.8, 0.050, 0.010, 0.020],
            [0.7, 0.055, -0.025, -0.015],
            [0.6, 0.060, 0.030, 0.010],
            [0.5, 0.065, -0.010, -0.020],
        ])[None, :, :]
        self.model.beam = dict(
            frequencies_hz=self.model.f,
            slice_parameters=self.parameters.copy(),
        )

    def test_default_centres_every_profile_and_preserves_measured_shape_parameters(self):
        field = self.model.beam_field()[0].reshape(self.model.X.shape, order="F")
        radius_squared = self.model.X**2 + self.model.Y**2
        expected = sum(
            amplitude * np.exp(-radius_squared / width**2)
            for amplitude, width, _, _ in self.parameters[0]
        )
        # A single whole-plane normalization may change only the common scale.
        expected *= field[80, 80].real / expected[80, 80]
        np.testing.assert_allclose(field, expected, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(field, field[::-1, :], rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(field, field[:, ::-1], rtol=1e-13, atol=1e-15)
        self.assertEqual(np.unravel_index(np.argmax(abs(field)), field.shape), (80, 80))
        self.assertAlmostEqual(float(np.sum(abs(field)**2)), 1.0, places=12)
        np.testing.assert_array_equal(self.model.beam["slice_parameters"], self.parameters)

    def test_measured_offsets_are_only_used_when_explicitly_requested(self):
        measured = self.model.beam_field(center_scale=1.0)
        centred = self.model.beam_field()
        self.assertGreater(float(np.max(abs(measured - centred))), 1e-3)
        self.model.beam["slice_parameters"][:, :, 2:] = 0
        np.testing.assert_allclose(self.model.beam_field(), centred, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
