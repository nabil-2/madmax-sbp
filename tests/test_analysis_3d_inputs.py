"""Regression checks for the measured-S22 3D analysis path."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

from src.analysis_3d_inputs import BACKGROUND_METHOD, endpoint_background, load_measurement, prepare_measurement
from src.analysis_3d_exports import reprofile_saved_geometry
from src.analysis_3d import EndpointFitProblem
from src.analysis_3d_plots import load as load_plot_result


class MeasuredReflectionTests(unittest.TestCase):
    def test_endpoint_background_ignores_illuminated_measurements(self):
        position = np.linspace(-0.2, 0.2, 30)
        baseline = (0.4 + 0.2j) + (0.03 - 0.01j) * position
        measured = baseline[:, None].copy()
        measured[5:-5] += 12 + 7j
        background, reference, _ = endpoint_background(position, measured)
        np.testing.assert_allclose(background[:, 0], baseline, atol=1e-14)
        np.testing.assert_array_equal(np.flatnonzero(reference), np.r_[np.arange(5), np.arange(25, 30)])
        measured[5:-5] *= 100
        changed, _, _ = endpoint_background(position, measured)
        np.testing.assert_array_equal(changed, background)

    def test_preparation_preserves_complex_samples_without_calibration_band(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frequencies = np.array([19e9, 20.1e9, 20.4e9, 23e9])
            rng = np.random.default_rng(41)
            raw = rng.normal(size=(48, 4)) + 1j * rng.normal(size=(48, 4))
            with h5py.File(root / "s11_traces.h5", "w") as handle:
                handle["frequencies_Hz"] = frequencies
                handle["s_parameter"] = raw
            position = np.tile(np.linspace(0, 0.3, 12), 4)
            pd.DataFrame(dict(
                measurement_json=[json.dumps({"trace_index": i}) for i in range(48)],
                sub_thread_index=np.repeat(np.arange(4), 12), position_m=position,
                position_x_m=position, position_y_m=np.zeros(48), position_z_m=np.zeros(48),
            )).to_csv(root / "scan_log.csv", index=False)
            profile = (np.array([0.0, 0.9, 0.0]), np.array([0j, 0j, 1+0j]), 0.0, 0.0, True)
            with patch("src.analysis_3d_inputs.fit_profile", return_value=profile):
                path = prepare_measurement(root, root / "input.npz")
            result = load_measurement(path)
            np.testing.assert_array_equal(result["observed_S22"], raw[:, 1:3])
            np.testing.assert_array_equal(result["frequency_hz"], frequencies[1:3])
            self.assertNotIn("corrected_S22", result)
            self.assertEqual(result["reflection_preprocessing"].item(), "none")
            self.assertEqual(result["background_method"].item(), BACKGROUND_METHOD)
            for thread in range(4):
                self.assertEqual(result["background_reference_mask"][result["thread_index"] == thread].sum(), 10)

    def test_legacy_corrected_inputs_rejected_before_model_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.npz"
            np.savez(path, corrected_S22=np.ones((4, 2)))
            with patch("src.analysis_3d.TiltedReceiveModel") as model:
                with self.assertRaisesRegex(ValueError, "Regenerate"):
                    EndpointFitProblem("unused-bank", path, "unused-beam")
                model.assert_not_called()

    def test_plot_loader_rejects_historical_corrected_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "historical.npz"
            np.savez(path, observed_S22=np.ones((2, 4)))
            with self.assertRaisesRegex(ValueError, "removed template correction"):
                load_plot_result(path)

    def test_reprofiling_recovers_raw_signal_and_preserves_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            frequency = np.array([20.7e9, 20.8e9, 20.9e9])
            thread = np.repeat(np.arange(4), 16)
            u = np.tile(np.linspace(-1, 1, 16), 4)
            shape = (1 + thread / 4) * np.exp(-2 * u**2 + 0.4j * u**2)
            reference = np.tile(np.r_[np.ones(5, bool), np.zeros(6, bool), np.ones(5, bool)], 4)
            shape[reference] = 0
            template = np.array([1, 0.8+0.2j, 1.1-0.3j])[:, None] * shape
            expected = np.array([1.2+0.3j, 0.7-0.2j, 1.1+0.4j])
            background = (0.1 + 0.02j * thread + 0.03 * u)[None, :]
            observed = expected[:, None] * template + background
            xy = np.column_stack([u / 10, np.zeros(len(u))])
            saved = dict(
                frequency_hz=frequency, trace_index=np.arange(64), thread_index=thread,
                signed_line_m=u / 10, xy_m=xy, template_S22=template,
                eligible=np.array([True, False, True]),
                beta2_received_geometry_only=np.array([10.0, 20.0, 30.0]),
                parameters=np.arange(17), directional_waves_V_per_m=np.ones((3, 64, 2)),
            )
            path = Path(directory) / "raw.npz"
            np.savez(path, frequency_hz=frequency, observed_S22=observed.T,
                     reflection_preprocessing="none", trace_index=np.arange(64),
                     thread_index=thread, signed_line_m=u / 10,
                     transverse_xyz_log_m=np.column_stack([xy, np.zeros(64)]),
                     transient_trace_mask=np.zeros(64, bool),
                     fixed_background_S22=np.broadcast_to(background, observed.shape).T,
                     background_reference_mask=reference, background_method=BACKGROUND_METHOD,
                     shared_fit_mask=np.array([True, True, True]))
            result = reprofile_saved_geometry(saved, path)
            np.testing.assert_allclose(result["quadratic_scale"], expected, atol=1e-13)
            np.testing.assert_allclose(result["predicted_S22"], observed, atol=1e-13)
            np.testing.assert_allclose(result["fitted_background"], np.broadcast_to(background, observed.shape), atol=1e-13)
            np.testing.assert_allclose(result["relative_residual"], 0, atol=1e-13)
            np.testing.assert_array_equal(result["parameters"], saved["parameters"])
            np.testing.assert_array_equal(result["directional_waves_V_per_m"], saved["directional_waves_V_per_m"])
            np.testing.assert_array_equal(result["geometry_fit_mask"], saved["eligible"])
            self.assertFalse(result["geometry_refitted"])
            # A wrong model may change the scale and residual, never the
            # physical background determined from the off-beam endpoints.
            mismatched = dict(saved, template_S22=template * (1 + 0.5 * u[None, :]))
            changed = reprofile_saved_geometry(mismatched, path)
            np.testing.assert_array_equal(changed["fitted_background"], result["fitted_background"])


if __name__ == "__main__":
    unittest.main()
