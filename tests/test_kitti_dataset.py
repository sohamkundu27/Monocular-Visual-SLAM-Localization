"""Tests for KITTI calibration parsing and the odometry dataset loader.

All tests run against the synthetic tree built by ``kitti_fixture`` so no
dataset download is required.
"""

from __future__ import annotations

import numpy as np
import pytest
from kitti_fixture import CALIB_TEXT, KITTI_00_CX, KITTI_00_CY, KITTI_00_FX, write_kitti_sequence

from monocular_slam.datasets.calibration import (
    CalibrationError,
    CameraCalibration,
    calibration_for_camera,
    parse_kitti_calib,
)
from monocular_slam.datasets.kitti import (
    DatasetError,
    FrameSelection,
    KittiOdometryDataset,
)


@pytest.fixture
def kitti_root(tmp_path):
    """A 12-frame synthetic sequence 00 with ground truth."""
    return write_kitti_sequence(tmp_path / "dataset", "00", n_frames=12)


@pytest.fixture
def calib_file(tmp_path):
    path = tmp_path / "calib.txt"
    path.write_text(CALIB_TEXT, encoding="utf-8")
    return path


class TestCalibParsing:
    def test_parses_all_projection_matrices(self, calib_file):
        calib = parse_kitti_calib(calib_file)
        assert set(calib) == {"P0", "P1", "P2", "P3", "Tr"}
        for key in ("P0", "P1", "P2", "P3"):
            assert calib[key].shape == (3, 4)

    def test_tr_is_promoted_to_4x4(self, calib_file):
        Tr = parse_kitti_calib(calib_file)["Tr"]
        assert Tr.shape == (4, 4)
        np.testing.assert_allclose(Tr[3], [0, 0, 0, 1])

    def test_intrinsics_match_kitti_sequence_00(self, calib_file):
        cam = calibration_for_camera(parse_kitti_calib(calib_file), "image_0")
        assert cam.fx == pytest.approx(KITTI_00_FX)
        assert cam.fy == pytest.approx(KITTI_00_FX)
        assert cam.cx == pytest.approx(KITTI_00_CX)
        assert cam.cy == pytest.approx(KITTI_00_CY)

    def test_K_is_upper_triangular_with_unit_scale(self, calib_file):
        K = calibration_for_camera(parse_kitti_calib(calib_file), "image_0").K
        assert K.shape == (3, 3)
        assert K[1, 0] == 0.0 and K[2, 0] == 0.0 and K[2, 1] == 0.0
        assert K[2, 2] == pytest.approx(1.0)

    def test_stereo_baseline_matches_kitti(self, calib_file):
        """P1's fourth column encodes -fx * baseline; KITTI's is ~0.54 m."""
        calib = parse_kitti_calib(calib_file)
        assert calibration_for_camera(calib, "image_0").baseline_m == pytest.approx(0.0)
        assert calibration_for_camera(calib, "image_1").baseline_m == pytest.approx(0.537, abs=1e-3)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(CalibrationError, match="not found"):
            parse_kitti_calib(tmp_path / "nope.txt")

    def test_wrong_value_count_raises(self, tmp_path):
        path = tmp_path / "calib.txt"
        path.write_text("P0: 1 2 3\n", encoding="utf-8")
        with pytest.raises(CalibrationError, match="expected 12"):
            parse_kitti_calib(path)

    def test_missing_colon_raises(self, tmp_path):
        path = tmp_path / "calib.txt"
        path.write_text("P0 1 2 3\n", encoding="utf-8")
        with pytest.raises(CalibrationError, match="expected 'key: values'"):
            parse_kitti_calib(path)

    def test_non_finite_values_raise(self, tmp_path):
        path = tmp_path / "calib.txt"
        path.write_text("P0: " + " ".join(["nan"] * 12) + "\n", encoding="utf-8")
        with pytest.raises(CalibrationError, match="non-finite"):
            parse_kitti_calib(path)

    def test_blank_lines_ignored(self, tmp_path):
        path = tmp_path / "calib.txt"
        path.write_text("\n# comment\n" + CALIB_TEXT + "\n", encoding="utf-8")
        assert "P0" in parse_kitti_calib(path)

    def test_unknown_camera_raises(self, calib_file):
        with pytest.raises(CalibrationError, match="Unknown camera"):
            calibration_for_camera(parse_kitti_calib(calib_file), "image_9")


class TestCameraModel:
    @pytest.fixture
    def cam(self):
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        P = np.hstack([K, np.zeros((3, 1))])
        return CameraCalibration(K=K, P=P)

    def test_normalize_points_centres_principal_point(self, cam):
        normalized = cam.normalize_points(np.array([[320.0, 240.0], [820.0, 240.0]]))
        np.testing.assert_allclose(normalized[0], [0.0, 0.0])
        np.testing.assert_allclose(normalized[1], [1.0, 0.0])

    def test_project_round_trips_with_normalize(self, cam, rng):
        pts_cam = np.column_stack(
            [rng.normal(size=20), rng.normal(size=20), rng.uniform(3.0, 30.0, size=20)]
        )
        uv = cam.project(pts_cam)
        normalized = cam.normalize_points(uv)
        np.testing.assert_allclose(normalized, pts_cam[:, :2] / pts_cam[:, 2:3], atol=1e-9)

    def test_points_behind_camera_become_nan(self, cam):
        uv = cam.project(np.array([[1.0, 1.0, -5.0], [1.0, 1.0, 5.0]]))
        assert np.all(np.isnan(uv[0]))
        assert np.all(np.isfinite(uv[1]))

    def test_invalid_shapes_rejected(self, cam):
        with pytest.raises(CalibrationError):
            CameraCalibration(K=np.eye(2), P=np.zeros((3, 4)))
        with pytest.raises(CalibrationError, match="Focal lengths"):
            CameraCalibration(K=np.diag([-1.0, 1.0, 1.0]), P=np.zeros((3, 4)))
        with pytest.raises(ValueError):
            cam.project(np.zeros((4, 2)))


class TestFrameSelection:
    def test_default_selects_everything(self):
        np.testing.assert_array_equal(FrameSelection().apply(5), np.arange(5))

    def test_step_subsamples(self):
        np.testing.assert_array_equal(FrameSelection(step=3).apply(10), [0, 3, 6, 9])

    def test_start_offsets(self):
        np.testing.assert_array_equal(FrameSelection(start=2, step=2).apply(8), [2, 4, 6])

    def test_max_frames_truncates(self):
        assert len(FrameSelection(max_frames=4).apply(100)) == 4

    def test_max_frames_applies_after_step(self):
        np.testing.assert_array_equal(FrameSelection(step=5, max_frames=3).apply(100), [0, 5, 10])

    @pytest.mark.parametrize("kwargs", [{"step": 0}, {"start": -1}, {"start": 99}])
    def test_invalid_selection_raises(self, kwargs):
        with pytest.raises(ValueError):
            FrameSelection(**kwargs).apply(10)


class TestDatasetLoader:
    def test_loads_all_frames(self, kitti_root):
        ds = KittiOdometryDataset(kitti_root, "00")
        assert len(ds) == 12
        np.testing.assert_array_equal(ds.frame_ids, np.arange(12))

    def test_sequence_id_is_normalised(self, kitti_root):
        assert KittiOdometryDataset(kitti_root, 0).sequence == "00"
        assert KittiOdometryDataset(kitti_root, "0").sequence == "00"
        assert KittiOdometryDataset(kitti_root, "00").sequence == "00"

    def test_images_are_grayscale_uint8(self, kitti_root):
        image = KittiOdometryDataset(kitti_root, "00")[0]
        assert image.ndim == 2
        assert image.dtype == np.uint8

    def test_colour_mode_returns_three_channels(self, kitti_root):
        image = KittiOdometryDataset(kitti_root, "00", grayscale=False)[0]
        assert image.ndim == 3 and image.shape[2] == 3

    def test_image_size_property(self, kitti_root):
        assert KittiOdometryDataset(kitti_root, "00").image_size == (320, 120)

    def test_intrinsics_exposed(self, kitti_root):
        ds = KittiOdometryDataset(kitti_root, "00")
        assert ds.K.shape == (3, 3)
        assert ds.calibration.fx == pytest.approx(KITTI_00_FX)

    def test_timestamps_align_with_frames(self, kitti_root):
        ds = KittiOdometryDataset(kitti_root, "00", frame_step=4)
        assert len(ds.timestamps) == len(ds)
        np.testing.assert_allclose(ds.timestamps[1] - ds.timestamps[0], 4 * 0.1037, atol=1e-6)

    def test_missing_times_file_is_tolerated(self, tmp_path):
        root = write_kitti_sequence(tmp_path / "d", "00", n_frames=5, with_times=False)
        assert KittiOdometryDataset(root, "00").timestamps is None

    def test_subsampling_selects_matching_frames(self, kitti_root):
        ds = KittiOdometryDataset(kitti_root, "00", start_frame=2, frame_step=3, max_frames=3)
        np.testing.assert_array_equal(ds.frame_ids, [2, 5, 8])
        assert len(ds.image_paths) == 3
        assert ds.image_paths[0].name == "000002.png"

    def test_iteration_yields_every_selected_frame(self, kitti_root):
        ds = KittiOdometryDataset(kitti_root, "00", max_frames=4)
        assert sum(1 for _ in ds) == 4

    def test_missing_sequence_raises_with_hint(self, kitti_root):
        with pytest.raises(DatasetError, match="Available sequences"):
            KittiOdometryDataset(kitti_root, "07")

    def test_missing_camera_raises(self, kitti_root):
        with pytest.raises(DatasetError, match="Image directory not found"):
            KittiOdometryDataset(kitti_root, "00", camera="image_3")

    def test_describe_reports_real_values(self, kitti_root):
        info = KittiOdometryDataset(kitti_root, "00").describe()
        assert info["sequence"] == "00"
        assert info["frames_selected"] == 12
        assert info["has_ground_truth"] is True
        assert info["ground_truth_path_length_m"] > 0


class TestGroundTruth:
    def test_ground_truth_matches_frame_count(self, kitti_root):
        gt = KittiOdometryDataset(kitti_root, "00").ground_truth
        assert len(gt) == 12
        assert gt.is_valid()

    def test_ground_truth_starts_at_identity(self, kitti_root):
        gt = KittiOdometryDataset(kitti_root, "00").ground_truth
        np.testing.assert_allclose(gt[0], np.eye(4), atol=1e-9)

    def test_ground_truth_is_subsampled_with_frames(self, kitti_root):
        ds = KittiOdometryDataset(kitti_root, "00", frame_step=2)
        gt = ds.ground_truth
        assert len(gt) == len(ds)
        np.testing.assert_array_equal(gt.frame_ids, ds.frame_ids)

    def test_scales_are_positive_and_match_step(self, kitti_root):
        scales = KittiOdometryDataset(kitti_root, "00").ground_truth_scales()
        assert len(scales) == 11
        np.testing.assert_allclose(scales, 0.9, atol=1e-6)

    def test_subsampled_scales_grow_with_step(self, kitti_root):
        """Skipping frames means each transition covers more ground."""
        scales = KittiOdometryDataset(kitti_root, "00", frame_step=3).ground_truth_scales()
        assert np.all(scales > 2.0)

    def test_missing_ground_truth_returns_none(self, tmp_path):
        root = write_kitti_sequence(tmp_path / "d", "00", n_frames=5, with_ground_truth=False)
        ds = KittiOdometryDataset(root, "00")
        assert ds.has_ground_truth is False
        assert ds.ground_truth is None
        assert ds.ground_truth_scales() is None

    def test_truncated_pose_file_raises(self, kitti_root):
        np.savetxt(kitti_root / "poses" / "00.txt", np.zeros((3, 12)), fmt="%.6e")
        with pytest.raises(DatasetError, match="poses but the sequence has"):
            _ = KittiOdometryDataset(kitti_root, "00").ground_truth

    def test_non_se3_pose_file_raises(self, kitti_root):
        bad = np.tile(np.arange(12.0), (12, 1))  # rotation block is not orthonormal
        np.savetxt(kitti_root / "poses" / "00.txt", bad, fmt="%.6e")
        with pytest.raises(DatasetError, match="not valid SE"):
            _ = KittiOdometryDataset(kitti_root, "00").ground_truth


class TestConfigIntegration:
    def test_from_config_round_trip(self, kitti_root):
        from monocular_slam.config import Config

        config = Config().with_overrides(
            {"dataset.path": str(kitti_root), "dataset.sequence": "00", "dataset.max_frames": 5}
        )
        ds = KittiOdometryDataset.from_config(config)
        assert len(ds) == 5
        assert ds.sequence == "00"
