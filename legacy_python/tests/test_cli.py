"""Tests for the command line interfaces and the config/logging plumbing."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from kitti_fixture import write_kitti_sequence
from monocular_slam.cli.common import _parse_scalar, config_from_args
from monocular_slam.cli.evaluate import build_parser as evaluate_parser
from monocular_slam.cli.evaluate import main as evaluate_main
from monocular_slam.cli.run_slam import build_parser as slam_parser
from monocular_slam.cli.run_slam import main as slam_main
from monocular_slam.cli.run_vo import build_parser as vo_parser
from monocular_slam.cli.run_vo import main as vo_main
from monocular_slam.config import Config, ConfigError
from monocular_slam.utils.logging import get_logger, setup_logging


@pytest.fixture
def kitti_root(tmp_path):
    return write_kitti_sequence(tmp_path / "dataset", "00", n_frames=25)


@pytest.fixture
def fast_config_file(tmp_path, kitti_root):
    """A config tuned to run the tiny fixture sequence quickly."""
    config = Config().with_overrides(
        {
            "dataset.path": str(kitti_root),
            "features.max_features": 300,
            "matcher.min_matches": 8,
            "odometry.min_inliers": 6,
            "odometry.min_inlier_ratio": 0.05,
            "loop_closure.enabled": False,
            "output.plot_3d": False,
            "output.root": str(tmp_path / "outputs"),
            "runtime.log_level": "ERROR",
            "runtime.progress_interval": 0,
            "evaluation.rpe_deltas": [1, 5],
        }
    )
    path = tmp_path / "test.yaml"
    config.to_yaml(path)
    return path


class TestParsers:
    @pytest.mark.parametrize("build", [slam_parser, vo_parser, evaluate_parser])
    def test_help_renders(self, build, capsys):
        with pytest.raises(SystemExit) as exit_info:
            build().parse_args(["--help"])
        assert exit_info.value.code == 0
        assert "usage:" in capsys.readouterr().out

    def test_slam_parser_accepts_the_documented_invocation(self):
        args = slam_parser().parse_args(
            ["--sequence", "00", "--dataset-path", "/data/kitti", "--config", "configs/kitti.yaml"]
        )
        assert args.sequence == "00"
        assert args.dataset_path == "/data/kitti"

    def test_short_flags_work(self):
        args = slam_parser().parse_args(["-s", "05", "-o", "/tmp/out", "-c", "configs/kitti.yaml"])
        assert args.sequence == "05" and args.output_dir == "/tmp/out"

    def test_evaluate_requires_sequence_and_path(self):
        with pytest.raises(SystemExit):
            evaluate_parser().parse_args(["traj.txt"])


class TestConfigFromArgs:
    def test_flags_override_the_config_file(self, fast_config_file):
        args = slam_parser().parse_args(
            ["-c", str(fast_config_file), "-s", "05", "--max-frames", "7"]
        )
        config = config_from_args(args)
        assert config.dataset.sequence == "05"
        assert config.dataset.max_frames == 7

    def test_set_override_reaches_nested_keys(self, fast_config_file):
        args = slam_parser().parse_args(
            ["-c", str(fast_config_file), "--set", "odometry.min_inliers=42"]
        )
        assert config_from_args(args).odometry.min_inliers == 42

    def test_multiple_set_overrides(self, fast_config_file):
        args = slam_parser().parse_args(
            [
                "-c", str(fast_config_file),
                "--set", "odometry.min_inliers=42",
                "--set", "features.max_features=999",
            ]
        )
        config = config_from_args(args)
        assert config.odometry.min_inliers == 42
        assert config.features.max_features == 999

    def test_no_loop_closure_flag_disables_it(self, fast_config_file):
        args = slam_parser().parse_args(["-c", str(fast_config_file), "--no-loop-closure"])
        assert config_from_args(args).loop_closure.enabled is False

    def test_no_plots_flag_disables_plots(self, fast_config_file):
        args = slam_parser().parse_args(["-c", str(fast_config_file), "--no-plots"])
        assert config_from_args(args).output.save_plots is False

    def test_missing_config_file_is_reported_clearly(self):
        args = slam_parser().parse_args(["-c", "/nonexistent/config.yaml"])
        with pytest.raises(ConfigError, match="Config file not found"):
            config_from_args(args)

    def test_malformed_set_is_rejected(self, fast_config_file):
        args = slam_parser().parse_args(["-c", str(fast_config_file), "--set", "nonsense"])
        with pytest.raises(ConfigError, match="KEY=VALUE"):
            config_from_args(args)

    def test_unknown_set_key_is_rejected(self, fast_config_file):
        args = slam_parser().parse_args(["-c", str(fast_config_file), "--set", "nope.key=1"])
        with pytest.raises(ConfigError, match="Unknown config"):
            config_from_args(args)


class TestScalarParsing:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("true", True), ("False", False), ("yes", True), ("off", False),
            ("42", 42), ("-7", -7), ("3.5", 3.5), ("1e-3", 0.001),
            ("none", None), ("null", None),
            ("magsac", "magsac"),
        ],
    )
    def test_parses_expected_types(self, text, expected):
        assert _parse_scalar(text) == expected

    def test_leading_zero_stays_a_string(self):
        """KITTI sequence ids like '00' and '05' must not become ints."""
        assert _parse_scalar("00") == "00"
        assert _parse_scalar("05") == "05"
        assert isinstance(_parse_scalar("00"), str)


class TestEndToEndCLI:
    def test_run_slam_writes_results_and_exits_zero(self, fast_config_file, tmp_path, capsys):
        out_dir = tmp_path / "slam_out"
        code = slam_main(["-c", str(fast_config_file), "-s", "00", "-o", str(out_dir)])
        assert code == 0
        assert (out_dir / "metrics.json").is_file()
        assert (out_dir / "run.log").is_file()
        assert "KITTI sequence 00 - results" in capsys.readouterr().out

    def test_run_slam_metrics_reflect_the_run(self, fast_config_file, tmp_path):
        out_dir = tmp_path / "slam_out"
        slam_main(["-c", str(fast_config_file), "-s", "00", "-o", str(out_dir)])
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["sequence"] == "00"
        assert metrics["frames"] == 25
        assert metrics["ground_truth_available"] is True

    def test_run_vo_disables_loop_closure(self, fast_config_file, tmp_path):
        out_dir = tmp_path / "vo_out"
        assert vo_main(["-c", str(fast_config_file), "-s", "00", "-o", str(out_dir)]) == 0
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["loop_closure"]["enabled"] is False
        assert metrics["loop_closures_detected"] == 0

    def test_max_frames_is_honoured(self, fast_config_file, tmp_path):
        out_dir = tmp_path / "short"
        slam_main(
            ["-c", str(fast_config_file), "-s", "00", "-o", str(out_dir), "--max-frames", "8"]
        )
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["frames"] == 8

    def test_missing_sequence_exits_with_code_two(self, fast_config_file, tmp_path, capsys):
        code = slam_main(["-c", str(fast_config_file), "-s", "42", "-o", str(tmp_path / "x")])
        assert code == 2
        assert "Dataset error" in capsys.readouterr().err

    def test_bad_config_exits_with_code_two(self, capsys):
        assert slam_main(["-c", "/nonexistent.yaml"]) == 2
        assert "Configuration error" in capsys.readouterr().err

    def test_evaluate_scores_a_saved_trajectory(
        self, fast_config_file, tmp_path, kitti_root, capsys
    ):
        out_dir = tmp_path / "slam_out"
        slam_main(["-c", str(fast_config_file), "-s", "00", "-o", str(out_dir)])

        json_out = tmp_path / "eval.json"
        code = evaluate_main(
            [
                str(out_dir / "trajectory_raw.txt"),
                "--sequence", "00",
                "--dataset-path", str(kitti_root),
                "--json", str(json_out),
                "--rpe-deltas", "1", "5",
                "--log-level", "ERROR",
            ]
        )
        assert code == 0
        assert "ATE RMSE" in capsys.readouterr().out
        results = json.loads(json_out.read_text(encoding="utf-8"))
        assert "trajectory_raw" in results

    def test_evaluate_rejects_a_length_mismatch(self, tmp_path, kitti_root, capsys):
        from monocular_slam.geometry.pose import Trajectory

        short = Trajectory(np.repeat(np.eye(4)[None], 3, axis=0))
        short.save_kitti(tmp_path / "short.txt")
        code = evaluate_main(
            [
                str(tmp_path / "short.txt"),
                "--sequence", "00",
                "--dataset-path", str(kitti_root),
                "--log-level", "ERROR",
            ]
        )
        assert code == 2
        assert "Length mismatch" in capsys.readouterr().err

    def test_evaluate_refuses_a_sequence_without_ground_truth(self, tmp_path, capsys):
        root = write_kitti_sequence(tmp_path / "d", "00", n_frames=6, with_ground_truth=False)
        from monocular_slam.geometry.pose import Trajectory

        Trajectory(np.repeat(np.eye(4)[None], 6, axis=0)).save_kitti(tmp_path / "t.txt")
        code = evaluate_main(
            [
                str(tmp_path / "t.txt"),
                "--sequence", "00",
                "--dataset-path", str(root),
                "--log-level", "ERROR",
            ]
        )
        assert code == 2
        assert "no ground-truth poses" in capsys.readouterr().err


class TestLogging:
    def test_file_log_stays_complete_when_the_console_is_quiet(self, tmp_path):
        """run.log must record the run even at a raised console threshold."""
        log_path = tmp_path / "run.log"
        setup_logging("ERROR", log_file=log_path)
        get_logger("monocular_slam.test").info("an informational message")
        logging.getLogger("monocular_slam").handlers[-1].flush()
        assert "an informational message" in log_path.read_text(encoding="utf-8")

    def test_file_level_can_be_raised_independently(self, tmp_path):
        log_path = tmp_path / "run.log"
        setup_logging("INFO", log_file=log_path, file_level="ERROR")
        get_logger("monocular_slam.test").info("should not be written")
        logging.getLogger("monocular_slam").handlers[-1].flush()
        assert log_path.read_text(encoding="utf-8") == ""

    def test_repeated_setup_does_not_duplicate_handlers(self, tmp_path):
        for _ in range(3):
            setup_logging("INFO", log_file=tmp_path / "run.log")
        assert len(logging.getLogger("monocular_slam").handlers) == 2

    def test_quiet_suppresses_only_the_console(self, tmp_path):
        setup_logging("INFO", log_file=tmp_path / "run.log", quiet=True)
        handlers = logging.getLogger("monocular_slam").handlers
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.FileHandler)

    def test_get_logger_strips_the_package_prefix(self):
        assert get_logger("monocular_slam.features.detector").name == (
            "monocular_slam.features.detector"
        )
        assert get_logger("monocular_slam").name == "monocular_slam"
