"""Tests for resume-metrics generation.

The property that matters most: unmeasured values must never become claims.
"""

from __future__ import annotations

import json

import pytest

from monocular_slam.cli.report import main as report_main
from monocular_slam.reporting import (
    RunSummary,
    load_run_summaries,
    measured_highlights,
    publish_figures,
    render_resume_metrics,
    results_table,
    resume_bullets,
    update_readme_results,
    write_resume_metrics,
)

FULL_METRICS = {
    "sequence": "00",
    "frames": 4541,
    "distance_km": 3.72,
    "successful_pose_rate_pct": 99.5,
    "ate_rmse_raw_m": 12.5,
    "ate_rmse_optimized_m": 5.0,
    "ate_reduction_pct": 60.0,
    "translational_drift_raw_pct": 2.2,
    "translational_drift_optimized_pct": 1.1,
    "rotational_drift_deg_per_100m": 0.5,
    "loop_closures_detected": 30,
    "avg_features_per_frame": 2980.0,
    "avg_matches_per_pair": 900.0,
    "avg_inlier_ratio_pct": 68.0,
    "runtime_fps": 14.5,
    "scale_uses_ground_truth": True,
    "alignment": "sim3",
    "loop_closure": {"n_keyframes": 1595},
}

NO_GROUND_TRUTH_METRICS = {
    "sequence": "11",
    "frames": 900,
    "distance_km": None,
    "successful_pose_rate_pct": 98.0,
    "ate_rmse_raw_m": None,
    "ate_rmse_optimized_m": None,
    "ate_reduction_pct": None,
    "translational_drift_raw_pct": None,
    "translational_drift_optimized_pct": None,
    "rotational_drift_deg_per_100m": None,
    "loop_closures_detected": 0,
    "avg_features_per_frame": 2900.0,
    "avg_matches_per_pair": 800.0,
    "avg_inlier_ratio_pct": 65.0,
    "runtime_fps": 15.0,
    "scale_uses_ground_truth": False,
    "alignment": "sim3",
    "loop_closure": {"n_keyframes": 300},
}


def write_run(root, sequence: str, metrics: dict):
    run_dir = root / f"sequence_{sequence}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run_dir


class TestLoading:
    def test_loads_a_run(self, tmp_path):
        write_run(tmp_path, "00", FULL_METRICS)
        summaries = load_run_summaries(tmp_path)
        assert len(summaries) == 1
        assert summaries[0].sequence == "00"
        assert summaries[0].ate_rmse_optimized_m == 5.0

    def test_loads_several_runs_sorted_by_sequence(self, tmp_path):
        write_run(tmp_path, "05", {**FULL_METRICS, "sequence": "05"})
        write_run(tmp_path, "00", FULL_METRICS)
        assert [s.sequence for s in load_run_summaries(tmp_path)] == ["00", "05"]

    def test_missing_directory_yields_nothing(self, tmp_path):
        assert load_run_summaries(tmp_path / "nope") == []

    def test_corrupt_metrics_file_is_skipped(self, tmp_path):
        write_run(tmp_path, "00", FULL_METRICS)
        bad = tmp_path / "sequence_99"
        bad.mkdir()
        (bad / "metrics.json").write_text("{not json", encoding="utf-8")
        assert len(load_run_summaries(tmp_path)) == 1


class TestResultsTable:
    def test_renders_measured_values(self, tmp_path):
        table = results_table([RunSummary.from_metrics(FULL_METRICS)])
        assert "| 00 " in table
        assert "4541" in table
        assert "5.00 m" in table

    def test_unmeasured_values_render_as_na(self):
        table = results_table([RunSummary.from_metrics(NO_GROUND_TRUTH_METRICS)])
        assert "n/a" in table
        # No fabricated zero or placeholder number in the ATE columns.
        assert "0.00 m" not in table

    def test_empty_input_explains_how_to_populate(self):
        assert "run_slam.py" in results_table([])


class TestHighlights:
    def test_highlights_use_measured_numbers(self):
        highlights = measured_highlights([RunSummary.from_metrics(FULL_METRICS)])
        joined = " ".join(highlights)
        assert "4,541 KITTI frames" in joined
        assert "3.72 km" in joined
        assert "5.00 m optimized ATE RMSE" in joined
        assert "60.0%" in joined
        assert "30 loop closures" in joined

    def test_no_ate_claim_without_ground_truth(self):
        joined = " ".join(measured_highlights([RunSummary.from_metrics(NO_GROUND_TRUTH_METRICS)]))
        assert "ATE" not in joined
        assert "drift" not in joined

    def test_no_loop_claim_when_none_were_found(self):
        joined = " ".join(measured_highlights([RunSummary.from_metrics(NO_GROUND_TRUTH_METRICS)]))
        assert "loop closures" not in joined

    def test_empty_input_yields_no_highlights(self):
        assert measured_highlights([]) == []


class TestResumeBullets:
    def test_generates_at_most_three(self):
        assert len(resume_bullets([RunSummary.from_metrics(FULL_METRICS)])) <= 3

    def test_bullets_carry_the_scale_caveat(self):
        joined = " ".join(resume_bullets([RunSummary.from_metrics(FULL_METRICS)]))
        assert "scale" in joined.lower()

    def test_no_scale_caveat_when_scale_was_not_borrowed(self):
        summary = RunSummary.from_metrics({**FULL_METRICS, "scale_uses_ground_truth": False})
        joined = " ".join(resume_bullets([summary]))
        assert "ground-truth-scaled" not in joined

    def test_no_accuracy_bullet_without_ground_truth(self):
        bullets = resume_bullets([RunSummary.from_metrics(NO_GROUND_TRUTH_METRICS)])
        joined = " ".join(bullets)
        assert "absolute trajectory error" not in joined.lower()
        assert bullets  # a system-description bullet is still valid

    def test_no_improvement_claim_without_an_improvement(self):
        summary = RunSummary.from_metrics(
            {**FULL_METRICS, "ate_reduction_pct": None, "loop_closures_detected": 0}
        )
        joined = " ".join(resume_bullets([summary]))
        assert "Cut absolute trajectory error" not in joined

    def test_empty_input_yields_no_bullets(self):
        assert resume_bullets([]) == []


class TestDocument:
    def test_document_contains_all_sections(self):
        document = render_resume_metrics([RunSummary.from_metrics(FULL_METRICS)])
        for section in ("# Resume metrics", "## Results", "## Measured highlights",
                        "## Suggested resume bullets", "## Metric definitions"):
            assert section in document

    def test_scale_caveat_is_prominent(self):
        document = render_resume_metrics([RunSummary.from_metrics(FULL_METRICS)])
        assert "Scale caveat" in document
        assert "cannot recover absolute scale" in document

    def test_no_runs_document_explains_itself(self):
        document = render_resume_metrics([])
        assert "No runs found" in document
        assert "run_slam.py" in document

    def test_write_creates_the_file(self, tmp_path):
        write_run(tmp_path, "00", FULL_METRICS)
        path = write_resume_metrics(tmp_path)
        assert path.is_file()
        assert "# Resume metrics" in path.read_text(encoding="utf-8")

    def test_custom_destination(self, tmp_path):
        write_run(tmp_path, "00", FULL_METRICS)
        path = write_resume_metrics(tmp_path, tmp_path / "custom" / "out.md")
        assert path.name == "out.md"


class TestReadmeInjection:
    def test_replaces_the_marked_block(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text(
            "# Title\n\n## Results\n\n<!-- RESULTS:START -->\nplaceholder\n"
            "<!-- RESULTS:END -->\n\n## Next\n",
            encoding="utf-8",
        )
        write_run(tmp_path, "00", FULL_METRICS)
        assert update_readme_results(readme, tmp_path) is True

        text = readme.read_text(encoding="utf-8")
        assert "placeholder" not in text
        assert "| 00 " in text
        # Surrounding content must be untouched.
        assert text.startswith("# Title")
        assert text.rstrip().endswith("## Next")

    def test_injection_is_idempotent(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text(
            "a\n<!-- RESULTS:START -->\nx\n<!-- RESULTS:END -->\nb\n", encoding="utf-8"
        )
        write_run(tmp_path, "00", FULL_METRICS)
        update_readme_results(readme, tmp_path)
        first = readme.read_text(encoding="utf-8")
        update_readme_results(readme, tmp_path)
        assert readme.read_text(encoding="utf-8") == first

    def test_missing_markers_are_left_alone(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("# No markers here\n", encoding="utf-8")
        assert update_readme_results(readme, tmp_path) is False
        assert readme.read_text(encoding="utf-8") == "# No markers here\n"

    def test_missing_readme_is_not_an_error(self, tmp_path):
        assert update_readme_results(tmp_path / "nope.md", tmp_path) is False

    def test_scale_caveat_appears_in_the_block(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n", encoding="utf-8")
        write_run(tmp_path, "00", FULL_METRICS)
        update_readme_results(readme, tmp_path)
        assert "cannot recover absolute scale" in readme.read_text(encoding="utf-8")

    def test_repo_readme_has_the_markers(self):
        """Guards against the markers being edited out of the real README."""
        from pathlib import Path as _Path

        readme = _Path(__file__).resolve().parent.parent / "README.md"
        text = readme.read_text(encoding="utf-8")
        assert "<!-- RESULTS:START -->" in text
        assert "<!-- RESULTS:END -->" in text


class TestRegressionReporting:
    """Sequences where optimization hurt must be reported, not quietly dropped."""

    REGRESSED = {
        **FULL_METRICS,
        "sequence": "01",
        "ate_rmse_raw_m": 16.2,
        "ate_rmse_optimized_m": 17.7,
        "ate_reduction_pct": -9.0,
        "loop_closures_detected": 1,
    }

    def test_regression_is_called_out(self, tmp_path):
        write_run(tmp_path, "01", self.REGRESSED)
        document = render_resume_metrics(load_run_summaries(tmp_path))
        assert "Where it does not help" in document
        assert "Sequence 01" in document
        assert "-9.0%" in document

    def test_no_note_when_everything_improved(self, tmp_path):
        write_run(tmp_path, "00", FULL_METRICS)
        assert "Where it does not help" not in render_resume_metrics(load_run_summaries(tmp_path))

    def test_highlights_still_use_the_best_run(self, tmp_path):
        write_run(tmp_path, "00", FULL_METRICS)
        write_run(tmp_path, "01", self.REGRESSED)
        summaries = load_run_summaries(tmp_path)
        joined = " ".join(measured_highlights(summaries))
        assert "60.0%" in joined  # the sequence-00 improvement, not the regression
        assert "-9.0" not in joined

    def test_readme_block_also_carries_the_note(self, tmp_path):
        from monocular_slam.reporting import render_readme_results

        write_run(tmp_path, "01", self.REGRESSED)
        assert "Where it does not help" in render_readme_results(load_run_summaries(tmp_path))


class TestFigurePublishing:
    def test_copies_known_figures(self, tmp_path):
        run_dir = write_run(tmp_path, "00", FULL_METRICS)
        for name in ("trajectory_comparison.png", "loop_closures.png", "error_over_time.png"):
            (run_dir / name).write_bytes(b"fake png bytes")
        (run_dir / "diagnostics.png").write_bytes(b"not published")

        written = publish_figures(tmp_path, tmp_path / "docs")
        assert len(written) == 3
        assert (tmp_path / "docs" / "sequence_00" / "trajectory_comparison.png").is_file()
        assert not (tmp_path / "docs" / "sequence_00" / "diagnostics.png").exists()

    def test_missing_figures_are_skipped(self, tmp_path):
        write_run(tmp_path, "00", FULL_METRICS)
        assert publish_figures(tmp_path, tmp_path / "docs") == []

    def test_directories_without_metrics_are_ignored(self, tmp_path):
        stray = tmp_path / "sequence_99"
        stray.mkdir()
        (stray / "trajectory_comparison.png").write_bytes(b"x")
        assert publish_figures(tmp_path, tmp_path / "docs") == []

    def test_readme_embeds_published_figures(self, tmp_path, monkeypatch):
        import monocular_slam.reporting as reporting

        run_dir = write_run(tmp_path, "00", FULL_METRICS)
        for name, _caption in reporting.PUBLISHED_FIGURES:
            (run_dir / name).write_bytes(b"x")
        figure_dir = tmp_path / "docs" / "results"
        publish_figures(tmp_path, figure_dir)
        monkeypatch.setattr(reporting, "FIGURE_DIR", figure_dir)

        block = reporting.render_readme_results(load_run_summaries(tmp_path))
        assert "![" in block
        assert "trajectory_comparison.png" in block

    def test_readme_points_at_the_command_when_nothing_is_published(self, tmp_path, monkeypatch):
        import monocular_slam.reporting as reporting

        write_run(tmp_path, "00", FULL_METRICS)
        # Point at an empty directory: the repository's real docs/results/ may
        # already hold published figures from a benchmark run.
        monkeypatch.setattr(reporting, "FIGURE_DIR", tmp_path / "empty")
        block = reporting.render_readme_results(load_run_summaries(tmp_path))
        assert "--publish-figures" in block


class TestReportCLI:
    def test_exits_zero_when_runs_exist(self, tmp_path, capsys):
        write_run(tmp_path, "00", FULL_METRICS)
        assert report_main(["--outputs", str(tmp_path), "--log-level", "ERROR"]) == 0
        assert "Wrote" in capsys.readouterr().out

    def test_exits_nonzero_and_explains_when_no_runs(self, tmp_path, capsys):
        assert report_main(["--outputs", str(tmp_path), "--log-level", "ERROR"]) == 1
        assert "No metrics.json files found" in capsys.readouterr().err

    def test_print_flag_emits_the_document(self, tmp_path, capsys):
        write_run(tmp_path, "00", FULL_METRICS)
        report_main(["--outputs", str(tmp_path), "--print", "--log-level", "ERROR"])
        assert "# Resume metrics" in capsys.readouterr().out

    def test_help_renders(self, capsys):
        from monocular_slam.cli.report import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(["--help"])
        assert "usage:" in capsys.readouterr().out
