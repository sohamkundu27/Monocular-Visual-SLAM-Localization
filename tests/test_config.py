"""Tests for the typed configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from monocular_slam.config import Config, ConfigError

REPO_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "kitti.yaml"


class TestDefaults:
    def test_defaults_are_constructible(self):
        config = Config()
        assert config.dataset.sequence == "00"
        assert config.features.detector == "orb"
        assert config.evaluation.alignment == "sim3"

    def test_run_dir_substitutes_the_sequence(self):
        config = Config().with_overrides({"dataset.sequence": "07"})
        assert config.run_dir == Path("outputs/sequence_07")

    def test_tuple_fields_are_tuples(self):
        assert isinstance(Config().evaluation.rpe_deltas, tuple)


class TestShippedConfig:
    def test_repo_config_loads(self):
        assert REPO_CONFIG.is_file(), "configs/kitti.yaml is missing from the repository"
        Config.from_yaml(REPO_CONFIG)

    def test_repo_config_covers_every_section(self):
        """A section silently missing from the YAML would revert to defaults."""
        data = yaml.safe_load(REPO_CONFIG.read_text(encoding="utf-8"))
        expected = {f.name for f in Config.__dataclass_fields__.values()}
        assert expected <= set(data)

    def test_sequence_id_survives_as_a_string(self):
        """'00' must not be parsed as the integer 0."""
        sequence = Config.from_yaml(REPO_CONFIG).dataset.sequence
        assert sequence == "00"
        assert isinstance(sequence, str)

    def test_repo_config_round_trips(self, tmp_path):
        original = Config.from_yaml(REPO_CONFIG)
        reloaded = Config.from_yaml(original.to_yaml(tmp_path / "copy.yaml"))
        assert reloaded.to_dict() == original.to_dict()


class TestLoading:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            Config.from_yaml(tmp_path / "nope.yaml")

    def test_empty_file_gives_defaults(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert Config.from_yaml(path).dataset.sequence == Config().dataset.sequence

    def test_non_mapping_root_raises(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="must be a mapping"):
            Config.from_yaml(path)

    def test_partial_config_keeps_other_defaults(self, tmp_path):
        path = tmp_path / "partial.yaml"
        path.write_text("odometry:\n  min_inliers: 55\n", encoding="utf-8")
        config = Config.from_yaml(path)
        assert config.odometry.min_inliers == 55
        assert config.features.max_features == Config().features.max_features


class TestValidation:
    def test_unknown_section_is_rejected(self):
        with pytest.raises(ConfigError, match="Unknown config key"):
            Config.from_dict({"odometery": {}})

    def test_unknown_key_within_a_section_is_rejected(self):
        with pytest.raises(ConfigError, match="Unknown config key"):
            Config.from_dict({"odometry": {"min_inlyers": 3}})

    def test_error_message_lists_valid_keys(self):
        with pytest.raises(ConfigError, match="Valid keys"):
            Config.from_dict({"odometry": {"typo": 1}})

    def test_wrong_type_is_rejected(self):
        with pytest.raises(ConfigError, match="expects an int"):
            Config.from_dict({"odometry": {"min_inliers": "many"}})

    def test_null_for_a_non_optional_key_is_rejected(self):
        with pytest.raises(ConfigError, match="does not accept null"):
            Config.from_dict({"odometry": {"min_inliers": None}})

    def test_null_is_allowed_for_optional_keys(self):
        assert Config.from_dict({"dataset": {"max_frames": None}}).dataset.max_frames is None
        assert Config.from_dict({"matcher": {"max_distance": None}}).matcher.max_distance is None

    def test_section_given_a_scalar_is_rejected(self):
        with pytest.raises(ConfigError, match="Expected a mapping"):
            Config.from_dict({"odometry": 5})


class TestCoercion:
    @pytest.mark.parametrize("value,expected", [("true", True), ("no", False), (True, True)])
    def test_boolean_coercion(self, value, expected):
        config = Config.from_dict({"loop_closure": {"enabled": value}})
        assert config.loop_closure.enabled is expected

    def test_bad_boolean_is_rejected(self):
        with pytest.raises(ConfigError, match="expects a boolean"):
            Config.from_dict({"loop_closure": {"enabled": "maybe"}})

    def test_int_is_widened_to_float(self):
        config = Config.from_dict({"odometry": {"ransac_threshold_px": 1}})
        assert config.odometry.ransac_threshold_px == 1.0
        assert isinstance(config.odometry.ransac_threshold_px, float)

    def test_int_field_rejects_a_bare_bool(self):
        with pytest.raises(ConfigError, match="expects an int"):
            Config.from_dict({"odometry": {"min_inliers": True}})

    def test_tuple_field_accepts_a_yaml_list(self):
        config = Config.from_dict({"evaluation": {"rpe_deltas": [1, 2, 3]}})
        assert config.evaluation.rpe_deltas == (1, 2, 3)

    def test_tuple_field_rejects_a_scalar(self):
        with pytest.raises(ConfigError, match="expects a list"):
            Config.from_dict({"evaluation": {"rpe_deltas": 5}})


class TestOverrides:
    def test_dotted_override_applies(self):
        assert Config().with_overrides({"odometry.min_inliers": 99}).odometry.min_inliers == 99

    def test_override_leaves_the_original_untouched(self):
        config = Config()
        config.with_overrides({"odometry.min_inliers": 99})
        assert config.odometry.min_inliers == Config().odometry.min_inliers

    def test_none_values_are_skipped(self):
        """CLI flags default to None and must not clobber configured values."""
        config = Config().with_overrides({"dataset.sequence": "05"})
        assert config.with_overrides({"dataset.sequence": None}).dataset.sequence == "05"

    def test_unknown_key_is_rejected(self):
        with pytest.raises(ConfigError, match="Unknown config key"):
            Config().with_overrides({"odometry.nope": 1})

    def test_unknown_section_is_rejected(self):
        with pytest.raises(ConfigError, match="Unknown config section"):
            Config().with_overrides({"nope.key": 1})

    def test_multiple_overrides_all_apply(self):
        config = Config().with_overrides(
            {"dataset.sequence": "05", "features.max_features": 500, "loop_closure.enabled": False}
        )
        assert config.dataset.sequence == "05"
        assert config.features.max_features == 500
        assert config.loop_closure.enabled is False


class TestSerialisation:
    def test_to_dict_is_json_serialisable(self):
        import json

        json.loads(json.dumps(Config().to_dict()))

    def test_tuples_become_lists(self):
        assert isinstance(Config().to_dict()["evaluation"]["rpe_deltas"], list)

    def test_yaml_round_trip_preserves_everything(self, tmp_path):
        original = Config().with_overrides(
            {"dataset.sequence": "09", "odometry.min_inliers": 77, "evaluation.rpe_deltas": [2, 4]}
        )
        reloaded = Config.from_yaml(original.to_yaml(tmp_path / "c.yaml"))
        assert reloaded.to_dict() == original.to_dict()
        assert reloaded.evaluation.rpe_deltas == (2, 4)

    def test_to_yaml_creates_parent_directories(self, tmp_path):
        path = Config().to_yaml(tmp_path / "deep" / "nested" / "c.yaml")
        assert path.is_file()
