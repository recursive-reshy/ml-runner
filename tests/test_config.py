import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from ml_runner.config import (
    Config,
    Defaults,
    config_path,
    load_config,
    save_config,
    set_value,
)
from ml_runner.errors import UsageError


def make_config(**overrides: object) -> Config:
    fields: dict[str, object] = {
        "project": "my-project",
        "region": "asia-southeast1",
        "zone": "asia-southeast1-b",
        "bucket": "my-project-ml-runner",
        "service_account": "ml-runner-vm@my-project.iam.gserviceaccount.com",
    }
    fields.update(overrides)
    return Config.model_validate(fields)


def test_defaults_match_the_trd() -> None:
    d = Defaults()
    assert (d.gpu, d.spot, d.max_hours, d.disk_gb, d.max_source_mb) == (
        "t4",
        False,
        6.0,
        100,
        200,
    )
    assert make_config().version == 1


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_config(surprise=1)
    with pytest.raises(ValidationError):
        Defaults.model_validate({"surprise": 1})


@pytest.mark.parametrize("hours", [0.0, 0.09, 24.5])
def test_max_hours_must_be_between_0_1_and_24(hours: float) -> None:
    with pytest.raises(ValidationError):
        Defaults(max_hours=hours)


def test_gpu_must_be_t4_l4_or_none() -> None:
    with pytest.raises(ValidationError):
        Defaults.model_validate({"gpu": "a100"})


def test_zone_must_be_in_the_region() -> None:
    with pytest.raises(ValidationError):
        make_config(zone="europe-west4-a")


def test_path_uses_override_first_then_env_then_platformdirs() -> None:
    override = Path("/x/override.toml")
    env = {"ML_RUNNER_CONFIG": "/x/env.toml"}
    assert config_path(override, env) == override
    assert config_path(None, env) == Path("/x/env.toml")
    default = config_path(None, {})
    assert default.name == "config.toml"
    assert "ml-runner" in default.parts


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "config.toml"
    config = make_config(defaults={"gpu": "l4", "spot": True, "max_hours": 2.5})
    save_config(path, config)
    assert load_config(path) == config
    raw = tomllib.loads(path.read_text())
    assert raw["project"] == "my-project"
    assert raw["defaults"]["gpu"] == "l4"


def test_missing_file_says_run_init_first(tmp_path: Path) -> None:
    with pytest.raises(UsageError, match="ml-runner init"):
        load_config(tmp_path / "nope.toml")


def test_invalid_file_is_a_usage_error_naming_the_path(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('version = 1\nproject = "p"\nbogus = 1\n')
    with pytest.raises(UsageError, match="config.toml"):
        load_config(path)
    path.write_text("not = [valid")
    with pytest.raises(UsageError, match="config.toml"):
        load_config(path)


def test_set_value_edits_defaults_and_zone() -> None:
    config = make_config()
    assert set_value(config, "defaults.gpu", "l4").defaults.gpu == "l4"
    assert set_value(config, "defaults.spot", "true").defaults.spot is True
    assert set_value(config, "defaults.max_hours", "3").defaults.max_hours == 3.0
    assert set_value(config, "defaults.disk_gb", "50").defaults.disk_gb == 50
    assert set_value(config, "zone", "asia-southeast1-c").zone == "asia-southeast1-c"
    assert config.defaults.gpu == "t4"  # the original is unchanged


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("project", "other"),
        ("defaults.nope", "1"),
        ("nope", "1"),
        ("defaults.gpu", "a100"),
        ("defaults.max_hours", "99"),
        ("defaults.disk_gb", "lots"),
        ("zone", "europe-west4-a"),
    ],
)
def test_set_value_rejects_bad_keys_and_values(key: str, value: str) -> None:
    with pytest.raises(UsageError):
        set_value(make_config(), key, value)
