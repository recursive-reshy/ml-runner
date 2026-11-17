"""The local config file: model, path resolution, load and save (TRD §3)."""

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

import platformdirs
import tomli_w
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ml_runner.errors import UsageError

ENV_VAR = "ML_RUNNER_CONFIG"
SETTABLE_TOP_LEVEL = ("zone",)


class Defaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpu: Literal["t4", "l4", "none"] = "t4"
    spot: bool = False
    max_hours: float = Field(default=6.0, ge=0.1, le=24)
    disk_gb: int = Field(default=100, gt=0)
    max_source_mb: int = Field(default=200, gt=0)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    project: str
    region: str
    zone: str
    bucket: str
    service_account: str
    defaults: Defaults = Field(default_factory=Defaults)

    @model_validator(mode="after")
    def _zone_is_in_region(self) -> Self:
        if not self.zone.startswith(f"{self.region}-"):
            raise ValueError(f"zone {self.zone!r} is not in region {self.region!r}")
        return self


def config_path(override: Path | None, env: Mapping[str, str] = os.environ) -> Path:
    """`--config` flag, else `$ML_RUNNER_CONFIG`, else the platform config dir."""
    if override is not None:
        return override
    if env.get(ENV_VAR):
        return Path(env[ENV_VAR])
    return Path(platformdirs.user_config_dir("ml-runner")) / "config.toml"


def _describe(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in e['loc']) or 'config'}: {e['msg']}"
        for e in error.errors()
    )


def load_config(path: Path) -> Config:
    if not path.exists():
        raise UsageError("No config found. Run `ml-runner init` first.")
    try:
        data = tomllib.loads(path.read_text())
        return Config.model_validate(data)
    except tomllib.TOMLDecodeError as error:
        raise UsageError(f"Invalid config file {path}: {error}") from error
    except ValidationError as error:
        raise UsageError(f"Invalid config file {path}: {_describe(error)}") from error


def save_config(path: Path, config: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(config.model_dump(mode="json")))


def set_value(config: Config, key: str, value: str) -> Config:
    """Return a copy of `config` with `defaults.*` or `zone` set, validated."""
    data = config.model_dump(mode="json")
    if key in SETTABLE_TOP_LEVEL:
        data[key] = value
    elif (
        key.startswith("defaults.")
        and key.removeprefix("defaults.") in Defaults.model_fields
    ):
        data["defaults"][key.removeprefix("defaults.")] = value
    else:
        allowed = [
            *SETTABLE_TOP_LEVEL,
            *(f"defaults.{k}" for k in Defaults.model_fields),
        ]
        raise UsageError(f"Cannot set {key!r}. Settable keys: {', '.join(allowed)}")
    try:
        return Config.model_validate(data)
    except ValidationError as error:
        raise UsageError(f"Invalid value for {key}: {_describe(error)}") from error
