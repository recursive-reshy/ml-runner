"""Global CLI state and the helper commands use to get the config."""

from dataclasses import dataclass
from pathlib import Path

import typer

from ml_runner.config import Config, config_path, load_config


@dataclass(frozen=True)
class State:
    """Global options, available to commands as `ctx.obj`."""

    verbose: bool = False
    config_path: Path | None = None


def get_config_path(ctx: typer.Context) -> Path:
    state: State = ctx.obj
    return config_path(state.config_path)


def get_config(ctx: typer.Context) -> Config:
    """Load the config, or exit 2 with "Run `ml-runner init` first".

    Every command except `init`, `doctor` and `--version` calls this.
    """
    return load_config(get_config_path(ctx))
