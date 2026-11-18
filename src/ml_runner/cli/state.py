from dataclasses import dataclass
from pathlib import Path

import typer

from ml_runner.config import Config, config_path, load_config

# Global options, available to commands as `ctx.obj`.
@dataclass( frozen = True )
class State:
    verbose: bool = False
    config_path: Path | None = None

def get_config_path( ctx: typer.Context ) -> Path:
    state: State = ctx.obj

    return config_path( state.config_path )

# Exits 2 with "Run `ml-runner init` first" when there is no config.
# Every command except `init`, `doctor` and `--version` calls this.
def get_config( ctx: typer.Context ) -> Config:
    return load_config( get_config_path( ctx ) )
