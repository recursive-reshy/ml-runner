from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import typer

from ml_runner.config import Config, config_path, load_config
from ml_runner.gcloud import Runner

# Global options, available to commands as `ctx.obj`. `runner` and `ls_remote` are only set by tests,
# to swap in FakeGcloud and to keep the git ref check off the network.
@dataclass( frozen = True )
class State:
    verbose: bool = False
    config_path: Path | None = None
    runner: Runner | None = None
    ls_remote: Callable[ [ str, str ], str ] | None = None

def get_config_path( ctx: typer.Context ) -> Path:
    state: State = ctx.obj

    return config_path( state.config_path )

# Exits 2 with "Run `ml-runner init` first" when there is no config.
# Every command except `init`, `doctor` and `--version` calls this.
def get_config( ctx: typer.Context ) -> Config:
    return load_config( get_config_path( ctx ) )
