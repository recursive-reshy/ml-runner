"""The Typer app: parse, call one orchestration function, print (TRD §14)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from typer.core import TyperGroup

from ml_runner import __version__
from ml_runner.errors import MlRunnerError


@dataclass(frozen=True)
class State:
    """Global options, available to commands as `ctx.obj`."""

    verbose: bool = False
    config_path: Path | None = None


class MlRunnerGroup(TyperGroup):
    """Turns `MlRunnerError` into a message on stderr and its exit code."""

    # `ctx` is Any: typer vendors click, and its Context type is private.
    def invoke(self, ctx: Any) -> Any:
        try:
            return super().invoke(ctx)
        except MlRunnerError as error:
            typer.echo(f"Error: {error.message}", err=True)
            state = ctx.obj
            if error.detail and isinstance(state, State) and state.verbose:
                typer.echo(error.detail, err=True)
            raise typer.Exit(error.exit_code) from error


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"ml-runner {__version__}")
        raise typer.Exit()


def root(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Show raw gcloud output on errors.")
    ] = False,
    config: Annotated[
        Path | None, typer.Option("--config", help="Path to the config file.")
    ] = None,
) -> None:
    """Run any ML training repo on a disposable Google Cloud GPU VM."""
    ctx.obj = State(verbose=verbose, config_path=config)


app = typer.Typer(
    name="ml-runner", cls=MlRunnerGroup, no_args_is_help=True, add_completion=False
)
app.callback()(root)
