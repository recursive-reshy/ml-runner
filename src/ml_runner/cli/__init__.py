from pathlib import Path
from typing import Annotated, Any

import typer
from typer.core import TyperGroup

from ml_runner import __version__
from ml_runner.cli.config_cmd import config_app
from ml_runner.cli.doctor_cmd import doctor
from ml_runner.cli.init_cmd import init
from ml_runner.cli.state import State
from ml_runner.errors import MlRunnerError

# Turns `MlRunnerError` into a message on stderr and its exit code.
class MlRunnerGroup( TyperGroup ):
    # `ctx` is Any: typer vendors click, and its Context type is private.
    def invoke( self, ctx: Any ) -> Any:
        try:
            return super().invoke( ctx )
        except MlRunnerError as error:
            typer.echo( f"Error: {error.message}", err = True )
            state = ctx.obj
            if error.detail and isinstance( state, State ) and state.verbose:
                typer.echo( error.detail, err = True )

            raise typer.Exit( error.exit_code ) from error

def _version_callback( value: bool ) -> None:
    if value:
        typer.echo( f"ml-runner {__version__}" )
        raise typer.Exit()

def root(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option( "--version", callback = _version_callback, is_eager = True, help = "Show the version and exit." )
    ] = False,
    verbose: Annotated[ bool, typer.Option( "--verbose", help = "Show raw gcloud output on errors." ) ] = False,
    config: Annotated[ Path | None, typer.Option( "--config", help = "Path to the config file." ) ] = None
) -> None:
    preset_runner = ctx.obj.runner if isinstance( ctx.obj, State ) else None
    ctx.obj = State( verbose = verbose, config_path = config, runner = preset_runner )

app = typer.Typer( name = "ml-runner", cls = MlRunnerGroup, no_args_is_help = True, add_completion = False )
app.callback( help = "Run any ML training repo on a disposable Google Cloud GPU VM." )( root )
app.add_typer( config_app, name = "config" )
app.command( "doctor", help = "Check gcloud, the project, the bucket, permissions and GPU quota." )( doctor )
app.command( "init", help = "Set up the project, bucket and runner identity (safe to run again)." )( init )
