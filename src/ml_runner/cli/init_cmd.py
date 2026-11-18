from typing import Annotated

import typer

from ml_runner.cli.state import State
from ml_runner.gcloud import Gcloud
from ml_runner.setup import (
    DEFAULT_REGION,
    InitSettings,
    StepResult,
    build_steps,
    default_bucket,
    resolve_project,
    run_steps
)

DRY_RUN_HEADER = "Dry run: nothing will be created. These are the commands ml-runner would run:"
STATUS_LINES = { "exists": "✓ exists", "created": "+ created", "planned": "~ would create" }

def _print_result( result: StepResult ) -> None:
    typer.echo( f"{STATUS_LINES[ result.status ]}: {result.name}" )

def init(
    ctx: typer.Context,
    project: Annotated[
        str | None,
        typer.Option( "--project", help = "Project id (default: gcloud's current project)." )
    ] = None,
    region: Annotated[ str, typer.Option( "--region", help = "Region for the bucket and VMs." ) ] = DEFAULT_REGION,
    bucket: Annotated[
        str | None,
        typer.Option( "--bucket", help = "Bucket name (default: ml-runner-<project>)." )
    ] = None,
    yes: Annotated[ bool, typer.Option( "--yes", help = "Do not ask for confirmation." ) ] = False,
    dry_run: Annotated[
        bool,
        typer.Option( "--dry-run", help = "Print the gcloud commands without running them." )
    ] = False
) -> None:
    state: State = ctx.obj
    project_id = resolve_project( Gcloud( runner = state.runner ), project )
    settings = InitSettings( project_id, region, bucket or default_bucket( project_id ) )
    gcloud = Gcloud( project = project_id, dry_run = dry_run, runner = state.runner, echo = typer.echo )
    if dry_run:
        typer.echo( DRY_RUN_HEADER )

    def confirm( pending: list[ str ] ) -> bool:
        if yes:
            return True

        typer.echo( "ml-runner will change project " + project_id + ":" )
        for name in pending:
            typer.echo( f"  - {name}" )

        return typer.confirm( "Continue?" )

    for result in run_steps( build_steps( gcloud, settings ), confirm, is_dry_run = dry_run ):
        _print_result( result )
