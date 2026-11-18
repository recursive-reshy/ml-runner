from pathlib import Path
from typing import Annotated

import typer

from ml_runner.checks import BUDGETS_URL, read_gpu_quotas, summarise_quotas
from ml_runner.cli.state import State, get_config_path
from ml_runner.config import Config, load_config, save_config
from ml_runner.errors import UsageError
from ml_runner.gcloud import Gcloud, GcloudError
from ml_runner.setup import (
    DEFAULT_REGION,
    InitSettings,
    StepResult,
    build_config,
    build_steps,
    choose_zone,
    default_bucket,
    resolve_project,
    run_steps
)

DRY_RUN_HEADER = "Dry run: nothing will be created. These are the commands ml-runner would run:"
STATUS_LINES = { "exists": "✓ exists", "created": "+ created", "planned": "~ would create" }

def _print_result( result: StepResult ) -> None:
    typer.echo( f"{STATUS_LINES[ result.status ]}: {result.name}" )

# An unreadable config must not block `init`, the command that rewrites it.
def _existing_config( path: Path ) -> Config | None:
    if not path.exists():
        return None

    try:
        return load_config( path )
    except UsageError:
        return None

def _print_quotas( gcloud: Gcloud, region: str ) -> None:
    typer.echo( "GPU quota in " + region + ":" )
    try:
        quotas = read_gpu_quotas( gcloud, region )
    except GcloudError as error:
        typer.echo( f"Could not read quotas: {error.message}" )
        return

    for line in summarise_quotas( quotas ):
        typer.echo( f"  {line}" )

def init(
    ctx: typer.Context,
    project: Annotated[
        str | None,
        typer.Option( "--project", help = "Project id (default: gcloud's current project)." )
    ] = None,
    region: Annotated[ str, typer.Option( "--region", help = "Region for the bucket and VMs." ) ] = DEFAULT_REGION,
    zone: Annotated[ str | None, typer.Option( "--zone", help = "Zone (default: the first with a T4 GPU)." ) ] = None,
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
    config_file = get_config_path( ctx )
    existing = _existing_config( config_file )
    # A re-run keeps the zone the user already chose for this region.
    kept_zone = existing.zone if existing and existing.region == region else None
    zone_name = choose_zone( gcloud, region, zone or kept_zone )
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

    config = build_config( settings, zone_name, existing )
    if dry_run:
        typer.echo( f"Would write config to {config_file} (zone {zone_name})." )
    else:
        save_config( config_file, config )
        typer.echo( f"Config written to {config_file} (zone {zone_name})." )

    _print_quotas( gcloud, region )
    typer.echo( f"Set a budget alert: {BUDGETS_URL}" )
