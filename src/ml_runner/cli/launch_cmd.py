import tempfile
from pathlib import Path
from typing import Annotated

import typer

from ml_runner.cli.state import State, get_config
from ml_runner.gcloud import Gcloud
from ml_runner.launching import LaunchFlags, describe_plan, execute_launch, prepare_launch, resolve_options
from ml_runner.source import ls_remote

DRY_RUN_HEADER = "Dry run: nothing will be uploaded or created. This is the plan:"
COMMANDS_HEADER = "These are the commands ml-runner would run:"
FOLLOW_COMMANDS = ( "status {run}", "logs {run} --follow", "fetch {run}", "cancel {run}" )

def launch(
    ctx: typer.Context,
    command: Annotated[ list[ str ], typer.Argument( help = "The command to run, after `--`." ) ],
    gpu: Annotated[ str | None, typer.Option( "--gpu", help = "t4, l4 or none (default from config)." ) ] = None,
    spot: Annotated[
        bool,
        typer.Option( "--spot", help = "Use a Spot VM: cheaper, but it can be pre-empted." )
    ] = False,
    machine_type: Annotated[
        str | None,
        typer.Option( "--machine-type", help = "Override the machine type (T4 needs n1-*, L4 needs g2-*)." )
    ] = None,
    disk_gb: Annotated[ int | None, typer.Option( "--disk-gb", help = "Boot disk size in GB." ) ] = None,
    zone: Annotated[ str | None, typer.Option( "--zone", help = "Try this zone first." ) ] = None,
    max_hours: Annotated[
        float | None,
        typer.Option( "--max-hours", help = "Stop the job after this many hours (default from config)." )
    ] = None,
    allow_long: Annotated[ bool, typer.Option( "--allow-long", help = "Allow --max-hours above 24." ) ] = False,
    name: Annotated[
        str | None,
        typer.Option( "--name", help = "Name of the run (default from the command)." )
    ] = None,
    experiment: Annotated[
        str | None,
        typer.Option( "--experiment", help = "Group runs under this name (default: the repo or folder name)." )
    ] = None,
    env: Annotated[
        list[ str ] | None,
        typer.Option( "--env", help = "KEY=VALUE for the run; repeatable. Not for secrets (visible in the project)." )
    ] = None,
    src: Annotated[
        Path | None,
        typer.Option( "--src", help = "Folder to upload as a snapshot (default: the current folder)." )
    ] = None,
    repo: Annotated[ str | None, typer.Option( "--repo", help = "Public https git URL to clone on the VM." ) ] = None,
    ref: Annotated[
        str | None,
        typer.Option( "--ref", help = "Branch, tag or commit to check out with --repo." )
    ] = None,
    max_source_mb: Annotated[
        int | None,
        typer.Option( "--max-source-mb", help = "Largest snapshot to upload (default from config)." )
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option( "--dry-run", help = "Print the plan and the gcloud commands without running them." )
    ] = False
) -> None:
    state: State = ctx.obj
    config = get_config( ctx )
    options = resolve_options(
        config,
        LaunchFlags(
            command = command,
            gpu = gpu,
            is_spot = spot,
            machine_type = machine_type,
            disk_gb = disk_gb,
            zone = zone,
            max_hours = max_hours,
            allow_long = allow_long,
            name = name,
            experiment = experiment,
            env = env or [],
            src = src,
            repo = repo,
            ref = ref,
            max_source_mb = max_source_mb
        )
    )
    gcloud = Gcloud( project = config.project, dry_run = dry_run, runner = state.runner, echo = typer.echo )
    with tempfile.TemporaryDirectory( prefix = "ml-runner-" ) as directory:
        plan = prepare_launch(
            gcloud,
            config,
            options,
            Path( directory ),
            ref_check = state.ls_remote or ls_remote
        )
        typer.echo( DRY_RUN_HEADER if dry_run else "Launching:" )
        for line in describe_plan( plan ):
            typer.echo( line )

        if dry_run:
            typer.echo( COMMANDS_HEADER )

        result = execute_launch( gcloud, plan )

    if dry_run:
        return

    run = plan.runs[ 0 ]
    typer.echo( f"Launched job {plan.job.job_id} on {run.machine.vm} in {result.zone}." )
    typer.echo( "Follow it with:" )
    for template in FOLLOW_COMMANDS:
        typer.echo( f"  ml-runner {template.format( run = run.run_id )}" )
