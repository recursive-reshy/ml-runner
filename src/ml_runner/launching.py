import contextlib
import random
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast, get_args

from ml_runner import __version__, layout
from ml_runner.config import Config
from ml_runner.errors import MlRunnerError, UsageError
from ml_runner.gcloud import Gcloud
from ml_runner.ids import experiment_name, job_id, run_id, slugify, vm_name
from ml_runner.machines import (
    create_in_first_available_zone,
    image_path,
    profile_for,
    zone_candidates,
    zones_offering
)
from ml_runner.records import (
    Gpu,
    JobRun,
    JobSpec,
    Machine,
    RunRecord,
    RunStatus,
    Source,
    SourceKind,
    Times
)
from ml_runner.source import (
    Snapshot,
    build_snapshot,
    check_git_source,
    ls_remote,
    upload_snapshot,
    validate_source_flags
)
from ml_runner.vmspec import VmFiles, VmSpec, build_create_args, runtime_cap_minutes

MIN_HOURS = 0.1
MAX_HOURS = 24
ENV_NAME = re.compile( r"[A-Z_][A-Z0-9_]*" )
RESERVED_ENV_NAMES = ( "OUTPUT_DIR", "CHECKPOINT_DIR", "RUN_ID" )
RESERVED_ENV_PREFIX = "ML_RUNNER"
VM_DIR = Path( __file__ ).parent / "vm"
JOB_FILE = "job.json"
METADATA_FILE = "mlr-job.json"
HASH_PREFIX_LENGTH = 12
BYTES_PER_MB = 1024 * 1024

# What the user typed. `None` and empty values mean "not given", so the config defaults apply.
@dataclass( frozen = True )
class LaunchFlags:
    command: list[ str ]
    gpu: str | None = None
    is_spot: bool = False
    machine_type: str | None = None
    disk_gb: int | None = None
    zone: str | None = None
    max_hours: float | None = None
    allow_long: bool = False
    name: str | None = None
    experiment: str | None = None
    env: list[ str ] = field( default_factory = list )
    src: Path | None = None
    repo: str | None = None
    ref: str | None = None
    max_source_mb: int | None = None

@dataclass( frozen = True )
class LaunchOptions:
    command: list[ str ]
    gpu: Gpu
    is_spot: bool
    machine_type: str | None
    disk_gb: int
    zone: str | None
    max_hours: float
    name: str | None
    experiment: str | None
    env: dict[ str, str ]
    src: Path | None
    repo: str | None
    ref: str | None
    max_source_mb: int

@dataclass( frozen = True )
class LaunchPlan:
    job: JobSpec
    runs: list[ RunRecord ]
    snapshot: Snapshot | None
    spec: VmSpec
    zones: list[ str ]
    work_dir: Path

@dataclass( frozen = True )
class LaunchResult:
    zone: str

def _parse_env( pairs: list[ str ] ) -> dict[ str, str ]:
    env: dict[ str, str ] = {}
    for pair in pairs:
        name, separator, value = pair.partition( "=" )
        if not separator:
            raise UsageError( f"--env needs KEY=VALUE, got {pair!r}." )
        if ENV_NAME.fullmatch( name ) is None:
            raise UsageError( f"--env name {name!r} must be upper case letters, digits and underscores." )
        if name in RESERVED_ENV_NAMES or name.startswith( RESERVED_ENV_PREFIX ):
            raise UsageError(
                f"--env cannot set {name}: ml-runner sets {', '.join( RESERVED_ENV_NAMES )} and ML_RUNNER*."
            )
        if name in env:
            raise UsageError( f"--env sets {name} twice." )

        env[ name ] = value

    return env

# Flag > config defaults (TRD §3.2). Every refusal is a usage error, before anything is read or created.
def resolve_options( config: Config, flags: LaunchFlags ) -> LaunchOptions:
    defaults = config.defaults
    if not flags.command:
        raise UsageError( "Give the command to run after `--`, for example: ml-runner launch -- python train.py" )

    gpu_name = flags.gpu or defaults.gpu
    if gpu_name not in get_args( Gpu ):
        raise UsageError( f"--gpu must be one of {', '.join( get_args( Gpu ) )}, got {gpu_name!r}." )

    gpu = cast( Gpu, gpu_name )
    max_hours = flags.max_hours if flags.max_hours is not None else defaults.max_hours
    if max_hours < MIN_HOURS:
        raise UsageError( f"--max-hours must be at least {MIN_HOURS}." )
    if max_hours > MAX_HOURS and not flags.allow_long:
        raise UsageError( f"--max-hours above {MAX_HOURS} needs --allow-long." )

    disk_gb = flags.disk_gb if flags.disk_gb is not None else defaults.disk_gb
    if disk_gb <= 0:
        raise UsageError( "--disk-gb must be positive." )

    max_source_mb = flags.max_source_mb if flags.max_source_mb is not None else defaults.max_source_mb
    if max_source_mb <= 0:
        raise UsageError( "--max-source-mb must be positive." )
    if flags.zone is not None and not flags.zone.startswith( f"{config.region}-" ):
        raise UsageError( f"Zone {flags.zone} is not in region {config.region}." )

    validate_source_flags( flags.src, flags.repo, flags.ref )
    profile_for( gpu, flags.machine_type )

    return LaunchOptions(
        command = flags.command,
        gpu = gpu,
        is_spot = flags.is_spot or defaults.spot,
        machine_type = flags.machine_type,
        disk_gb = disk_gb,
        zone = flags.zone,
        max_hours = max_hours,
        name = flags.name,
        experiment = flags.experiment,
        env = _parse_env( flags.env ),
        src = flags.src,
        repo = flags.repo,
        ref = flags.ref,
        max_source_mb = max_source_mb
    )

def _resolve_source(
    options: LaunchOptions,
    work_dir: Path,
    ref_check: Callable[ [ str, str ], str ]
) -> tuple[ Source, Snapshot | None, str ]:
    if options.repo is not None and options.ref is not None:
        git_source = check_git_source( options.repo, options.ref, ref_check )
        source = Source( kind = SourceKind.GIT, repo = git_source.repo, ref = git_source.ref )

        return ( source, None, experiment_name( options.experiment, options.repo, "" ) )

    src = options.src or Path( "." )
    snapshot = build_snapshot( src, options.max_source_mb, work_dir )
    folder = src.resolve().name
    source = Source(
        kind = SourceKind.SNAPSHOT,
        path = folder,
        sha256 = snapshot.sha256,
        git_commit = snapshot.git_commit,
        git_dirty = snapshot.git_dirty
    )

    return ( source, snapshot, experiment_name( options.experiment, None, folder ) )

# Everything that can be decided and checked without changing anything: source, IDs, records, zones.
# It reads from the cloud (accelerator types) but never writes, so `--dry-run` can show it.
def prepare_launch(
    gcloud: Gcloud,
    config: Config,
    options: LaunchOptions,
    work_dir: Path,
    *,
    now: datetime | None = None,
    rng: random.Random | None = None,
    ref_check: Callable[ [ str, str ], str ] = ls_remote
) -> LaunchPlan:
    clock = now or datetime.now( UTC )
    source, snapshot, experiment = _resolve_source( options, work_dir, ref_check )
    profile = profile_for( options.gpu, options.machine_type )
    offering = None
    if profile.zone_accelerator is not None:
        rows = gcloud.json( "compute", "accelerator-types", "list", f"--filter=zone~{config.region}-" ) or []
        offering = zones_offering( rows, profile.zone_accelerator )

    zones = zone_candidates( options.zone or config.zone, offering, is_explicit = options.zone is not None )
    job = job_id( clock, rng )
    run_name = slugify( options.name or " ".join( options.command ) )
    run = run_id( run_name, clock, rng )
    machine = Machine(
        vm = vm_name( job ),
        zone = zones[ 0 ],
        machine_type = profile.machine_type,
        gpu = options.gpu,
        spot = options.is_spot,
        image = image_path( profile )
    )
    record = RunRecord(
        run_id = run,
        job_id = job,
        experiment = experiment,
        name = run_name,
        command = options.command,
        env_names = list( options.env ),
        status = RunStatus.PROVISIONING,
        source = source,
        machine = machine,
        ml_runner_version = __version__,
        times = Times( created = clock )
    )
    spec = JobSpec(
        job_id = job,
        bucket = config.bucket,
        max_hours = options.max_hours,
        stop_on_failure = True,
        source = source,
        runs = [ JobRun( run_id = run, experiment = experiment, command = options.command ) ],
        gpu = options.gpu,
        env = options.env
    )
    vm_spec = VmSpec(
        job_id = job,
        zone = zones[ 0 ],
        profile = profile,
        is_spot = options.is_spot,
        disk_gb = options.disk_gb,
        max_hours = options.max_hours,
        service_account = config.service_account,
        version = __version__
    )

    return LaunchPlan( spec, [ record ], snapshot, vm_spec, zones, work_dir )

def _describe_source( run: RunRecord, snapshot: Snapshot | None ) -> str:
    source = run.source
    if snapshot is None:
        return f"git {source.repo} at {source.ref}"

    text = (
        f"snapshot of {source.path}: {snapshot.file_count} files, {snapshot.size_bytes / BYTES_PER_MB:.1f} MB, "
        f"sha256 {( source.sha256 or '' )[ :HASH_PREFIX_LENGTH ]}"
    )
    if source.git_commit:
        text += f", commit {source.git_commit[ :HASH_PREFIX_LENGTH ]}{' (dirty)' if source.git_dirty else ''}"

    return text

def describe_plan( plan: LaunchPlan ) -> list[ str ]:
    run = plan.runs[ 0 ]
    machine = run.machine
    others = [ zone for zone in plan.zones if zone != machine.zone ]
    lines = [
        f"Job:        {plan.job.job_id} (VM {machine.vm})",
        f"Run:        {run.run_id}",
        f"Experiment: {run.experiment}",
        f"Command:    {shlex.join( run.command )}",
        f"Source:     {_describe_source( run, plan.snapshot )}",
        f"Machine:    {machine.gpu} ({machine.machine_type}), {'Spot' if machine.spot else 'on-demand'}, "
        f"{plan.spec.disk_gb} GB disk",
        f"Zone:       {machine.zone}" + ( f" (then {', '.join( others )} if it has no capacity)" if others else "" ),
        f"Limit:      {plan.job.max_hours} h; Google deletes the VM after "
        f"{runtime_cap_minutes( plan.job.max_hours )} minutes at the latest"
    ]
    if run.env_names:
        lines.append(
            f"Env:        {', '.join( run.env_names )} (values go in instance metadata, never in the bucket)"
        )

    return lines

def _upload_text( gcloud: Gcloud, work_dir: Path, filename: str, text: str, uri: str ) -> None:
    path = work_dir / filename
    path.write_text( text )
    gcloud.run( "storage", "cp", str( path ), uri, mutating = True )

def _upload_run( gcloud: Gcloud, plan: LaunchPlan, run: RunRecord ) -> None:
    uri = layout.run_json( plan.job.bucket, run.experiment, run.run_id )
    _upload_text( gcloud, plan.work_dir, f"run-{run.run_id}.json", run.to_json(), uri )

def _mark_failed( gcloud: Gcloud, plan: LaunchPlan, message: str ) -> None:
    for run in plan.runs:
        _upload_run( gcloud, plan, run.model_copy( update = { "status": RunStatus.FAILED, "message": message } ) )

# Uploads, then creates the VM, moving to the next zone when one has no capacity. If no VM can be created, the
# runs are marked failed (job.json stays) and the error is raised.
def execute_launch( gcloud: Gcloud, plan: LaunchPlan ) -> LaunchResult:
    bucket = plan.job.bucket
    if plan.snapshot is not None:
        upload_snapshot( gcloud, bucket, plan.snapshot )

    _upload_text( gcloud, plan.work_dir, JOB_FILE, plan.job.to_json(), layout.job_json( bucket, plan.job.job_id ) )
    for run in plan.runs:
        _upload_run( gcloud, plan, run )

    metadata = plan.work_dir / METADATA_FILE
    metadata.write_text( plan.job.metadata_json() )
    files = VmFiles(
        startup = VM_DIR / "startup.sh",
        shutdown = VM_DIR / "shutdown.sh",
        agent = VM_DIR / "agent.py",
        job = metadata
    )

    def create( zone: str ) -> str:
        return gcloud.run( *build_create_args( replace( plan.spec, zone = zone ), files ), mutating = True )

    try:
        zone, _ = create_in_first_available_zone( plan.zones, create )
    except MlRunnerError as error:
        # The original error is the useful one, so a failure to record it is not allowed to replace it.
        with contextlib.suppress( MlRunnerError ):
            _mark_failed( gcloud, plan, error.message )

        raise

    if zone != plan.spec.zone:
        for run in plan.runs:
            moved = run.machine.model_copy( update = { "zone": zone } )
            _upload_run( gcloud, plan, run.model_copy( update = { "machine": moved } ) )

    return LaunchResult( zone )
