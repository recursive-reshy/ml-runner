# On-VM agent: standard library only, Python >= 3.10, imports nothing from the CLI (ADR 0006).
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

DEFAULT_HOME = "/opt/mlr"
HOME_VARIABLE = "MLR_HOME"
METADATA_FILE_VARIABLE = "MLR_METADATA_FILE"
METADATA_URL = "http://metadata.google.internal/computeMetadata/v1"
METADATA_HEADER = ( "Metadata-Flavor", "Google" )
METADATA_TIMEOUT_SECONDS = 10
JOB_KEY = "mlr-job"
JOB_ATTRIBUTE = f"instance/attributes/{JOB_KEY}"
NAME_KEY = "name"
ZONE_KEY = "zone"
STATE_FILE = "state.json"
SOURCE_DIR = "src"
SOURCE_ARCHIVE = "source.tar.gz"
UV_INSTALL_URL = "https://astral.sh/uv/install.sh"
UV_USER_BIN = Path( ".local" ) / "bin"
INSTALL_FAILED = "dependency install failed"
GPU_NOT_READY = "GPU driver not ready"
GPU_POLL_SECONDS = 10
GPU_WAIT_SECONDS = 600
GPU_POLL_VARIABLE = "MLR_GPU_POLL_SECONDS"
GPU_WAIT_VARIABLE = "MLR_GPU_WAIT_SECONDS"
NO_GPU = "none"
FETCH_FAILED = "source fetch failed"
AGENT_LOG = "agent.log"
RUNS_DIR = "runs"
RUN_LOG = "run.log"
RUN_RECORD = "run.json"
LOG_SYNC_SECONDS = 30
DATA_SYNC_SECONDS = 300
RETRY_SECONDS = 5
KILL_GRACE_SECONDS = 30
FINAL_SYNC_ATTEMPTS = 3
LOG_SYNC_VARIABLE = "MLR_LOG_SYNC_SECONDS"
DATA_SYNC_VARIABLE = "MLR_DATA_SYNC_SECONDS"
RETRY_VARIABLE = "MLR_RETRY_SECONDS"
KILL_GRACE_VARIABLE = "MLR_KILL_GRACE_SECONDS"
MAX_SECONDS_VARIABLE = "MLR_MAX_SECONDS"
SECONDS_PER_HOUR = 3600
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
RECORD_INDENT = 2
SUCCEEDED = "succeeded"
FAILED = "failed"
RUNNING = "running"
CANCELLED = "cancelled"
TIMED_OUT = "timed_out"
SKIPPED_MESSAGE = "skipped after an earlier failure"
TIMED_OUT_MESSAGE = "job exceeded max hours"
CANCELLED_MESSAGE = "job cancelled"

class AgentError( Exception ):
    # What `run.json` shows when this error stops a job before its runs start.
    short = "agent error"

class FetchError( AgentError ):
    short = FETCH_FAILED

class InstallError( AgentError ):
    short = INSTALL_FAILED

class GpuError( AgentError ):
    short = GPU_NOT_READY

    def __init__( self ) -> None:
        super().__init__( GPU_NOT_READY )

@dataclass
class Source:
    kind: str
    sha256: str | None = None
    repo: str | None = None
    ref: str | None = None

@dataclass
class JobRun:
    run_id: str
    experiment: str
    command: list[ str ]

@dataclass
class Job:
    job_id: str
    bucket: str
    max_hours: float
    stop_on_failure: bool
    gpu: str
    source: Source
    runs: list[ JobRun ]
    env: dict[ str, str ] = field( default_factory = dict )

@dataclass
class Metadata:
    job: Job
    instance: str
    zone: str

@dataclass
class Environment:
    prefix: list[ str ]
    path_prepend: str | None

@dataclass
class State:
    job_id: str
    phase: str
    current_run: str | None = None

# Bucket paths. The CLI builds the same ones in layout.py; a drift test keeps them equal.
def source_uri( bucket: str, sha256: str ) -> str:
    return f"gs://{bucket}/sources/{sha256}.tar.gz"

def run_dir_uri( bucket: str, experiment: str, run_id: str ) -> str:
    return f"gs://{bucket}/runs/{experiment}/{run_id}"

def home_dir() -> Path:
    return Path( os.environ.get( HOME_VARIABLE, DEFAULT_HOME ) )

def metadata_request( path: str ) -> urllib.request.Request:
    request = urllib.request.Request( f"{METADATA_URL}/{path}" )
    request.add_header( *METADATA_HEADER )

    return request

def read_metadata_value( key: str, path: str ) -> str:
    file_name = os.environ.get( METADATA_FILE_VARIABLE )

    if file_name is not None:
        try:
            values = json.loads( Path( file_name ).read_text() )
        except ( OSError, ValueError ) as error:
            raise AgentError( f"cannot read metadata file: {error}" ) from error

        if key not in values:
            raise AgentError( f"metadata key {key} is missing" )

        return values[ key ]

    try:
        with urllib.request.urlopen( metadata_request( path ), timeout = METADATA_TIMEOUT_SECONDS ) as response:
            return response.read().decode()
    except OSError as error:
        raise AgentError( f"cannot read metadata {key}: {error}" ) from error

def parse_job( text: str ) -> Job:
    values: dict[ str, Any ] = json.loads( text )
    source = values[ "source" ]

    return Job(
        job_id = values[ "job_id" ],
        bucket = values[ "bucket" ],
        max_hours = values[ "max_hours" ],
        stop_on_failure = values[ "stop_on_failure" ],
        gpu = values[ "gpu" ],
        source = Source(
            kind = source[ "kind" ],
            sha256 = source.get( "sha256" ),
            repo = source.get( "repo" ),
            ref = source.get( "ref" )
        ),
        runs = [
            JobRun( run_id = run[ "run_id" ], experiment = run[ "experiment" ], command = run[ "command" ] )
            for run in values[ "runs" ]
        ],
        env = values.get( "env", {} )
    )

def read_metadata() -> Metadata:
    job = parse_job( read_metadata_value( JOB_KEY, JOB_ATTRIBUTE ) )
    instance = read_metadata_value( NAME_KEY, "instance/name" )
    zone = read_metadata_value( ZONE_KEY, "instance/zone" )

    # The server answers `projects/<number>/zones/<zone>`; only the last part is the zone.
    return Metadata( job = job, instance = instance, zone = zone.rsplit( "/", 1 )[ -1 ] )

# Written through a temp file so a reader (finalize, on pre-emption) never sees half a file.
def write_state( home: Path, state: State ) -> None:
    target = home / STATE_FILE
    temporary = home / f"{STATE_FILE}.tmp"
    temporary.write_text( json.dumps( asdict( state ) ) )
    temporary.replace( target )

def read_state( home: Path ) -> State | None:
    target = home / STATE_FILE
    if not target.exists():
        return None

    return State( **json.loads( target.read_text() ) )

def run_command( argv: list[ str ], step: str, cwd: Path | None = None ) -> None:
    try:
        completed = subprocess.run( argv, capture_output = True, text = True, cwd = cwd )
    except OSError as error:
        raise AgentError( f"{step} failed: {error}" ) from error

    if completed.returncode != 0:
        raise AgentError( f"{step} failed ({completed.returncode}): {completed.stderr.strip()}" )

def fetch_source( job: Job, home: Path ) -> Path:
    try:
        return download_source( job, home )
    except AgentError as error:
        raise FetchError( str( error ) ) from error

def download_source( job: Job, home: Path ) -> Path:
    source_dir = home / SOURCE_DIR

    if job.source.kind == "git":
        assert job.source.repo is not None and job.source.ref is not None
        run_command( [ "git", "clone", job.source.repo, str( source_dir ) ], "fetch (git clone)" )
        run_command( [ "git", "-C", str( source_dir ), "checkout", job.source.ref ], "fetch (git checkout)" )

        return source_dir

    assert job.source.sha256 is not None
    archive = home / SOURCE_ARCHIVE
    source_dir.mkdir( parents = True, exist_ok = True )
    uri = source_uri( job.bucket, job.source.sha256 )
    run_command( [ "gcloud", "storage", "cp", uri, str( archive ) ], "fetch (download)" )
    run_command( [ "tar", "-xzf", str( archive ), "-C", str( source_dir ) ], "fetch (extract)" )

    return source_dir

def install_uv() -> None:
    # pipefail: a failed download must not look like an empty, successful script.
    script = f"set -o pipefail; curl -LsSf {UV_INSTALL_URL} | sh"
    try:
        run_command( [ "bash", "-c", script ], "uv installer" )
    except AgentError:
        run_command( [ "python3", "-m", "pip", "install", "--user", "uv" ], "uv pip install" )

    os.environ[ "PATH" ] = os.pathsep.join( [ str( Path.home() / UV_USER_BIN ), os.environ[ "PATH" ] ] )

def ensure_uv() -> None:
    if shutil.which( "uv" ) is not None:
        return

    try:
        install_uv()
    except AgentError as error:
        raise InstallError( f"{INSTALL_FAILED}: {error}" ) from error

def install_dependencies( source_dir: Path ) -> Environment:
    ensure_uv()
    has_project = ( source_dir / "pyproject.toml" ).exists() or ( source_dir / "uv.lock" ).exists()
    has_requirements = ( source_dir / "requirements.txt" ).exists()

    if not has_project and not has_requirements:
        raise InstallError( f"{INSTALL_FAILED}: no pyproject.toml or requirements.txt" )

    try:
        if has_project:
            sync = [ "uv", "sync", "--frozen" ] if ( source_dir / "uv.lock" ).exists() else [ "uv", "sync" ]
            run_command( sync, "uv sync", cwd = source_dir )

            return Environment( prefix = [ "uv", "run", "--no-sync", "--" ], path_prepend = None )

        run_command( [ "uv", "venv" ], "uv venv", cwd = source_dir )
        run_command( [ "uv", "pip", "install", "-r", "requirements.txt" ], "uv pip install", cwd = source_dir )
    except AgentError as error:
        raise InstallError( f"{INSTALL_FAILED}: {error}" ) from error

    return Environment( prefix = [], path_prepend = str( source_dir / ".venv" / "bin" ) )

def is_gpu_ready() -> bool:
    try:
        return subprocess.run( [ "nvidia-smi" ], capture_output = True ).returncode == 0
    except OSError:
        return False

def wait_for_gpu( gpu: str ) -> None:
    if gpu == NO_GPU:
        return

    poll_seconds = float( os.environ.get( GPU_POLL_VARIABLE, GPU_POLL_SECONDS ) )
    wait_seconds = float( os.environ.get( GPU_WAIT_VARIABLE, GPU_WAIT_SECONDS ) )
    deadline = time.monotonic() + wait_seconds

    while not is_gpu_ready():
        if time.monotonic() >= deadline:
            raise GpuError()

        time.sleep( poll_seconds )

@dataclass
class Outcome:
    exit_code: int | None
    is_timed_out: bool
    message: str | None = None

class Ticker( threading.Thread ):
    def __init__( self, interval: float, action: Callable[ [], None ] ) -> None:
        super().__init__( daemon = True )
        self.interval = interval
        self.action = action
        self.is_stopped = threading.Event()

    def run( self ) -> None:
        while not self.is_stopped.wait( self.interval ):
            try:
                self.action()
            except AgentError as error:
                log( f"sync failed, will retry: {error}" )

    def stop( self ) -> None:
        self.is_stopped.set()
        self.join()

def utc_now() -> datetime:
    return datetime.now( timezone.utc )

def seconds_from( variable: str, default: float ) -> float:
    return float( os.environ.get( variable, default ) )

def log( message: str ) -> None:
    print( message, flush = True )
    with open( home_dir() / AGENT_LOG, "a" ) as agent_log:
        agent_log.write( f"{message}\n" )

def run_run_dir( home: Path, run: JobRun ) -> Path:
    return home / RUNS_DIR / run.run_id

def record_uri( job: Job, run: JobRun ) -> str:
    return f"{run_dir_uri( job.bucket, run.experiment, run.run_id )}/{RUN_RECORD}"

def log_uri( job: Job, run: JobRun ) -> str:
    return f"{run_dir_uri( job.bucket, run.experiment, run.run_id )}/{RUN_LOG}"

def outputs_uri( job: Job, run: JobRun ) -> str:
    return f"{run_dir_uri( job.bucket, run.experiment, run.run_id )}/outputs"

def checkpoints_uri( job: Job, run: JobRun ) -> str:
    return f"{run_dir_uri( job.bucket, run.experiment, run.run_id )}/checkpoints"

def cancel_uri( job: Job ) -> str:
    return f"gs://{job.bucket}/jobs/{job.job_id}/cancel"

def agent_log_uri( job: Job ) -> str:
    return f"gs://{job.bucket}/jobs/{job.job_id}/{AGENT_LOG}"

def retry( action: Callable[ [], None ], attempts: int ) -> bool:
    for attempt in range( 1, attempts + 1 ):
        try:
            action()

            return True
        except AgentError as error:
            log( f"attempt {attempt} of {attempts} failed: {error}" )
            if attempt < attempts:
                time.sleep( seconds_from( RETRY_VARIABLE, RETRY_SECONDS ) )

    return False

def upload_file( path: Path, uri: str ) -> None:
    run_command( [ "gcloud", "storage", "cp", str( path ), uri ], f"upload {path.name}" )

def sync_log( job: Job, run: JobRun, home: Path ) -> None:
    upload_file( run_run_dir( home, run ) / RUN_LOG, log_uri( job, run ) )

def sync_folder( local: Path, uri: str ) -> None:
    run_command( [ "gcloud", "storage", "rsync", "-r", str( local ), uri ], f"sync {local.name}" )

def sync_data( job: Job, run: JobRun, home: Path ) -> None:
    sync_folder( run_run_dir( home, run ) / "outputs", outputs_uri( job, run ) )
    sync_folder( run_run_dir( home, run ) / "checkpoints", checkpoints_uri( job, run ) )

def final_sync( job: Job, run: JobRun, home: Path ) -> None:
    run_dir = run_run_dir( home, run )
    steps: list[ tuple[ str, Callable[ [], None ] ] ] = [
        ( RUN_LOG, partial( upload_file, run_dir / RUN_LOG, log_uri( job, run ) ) ),
        ( "outputs", partial( sync_folder, run_dir / "outputs", outputs_uri( job, run ) ) ),
        ( "checkpoints", partial( sync_folder, run_dir / "checkpoints", checkpoints_uri( job, run ) ) )
    ]

    for name, action in steps:
        if not retry( action, FINAL_SYNC_ATTEMPTS ):
            log( f"final sync of {name} for {run.run_id} gave up" )

def load_record( job: Job, run: JobRun, home: Path ) -> dict[ str, Any ]:
    path = run_run_dir( home, run ) / RUN_RECORD
    if not path.exists():
        path.parent.mkdir( parents = True, exist_ok = True )
        run_command( [ "gcloud", "storage", "cp", record_uri( job, run ), str( path ) ], "download run.json" )

    return json.loads( path.read_text() )

# The launcher wrote the full record as `provisioning`; the agent only changes status, times, exit code and message.
def update_record(
    job: Job,
    run: JobRun,
    home: Path,
    changes: dict[ str, Any ],
    times: dict[ str, Any ] | None = None
) -> None:
    values = load_record( job, run, home )
    values.update( changes )
    values[ "times" ].update( times or {} )
    path = run_run_dir( home, run ) / RUN_RECORD
    path.write_text( json.dumps( values, indent = RECORD_INDENT ) )
    if not retry( lambda: upload_file( path, record_uri( job, run ) ), FINAL_SYNC_ATTEMPTS ):
        log( f"could not upload the record of {run.run_id}" )

def mark_runs( job: Job, runs: list[ JobRun ], home: Path, status: str, message: str ) -> None:
    for run in runs:
        try:
            update_record(
                job,
                run,
                home,
                { "status": status, "message": message },
                { "finished": utc_now().strftime( TIME_FORMAT ) }
            )
        except AgentError as error:
            log( f"cannot record {run.run_id}: {error}" )

def is_cancelled( job: Job ) -> bool:
    argv = [ "gcloud", "storage", "objects", "describe", cancel_uri( job ) ]

    return subprocess.run( argv, capture_output = True ).returncode == 0

def run_environment( job: Job, run: JobRun, home: Path, environment: Environment ) -> dict[ str, str ]:
    run_dir = run_run_dir( home, run )
    values = { **os.environ, **job.env }
    values.update( {
        "OUTPUT_DIR": str( run_dir / "outputs" ),
        "CHECKPOINT_DIR": str( run_dir / "checkpoints" ),
        "RUN_ID": run.run_id,
        "ML_RUNNER": "1",
        "ML_RUNNER_GPU": job.gpu
    } )
    if environment.path_prepend is not None:
        values[ "PATH" ] = os.pathsep.join( [ environment.path_prepend, values[ "PATH" ] ] )

    return values

def stop_process( process: "subprocess.Popen[str]" ) -> None:
    # The command runs in its own process group so `uv run` and its children stop together.
    grace_seconds = seconds_from( KILL_GRACE_VARIABLE, KILL_GRACE_SECONDS )
    for signal_number, wait_seconds in [ ( signal.SIGTERM, grace_seconds ), ( signal.SIGKILL, None ) ]:
        try:
            os.killpg( process.pid, signal_number )
            process.wait( timeout = wait_seconds )

            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue

def copy_output( process: "subprocess.Popen[str]", log_path: Path ) -> None:
    assert process.stdout is not None
    with open( log_path, "a", buffering = 1 ) as run_log:
        for line in process.stdout:
            run_log.write( line )
            log( line.rstrip( "\n" ) )

def execute(
    argv: list[ str ],
    source_dir: Path,
    values: dict[ str, str ],
    log_path: Path,
    deadline: float
) -> Outcome:
    try:
        process = subprocess.Popen(
            argv,
            cwd = source_dir,
            env = values,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            start_new_session = True
        )
    except OSError as error:
        return Outcome( exit_code = None, is_timed_out = False, message = f"cannot start command: {error}" )

    reader = threading.Thread( target = copy_output, args = ( process, log_path ), daemon = True )
    reader.start()

    try:
        exit_code = process.wait( timeout = max( deadline - time.monotonic(), 0 ) )
    except subprocess.TimeoutExpired:
        stop_process( process )
        reader.join( timeout = seconds_from( KILL_GRACE_VARIABLE, KILL_GRACE_SECONDS ) )

        return Outcome( exit_code = None, is_timed_out = True, message = TIMED_OUT_MESSAGE )

    reader.join()

    return Outcome( exit_code = exit_code, is_timed_out = False )

def outcome_status( outcome: Outcome ) -> str:
    if outcome.is_timed_out:
        return TIMED_OUT

    return SUCCEEDED if outcome.exit_code == 0 else FAILED

def run_one(
    job: Job,
    run: JobRun,
    home: Path,
    source_dir: Path,
    environment: Environment,
    deadline: float
) -> str:
    run_dir = run_run_dir( home, run )
    for name in [ "outputs", "checkpoints" ]:
        shutil.rmtree( run_dir / name, ignore_errors = True )
        ( run_dir / name ).mkdir( parents = True )
    log_path = run_dir / RUN_LOG
    log_path.write_text( "" )

    write_state( home, State( job_id = job.job_id, phase = RUNNING, current_run = run.run_id ) )
    started = utc_now()
    update_record( job, run, home, { "status": RUNNING }, { "started": started.strftime( TIME_FORMAT ) } )
    log( f"run {run.run_id}: {' '.join( run.command )}" )

    tickers = [
        Ticker( seconds_from( LOG_SYNC_VARIABLE, LOG_SYNC_SECONDS ), lambda: sync_log( job, run, home ) ),
        Ticker( seconds_from( DATA_SYNC_VARIABLE, DATA_SYNC_SECONDS ), lambda: sync_data( job, run, home ) )
    ]
    for ticker in tickers:
        ticker.start()

    values = run_environment( job, run, home, environment )
    outcome = execute( environment.prefix + run.command, source_dir, values, log_path, deadline )
    finished = utc_now()
    for ticker in tickers:
        ticker.stop()

    final_sync( job, run, home )
    status = outcome_status( outcome )
    update_record(
        job,
        run,
        home,
        {
            "status": status,
            "exit_code": outcome.exit_code,
            "duration_seconds": ( finished - started ).total_seconds(),
            "message": outcome.message
        },
        { "finished": finished.strftime( TIME_FORMAT ) }
    )
    log( f"run {run.run_id}: {status}" )

    return status

def run_job( metadata: Metadata, home: Path, source_dir: Path, environment: Environment ) -> None:
    job = metadata.job
    max_seconds = seconds_from( MAX_SECONDS_VARIABLE, job.max_hours * SECONDS_PER_HOUR )
    deadline = time.monotonic() + max_seconds

    for index, run in enumerate( job.runs ):
        if is_cancelled( job ):
            mark_runs( job, job.runs[ index: ], home, CANCELLED, CANCELLED_MESSAGE )

            return

        if time.monotonic() >= deadline:
            mark_runs( job, job.runs[ index: ], home, TIMED_OUT, TIMED_OUT_MESSAGE )

            return

        status = run_one( job, run, home, source_dir, environment, deadline )
        remaining = job.runs[ index + 1: ]

        if status == TIMED_OUT:
            mark_runs( job, remaining, home, TIMED_OUT, TIMED_OUT_MESSAGE )

            return

        if status == FAILED and job.stop_on_failure:
            mark_runs( job, remaining, home, FAILED, SKIPPED_MESSAGE )

            return

def self_delete( metadata: Metadata ) -> None:
    delete = [ "gcloud", "compute", "instances", "delete", metadata.instance, f"--zone={metadata.zone}", "--quiet" ]
    try:
        run_command( delete, "self-delete" )
    except AgentError as error:
        log( f"{error}; powering off instead" )
        try:
            run_command( [ "shutdown", "-h", "now" ], "shutdown" )
        except AgentError as shutdown_error:
            log( str( shutdown_error ) )

def finish( metadata: Metadata, home: Path ) -> None:
    write_state( home, State( job_id = metadata.job.job_id, phase = "done" ) )
    retry( lambda: upload_file( home / AGENT_LOG, agent_log_uri( metadata.job ) ), FINAL_SYNC_ATTEMPTS )
    self_delete( metadata )

def prepare_and_run( metadata: Metadata, home: Path ) -> None:
    job = metadata.job
    try:
        write_state( home, State( job_id = job.job_id, phase = "fetching" ) )
        source_dir = fetch_source( job, home )
        write_state( home, State( job_id = job.job_id, phase = "installing" ) )
        environment = install_dependencies( source_dir )
        write_state( home, State( job_id = job.job_id, phase = "waiting_for_gpu" ) )
        wait_for_gpu( job.gpu )
    except AgentError as error:
        log( f"job cannot start: {error}" )
        mark_runs( job, job.runs, home, FAILED, error.short )

        return

    run_job( metadata, home, source_dir, environment )

def start() -> int:
    home = home_dir()
    home.mkdir( parents = True, exist_ok = True )
    metadata = read_metadata()

    try:
        prepare_and_run( metadata, home )
    except AgentError as error:
        log( f"agent stopped: {error}" )
    finally:
        finish( metadata, home )

    return 0

def main( argv: list[ str ] ) -> int:
    if argv[ 1: 2 ] != [ "start" ]:
        print( "usage: agent.py start", file = sys.stderr )

        return 2

    try:
        return start()
    except AgentError as error:
        print( f"agent: {error}", file = sys.stderr )

        return 1

if __name__ == "__main__":
    sys.exit( main( sys.argv ) )
