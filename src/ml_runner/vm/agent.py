# On-VM agent: standard library only, Python >= 3.10, imports nothing from the CLI (ADR 0006).
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
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

class AgentError( Exception ):
    pass

class InstallError( AgentError ):
    pass

class GpuError( AgentError ):
    pass

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
        values = json.loads( Path( file_name ).read_text() )
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
            raise GpuError( GPU_NOT_READY )

        time.sleep( poll_seconds )

def start() -> int:
    home = home_dir()
    metadata = read_metadata()
    write_state( home, State( job_id = metadata.job.job_id, phase = "fetching" ) )
    fetch_source( metadata.job, home )

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
