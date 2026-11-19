import stat
import sys
from pathlib import Path
from typing import Any

# A gcloud that keeps a "bucket" in a local folder: gs://<bucket>/<path> is $FAKE_BUCKET/<bucket>/<path>.
# Every call's argv is logged. FAKE_GCLOUD_FAIL is a comma-separated list of substrings (or `*`) that make a call
# fail; FAKE_GCLOUD_DELAY sleeps before rsync.
FAKE_GCLOUD = f"""#!{sys.executable}
import os
import shutil
import sys
import time
from pathlib import Path

args = sys.argv[ 1: ]
joined = " ".join( args )
with open( Path( os.environ[ "FAKE_CALLS" ] ) / "gcloud.log", "a" ) as log:
    log.write( joined + "\\n" )

for pattern in filter( None, os.environ.get( "FAKE_GCLOUD_FAIL", "" ).split( "," ) ):
    if pattern == "*" or pattern in joined:
        print( "fake gcloud: failing on purpose", file = sys.stderr )
        sys.exit( 1 )

bucket = Path( os.environ[ "FAKE_BUCKET" ] )

def local( path ):
    return bucket / path[ len( "gs://" ): ] if path.startswith( "gs://" ) else Path( path )

if args[ :2 ] == [ "storage", "cp" ]:
    source, target = local( args[ 2 ] ), local( args[ 3 ] )
    if not source.is_file():
        print( "not found: " + args[ 2 ], file = sys.stderr )
        sys.exit( 1 )
    target.parent.mkdir( parents = True, exist_ok = True )
    shutil.copyfile( source, target )
elif args[ :2 ] == [ "storage", "rsync" ]:
    time.sleep( float( os.environ.get( "FAKE_GCLOUD_DELAY", "0" ) ) )
    source, target = local( args[ 3 ] ), local( args[ 4 ] )
    if source.is_dir():
        shutil.copytree( source, target, dirs_exist_ok = True )
elif args[ :3 ] == [ "storage", "objects", "describe" ]:
    sys.exit( 0 if local( args[ 3 ] ).exists() else 1 )
"""

FAKE_GIT = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_CALLS/git.log"
if [[ "$1" == "clone" ]]; then
    mkdir -p "$3"
fi
exit "${FAKE_GIT_EXIT:-0}"
"""

# A tool that logs its argv to <name>.log and exits FAKE_<NAME>_EXIT (default 0).
LOGGING_TOOL = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_CALLS/NAME.log"
exit "${FAKE_UPPER_EXIT:-0}"
"""

# Fails its first FAKE_NVIDIA_FAILS calls, like a driver that is still installing.
FAKE_NVIDIA_SMI = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_CALLS/nvidia-smi.log"
calls=$(wc -l < "$FAKE_CALLS/nvidia-smi.log")
if (( calls <= ${FAKE_NVIDIA_FAILS:-0} )); then
    exit 1
fi
"""

BUCKET = "my-bucket"

class FakeBin:
    def __init__( self, root: Path ) -> None:
        self.bin = root / "bin"
        self.calls = root / "calls"
        self.bucket = root / "bucket"
        self.bin.mkdir()
        self.calls.mkdir()
        self.bucket.mkdir()

    def add( self, name: str, script: str ) -> None:
        path = self.bin / name
        path.write_text( script )
        path.chmod( path.stat().st_mode | stat.S_IEXEC )

    def put( self, uri_path: str, source: Path ) -> None:
        target = self.bucket / uri_path
        target.parent.mkdir( parents = True, exist_ok = True )
        target.write_bytes( source.read_bytes() )

    def has( self, uri_path: str ) -> bool:
        return ( self.bucket / uri_path ).exists()

    def read( self, uri_path: str ) -> str:
        return ( self.bucket / uri_path ).read_text()

    def add_logging( self, name: str ) -> None:
        upper = name.upper().replace( "-", "_" )
        self.add( name, LOGGING_TOOL.replace( "NAME", name ).replace( "UPPER", upper ) )

    def lines( self, name: str ) -> list[ str ]:
        log = self.calls / f"{name}.log"

        return log.read_text().splitlines() if log.exists() else []

JOB = {
    "schema": 1,
    "job_id": "20261120-143005-7c21",
    "bucket": "my-bucket",
    "max_hours": 6,
    "stop_on_failure": True,
    "source": { "kind": "snapshot", "path": "proj", "sha256": "a" * 64 },
    "runs": [ { "run_id": "R1", "experiment": "exp", "command": [ "python", "train.py" ] } ],
    "env": { "KEY": "VALUE" },
    "gpu": "t4"
}

def launcher_record( run_id: str, experiment: str ) -> dict:
    return {
        "schema": 1,
        "run_id": run_id,
        "job_id": JOB[ "job_id" ],
        "experiment": experiment,
        "name": run_id,
        "command": [ "python", "train.py" ],
        "env_names": [ "KEY" ],
        "status": "provisioning",
        "exit_code": None,
        "source": { "kind": "snapshot", "path": "proj", "sha256": "a" * 64 },
        "machine": {
            "vm": "mlr-20261120-143005-7c21",
            "zone": "asia-southeast1-a",
            "machine_type": "n1-standard-4",
            "gpu": "t4",
            "spot": True,
            "image": "projects/p/global/images/family/f"
        },
        "ml_runner_version": "0.1.0",
        "times": { "created": "2026-11-20T14:30:05Z", "started": None, "finished": None },
        "duration_seconds": None,
        "message": None
    }

def python_command( code: str ) -> list[ str ]:
    return [ sys.executable, "-c", code ]

def make_runs( *commands: list[ str ] ) -> list[ dict[ str, Any ] ]:
    return [
        { "run_id": f"R{index}", "experiment": "exp", "command": command }
        for index, command in enumerate( commands, start = 1 )
    ]
