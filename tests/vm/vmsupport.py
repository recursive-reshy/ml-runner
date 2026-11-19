import stat
from pathlib import Path

# Logs every call's argv as one line, then copies FAKE_CP_SOURCE for `storage cp`, exiting with FAKE_GCLOUD_EXIT.
FAKE_GCLOUD = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_CALLS/gcloud.log"
if [[ "$1 $2" == "storage cp" && -n "${FAKE_CP_SOURCE:-}" ]]; then
    cp "$FAKE_CP_SOURCE" "$4"
fi
exit "${FAKE_GCLOUD_EXIT:-0}"
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

class FakeBin:
    def __init__( self, root: Path ) -> None:
        self.bin = root / "bin"
        self.calls = root / "calls"
        self.bin.mkdir()
        self.calls.mkdir()

    def add( self, name: str, script: str ) -> None:
        path = self.bin / name
        path.write_text( script )
        path.chmod( path.stat().st_mode | stat.S_IEXEC )

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
