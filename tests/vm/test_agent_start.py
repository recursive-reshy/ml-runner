import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from vmsupport import FAKE_NVIDIA_SMI, JOB, FakeBin, launcher_record

BUCKET = "my-bucket"
JOB_ID = JOB[ "job_id" ]
INSTANCE = "mlr-20261120-143005-7c21"
SHA = "a" * 64

def make_snapshot( directory: Path, files: dict[ str, str ] ) -> Path:
    archive = directory / "snapshot.tar.gz"
    with tarfile.open( archive, "w:gz" ) as bundle:
        for name, content in files.items():
            member = directory / name
            member.write_text( content )
            bundle.add( member, arcname = name )

    return archive

@pytest.fixture
def start_job( agent: ModuleType, home: Path, fake_bin: FakeBin, tmp_path: Path, monkeypatch: pytest.MonkeyPatch ):
    fake_bin.add_logging( "uv" )
    fake_bin.add_logging( "shutdown" )
    monkeypatch.setenv( "MLR_RETRY_SECONDS", "0" )
    monkeypatch.setenv( "MLR_GPU_POLL_SECONDS", "0.01" )
    monkeypatch.setenv( "MLR_GPU_WAIT_SECONDS", "0.1" )

    def start( files: dict[ str, str ] | None = None, **overrides: Any ) -> int:
        files = { "requirements.txt": "" } if files is None else files
        fake_bin.put( f"{BUCKET}/sources/{SHA}.tar.gz", make_snapshot( tmp_path, files ) )
        runs = [ { "run_id": "R1", "experiment": "exp", "command": [ sys.executable, "-c", "print( 'ok' )" ] } ]
        job = { **JOB, "gpu": "none", "runs": runs, **overrides }
        ( tmp_path / "record.json" ).write_text( json.dumps( launcher_record( "R1", "exp" ) ) )
        fake_bin.put( f"{BUCKET}/runs/exp/R1/run.json", tmp_path / "record.json" )
        metadata = tmp_path / "metadata.json"
        metadata.write_text( json.dumps( {
            "mlr-job": json.dumps( job ),
            "name": INSTANCE,
            "zone": "projects/123/zones/asia-southeast1-a"
        } ) )
        monkeypatch.setenv( "MLR_METADATA_FILE", str( metadata ) )

        return agent.main( [ "agent.py", "start" ] )

    return start

def saved_record( fake_bin: FakeBin ) -> dict[ str, Any ]:
    return json.loads( fake_bin.read( f"{BUCKET}/runs/exp/R1/run.json" ) )

def test_a_whole_job_runs_uploads_its_log_and_deletes_its_vm(
    agent: ModuleType,
    start_job,
    fake_bin: FakeBin,
    home: Path
) -> None:
    assert start_job() == 0

    assert saved_record( fake_bin )[ "status" ] == "succeeded"
    assert fake_bin.has( f"{BUCKET}/jobs/{JOB_ID}/agent.log" )
    assert fake_bin.lines( "gcloud" )[ -1 ] == f"compute instances delete {INSTANCE} --zone=asia-southeast1-a --quiet"
    assert agent.read_state( home ).phase == "done"
    assert fake_bin.lines( "shutdown" ) == []

def test_a_failed_self_delete_powers_the_vm_off(
    start_job,
    fake_bin: FakeBin,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv( "FAKE_GCLOUD_FAIL", "compute" )

    start_job()

    assert fake_bin.lines( "shutdown" ) == [ "-h now" ]

def test_a_missing_dependency_file_fails_every_run( start_job, fake_bin: FakeBin ) -> None:
    assert start_job( files = { "train.py": "" } ) == 0

    saved = saved_record( fake_bin )
    assert saved[ "status" ] == "failed"
    assert saved[ "message" ] == "dependency install failed"
    assert fake_bin.has( f"{BUCKET}/jobs/{JOB_ID}/agent.log" )
    assert fake_bin.lines( "gcloud" )[ -1 ].startswith( "compute instances delete" )

def test_a_failed_uv_sync_fails_every_run( start_job, fake_bin: FakeBin, monkeypatch: pytest.MonkeyPatch ) -> None:
    monkeypatch.setenv( "FAKE_UV_EXIT", "1" )

    start_job()

    assert saved_record( fake_bin )[ "message" ] == "dependency install failed"

def test_a_source_that_cannot_be_fetched_fails_every_run(
    start_job,
    fake_bin: FakeBin,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv( "FAKE_GIT_EXIT", "1" )

    start_job( source = { "kind": "git", "repo": "https://github.com/x/y", "ref": "main" } )

    assert saved_record( fake_bin )[ "message" ] == "source fetch failed"

def test_a_gpu_that_never_appears_fails_every_run(
    start_job,
    fake_bin: FakeBin,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bin.add( "nvidia-smi", FAKE_NVIDIA_SMI )
    monkeypatch.setenv( "FAKE_NVIDIA_FAILS", "999" )

    start_job( gpu = "t4" )

    saved = saved_record( fake_bin )
    assert saved[ "status" ] == "failed"
    assert saved[ "message" ] == "GPU driver not ready"

def test_the_requirements_environment_puts_its_bin_first_on_the_path(
    start_job,
    fake_bin: FakeBin,
    home: Path
) -> None:
    # The run's own PATH must start with the venv's bin directory.
    code = (
        "import os, pathlib\n"
        "pathlib.Path( os.environ[ 'OUTPUT_DIR' ], 'path.txt' ).write_text( os.environ[ 'PATH' ] )"
    )

    start_job( runs = [ { "run_id": "R1", "experiment": "exp", "command": [ sys.executable, "-c", code ] } ] )

    assert fake_bin.read( f"{BUCKET}/runs/exp/R1/outputs/path.txt" ).startswith( str( home / "src" / ".venv" / "bin" ) )

def test_an_unreadable_job_exits_one(
    agent: ModuleType,
    home: Path,
    fake_bin: FakeBin,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv( "MLR_METADATA_FILE", raising = False )
    monkeypatch.setenv( "MLR_METADATA_FILE", "/no/such/file" )

    assert agent.main( [ "agent.py", "start" ] ) == 1
