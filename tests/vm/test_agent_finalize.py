import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from vmsupport import BUCKET, JOB, FakeBin, launcher_record, make_runs

JOB_ID = JOB[ "job_id" ]
INSTANCE = "mlr-20261120-143005-7c21"

@pytest.fixture
def finalize_job( agent: ModuleType, home: Path, fake_bin: FakeBin, tmp_path: Path, monkeypatch: pytest.MonkeyPatch ):
    monkeypatch.setenv( "MLR_RETRY_SECONDS", "0" )
    fake_bin.add_logging( "shutdown" )

    def prepare( current: str | None, phase: str = "running", started_ago: float = 60 ) -> None:
        runs = make_runs( [ sys.executable ], [ sys.executable ], [ sys.executable ] )
        metadata = tmp_path / "metadata.json"
        metadata.write_text( json.dumps( {
            "mlr-job": json.dumps( { **JOB, "runs": runs } ),
            "name": INSTANCE,
            "zone": "projects/123/zones/asia-southeast1-a"
        } ) )
        monkeypatch.setenv( "MLR_METADATA_FILE", str( metadata ) )
        for run in runs:
            ( tmp_path / "record.json" ).write_text( json.dumps( launcher_record( run[ "run_id" ], "exp" ) ) )
            fake_bin.put( f"{BUCKET}/runs/exp/{run[ 'run_id' ]}/run.json", tmp_path / "record.json" )
        if current is not None:
            outputs = home / "runs" / current / "outputs"
            outputs.mkdir( parents = True )
            ( outputs / "result.txt" ).write_text( "partial" )
            ( home / "runs" / current / "checkpoints" ).mkdir()
            ( home / "runs" / current / "run.log" ).write_text( "line\n" )
        started_at = time.time() - started_ago
        state = agent.State( job_id = JOB_ID, phase = phase, current_run = current, started_at = started_at )
        agent.write_state( home, state )

    return prepare

def saved( fake_bin: FakeBin, run_id: str ) -> dict[ str, Any ]:
    return json.loads( fake_bin.read( f"{BUCKET}/runs/exp/{run_id}/run.json" ) )

def test_a_job_that_is_done_is_left_alone( agent: ModuleType, finalize_job, fake_bin: FakeBin ) -> None:
    finalize_job( current = "R1", phase = "done" )

    assert agent.main( [ "agent.py", "finalize" ] ) == 0
    assert fake_bin.lines( "gcloud" ) == []

def test_without_a_state_file_there_is_nothing_to_finalize( agent: ModuleType, home: Path, fake_bin: FakeBin ) -> None:
    assert agent.main( [ "agent.py", "finalize" ] ) == 0
    assert fake_bin.lines( "gcloud" ) == []

def test_a_preempted_job_marks_the_current_and_remaining_runs(
    agent: ModuleType,
    finalize_job,
    fake_bin: FakeBin
) -> None:
    finalize_job( current = "R2" )
    first = { **saved( fake_bin, "R1" ), "status": "succeeded", "exit_code": 0 }
    ( fake_bin.bucket / BUCKET / "runs" / "exp" / "R1" / "run.json" ).write_text( json.dumps( first ) )

    assert agent.main( [ "agent.py", "finalize" ] ) == 0

    assert saved( fake_bin, "R1" )[ "status" ] == "succeeded"
    current = saved( fake_bin, "R2" )
    assert current[ "status" ] == "preempted"
    assert current[ "message" ] == "VM preempted"
    assert current[ "times" ][ "finished" ] is not None
    assert saved( fake_bin, "R3" )[ "status" ] == "preempted"
    assert saved( fake_bin, "R3" )[ "message" ] == "job ended early"

def test_the_current_run_gets_a_last_sync( agent: ModuleType, finalize_job, fake_bin: FakeBin ) -> None:
    finalize_job( current = "R1" )

    agent.main( [ "agent.py", "finalize" ] )

    assert fake_bin.read( f"{BUCKET}/runs/exp/R1/outputs/result.txt" ) == "partial"
    assert fake_bin.read( f"{BUCKET}/runs/exp/R1/run.log" ) == "line\n"

def test_a_cancel_marker_makes_the_status_cancelled(
    agent: ModuleType,
    finalize_job,
    fake_bin: FakeBin,
    tmp_path: Path
) -> None:
    finalize_job( current = "R1" )
    ( tmp_path / "marker" ).write_text( "" )
    fake_bin.put( f"{BUCKET}/jobs/{JOB_ID}/cancel", tmp_path / "marker" )

    agent.main( [ "agent.py", "finalize" ] )

    assert saved( fake_bin, "R1" )[ "status" ] == "cancelled"
    assert saved( fake_bin, "R3" )[ "status" ] == "cancelled"

def test_a_job_past_its_limit_is_timed_out( agent: ModuleType, finalize_job, fake_bin: FakeBin ) -> None:
    finalize_job( current = "R1", started_ago = 7 * 3600 )

    agent.main( [ "agent.py", "finalize" ] )

    assert saved( fake_bin, "R1" )[ "status" ] == "timed_out"

def test_a_job_that_ended_during_setup_marks_every_run( agent: ModuleType, finalize_job, fake_bin: FakeBin ) -> None:
    finalize_job( current = None, phase = "installing" )

    agent.main( [ "agent.py", "finalize" ] )

    for run_id in [ "R1", "R2", "R3" ]:
        assert saved( fake_bin, run_id )[ "status" ] == "preempted"
        assert saved( fake_bin, run_id )[ "message" ] == "job ended early"

def test_after_a_reboot_the_reason_is_the_message_and_the_vm_is_deleted(
    agent: ModuleType,
    finalize_job,
    fake_bin: FakeBin
) -> None:
    finalize_job( current = "R1" )

    assert agent.main( [ "agent.py", "finalize", "--reason", "VM restarted" ] ) == 0

    assert saved( fake_bin, "R1" )[ "message" ] == "VM restarted"
    assert fake_bin.lines( "gcloud" )[ -1 ] == f"compute instances delete {INSTANCE} --zone=asia-southeast1-a --quiet"
    assert agent.read_state( agent.home_dir() ).phase == "done"

def test_a_second_finalize_does_nothing( agent: ModuleType, finalize_job, fake_bin: FakeBin ) -> None:
    finalize_job( current = "R1" )
    agent.main( [ "agent.py", "finalize" ] )
    calls = len( fake_bin.lines( "gcloud" ) )

    agent.main( [ "agent.py", "finalize" ] )

    assert len( fake_bin.lines( "gcloud" ) ) == calls

def test_a_slow_sync_is_cut_off_inside_the_budget(
    agent: ModuleType,
    finalize_job,
    fake_bin: FakeBin,
    home: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    finalize_job( current = "R1" )
    monkeypatch.setenv( "FAKE_GCLOUD_DELAY", "5" )
    monkeypatch.setenv( "MLR_FINALIZE_SECONDS", "0.6" )
    started = time.monotonic()

    agent.main( [ "agent.py", "finalize" ] )

    assert time.monotonic() - started < 4
    assert saved( fake_bin, "R1" )[ "status" ] == "preempted"
    assert "out of time" in ( home / "agent.log" ).read_text()
