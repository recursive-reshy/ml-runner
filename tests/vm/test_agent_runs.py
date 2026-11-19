import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from vmsupport import JOB, FakeBin, make_runs, python_command

BUCKET = "my-bucket"
JOB_ID = JOB[ "job_id" ]

def record( fake_bin: FakeBin, run_id: str ) -> dict[ str, Any ]:
    return json.loads( fake_bin.read( f"{BUCKET}/runs/exp/{run_id}/run.json" ) )

def test_a_successful_run_is_recorded_with_times_and_exit_code(
    agent: ModuleType,
    job_runner,
    fake_bin: FakeBin,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks = iter( [
        datetime( 2026, 11, 20, 14, 40, 0, tzinfo = timezone.utc ),
        datetime( 2026, 11, 20, 14, 52, 22, tzinfo = timezone.utc )
    ] )
    monkeypatch.setattr( agent, "utc_now", lambda: next( ticks ) )

    job_runner( make_runs( python_command( "print( 'hi' )" ) ) )

    saved = record( fake_bin, "R1" )
    assert saved[ "status" ] == "succeeded"
    assert saved[ "exit_code" ] == 0
    assert saved[ "times" ] == {
        "created": "2026-11-20T14:30:05Z",
        "started": "2026-11-20T14:40:00Z",
        "finished": "2026-11-20T14:52:22Z"
    }
    assert saved[ "duration_seconds" ] == 742.0
    assert saved[ "command" ] == [ "python", "train.py" ]

def test_the_running_record_is_uploaded_before_the_command_starts( job_runner, fake_bin: FakeBin ) -> None:
    marker = f"open( '{fake_bin.calls / 'gcloud.log'}', 'a' ).write( 'COMMAND-RAN\\n' )"

    job_runner( make_runs( python_command( marker ) ) )

    lines = fake_bin.lines( "gcloud" )
    record_uri = "gs://my-bucket/runs/exp/R1/run.json"
    upload = next( index for index, line in enumerate( lines ) if line.endswith( record_uri ) )
    assert upload < lines.index( "COMMAND-RAN" )

def test_the_command_gets_the_run_contract_and_the_job_env( job_runner, fake_bin: FakeBin, home: Path ) -> None:
    code = (
        "import json, os, pathlib\n"
        "keys = [ 'OUTPUT_DIR', 'CHECKPOINT_DIR', 'RUN_ID', 'ML_RUNNER', 'ML_RUNNER_GPU', 'KEY' ]\n"
        "seen = { key: os.environ[ key ] for key in keys }\n"
        "pathlib.Path( os.environ[ 'OUTPUT_DIR' ], 'env.json' ).write_text( json.dumps( seen ) )\n"
        "pathlib.Path( os.environ[ 'CHECKPOINT_DIR' ], 'step-1.txt' ).write_text( 'x' )\n"
    )

    job_runner( make_runs( python_command( code ) ) )

    seen = json.loads( fake_bin.read( f"{BUCKET}/runs/exp/R1/outputs/env.json" ) )
    assert seen == {
        "OUTPUT_DIR": str( home / "runs" / "R1" / "outputs" ),
        "CHECKPOINT_DIR": str( home / "runs" / "R1" / "checkpoints" ),
        "RUN_ID": "R1",
        "ML_RUNNER": "1",
        "ML_RUNNER_GPU": "t4",
        "KEY": "VALUE"
    }
    assert fake_bin.has( f"{BUCKET}/runs/exp/R1/checkpoints/step-1.txt" )

def test_the_command_runs_in_the_source_directory_without_a_shell(
    job_runner,
    fake_bin: FakeBin,
    tmp_path: Path
) -> None:
    code = "import os, pathlib; pathlib.Path( os.environ[ 'OUTPUT_DIR' ], 'cwd.txt' ).write_text( os.getcwd() )"

    job_runner( make_runs( [ sys.executable, "-c", code, "$(echo no-shell)" ] ) )

    seen = Path( fake_bin.read( f"{BUCKET}/runs/exp/R1/outputs/cwd.txt" ) )
    assert seen.resolve() == ( tmp_path / "source" ).resolve()

def test_each_run_starts_with_empty_directories( job_runner, home: Path ) -> None:
    stale = home / "runs" / "R1" / "outputs"
    stale.mkdir( parents = True )
    ( stale / "old.txt" ).write_text( "old" )

    job_runner( make_runs( python_command( "pass" ) ) )

    assert not ( stale / "old.txt" ).exists()

def test_a_failing_command_is_recorded_as_failed( job_runner, fake_bin: FakeBin ) -> None:
    job_runner( make_runs( python_command( "import sys; sys.exit( 3 )" ) ) )

    saved = record( fake_bin, "R1" )
    assert saved[ "status" ] == "failed"
    assert saved[ "exit_code" ] == 3

def test_a_command_that_cannot_start_is_recorded_as_failed( job_runner, fake_bin: FakeBin ) -> None:
    job_runner( make_runs( [ "/no/such/program" ] ) )

    saved = record( fake_bin, "R1" )
    assert saved[ "status" ] == "failed"
    assert "cannot start" in saved[ "message" ]

def test_stdout_and_stderr_go_to_the_run_log_and_the_agent_log( job_runner, fake_bin: FakeBin, home: Path ) -> None:
    code = "import sys; print( 'to-out' ); print( 'to-err', file = sys.stderr )"

    job_runner( make_runs( python_command( code ) ) )

    run_log = fake_bin.read( f"{BUCKET}/runs/exp/R1/run.log" )
    assert "to-out" in run_log and "to-err" in run_log
    agent_log = ( home / "agent.log" ).read_text()
    assert "to-out" in agent_log and "to-err" in agent_log

def test_the_log_is_synced_while_the_run_is_going(
    job_runner,
    fake_bin: FakeBin,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv( "MLR_LOG_SYNC_SECONDS", "0.05" )

    job_runner( make_runs( python_command( "import time; print( 'start', flush = True ); time.sleep( 1 )" ) ) )

    uploads = [ line for line in fake_bin.lines( "gcloud" ) if line.endswith( "gs://my-bucket/runs/exp/R1/run.log" ) ]
    assert len( uploads ) >= 3

def test_outputs_are_synced_while_the_run_is_going(
    job_runner,
    fake_bin: FakeBin,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv( "MLR_DATA_SYNC_SECONDS", "0.05" )

    job_runner( make_runs( python_command( "import time; time.sleep( 1 )" ) ) )

    syncs = [ line for line in fake_bin.lines( "gcloud" ) if line.endswith( "gs://my-bucket/runs/exp/R1/outputs" ) ]
    assert len( syncs ) >= 3
    assert all( line.startswith( "storage rsync -r " ) for line in syncs )
    assert not any( "delete-unmatched" in line for line in fake_bin.lines( "gcloud" ) )

def test_a_failing_sync_does_not_stop_the_run_and_the_final_sync_retries(
    job_runner,
    fake_bin: FakeBin,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv( "FAKE_GCLOUD_FAIL", "rsync" )

    job_runner( make_runs( python_command( "pass" ) ) )

    assert record( fake_bin, "R1" )[ "status" ] == "succeeded"
    attempts = [ line for line in fake_bin.lines( "gcloud" ) if line.endswith( "gs://my-bucket/runs/exp/R1/outputs" ) ]
    assert len( attempts ) == 3

def test_stop_on_failure_skips_the_remaining_runs( job_runner, fake_bin: FakeBin ) -> None:
    runs = make_runs( python_command( "import sys; sys.exit( 1 )" ), python_command( "pass" ) )

    job_runner( runs, stop_on_failure = True )

    second = record( fake_bin, "R2" )
    assert second[ "status" ] == "failed"
    assert second[ "message" ] == "skipped after an earlier failure"
    assert second[ "times" ][ "started" ] is None

def test_without_stop_on_failure_the_next_run_still_runs( job_runner, fake_bin: FakeBin ) -> None:
    runs = make_runs( python_command( "import sys; sys.exit( 1 )" ), python_command( "pass" ) )

    job_runner( runs, stop_on_failure = False )

    assert record( fake_bin, "R1" )[ "status" ] == "failed"
    assert record( fake_bin, "R2" )[ "status" ] == "succeeded"

def test_a_cancel_marker_stops_the_job_before_the_next_run( job_runner, fake_bin: FakeBin, tmp_path: Path ) -> None:
    marker = tmp_path / "marker"
    marker.write_text( "" )
    fake_bin.put( f"{BUCKET}/jobs/{JOB_ID}/cancel", marker )

    job_runner( make_runs( python_command( "pass" ), python_command( "pass" ) ) )

    for run_id in [ "R1", "R2" ]:
        saved = record( fake_bin, run_id )
        assert saved[ "status" ] == "cancelled"
        assert saved[ "times" ][ "started" ] is None

def test_a_run_past_the_job_limit_is_terminated_and_timed_out(
    job_runner,
    fake_bin: FakeBin,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv( "MLR_MAX_SECONDS", "0.3" )
    started = time.monotonic()

    job_runner( make_runs( python_command( "import time; time.sleep( 60 )" ), python_command( "pass" ) ) )

    assert time.monotonic() - started < 10
    assert record( fake_bin, "R1" )[ "status" ] == "timed_out"
    second = record( fake_bin, "R2" )
    assert second[ "status" ] == "timed_out"
    assert second[ "times" ][ "started" ] is None

def test_a_process_that_ignores_sigterm_is_killed_after_the_grace_period(
    job_runner,
    fake_bin: FakeBin,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv( "MLR_MAX_SECONDS", "0.3" )
    monkeypatch.setenv( "MLR_KILL_GRACE_SECONDS", "0.3" )
    code = "import signal, time; signal.signal( signal.SIGTERM, signal.SIG_IGN ); time.sleep( 60 )"
    started = time.monotonic()

    job_runner( make_runs( python_command( code ) ) )

    assert time.monotonic() - started < 10
    assert record( fake_bin, "R1" )[ "status" ] == "timed_out"
