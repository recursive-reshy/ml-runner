from types import ModuleType

import pytest

from vmsupport import BUCKET, FakeBin, make_runs, python_command

# Runs only where the CLI package is importable (not in the Python 3.10 agent-only step).
@pytest.fixture
def records_module() -> ModuleType:
    return pytest.importorskip( "ml_runner.records" )

def test_every_record_the_agent_writes_is_a_valid_run_record(
    records_module: ModuleType,
    job_runner,
    fake_bin: FakeBin,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv( "MLR_MAX_SECONDS", "5" )
    runs = make_runs(
        python_command( "pass" ),
        python_command( "import sys; sys.exit( 2 )" ),
        python_command( "pass" )
    )

    job_runner( runs, stop_on_failure = True )

    for run in runs:
        text = fake_bin.read( f"{BUCKET}/runs/exp/{run[ 'run_id' ]}/run.json" )
        record = records_module.RunRecord.from_json( text )
        assert record.status.value in { "succeeded", "failed" }
    assert records_module.RunRecord.from_json( fake_bin.read( f"{BUCKET}/runs/exp/R2/run.json" ) ).exit_code == 2

def test_timed_out_records_are_valid(
    records_module: ModuleType,
    job_runner,
    fake_bin: FakeBin,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv( "MLR_MAX_SECONDS", "0.3" )
    monkeypatch.setenv( "MLR_KILL_GRACE_SECONDS", "0.3" )

    job_runner( make_runs( python_command( "import time; time.sleep( 60 )" ), python_command( "pass" ) ) )

    for run_id in [ "R1", "R2" ]:
        record = records_module.RunRecord.from_json( fake_bin.read( f"{BUCKET}/runs/exp/{run_id}/run.json" ) )
        assert record.status.value == "timed_out"

def test_agent_status_words_are_all_run_statuses( records_module: ModuleType, agent: ModuleType ) -> None:
    known = { status.value for status in records_module.RunStatus }

    assert { agent.SUCCEEDED, agent.FAILED, agent.RUNNING, agent.CANCELLED, agent.TIMED_OUT, agent.PREEMPTED } <= known

def test_the_job_json_the_cli_writes_is_what_the_agent_parses( records_module: ModuleType, agent: ModuleType ) -> None:
    spec = records_module.JobSpec.model_validate( {
        "schema": 1,
        "job_id": "J",
        "bucket": "b",
        "max_hours": 6,
        "stop_on_failure": True,
        "source": { "kind": "git", "repo": "https://github.com/x/y", "ref": "main" },
        "runs": [ { "run_id": "R", "experiment": "e", "command": [ "python", "t.py" ] } ],
        "gpu": "t4",
        "env": { "KEY": "VALUE" }
    } )

    from_metadata = agent.parse_job( spec.metadata_json() )
    from_bucket = agent.parse_job( spec.to_json() )

    assert from_metadata.env == { "KEY": "VALUE" }
    assert from_bucket.env == {}
    assert from_metadata.runs[ 0 ].command == [ "python", "t.py" ]
    assert from_metadata.source.repo == "https://github.com/x/y"
