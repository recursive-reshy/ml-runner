import json

import pytest
from pydantic import ValidationError

from ml_runner.records import JobRun, JobSpec, Machine, RunRecord, RunStatus, Source, SourceKind, Times

SECRET = "s3cret-token-value"

RUN_JSON = {
    "schema": 1,
    "run_id": "20261120-143005-train-seed-42-3f9a",
    "job_id": "20261120-143005-7c21",
    "experiment": "nlp-emotion",
    "name": "train-seed-42",
    "command": [ "python", "train.py", "--seed", "42" ],
    "env_names": [ "WANDB_MODE" ],
    "status": "succeeded",
    "exit_code": 0,
    "source": {
        "kind": "snapshot",
        "path": "nlp-emotion",
        "sha256": "ab" * 32,
        "git_commit": "abc123",
        "git_dirty": True,
        "repo": None,
        "ref": None
    },
    "machine": {
        "vm": "mlr-20261120-143005-7c21",
        "zone": "asia-southeast1-b",
        "machine_type": "n1-standard-4",
        "gpu": "t4",
        "spot": True,
        "image": "projects/deeplearning-platform-release/global/images/family/common-cu129-ubuntu-2404-nvidia-580"
    },
    "ml_runner_version": "0.1.0",
    "times": {
        "created": "2026-11-20T14:30:05Z",
        "started": "2026-11-20T14:33:00Z",
        "finished": "2026-11-20T14:45:22Z"
    },
    "duration_seconds": 742.1,
    "message": None
}

JOB_JSON = {
    "schema": 1,
    "job_id": "20261120-143005-7c21",
    "bucket": "my-bucket",
    "max_hours": 6,
    "stop_on_failure": True,
    "source": {
        "kind": "git",
        "path": None,
        "sha256": None,
        "git_commit": None,
        "git_dirty": None,
        "repo": "https://github.com/x/y",
        "ref": "main"
    },
    "runs": [
        {
            "run_id": "20261120-143005-train-seed-42-3f9a",
            "experiment": "nlp-emotion",
            "command": [ "python", "train.py" ]
        }
    ],
    "gpu": "t4"
}

def make_job( env: dict[ str, str ] | None = None ) -> JobSpec:
    return JobSpec.model_validate( { **JOB_JSON, "env": env or {} } )

def test_a_run_record_round_trips_through_json() -> None:
    record = RunRecord.model_validate( RUN_JSON )
    assert RunRecord.from_json( record.to_json() ) == record
    assert json.loads( record.to_json() ) == RUN_JSON

def test_a_job_spec_round_trips_through_json() -> None:
    job = make_job()
    assert JobSpec.from_json( job.to_json() ) == job
    assert json.loads( job.to_json() ) == JOB_JSON

def test_the_schema_key_is_called_schema_in_json() -> None:
    assert json.loads( RunRecord.model_validate( RUN_JSON ).to_json() )[ "schema" ] == 1
    assert json.loads( make_job().to_json() )[ "schema" ] == 1

def test_the_status_values_are_exactly_the_stored_ones() -> None:
    assert [ status.value for status in RunStatus ] == [
        "provisioning",
        "running",
        "succeeded",
        "failed",
        "preempted",
        "cancelled",
        "timed_out"
    ]

def test_lost_is_not_a_storable_status() -> None:
    with pytest.raises( ValidationError ):
        RunRecord.model_validate( { **RUN_JSON, "status": "lost" } )

def test_unknown_fields_are_rejected() -> None:
    with pytest.raises( ValidationError ):
        RunRecord.model_validate( { **RUN_JSON, "surprise": 1 } )
    with pytest.raises( ValidationError ):
        JobSpec.model_validate( { **JOB_JSON, "surprise": 1 } )

def test_an_unknown_schema_version_is_rejected() -> None:
    with pytest.raises( ValidationError ):
        RunRecord.model_validate( { **RUN_JSON, "schema": 2 } )

def test_a_provisioning_record_has_no_start_finish_or_exit_code() -> None:
    record = RunRecord.model_validate(
        {
            **RUN_JSON,
            "status": "provisioning",
            "exit_code": None,
            "duration_seconds": None,
            "times": { "created": "2026-11-20T14:30:05Z", "started": None, "finished": None }
        }
    )
    assert record.status == RunStatus.PROVISIONING
    assert record.times.started is None
    assert record.exit_code is None

def test_env_values_never_reach_the_bucket_json() -> None:
    job = make_job( { "TOKEN": SECRET } )
    assert SECRET not in job.to_json()
    assert SECRET not in json.dumps( job.model_dump( mode = "json" ) )
    assert "env" not in json.loads( job.to_json() )

def test_env_values_travel_only_in_the_metadata_json() -> None:
    job = make_job( { "TOKEN": SECRET } )
    assert json.loads( job.metadata_json() )[ "env" ] == { "TOKEN": SECRET }

def test_the_metadata_json_is_the_job_json_plus_env() -> None:
    job = make_job( { "TOKEN": SECRET } )
    metadata = json.loads( job.metadata_json() )
    assert { key: value for key, value in metadata.items() if key != "env" } == json.loads( job.to_json() )

def test_the_metadata_json_reads_back_with_its_env() -> None:
    job = make_job( { "TOKEN": SECRET } )
    assert JobSpec.from_json( job.metadata_json() ).env == { "TOKEN": SECRET }

def test_a_run_record_keeps_env_names_not_values() -> None:
    record = RunRecord.model_validate( RUN_JSON )
    assert record.env_names == [ "WANDB_MODE" ]
    assert "env" not in json.loads( record.to_json() )

def test_the_models_can_be_built_in_code() -> None:
    source = Source( kind = SourceKind.GIT, repo = "https://github.com/x/y", ref = "main" )
    machine = Machine(
        vm = "mlr-a",
        zone = "asia-southeast1-a",
        machine_type = "g2-standard-4",
        gpu = "l4",
        spot = False,
        image = "projects/p/global/images/family/f"
    )
    run = JobRun( run_id = "r", experiment = "e", command = [ "echo" ] )
    job = JobSpec(
        job_id = "j",
        bucket = "b",
        max_hours = 1.5,
        stop_on_failure = True,
        source = source,
        runs = [ run ],
        gpu = "l4"
    )
    assert machine.gpu == "l4"
    assert Times( created = RunRecord.model_validate( RUN_JSON ).times.created ).started is None
    assert JobSpec.from_json( job.to_json() ) == job
