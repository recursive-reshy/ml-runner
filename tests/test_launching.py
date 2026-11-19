import json
import random
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from ml_runner.config import Defaults
from ml_runner.errors import MlRunnerError, UsageError
from ml_runner.gcloud import Gcloud, GcloudError
from ml_runner.launching import LaunchFlags, LaunchPlan, describe_plan, execute_launch, prepare_launch, resolve_options
from ml_runner.machines import NoCapacityError
from ml_runner.records import JobSpec, RunRecord, RunStatus, SourceKind
from tests.fakes.fake_gcloud import FakeGcloud
from tests.fakes.launch import COMMAND, REPO, CapacityRunner, found_ref, make_config, make_launch_fake
from tests.fakes.project import BUCKET, PROJECT, SERVICE_ACCOUNT, fixture, mutating_calls

NOW = datetime( 2026, 11, 20, 14, 30, 5, tzinfo = UTC )
SECRET = "s3cret-token-value"

def flags( **changes: Any ) -> LaunchFlags:
    return replace( LaunchFlags( command = COMMAND ), **changes )

# ----- flags and validation -----

def test_the_config_defaults_fill_in_what_the_flags_leave_out() -> None:
    config = make_config( Defaults( gpu = "l4", spot = True, max_hours = 3, disk_gb = 80, max_source_mb = 50 ) )
    options = resolve_options( config, flags() )
    assert ( options.gpu, options.is_spot, options.max_hours, options.disk_gb, options.max_source_mb ) == (
        "l4",
        True,
        3,
        80,
        50
    )

def test_flags_beat_the_config_defaults() -> None:
    config = make_config( Defaults( gpu = "l4", max_hours = 3, disk_gb = 80, max_source_mb = 50 ) )
    options = resolve_options(
        config,
        flags( gpu = "none", is_spot = True, max_hours = 1.5, disk_gb = 200, max_source_mb = 500 )
    )
    assert ( options.gpu, options.is_spot, options.max_hours, options.disk_gb, options.max_source_mb ) == (
        "none",
        True,
        1.5,
        200,
        500
    )

@pytest.mark.parametrize(
    ( "changes", "needle" ),
    [
        ( { "gpu": "a100" }, "--gpu" ),
        ( { "max_hours": 25 }, "--allow-long" ),
        ( { "max_hours": 0.01 }, "--max-hours" ),
        ( { "disk_gb": 0 }, "--disk-gb" ),
        ( { "max_source_mb": 0 }, "--max-source-mb" ),
        ( { "zone": "europe-west4-a" }, "europe-west4-a" ),
        ( { "machine_type": "g2-standard-4" }, "N1" ),
        ( { "repo": REPO, "src": Path( "." ), "ref": "main" }, "--repo" ),
        ( { "repo": REPO }, "--ref" ),
        ( { "ref": "main" }, "--repo" )
    ]
)
def test_bad_flags_are_usage_errors( changes: dict[ str, Any ], needle: str ) -> None:
    with pytest.raises( UsageError, match = needle ):
        resolve_options( make_config(), flags( **changes ) )

def test_an_empty_command_is_a_usage_error() -> None:
    with pytest.raises( UsageError, match = "command" ):
        resolve_options( make_config(), LaunchFlags( command = [] ) )

def test_more_than_24_hours_needs_allow_long() -> None:
    assert resolve_options( make_config(), flags( max_hours = 48, allow_long = True ) ).max_hours == 48

def test_env_pairs_become_a_dict() -> None:
    options = resolve_options( make_config(), flags( env = [ "WANDB_MODE=offline", "EMPTY=", "URL=a=b" ] ) )
    assert options.env == { "WANDB_MODE": "offline", "EMPTY": "", "URL": "a=b" }

@pytest.mark.parametrize(
    "pair",
    [
        "novalue",
        "lower=1",
        "1START=1",
        "BAD-NAME=1",
        "OUTPUT_DIR=x",
        "CHECKPOINT_DIR=x",
        "RUN_ID=x",
        "ML_RUNNER=1",
        "ML_RUNNER_GPU=t4"
    ]
)
def test_bad_or_reserved_env_names_are_refused( pair: str ) -> None:
    with pytest.raises( UsageError, match = "--env" ):
        resolve_options( make_config(), flags( env = [ pair ] ) )

def test_a_repeated_env_name_is_refused() -> None:
    with pytest.raises( UsageError, match = "twice" ):
        resolve_options( make_config(), flags( env = [ "A=1", "A=2" ] ) )

# ----- preparing the plan -----

def make_project( tmp_path: Path ) -> Path:
    src = tmp_path / "nlp-emotion"
    src.mkdir()
    ( src / "pyproject.toml" ).write_text( "[project]\n" )
    ( src / "train.py" ).write_text( "print( 1 )" )

    return src

def make_work( tmp_path: Path ) -> Path:
    work = tmp_path / "work"
    work.mkdir()

    return work

def make_gcloud( fake: FakeGcloud, lines: list[ str ] | None = None, *, dry_run: bool = False ) -> Gcloud:
    echo = lines.append if lines is not None else print

    return Gcloud( project = PROJECT, dry_run = dry_run, runner = fake, echo = echo )

def prepare( tmp_path: Path, fake: FakeGcloud, **changes: Any ) -> LaunchPlan:
    config = make_config()
    options = resolve_options( config, flags( src = make_project( tmp_path ), **changes ) )

    return prepare_launch(
        make_gcloud( fake ),
        config,
        options,
        make_work( tmp_path ),
        now = NOW,
        rng = random.Random( 1 ),
        ref_check = found_ref
    )

def test_a_snapshot_plan_has_a_job_one_run_and_the_source_hash( tmp_path: Path ) -> None:
    plan = prepare( tmp_path, make_launch_fake() )
    run = plan.runs[ 0 ]
    assert plan.job.job_id.startswith( "20261120-143005-" )
    assert run.run_id.startswith( "20261120-143005-python-train-py-seed-42-" )
    assert [ job_run.run_id for job_run in plan.job.runs ] == [ run.run_id ]
    assert run.job_id == plan.job.job_id
    assert run.experiment == "nlp-emotion"
    assert run.status == RunStatus.PROVISIONING
    assert run.command == COMMAND
    assert run.source.kind == SourceKind.SNAPSHOT
    assert run.source.path == "nlp-emotion"
    assert plan.snapshot is not None and run.source.sha256 == plan.snapshot.sha256
    assert plan.job.stop_on_failure is True
    assert plan.job.bucket == BUCKET

def test_names_and_experiments_can_be_chosen( tmp_path: Path ) -> None:
    plan = prepare( tmp_path, make_launch_fake(), name = "Seed 42!", experiment = "My Experiment" )
    run = plan.runs[ 0 ]
    assert ( run.name, run.experiment ) == ( "seed-42", "my-experiment" )
    assert "-seed-42-" in run.run_id

def test_a_git_plan_has_no_snapshot_and_names_the_experiment_after_the_repo( tmp_path: Path ) -> None:
    config = make_config()
    options = resolve_options( config, flags( repo = REPO, ref = "main" ) )
    plan = prepare_launch(
        make_gcloud( make_launch_fake() ),
        config,
        options,
        make_work( tmp_path ),
        now = NOW,
        rng = random.Random( 1 ),
        ref_check = found_ref
    )
    run = plan.runs[ 0 ]
    assert plan.snapshot is None
    assert ( run.source.kind, run.source.repo, run.source.ref ) == ( SourceKind.GIT, REPO, "main" )
    assert run.experiment == "nlp-emotion"

def test_a_missing_git_ref_is_refused_before_anything_else( tmp_path: Path ) -> None:
    config = make_config()
    options = resolve_options( config, flags( repo = REPO, ref = "nope" ) )
    fake = make_launch_fake()
    with pytest.raises( UsageError, match = "nope" ):
        prepare_launch(
            make_gcloud( fake ),
            config,
            options,
            make_work( tmp_path ),
            now = NOW,
            ref_check = lambda url, ref: ""
        )
    assert fake.calls == []

def test_the_machine_of_each_run_is_the_first_zone_choice( tmp_path: Path ) -> None:
    plan = prepare( tmp_path, make_launch_fake(), gpu = "t4", is_spot = True )
    machine = plan.runs[ 0 ].machine
    assert ( machine.zone, machine.machine_type, machine.gpu, machine.spot ) == (
        "asia-southeast1-b",
        "n1-standard-4",
        "t4",
        True
    )
    assert machine.vm == f"mlr-{plan.job.job_id}"
    assert plan.zones == [ "asia-southeast1-b", "asia-southeast1-a", "asia-southeast1-c" ]

def test_an_l4_job_finds_its_zones_by_the_l4_accelerator_name( tmp_path: Path ) -> None:
    plan = prepare( tmp_path, make_launch_fake(), gpu = "l4" )
    assert plan.zones[ 0 ] == "asia-southeast1-b"
    assert plan.runs[ 0 ].machine.machine_type == "g2-standard-4"

def test_a_cpu_job_asks_for_no_zones_and_uses_the_chosen_one( tmp_path: Path ) -> None:
    fake = make_launch_fake()
    plan = prepare( tmp_path, fake, gpu = "none", zone = "asia-southeast1-c" )
    assert plan.zones == [ "asia-southeast1-c" ]
    assert not [ call for call in fake.calls if "accelerator-types" in call ]

def test_a_snapshot_is_checked_against_the_size_limit( tmp_path: Path ) -> None:
    src = make_project( tmp_path )
    ( src / "big.bin" ).write_bytes( b"x" * ( 2 * 1024 * 1024 ) )
    config = make_config()
    options = resolve_options( config, flags( src = src, max_source_mb = 1 ) )
    with pytest.raises( UsageError, match = "big.bin" ):
        prepare_launch( make_gcloud( make_launch_fake() ), config, options, make_work( tmp_path ), now = NOW )

def test_the_plan_description_names_the_job_the_source_and_the_machine( tmp_path: Path ) -> None:
    plan = prepare( tmp_path, make_launch_fake(), is_spot = True )
    text = "\n".join( describe_plan( plan ) )
    for needle in ( plan.job.job_id, plan.runs[ 0 ].run_id, "nlp-emotion", "python train.py --seed 42", "t4", "Spot" ):
        assert needle in text

# ----- executing it -----

def subcommands( fake: FakeGcloud ) -> list[ str ]:
    return [ " ".join( call[ 1:3 ] ) for call in mutating_calls( fake ) ]

def uploaded( fake: FakeGcloud ) -> list[ str ]:
    return [ call[ 4 ] for call in mutating_calls( fake ) if call[ 1:3 ] == [ "storage", "cp" ] ]

def test_a_launch_uploads_the_snapshot_then_job_and_run_json_then_creates_the_vm( tmp_path: Path ) -> None:
    fake = make_launch_fake()
    plan = prepare( tmp_path, fake )
    result = execute_launch( make_gcloud( fake ), plan )
    run = plan.runs[ 0 ]
    assert subcommands( fake ) == [ "storage cp", "storage cp", "storage cp", "compute instances" ]
    assert plan.snapshot is not None
    assert uploaded( fake ) == [
        f"gs://{BUCKET}/sources/{plan.snapshot.sha256}.tar.gz",
        f"gs://{BUCKET}/jobs/{plan.job.job_id}/job.json",
        f"gs://{BUCKET}/runs/nlp-emotion/{run.run_id}/run.json"
    ]
    assert result.zone == "asia-southeast1-b"

def test_the_uploaded_job_and_run_files_hold_the_records( tmp_path: Path ) -> None:
    fake = make_launch_fake()
    plan = prepare( tmp_path, fake )
    execute_launch( make_gcloud( fake ), plan )
    work = plan.work_dir
    assert JobSpec.from_json( ( work / "job.json" ).read_text() ) == plan.job
    uploaded_run = RunRecord.from_json( ( work / f"run-{plan.runs[ 0 ].run_id}.json" ).read_text() )
    assert uploaded_run.status == RunStatus.PROVISIONING

def test_an_existing_snapshot_is_not_uploaded_again( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fake.register_json( [ "compute", "accelerator-types", "list" ], fixture( "accelerator_types" ) )
    fake.register_json( [ "storage", "objects", "describe" ], { "name": "sources/x" } )
    fake.register( [ "storage", "cp" ] )
    fake.register( [ "compute", "instances", "create" ] )
    plan = prepare( tmp_path, fake )
    execute_launch( make_gcloud( fake ), plan )
    assert subcommands( fake ) == [ "storage cp", "storage cp", "compute instances" ]

def test_a_git_launch_uploads_no_snapshot( tmp_path: Path ) -> None:
    fake = make_launch_fake()
    config = make_config()
    options = resolve_options( config, flags( repo = REPO, ref = "main" ) )
    plan = prepare_launch(
        make_gcloud( fake ),
        config,
        options,
        make_work( tmp_path ),
        now = NOW,
        rng = random.Random( 1 ),
        ref_check = found_ref
    )
    execute_launch( make_gcloud( fake ), plan )
    assert subcommands( fake ) == [ "storage cp", "storage cp", "compute instances" ]
    assert not [ call for call in fake.calls if "objects" in call ]

def test_the_create_command_has_the_cap_the_labels_and_the_metadata_files( tmp_path: Path ) -> None:
    fake = make_launch_fake()
    plan = prepare( tmp_path, fake, is_spot = True, max_hours = 2 )
    execute_launch( make_gcloud( fake ), plan )
    create = [ call for call in fake.calls if call[ 1:4 ] == [ "compute", "instances", "create" ] ][ 0 ]
    assert f"mlr-{plan.job.job_id}" in create
    assert "--max-run-duration=135m" in create
    assert "--provisioning-model=SPOT" in create
    assert f"--labels=ml-runner=1,mlr-job={plan.job.job_id}" in create
    assert f"--service-account={SERVICE_ACCOUNT}" in create
    assert f"--project={PROJECT}" in create
    from_files = [ arg for arg in create if arg.startswith( "--metadata-from-file=" ) ][ 0 ]
    assert f"mlr-job={plan.work_dir / 'mlr-job.json'}" in from_files
    for key in ( "startup-script=", "shutdown-script=", "mlr-agent=" ):
        assert key in from_files

def creates( fake: FakeGcloud ) -> list[ list[ str ] ]:
    return [ call for call in fake.calls if call[ 1:4 ] == [ "compute", "instances", "create" ] ]

def test_an_exhausted_zone_falls_back_and_the_run_record_names_the_zone_used( tmp_path: Path ) -> None:
    fake = make_launch_fake()
    plan = prepare( tmp_path, fake )
    gcloud = Gcloud( project = PROJECT, runner = CapacityRunner( fake, [ "asia-southeast1-b" ] ), echo = print )
    result = execute_launch( gcloud, plan )
    assert result.zone == "asia-southeast1-a"
    uploaded_run = RunRecord.from_json( ( plan.work_dir / f"run-{plan.runs[ 0 ].run_id}.json" ).read_text() )
    assert uploaded_run.machine.zone == "asia-southeast1-a"
    assert uploaded_run.status == RunStatus.PROVISIONING

def test_when_no_zone_has_capacity_the_runs_are_marked_failed_and_the_error_is_raised( tmp_path: Path ) -> None:
    fake = make_launch_fake()
    plan = prepare( tmp_path, fake )
    gcloud = Gcloud( project = PROJECT, runner = CapacityRunner( fake, plan.zones ), echo = print )
    with pytest.raises( NoCapacityError ):
        execute_launch( gcloud, plan )
    failed = RunRecord.from_json( ( plan.work_dir / f"run-{plan.runs[ 0 ].run_id}.json" ).read_text() )
    assert failed.status == RunStatus.FAILED
    assert failed.message is not None and "capacity" in failed.message
    assert [ call for call in fake.calls if "delete" in call ] == []
    assert ( plan.work_dir / "job.json" ).exists()

def test_a_quota_error_stops_at_the_first_zone_and_marks_the_run_failed( tmp_path: Path ) -> None:
    fake = make_launch_fake()
    plan = prepare( tmp_path, fake )
    gcloud = Gcloud( project = PROJECT, runner = CapacityRunner( fake, plan.zones, "quota_exceeded" ), echo = print )
    with pytest.raises( GcloudError, match = "ml-runner doctor" ):
        execute_launch( gcloud, plan )
    assert len( creates( fake ) ) == 0
    failed = RunRecord.from_json( ( plan.work_dir / f"run-{plan.runs[ 0 ].run_id}.json" ).read_text() )
    assert failed.status == RunStatus.FAILED
    assert failed.message is not None and "quota" in failed.message

def test_the_failed_record_is_uploaded_again_to_the_same_path( tmp_path: Path ) -> None:
    fake = make_launch_fake()
    plan = prepare( tmp_path, fake )
    gcloud = Gcloud( project = PROJECT, runner = CapacityRunner( fake, plan.zones ), echo = print )
    with pytest.raises( MlRunnerError ):
        execute_launch( gcloud, plan )
    run_path = f"gs://{BUCKET}/runs/nlp-emotion/{plan.runs[ 0 ].run_id}/run.json"
    assert uploaded( fake ).count( run_path ) == 2

def test_env_values_reach_only_the_metadata_file( tmp_path: Path ) -> None:
    fake = make_launch_fake()
    plan = prepare( tmp_path, fake, env = [ f"TOKEN={SECRET}" ] )
    execute_launch( make_gcloud( fake ), plan )
    assert json.loads( ( plan.work_dir / "mlr-job.json" ).read_text() )[ "env" ] == { "TOKEN": SECRET }
    for path in plan.work_dir.iterdir():
        if path.name != "mlr-job.json" and path.suffix == ".json":
            assert SECRET not in path.read_text()
    assert SECRET not in json.dumps( fake.calls )
    assert plan.runs[ 0 ].env_names == [ "TOKEN" ]

def test_a_dry_run_changes_nothing_and_echoes_every_mutating_command( tmp_path: Path ) -> None:
    fake = make_launch_fake()
    lines: list[ str ] = []
    plan = prepare( tmp_path, fake )
    execute_launch( make_gcloud( fake, lines, dry_run = True ), plan )
    assert mutating_calls( fake ) == []
    assert [ " ".join( line.split()[ 1:3 ] ) for line in lines ] == [
        "storage cp",
        "storage cp",
        "storage cp",
        "compute instances"
    ]
