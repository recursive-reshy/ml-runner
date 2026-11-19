from pathlib import Path

from typer.testing import CliRunner

from ml_runner.cli import app
from ml_runner.cli.state import State
from ml_runner.config import save_config
from ml_runner.gcloud import Runner
from tests.fakes.fake_gcloud import FakeGcloud
from tests.fakes.launch import CapacityRunner, found_ref, make_config, make_launch_fake
from tests.fakes.project import mutating_calls

runner = CliRunner()
ZONES = [ "asia-southeast1-a", "asia-southeast1-b", "asia-southeast1-c" ]

def make_project( tmp_path: Path ) -> Path:
    src = tmp_path / "nlp-emotion"
    src.mkdir()
    ( src / "pyproject.toml" ).write_text( "[project]\n" )
    ( src / "train.py" ).write_text( "print( 1 )" )

    return src

def invoke(
    tmp_path: Path,
    *args: str,
    fake: FakeGcloud | None = None,
    gcloud: Runner | None = None,
    has_config: bool = True,
    ref_output: str | None = None
):
    config_file = tmp_path / "config.toml"
    if has_config:
        save_config( config_file, make_config() )

    state = State(
        runner = gcloud or fake or make_launch_fake(),
        ls_remote = found_ref if ref_output is None else lambda url, ref: ref_output
    )

    return runner.invoke(
        app,
        [ "launch", *args ],
        obj = state,
        env = { "ML_RUNNER_CONFIG": str( config_file ) }
    )

def test_launch_without_a_config_says_to_run_init( tmp_path: Path ) -> None:
    result = invoke( tmp_path, "--", "python", "train.py", has_config = False )
    assert result.exit_code == 2
    assert "ml-runner init" in result.output

def test_launch_without_a_command_is_a_usage_error( tmp_path: Path ) -> None:
    result = invoke( tmp_path, "--src", str( make_project( tmp_path ) ) )
    assert result.exit_code == 2

def test_a_launch_prints_the_ids_the_vm_the_zone_and_how_to_follow_it( tmp_path: Path ) -> None:
    fake = make_launch_fake()
    result = invoke( tmp_path, "--src", str( make_project( tmp_path ) ), "--", "python", "train.py", fake = fake )
    assert result.exit_code == 0, result.output
    create = [ call for call in fake.calls if call[ 1:4 ] == [ "compute", "instances", "create" ] ][ 0 ]
    vm = create[ 4 ]
    assert vm.startswith( "mlr-" )
    for needle in ( vm, vm.removeprefix( "mlr-" ), "python-train-py", "asia-southeast1-b", "ml-runner status" ):
        assert needle in result.output
    assert "ml-runner logs" in result.output and "ml-runner cancel" in result.output
    assert [ " ".join( call[ 1:3 ] ) for call in mutating_calls( fake ) ] == [
        "storage cp",
        "storage cp",
        "storage cp",
        "compute instances"
    ]

def test_the_flags_reach_the_create_command( tmp_path: Path ) -> None:
    fake = make_launch_fake()
    result = invoke(
        tmp_path,
        "--src",
        str( make_project( tmp_path ) ),
        "--gpu",
        "l4",
        "--spot",
        "--max-hours",
        "2",
        "--disk-gb",
        "50",
        "--env",
        "WANDB_MODE=offline",
        "--name",
        "first",
        "--experiment",
        "emotion",
        "--",
        "python",
        "train.py",
        fake = fake
    )
    assert result.exit_code == 0, result.output
    create = [ call for call in fake.calls if call[ 1:4 ] == [ "compute", "instances", "create" ] ][ 0 ]
    for flag in ( "--machine-type=g2-standard-4", "--provisioning-model=SPOT", "--max-run-duration=135m" ):
        assert flag in create
    assert "--boot-disk-size=50GB" in create
    assert "-first-" in result.output and "emotion" in result.output

def test_bad_usage_exits_2_and_creates_nothing( tmp_path: Path ) -> None:
    fake = make_launch_fake()
    src = str( make_project( tmp_path ) )
    for args in (
        [ "--gpu", "a100", "--src", src, "--", "x" ],
        [ "--max-hours", "30", "--src", src, "--", "x" ],
        [ "--env", "lower=1", "--src", src, "--", "x" ],
        [ "--repo", "https://github.com/example/x", "--src", src, "--ref", "main", "--", "x" ],
        [ "--repo", "https://github.com/example/x", "--", "x" ]
    ):
        result = invoke( tmp_path, *args, fake = fake )
        assert result.exit_code == 2, ( args, result.output )

    assert fake.calls == []

def test_a_missing_git_ref_is_refused_before_any_gcloud_call( tmp_path: Path ) -> None:
    fake = make_launch_fake()
    result = invoke(
        tmp_path,
        "--repo",
        "https://github.com/example/nlp-emotion",
        "--ref",
        "nope",
        "--",
        "python",
        "train.py",
        fake = fake,
        ref_output = ""
    )
    assert result.exit_code == 2
    assert "nope" in result.output
    assert fake.calls == []

def test_a_git_launch_uploads_no_snapshot( tmp_path: Path ) -> None:
    fake = make_launch_fake()
    result = invoke(
        tmp_path,
        "--repo",
        "https://github.com/example/nlp-emotion",
        "--ref",
        "main",
        "--",
        "python",
        "train.py",
        fake = fake
    )
    assert result.exit_code == 0, result.output
    assert not [ call for call in fake.calls if "objects" in call ]

def test_a_dry_run_prints_the_plan_and_every_command_and_changes_nothing( tmp_path: Path ) -> None:
    fake = make_launch_fake()
    src = str( make_project( tmp_path ) )
    result = invoke( tmp_path, "--dry-run", "--src", src, "--", "python", "train.py", fake = fake )
    assert result.exit_code == 0, result.output
    assert mutating_calls( fake ) == []
    for needle in (
        "Dry run",
        "Job:",
        "Run:",
        "Source:     snapshot of nlp-emotion",
        "Machine:    t4",
        "gcloud storage cp",
        "/jobs/",
        "gcloud compute instances create mlr-"
    ):
        assert needle in result.output, needle
    assert "ml-runner status" not in result.output

def test_a_git_dry_run_names_the_repo_and_the_ref( tmp_path: Path ) -> None:
    fake = make_launch_fake()
    result = invoke(
        tmp_path,
        "--dry-run",
        "--repo",
        "https://github.com/example/nlp-emotion",
        "--ref",
        "main",
        "--",
        "python",
        "train.py",
        fake = fake
    )
    assert result.exit_code == 0, result.output
    assert "git https://github.com/example/nlp-emotion at main" in result.output
    assert "/sources/" not in result.output

def test_a_dry_run_never_prints_env_values( tmp_path: Path ) -> None:
    result = invoke(
        tmp_path,
        "--dry-run",
        "--src",
        str( make_project( tmp_path ) ),
        "--env",
        "TOKEN=s3cret-token-value",
        "--",
        "python",
        "train.py"
    )
    assert result.exit_code == 0, result.output
    assert "s3cret-token-value" not in result.output
    assert "TOKEN" in result.output

def test_no_capacity_anywhere_exits_1_with_the_zones_tried( tmp_path: Path ) -> None:
    fake = make_launch_fake()
    result = invoke(
        tmp_path,
        "--src",
        str( make_project( tmp_path ) ),
        "--",
        "python",
        "train.py",
        gcloud = CapacityRunner( fake, ZONES )
    )
    assert result.exit_code == 1
    assert "capacity" in result.output
    for zone in ZONES:
        assert zone in result.output

def test_a_quota_error_exits_1_and_points_at_doctor( tmp_path: Path ) -> None:
    fake = make_launch_fake()
    result = invoke(
        tmp_path,
        "--src",
        str( make_project( tmp_path ) ),
        "--",
        "python",
        "train.py",
        gcloud = CapacityRunner( fake, ZONES, "quota_exceeded" )
    )
    assert result.exit_code == 1
    assert "ml-runner doctor" in result.output
