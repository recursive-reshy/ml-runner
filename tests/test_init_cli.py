from pathlib import Path

from typer.testing import CliRunner

from ml_runner.cli import app
from ml_runner.cli.state import State
from tests.fakes.fake_gcloud import FakeGcloud
from tests.fakes.project import BUCKET, PROJECT, mutating_calls, register_mutations, register_project

runner = CliRunner()

def invoke( fake: FakeGcloud, tmp_path: Path, *args: str, input: str | None = None ):
    env = { "ML_RUNNER_CONFIG": str( tmp_path / "missing.toml" ) }

    return runner.invoke( app, [ "init", *args ], obj = State( runner = fake ), env = env, input = input )

def fresh_project( fake: FakeGcloud ) -> None:
    register_project(
        fake,
        has_apis = False,
        has_bucket = False,
        has_service_account = False,
        has_role = False,
        has_project_binding = False
    )
    register_mutations( fake )

def test_init_works_without_a_config_file_and_creates_everything_with_yes( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fresh_project( fake )
    result = invoke( fake, tmp_path, "--project", PROJECT, "--yes" )
    assert result.exit_code == 0, result.output
    assert f"created: Bucket gs://{BUCKET}" in result.output
    assert len( mutating_calls( fake ) ) == 7

def test_init_again_reports_exists_and_changes_nothing( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    register_project( fake )
    result = invoke( fake, tmp_path, "--project", PROJECT )
    assert result.exit_code == 0, result.output
    assert f"exists: Bucket gs://{BUCKET}" in result.output
    assert "created" not in result.output
    assert mutating_calls( fake ) == []

def test_init_dry_run_prints_a_header_and_the_commands_and_creates_nothing( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fresh_project( fake )
    result = invoke( fake, tmp_path, "--project", PROJECT, "--dry-run" )
    assert result.exit_code == 0, result.output
    assert "nothing will be created" in result.output.lower()
    assert f"gcloud storage buckets create gs://{BUCKET}" in result.output
    assert "would create" in result.output
    assert mutating_calls( fake ) == []

def test_init_asks_once_and_stops_on_no( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fresh_project( fake )
    result = invoke( fake, tmp_path, "--project", PROJECT, input = "n\n" )
    assert result.exit_code == 1
    assert "Nothing was changed" in result.output
    assert mutating_calls( fake ) == []

def test_init_proceeds_on_yes( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fresh_project( fake )
    result = invoke( fake, tmp_path, "--project", PROJECT, input = "y\n" )
    assert result.exit_code == 0, result.output
    assert len( mutating_calls( fake ) ) == 7

def test_init_uses_the_gcloud_config_project_when_no_flag( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fake.register( [ "config", "get-value", "project" ], stdout = f"{PROJECT}\n" )
    register_project( fake )
    result = invoke( fake, tmp_path )
    assert result.exit_code == 0, result.output

def test_init_with_no_project_anywhere_exits_2( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fake.register( [ "config", "get-value", "project" ], stdout = "" )
    result = invoke( fake, tmp_path )
    assert result.exit_code == 2
    assert "--project" in result.output

def test_init_honours_region_and_bucket_flags( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fresh_project( fake )
    result = invoke( fake, tmp_path, "--project", PROJECT, "--region", "us-central1", "--bucket", "my-own", "--yes" )
    assert result.exit_code == 0, result.output
    create = next( call for call in mutating_calls( fake ) if call[ 1:4 ] == [ "storage", "buckets", "create" ] )
    assert "gs://my-own" in create
    assert "--location=us-central1" in create
