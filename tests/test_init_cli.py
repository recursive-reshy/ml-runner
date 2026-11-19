from pathlib import Path

from typer.testing import CliRunner

from ml_runner.cli import app
from ml_runner.cli.state import State
from ml_runner.config import Defaults, load_config, save_config
from ml_runner.setup import InitSettings, build_config
from tests.fakes.fake_gcloud import FakeGcloud
from tests.fakes.project import (
    BUCKET,
    PROJECT,
    mutating_calls,
    register_mutations,
    register_project,
    register_zone_and_quotas
)

runner = CliRunner()

def invoke( fake: FakeGcloud, tmp_path: Path, *args: str, input: str | None = None ):
    env = { "ML_RUNNER_CONFIG": str( tmp_path / "config.toml" ) }
    register_zone_and_quotas( fake )

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
    result = invoke(
        fake,
        tmp_path,
        "--project",
        PROJECT,
        "--region",
        "us-central1",
        "--zone",
        "us-central1-a",
        "--bucket",
        "my-own",
        "--yes"
    )
    assert result.exit_code == 0, result.output
    create = next( call for call in mutating_calls( fake ) if call[ 1:4 ] == [ "storage", "buckets", "create" ] )
    assert "gs://my-own" in create
    assert "--location=us-central1" in create

def test_init_writes_the_config_and_prints_its_path( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fresh_project( fake )
    result = invoke( fake, tmp_path, "--project", PROJECT, "--yes" )
    assert result.exit_code == 0, result.output
    path = tmp_path / "config.toml"
    assert str( path ) in result.output
    config = load_config( path )
    assert ( config.project, config.region, config.zone, config.bucket ) == (
        PROJECT,
        "asia-southeast1",
        "asia-southeast1-a",
        BUCKET
    )

def test_init_dry_run_writes_no_config( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fresh_project( fake )
    result = invoke( fake, tmp_path, "--project", PROJECT, "--dry-run" )
    assert result.exit_code == 0, result.output
    assert not ( tmp_path / "config.toml" ).exists()
    assert "would write" in result.output.lower()

def test_init_declined_writes_no_config( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fresh_project( fake )
    invoke( fake, tmp_path, "--project", PROJECT, input = "n\n" )
    assert not ( tmp_path / "config.toml" ).exists()

def test_init_honours_the_zone_flag( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    register_project( fake )
    result = invoke( fake, tmp_path, "--project", PROJECT, "--zone", "asia-southeast1-c" )
    assert result.exit_code == 0, result.output
    assert load_config( tmp_path / "config.toml" ).zone == "asia-southeast1-c"

def test_init_again_keeps_the_users_defaults_and_zone( tmp_path: Path ) -> None:
    settings = InitSettings( PROJECT, "asia-southeast1", BUCKET )
    existing = build_config( settings, "asia-southeast1-c", None ).model_copy(
        update = { "defaults": Defaults( gpu = "l4" ) }
    )
    save_config( tmp_path / "config.toml", existing )
    fake = FakeGcloud()
    register_project( fake )
    result = invoke( fake, tmp_path, "--project", PROJECT )
    assert result.exit_code == 0, result.output
    config = load_config( tmp_path / "config.toml" )
    assert config.defaults.gpu == "l4"
    assert config.zone == "asia-southeast1-c"

def test_init_prints_the_quota_summary_and_the_budgets_link( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    register_project( fake )
    result = invoke( fake, tmp_path, "--project", PROJECT )
    assert "T4: 0" in result.output
    assert "GPUS_ALL_REGIONS" in result.output
    assert "https://console.cloud.google.com/iam-admin/quotas" in result.output
    assert "https://console.cloud.google.com/billing/budgets" in result.output

def test_init_still_succeeds_when_quotas_cannot_be_read( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fake.register( [ "compute", "regions", "describe" ], stderr = "ERROR: PERMISSION_DENIED 403", returncode = 1 )
    register_project( fake )
    result = invoke( fake, tmp_path, "--project", PROJECT )
    assert result.exit_code == 0, result.output
    assert "could not read quotas" in result.output.lower()
    assert ( tmp_path / "config.toml" ).exists()

def test_init_with_no_gpu_zone_fails_before_changing_anything( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fake.register_json( [ "compute", "accelerator-types", "list" ], [] )
    fresh_project( fake )
    result = invoke( fake, tmp_path, "--project", PROJECT, "--yes" )
    assert result.exit_code == 2
    assert "--region" in result.output
    assert mutating_calls( fake ) == []
