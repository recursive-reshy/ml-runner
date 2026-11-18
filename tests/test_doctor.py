from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pytest
from typer.testing import CliRunner

from ml_runner.checks import BUDGETS_URL, GLOBAL_METRIC, QUOTA_URL, Check, run_checks
from ml_runner.cli import app
from ml_runner.cli.state import State
from ml_runner.config import Defaults, save_config
from ml_runner.gcloud import Completed
from ml_runner.setup import InitSettings, build_config
from tests.fakes.fake_gcloud import FakeGcloud
from tests.fakes.project import (
    BUCKET,
    PROJECT,
    fixture,
    mutating_calls,
    register_healthy,
    stderr
)

ZONE = "asia-southeast1-b"
SETTINGS = InitSettings( PROJECT, "asia-southeast1", BUCKET )

def write_config( tmp_path: Path, defaults: Defaults | None = None ) -> Path:
    path = tmp_path / "config.toml"
    config = build_config( SETTINGS, ZONE, None )
    save_config( path, config.model_copy( update = { "defaults": defaults or Defaults() } ) )

    return path

def check_all( fake: FakeGcloud, config_file: Path ) -> dict[ str, Check ]:
    return { check.title: check for check in run_checks( fake, config_file ) }

def statuses( checks: dict[ str, Check ] ) -> dict[ str, str ]:
    return { title: check.status for title, check in checks.items() }

def test_a_healthy_project_passes_everything_except_the_budget_reminder( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    register_healthy( fake )
    checks = check_all( fake, write_config( tmp_path ) )
    assert [ title for title, status in statuses( checks ).items() if status != "pass" ] == [ "Budget" ]
    assert checks[ "Budget" ].status == "warn"
    assert BUDGETS_URL in checks[ "Budget" ].fix
    assert mutating_calls( fake ) == []

def test_a_missing_gcloud_is_one_failure_with_an_install_hint( tmp_path: Path ) -> None:
    def runner( argv: Sequence[ str ], *, timeout: float ) -> Completed:
        raise FileNotFoundError( "gcloud" )

    checks = run_checks( runner, write_config( tmp_path ) )
    assert [ ( check.title, check.status ) for check in checks ] == [ ( "gcloud", "fail" ) ]
    assert "Install" in checks[ 0 ].fix

def test_an_old_gcloud_is_a_warning( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fake.register_json( [ "version" ], { "Google Cloud SDK": "400.0.0" } )
    register_healthy( fake )
    checks = check_all( fake, write_config( tmp_path ) )
    assert checks[ "gcloud" ].status == "warn"
    assert "590" in checks[ "gcloud" ].detail

def test_not_signed_in_fails_and_skips_the_rest( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fake.register_json( [ "auth", "list" ], [] )
    register_healthy( fake )
    checks = check_all( fake, write_config( tmp_path ) )
    assert checks[ "Signed in" ].status == "fail"
    assert "gcloud auth login" in checks[ "Signed in" ].fix
    assert "Project" not in checks
    assert checks[ "Other checks" ].status == "warn"

def test_a_missing_config_fails_with_the_init_hint( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    register_healthy( fake )
    checks = check_all( fake, tmp_path / "missing.toml" )
    assert checks[ "Config" ].status == "fail"
    assert "ml-runner init" in checks[ "Config" ].fix
    assert "Project" not in checks

def test_an_unknown_project_fails( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fake.register( [ "projects", "describe" ], stderr = stderr( "not_found" ), returncode = 1 )
    register_healthy( fake )
    assert check_all( fake, write_config( tmp_path ) )[ "Project" ].status == "fail"

def test_each_missing_api_fails_on_its_own( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fake.register_json( [ "services", "list" ], [] )
    register_healthy( fake )
    checks = check_all( fake, write_config( tmp_path ) )
    assert checks[ "API compute.googleapis.com" ].status == "fail"
    assert checks[ "API storage.googleapis.com" ].status == "fail"
    assert "ml-runner init" in checks[ "API compute.googleapis.com" ].fix

def test_a_missing_bucket_fails_the_bucket_and_its_binding( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fake.register( [ "storage", "buckets", "describe" ], stderr = stderr( "not_found_bucket" ), returncode = 1 )
    fake.register( [ "storage", "buckets", "get-iam-policy" ], stderr = stderr( "not_found_bucket" ), returncode = 1 )
    register_healthy( fake )
    checks = check_all( fake, write_config( tmp_path ) )
    assert checks[ "Bucket" ].status == "fail"
    assert checks[ "Bucket access" ].status == "fail"

def test_a_missing_service_account_fails( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fake.register(
        [ "iam", "service-accounts", "describe" ],
        stderr = stderr( "not_found_service_account" ),
        returncode = 1
    )
    register_healthy( fake )
    assert check_all( fake, write_config( tmp_path ) )[ "Service account" ].status == "fail"

def test_a_missing_self_delete_binding_fails( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fake.register_json( [ "projects", "get-iam-policy" ], fixture( "project_policy" ) )
    register_healthy( fake )
    checks = check_all( fake, write_config( tmp_path ) )
    assert checks[ "Self-delete role" ].status == "fail"
    assert "ml-runner init" in checks[ "Self-delete role" ].fix

def test_a_missing_default_network_is_only_a_warning( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fake.register( [ "compute", "networks", "describe" ], stderr = stderr( "not_found" ), returncode = 1 )
    register_healthy( fake )
    assert check_all( fake, write_config( tmp_path ) )[ "Network" ].status == "warn"

def test_a_global_gpu_quota_of_zero_warns_and_names_the_global_quota( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fake.register_json( [ "compute", "project-info", "describe" ], fixture( "project_info" ) )
    register_healthy( fake )
    quota = check_all( fake, write_config( tmp_path ) )[ "GPU quota" ]
    assert quota.status == "warn"
    assert GLOBAL_METRIC in quota.detail
    assert QUOTA_URL in quota.fix

def test_the_quota_checked_follows_the_default_gpu_and_spot( tmp_path: Path ) -> None:
    region = fixture( "region_describe" )
    assert isinstance( region, dict )
    spot_zero = {
        "quotas": [
            { **quota, "limit": 0.0 } if quota[ "metric" ] == "PREEMPTIBLE_NVIDIA_L4_GPUS" else quota
            for quota in region[ "quotas" ]
        ]
    }
    fake = FakeGcloud()
    fake.register_json( [ "compute", "regions", "describe" ], spot_zero )
    register_healthy( fake )
    on_demand = check_all( fake, write_config( tmp_path, Defaults( gpu = "l4" ) ) )[ "GPU quota" ]
    spot = check_all( fake, write_config( tmp_path, Defaults( gpu = "l4", spot = True ) ) )[ "GPU quota" ]
    assert on_demand.status == "pass"
    assert spot.status == "warn"
    assert "PREEMPTIBLE_NVIDIA_L4_GPUS" in spot.detail

def test_no_gpu_needs_no_gpu_quota( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fake.register_json( [ "compute", "project-info", "describe" ], fixture( "project_info" ) )
    register_healthy( fake )
    assert check_all( fake, write_config( tmp_path, Defaults( gpu = "none" ) ) )[ "GPU quota" ].status == "pass"

def test_unreadable_quotas_are_a_warning_not_a_failure( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fake.register( [ "compute", "regions", "describe" ], stderr = stderr( "permission_denied" ), returncode = 1 )
    register_healthy( fake )
    quota = check_all( fake, write_config( tmp_path ) )[ "GPU quota" ]
    assert quota.status == "warn"
    assert "Could not read" in quota.detail

runner = CliRunner()

def invoke( fake: FakeGcloud, config_file: Path ):
    return runner.invoke(
        app,
        [ "doctor" ],
        obj = State( runner = fake ),
        env = { "ML_RUNNER_CONFIG": str( config_file ) }
    )

def test_doctor_exits_0_when_there_are_only_warnings( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    register_healthy( fake )
    result = invoke( fake, write_config( tmp_path ) )
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output and "WARN" in result.output
    assert BUDGETS_URL in result.output

def test_doctor_exits_1_when_anything_fails_and_prints_the_fix( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    fake.register_json( [ "services", "list" ], [] )
    register_healthy( fake )
    result = invoke( fake, write_config( tmp_path ) )
    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "fix: " in result.output

def test_doctor_without_a_config_still_runs_and_exits_1( tmp_path: Path ) -> None:
    fake = FakeGcloud()
    register_healthy( fake )
    result = invoke( fake, tmp_path / "missing.toml" )
    assert result.exit_code == 1
    assert "ml-runner init" in result.output

@pytest.mark.parametrize( "gpu", [ "t4", "l4", "none" ] )
def test_doctor_never_makes_a_mutating_call( tmp_path: Path, gpu: Literal[ "t4", "l4", "none" ] ) -> None:
    fake = FakeGcloud()
    register_healthy( fake )
    invoke( fake, write_config( tmp_path, Defaults( gpu = gpu ) ) )
    assert mutating_calls( fake ) == []
