import json
import re
from pathlib import Path

import pytest

from ml_runner.gcloud import Gcloud, classify_error
from tests.fakes.fake_gcloud import FakeGcloud

FIXTURES = Path( __file__ ).parent / "fakes" / "gcloud"
STDERR = FIXTURES / "stderr"
PLACEHOLDER_PROJECT = "my-project"
PLACEHOLDER_NUMBER = "123456789012"
PLACEHOLDER_DOMAIN = "example.com"
EMAIL = re.compile( r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)" )
PROJECT_NUMBER = re.compile( r"\b\d{12}\b" )

def load( name: str ) -> object:
    return json.loads( ( FIXTURES / f"{name}.json" ).read_text() )

def fixture_files() -> list[ Path ]:
    return sorted( [ *FIXTURES.glob( "*.json" ), *STDERR.glob( "*.txt" ) ] )

@pytest.mark.parametrize( "path", fixture_files(), ids = lambda path: path.name )
def test_fixture_is_anonymised( path: Path ) -> None:
    text = path.read_text()
    for match in EMAIL.finditer( text ):
        domain = match.group( 1 )
        # service accounts are fine when they belong to the placeholder project
        assert domain == PLACEHOLDER_DOMAIN or domain.endswith( "gserviceaccount.com" ), match.group( 0 )
    assert set( PROJECT_NUMBER.findall( text ) ) <= { PLACEHOLDER_NUMBER }

@pytest.mark.parametrize( "path", sorted( FIXTURES.glob( "*.json" ) ), ids = lambda path: path.name )
def test_json_fixture_parses( path: Path ) -> None:
    assert json.loads( path.read_text() ) is not None

def test_version_fixture_has_the_sdk_version_key() -> None:
    fake = FakeGcloud()
    fake.register_json( [ "version" ], load( "version" ) )
    major, minor, patch = Gcloud( runner = fake ).version()
    assert major >= 500
    assert ( minor, patch ) == ( 0, 0 )

@pytest.mark.parametrize(
    "name",
    [ "not_found", "not_found_bucket", "not_found_service_account", "not_found_role" ]
)
def test_real_not_found_stderr_is_classified( name: str ) -> None:
    assert classify_error( 1, ( STDERR / f"{name}.txt" ).read_text() ) == "not_found"

def test_quota_fixtures_have_the_metrics_doctor_reads() -> None:
    region = load( "region_describe" )
    project = load( "project_info" )
    assert isinstance( region, dict ) and isinstance( project, dict )
    region_metrics = { quota[ "metric" ]: quota[ "limit" ] for quota in region[ "quotas" ] }
    project_metrics = { quota[ "metric" ]: quota[ "limit" ] for quota in project[ "quotas" ] }
    for metric in ( "NVIDIA_T4_GPUS", "PREEMPTIBLE_NVIDIA_T4_GPUS", "NVIDIA_L4_GPUS", "PREEMPTIBLE_NVIDIA_L4_GPUS" ):
        assert region_metrics[ metric ] == 1.0
    assert project_metrics[ "GPUS_ALL_REGIONS" ] == 0.0
