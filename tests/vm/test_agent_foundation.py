import json
import tarfile
from pathlib import Path
from types import ModuleType

import pytest

from vmsupport import JOB, FakeBin

def test_reads_job_instance_and_zone_from_the_metadata_file( agent: ModuleType, metadata_file: Path ) -> None:
    metadata = agent.read_metadata()

    assert metadata.instance == "mlr-20261120-143005-7c21"
    assert metadata.zone == "asia-southeast1-a"
    assert metadata.job.job_id == "20261120-143005-7c21"
    assert metadata.job.bucket == "my-bucket"
    assert metadata.job.max_hours == 6
    assert metadata.job.stop_on_failure is True
    assert metadata.job.gpu == "t4"
    assert metadata.job.env == { "KEY": "VALUE" }
    assert metadata.job.source.kind == "snapshot"
    assert metadata.job.source.sha256 == "a" * 64
    assert metadata.job.runs[ 0 ].run_id == "R1"
    assert metadata.job.runs[ 0 ].experiment == "exp"
    assert metadata.job.runs[ 0 ].command == [ "python", "train.py" ]

def test_a_missing_metadata_key_is_a_clear_error( agent: ModuleType, metadata_file: Path ) -> None:
    metadata_file.write_text( json.dumps( { "name": "x", "zone": "z" } ) )

    with pytest.raises( agent.AgentError, match = "mlr-job" ):
        agent.read_metadata()

def test_without_the_file_the_metadata_server_is_asked_with_the_flavor_header(
    agent: ModuleType,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv( "MLR_METADATA_FILE", raising = False )

    request = agent.metadata_request( "instance/attributes/mlr-job" )

    assert request.full_url == "http://metadata.google.internal/computeMetadata/v1/instance/attributes/mlr-job"
    assert request.get_header( "Metadata-flavor" ) == "Google"

def test_a_git_job_parses_its_repo_and_ref( agent: ModuleType ) -> None:
    job = agent.parse_job( json.dumps( {
        **JOB,
        "source": { "kind": "git", "repo": "https://github.com/x/y", "ref": "main" }
    } ) )

    assert job.source.kind == "git"
    assert job.source.repo == "https://github.com/x/y"
    assert job.source.ref == "main"

def test_home_defaults_to_opt_mlr( agent: ModuleType, monkeypatch: pytest.MonkeyPatch ) -> None:
    monkeypatch.delenv( "MLR_HOME", raising = False )

    assert agent.home_dir() == Path( "/opt/mlr" )

def test_state_round_trips_and_leaves_no_temp_file( agent: ModuleType, home: Path ) -> None:
    state = agent.State( job_id = "J", phase = "installing", current_run = "R1" )

    agent.write_state( home, state )

    assert agent.read_state( home ) == state
    assert [ path.name for path in home.iterdir() ] == [ "state.json" ]

def test_a_missing_state_file_reads_as_none( agent: ModuleType, home: Path ) -> None:
    assert agent.read_state( home ) is None

def make_archive( directory: Path ) -> Path:
    project = directory / "project"
    project.mkdir()
    ( project / "train.py" ).write_text( "print( 1 )\n" )
    archive = directory / "snapshot.tar.gz"
    with tarfile.open( archive, "w:gz" ) as bundle:
        bundle.add( project / "train.py", arcname = "train.py" )

    return archive

def test_snapshot_is_copied_from_the_bucket_and_extracted(
    agent: ModuleType,
    home: Path,
    fake_bin: FakeBin,
    tmp_path: Path,
    metadata_file: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv( "FAKE_CP_SOURCE", str( make_archive( tmp_path ) ) )
    job = agent.read_metadata().job

    source_dir = agent.fetch_source( job, home )

    assert source_dir == home / "src"
    assert ( source_dir / "train.py" ).read_text() == "print( 1 )\n"
    uri = f"gs://my-bucket/sources/{'a' * 64}.tar.gz"
    assert fake_bin.lines( "gcloud" ) == [ f"storage cp {uri} {home / 'source.tar.gz'}" ]

def test_git_source_is_cloned_and_checked_out( agent: ModuleType, home: Path, fake_bin: FakeBin ) -> None:
    job = agent.parse_job( json.dumps( {
        **JOB,
        "source": { "kind": "git", "repo": "https://github.com/x/y", "ref": "v1" }
    } ) )

    source_dir = agent.fetch_source( job, home )

    assert source_dir == home / "src"
    assert fake_bin.lines( "git" ) == [ f"clone https://github.com/x/y {source_dir}", f"-C {source_dir} checkout v1" ]

def test_a_failed_fetch_names_the_step(
    agent: ModuleType,
    home: Path,
    fake_bin: FakeBin,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv( "FAKE_GCLOUD_EXIT", "1" )
    job = agent.parse_job( json.dumps( JOB ) )

    with pytest.raises( agent.AgentError, match = "fetch" ):
        agent.fetch_source( job, home )

def test_agent_paths_follow_the_bucket_layout( agent: ModuleType ) -> None:
    assert agent.source_uri( "b", "s" ) == "gs://b/sources/s.tar.gz"
    assert agent.run_dir_uri( "b", "e", "r" ) == "gs://b/runs/e/r"

def test_start_records_the_fetching_state_and_fetches(
    agent: ModuleType,
    home: Path,
    fake_bin: FakeBin,
    tmp_path: Path,
    metadata_file: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv( "FAKE_CP_SOURCE", str( make_archive( tmp_path ) ) )

    assert agent.main( [ "agent.py", "start" ] ) == 0
    assert agent.read_state( home ) == agent.State( job_id = "20261120-143005-7c21", phase = "fetching" )
    assert ( home / "src" / "train.py" ).exists()

def test_a_failed_start_exits_one(
    agent: ModuleType,
    home: Path,
    fake_bin: FakeBin,
    metadata_file: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv( "FAKE_GCLOUD_EXIT", "1" )

    assert agent.main( [ "agent.py", "start" ] ) == 1
