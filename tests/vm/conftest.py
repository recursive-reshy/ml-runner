import importlib.util
import json
import os
from typing import Any
from pathlib import Path
from types import ModuleType

import pytest

from vmsupport import BUCKET, FAKE_GCLOUD, FAKE_GIT, JOB, FakeBin, launcher_record

AGENT = Path( __file__ ).parents[ 2 ] / "src" / "ml_runner" / "vm" / "agent.py"

@pytest.fixture
def agent() -> ModuleType:
    spec = importlib.util.spec_from_file_location( "agent", AGENT )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )

    return module

@pytest.fixture
def home( tmp_path: Path, monkeypatch: pytest.MonkeyPatch ) -> Path:
    path = tmp_path / "mlr"
    path.mkdir()
    monkeypatch.setenv( "MLR_HOME", str( path ) )

    return path

@pytest.fixture
def fake_bin( tmp_path: Path, monkeypatch: pytest.MonkeyPatch ) -> FakeBin:
    fake = FakeBin( _make( tmp_path / "fakes" ) )
    fake.add( "gcloud", FAKE_GCLOUD )
    fake.add( "git", FAKE_GIT )
    # Only the fakes and the system tools: a developer's real uv or nvidia-smi must never be found.
    monkeypatch.setenv( "PATH", os.pathsep.join( [ str( fake.bin ), "/usr/bin", "/bin" ] ) )
    monkeypatch.setenv( "FAKE_CALLS", str( fake.calls ) )
    monkeypatch.setenv( "FAKE_BUCKET", str( fake.bucket ) )

    return fake

def _make( path: Path ) -> Path:
    path.mkdir()

    return path

@pytest.fixture
def metadata_file( tmp_path: Path, monkeypatch: pytest.MonkeyPatch ) -> Path:
    path = tmp_path / "metadata.json"
    path.write_text( json.dumps( {
        "mlr-job": json.dumps( JOB ),
        "name": "mlr-20261120-143005-7c21",
        "zone": "projects/123/zones/asia-southeast1-a"
    } ) )
    monkeypatch.setenv( "MLR_METADATA_FILE", str( path ) )

    return path

@pytest.fixture
def job_runner( agent: ModuleType, home: Path, fake_bin: FakeBin, tmp_path: Path, monkeypatch: pytest.MonkeyPatch ):
    for variable, value in {
        "MLR_LOG_SYNC_SECONDS": "30",
        "MLR_DATA_SYNC_SECONDS": "30",
        "MLR_RETRY_SECONDS": "0",
        "MLR_KILL_GRACE_SECONDS": "5"
    }.items():
        monkeypatch.setenv( variable, value )

    source_dir = tmp_path / "source"
    source_dir.mkdir()

    def run( runs: list[ dict[ str, Any ] ], **overrides: Any ) -> Any:
        job = agent.parse_job( json.dumps( { **JOB, "runs": runs, **overrides } ) )
        for run in runs:
            record = launcher_record( run[ "run_id" ], run[ "experiment" ] )
            ( tmp_path / "record.json" ).write_text( json.dumps( record ) )
            fake_bin.put( f"{BUCKET}/runs/exp/{run[ 'run_id' ]}/run.json", tmp_path / "record.json" )
        metadata = agent.Metadata( job = job, instance = "mlr-vm", zone = "asia-southeast1-a" )

        return agent.run_job( metadata, home, source_dir, agent.Environment( prefix = [], path_prepend = None ) )

    return run

