import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import pytest

from vmsupport import FAKE_GCLOUD, FAKE_GIT, JOB, FakeBin

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
    monkeypatch.setenv( "PATH", f"{fake.bin}{os.pathsep}{os.environ[ 'PATH' ]}" )
    monkeypatch.setenv( "FAKE_CALLS", str( fake.calls ) )

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
