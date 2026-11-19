import json
import os
import subprocess
import sys
from pathlib import Path

from tests.fakes.project import mutating_calls
from tests.test_launch_cli import invoke
from tests.fakes.launch import make_launch_fake

EXAMPLE = Path( __file__ ).parents[ 1 ] / "examples" / "hello-gpu"

def train( tmp_path: Path, *args: str ) -> subprocess.CompletedProcess[ str ]:
    env = {
        **os.environ,
        "OUTPUT_DIR": str( tmp_path / "out" ),
        "CHECKPOINT_DIR": str( tmp_path / "checkpoints" ),
        "RUN_ID": "run-1",
        "PATH": "/usr/bin:/bin"
    }

    return subprocess.run(
        [ sys.executable, str( EXAMPLE / "train.py" ), *args ],
        capture_output = True,
        text = True,
        env = env,
        cwd = tmp_path
    )

def test_the_example_declares_its_dependencies_with_a_pyproject() -> None:
    text = ( EXAMPLE / "pyproject.toml" ).read_text()

    assert 'name = "hello-gpu"' in text
    assert "requires-python" in text

def test_a_cpu_run_writes_a_result_and_one_checkpoint_per_step( tmp_path: Path ) -> None:
    result = train( tmp_path, "--steps", "3", "--pause", "0" )

    assert result.returncode == 0, result.stderr
    summary = json.loads( ( tmp_path / "out" / "result.json" ).read_text() )
    assert summary[ "run_id" ] == "run-1"
    assert summary[ "steps" ] == 3
    assert summary[ "gpu" ] is None
    assert sorted( path.name for path in ( tmp_path / "checkpoints" ).iterdir() ) == [
        "step-1.txt",
        "step-2.txt",
        "step-3.txt"
    ]

def test_the_gpu_check_is_printed_to_the_log( tmp_path: Path ) -> None:
    result = train( tmp_path, "--steps", "1", "--pause", "0" )

    assert "GPU:" in result.stdout

def test_without_the_contract_variables_it_writes_next_to_itself( tmp_path: Path ) -> None:
    env = { "PATH": "/usr/bin:/bin" }

    result = subprocess.run(
        [ sys.executable, str( EXAMPLE / "train.py" ), "--steps", "1", "--pause", "0" ],
        capture_output = True,
        text = True,
        env = env,
        cwd = tmp_path
    )

    assert result.returncode == 0, result.stderr
    assert ( tmp_path / "out" / "result.json" ).exists()

def test_launch_dry_run_snapshots_the_example( tmp_path: Path ) -> None:
    fake = make_launch_fake()

    result = invoke( tmp_path, "--dry-run", "--src", str( EXAMPLE ), "--", "python", "train.py", fake = fake )

    assert result.exit_code == 0, result.output
    assert "snapshot of hello-gpu" in result.output
    assert mutating_calls( fake ) == []
