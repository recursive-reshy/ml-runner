import os
import subprocess
from pathlib import Path

import pytest

from vmsupport import FakeBin

HOOKS = Path( __file__ ).parents[ 2 ] / "src" / "ml_runner" / "vm"

# Runs the command that follows its first argument, like coreutils' timeout.
FAKE_TIMEOUT = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_CALLS/timeout.log"
shift
exec "$@"
"""

# Writes FAKE_AGENT_SOURCE to the file after `-o`, like curl would.
FAKE_CURL = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_CALLS/curl.log"
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "-o" ]]; then
        cp "$FAKE_AGENT_SOURCE" "$2"
    fi
    shift
done
"""

@pytest.fixture
def hooks( fake_bin: FakeBin, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch ):
    fake_bin.add( "curl", FAKE_CURL )
    fake_bin.add( "timeout", FAKE_TIMEOUT )
    fake_bin.add_logging( "python3" )
    source = tmp_path / "agent-source.py"
    source.write_text( "# the agent\n" )
    monkeypatch.setenv( "FAKE_AGENT_SOURCE", str( source ) )
    monkeypatch.setenv( "MLR_LOG", str( tmp_path / "mlr-agent.log" ) )

    def run( script: str ) -> subprocess.CompletedProcess[ str ]:
        argv = [ "bash", str( HOOKS / script ) ]

        return subprocess.run( argv, capture_output = True, text = True, env = os.environ.copy() )

    return run

@pytest.mark.parametrize( "script", [ "startup.sh", "shutdown.sh" ] )
def test_the_hooks_are_valid_bash_with_strict_mode( script: str ) -> None:
    assert subprocess.run( [ "bash", "-n", str( HOOKS / script ) ] ).returncode == 0
    assert "set -euo pipefail" in ( HOOKS / script ).read_text()

def test_startup_fetches_the_agent_and_starts_it( hooks, fake_bin: FakeBin, home: Path ) -> None:
    result = hooks( "startup.sh" )

    assert result.returncode == 0
    assert ( home / "agent.py" ).read_text() == "# the agent\n"
    assert "Metadata-Flavor: Google" in fake_bin.lines( "curl" )[ 0 ]
    assert "instance/attributes/mlr-agent" in fake_bin.lines( "curl" )[ 0 ]
    assert fake_bin.lines( "python3" ) == [ f"{home / 'agent.py'} start" ]
    assert ( home / "started" ).exists()

def test_startup_output_goes_to_the_log_file( hooks, fake_bin: FakeBin, tmp_path: Path ) -> None:
    fake_bin.add( "python3", "#!/usr/bin/env bash\necho agent-output\n" )

    hooks( "startup.sh" )

    assert "agent-output" in ( tmp_path / "mlr-agent.log" ).read_text()

def test_a_second_start_after_a_reboot_finalizes_instead( hooks, fake_bin: FakeBin, home: Path ) -> None:
    hooks( "startup.sh" )

    hooks( "startup.sh" )

    assert fake_bin.lines( "python3" ) == [
        f"{home / 'agent.py'} start",
        f"{home / 'agent.py'} finalize --reason VM restarted"
    ]

def test_a_failing_agent_makes_the_start_hook_fail( hooks, fake_bin: FakeBin, monkeypatch: pytest.MonkeyPatch ) -> None:
    monkeypatch.setenv( "FAKE_PYTHON3_EXIT", "1" )

    assert hooks( "startup.sh" ).returncode != 0

def test_shutdown_finalizes_within_the_time_limit( hooks, fake_bin: FakeBin, home: Path ) -> None:
    ( home / "agent.py" ).write_text( "# the agent\n" )

    result = hooks( "shutdown.sh" )

    assert result.returncode == 0
    assert fake_bin.lines( "timeout" ) == [ f"25 python3 {home / 'agent.py'} finalize" ]
    assert fake_bin.lines( "python3" ) == [ f"{home / 'agent.py'} finalize" ]

def test_shutdown_without_an_agent_does_nothing( hooks, fake_bin: FakeBin ) -> None:
    assert hooks( "shutdown.sh" ).returncode == 0
    assert fake_bin.lines( "python3" ) == []

def test_shutdown_never_fails_the_shutdown(
    hooks,
    fake_bin: FakeBin,
    home: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    ( home / "agent.py" ).write_text( "# the agent\n" )
    monkeypatch.setenv( "FAKE_PYTHON3_EXIT", "1" )

    assert hooks( "shutdown.sh" ).returncode == 0
