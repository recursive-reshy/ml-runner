import subprocess
import sys

import typer
from typer.testing import CliRunner

from ml_runner import __version__
from ml_runner.cli import MlRunnerGroup, app, root
from ml_runner.errors import EXIT_FAILED, EXIT_OK, EXIT_USAGE, MlRunnerError, UsageError

runner = CliRunner()

def test_exit_codes_are_0_1_2() -> None:
    assert ( EXIT_OK, EXIT_FAILED, EXIT_USAGE ) == ( 0, 1, 2 )

def test_help_lists_global_options() -> None:
    result = runner.invoke( app, [ "--help" ] )
    assert result.exit_code == 0
    assert "ml-runner" in result.output
    for option in ( "--version", "--verbose", "--config" ):
        assert option in result.output

def test_version_prints_name_and_version() -> None:
    result = runner.invoke( app, [ "--version" ] )
    assert result.exit_code == 0
    assert result.output.strip() == f"ml-runner {__version__}"

def test_unknown_option_is_a_usage_error() -> None:
    result = runner.invoke( app, [ "--nope" ] )
    assert result.exit_code == 2

# A private app with the real group class and root callback.
def _probe_app() -> typer.Typer:
    probe = typer.Typer( cls = MlRunnerGroup )
    probe.callback()( root )

    @probe.command()
    def ok( ctx: typer.Context ) -> None:
        typer.echo( f"verbose={ctx.obj.verbose} config={ctx.obj.config_path}" )

    @probe.command()
    def fail() -> None:
        raise MlRunnerError( "the operation failed", detail = "raw stderr here" )

    @probe.command()
    def usage() -> None:
        raise UsageError( "bad config" )

    return probe

def test_global_options_are_stored_in_state() -> None:
    result = runner.invoke( _probe_app(), [ "--verbose", "--config", "/tmp/c.toml", "ok" ] )
    assert result.exit_code == 0
    assert "verbose=True config=/tmp/c.toml" in result.output

def test_failed_operation_exits_1_and_hides_detail_by_default() -> None:
    result = runner.invoke( _probe_app(), [ "fail" ] )
    assert result.exit_code == 1
    assert "the operation failed" in result.output
    assert "raw stderr here" not in result.output

def test_verbose_shows_detail() -> None:
    result = runner.invoke( _probe_app(), [ "--verbose", "fail" ] )
    assert result.exit_code == 1
    assert "raw stderr here" in result.output

def test_usage_error_exits_2() -> None:
    result = runner.invoke( _probe_app(), [ "usage" ] )
    assert result.exit_code == 2
    assert "bad config" in result.output

def test_python_dash_m_works() -> None:
    done = subprocess.run(
        [ sys.executable, "-m", "ml_runner", "--version" ],
        capture_output = True,
        text = True,
        check = False
    )
    assert done.returncode == 0
    assert done.stdout.strip() == f"ml-runner {__version__}"
