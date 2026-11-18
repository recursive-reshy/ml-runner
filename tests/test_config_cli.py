import tomllib
from pathlib import Path

import typer
from typer.testing import CliRunner

from ml_runner.cli import MlRunnerGroup, app, root
from ml_runner.cli.state import get_config
from ml_runner.config import load_config, save_config
from tests.test_config import make_config

runner = CliRunner()

def _env( path: Path ) -> dict[ str, str ]:
    return { "ML_RUNNER_CONFIG": str( path ) }

def test_config_show_prints_path_and_values( tmp_path: Path ) -> None:
    path = tmp_path / "c.toml"
    save_config( path, make_config() )
    result = runner.invoke( app, [ "config", "show" ], env = _env( path ) )
    assert result.exit_code == 0
    assert str( path ) in result.output
    assert 'project = "my-project"' in result.output
    assert "gpu = " in result.output

def test_config_flag_overrides_env( tmp_path: Path ) -> None:
    flag_path, env_path = tmp_path / "flag.toml", tmp_path / "env.toml"
    save_config( flag_path, make_config( project = "from-flag" ) )
    save_config( env_path, make_config( project = "from-env" ) )
    result = runner.invoke( app, [ "--config", str( flag_path ), "config", "show" ], env = _env( env_path ) )
    assert "from-flag" in result.output

def test_config_set_writes_the_file( tmp_path: Path ) -> None:
    path = tmp_path / "c.toml"
    save_config( path, make_config() )
    result = runner.invoke( app, [ "config", "set", "defaults.gpu", "l4" ], env = _env( path ) )
    assert result.exit_code == 0
    assert load_config( path ).defaults.gpu == "l4"

def test_config_set_dry_run_changes_nothing( tmp_path: Path ) -> None:
    path = tmp_path / "c.toml"
    save_config( path, make_config() )
    before = path.read_text()
    result = runner.invoke( app, [ "config", "set", "--dry-run", "defaults.gpu", "l4" ], env = _env( path ) )
    assert result.exit_code == 0
    assert "defaults.gpu" in result.output
    assert path.read_text() == before

def test_config_set_bad_value_exits_2_and_leaves_file( tmp_path: Path ) -> None:
    path = tmp_path / "c.toml"
    save_config( path, make_config() )
    before = path.read_text()
    result = runner.invoke( app, [ "config", "set", "defaults.gpu", "a100" ], env = _env( path ) )
    assert result.exit_code == 2
    assert path.read_text() == before
    assert tomllib.loads( before )[ "defaults" ][ "gpu" ] == "t4"

def test_missing_config_exits_2_with_init_hint( tmp_path: Path ) -> None:
    result = runner.invoke( app, [ "config", "show" ], env = _env( tmp_path / "missing.toml" ) )
    assert result.exit_code == 2
    assert "ml-runner init" in result.output

def test_commands_that_need_config_get_it_from_get_config( tmp_path: Path ) -> None:
    path = tmp_path / "c.toml"
    save_config( path, make_config() )
    probe = typer.Typer( cls = MlRunnerGroup )
    probe.callback()( root )

    @probe.command()
    def needs( ctx: typer.Context ) -> None:
        typer.echo( get_config( ctx ).project )

    @probe.command()
    def exempt() -> None:
        typer.echo( "fine without config" )

    ok = runner.invoke( probe, [ "needs" ], env = _env( path ) )
    assert ok.exit_code == 0 and "my-project" in ok.output

    missing = _env( tmp_path / "none.toml" )
    assert runner.invoke( probe, [ "needs" ], env = missing ).exit_code == 2
    # A command that never asks for config (init, doctor, --version) still works.
    assert runner.invoke( probe, [ "exempt" ], env = missing ).exit_code == 0
