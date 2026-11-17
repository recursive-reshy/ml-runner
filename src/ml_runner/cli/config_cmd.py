"""`ml-runner config show` and `config set`."""

from typing import Annotated

import tomli_w
import typer

from ml_runner.cli.state import get_config, get_config_path
from ml_runner.config import save_config, set_value

config_app = typer.Typer(help="Show or change the local config.", no_args_is_help=True)


@config_app.command("show")
def show(ctx: typer.Context) -> None:
    """Print the config file path and its values."""
    config = get_config(ctx)
    typer.echo(f"# {get_config_path(ctx)}")
    typer.echo(tomli_w.dumps(config.model_dump(mode="json")).rstrip())


@config_app.command("set")
def set_(
    ctx: typer.Context,
    key: Annotated[str, typer.Argument(help="`zone` or `defaults.<name>`.")],
    value: Annotated[str, typer.Argument(help="The new value.")],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the change without saving it.")
    ] = False,
) -> None:
    """Change `zone` or a `defaults.*` value."""
    updated = set_value(get_config(ctx), key, value)
    if dry_run:
        typer.echo(f"Dry run: would set {key} = {value} in {get_config_path(ctx)}")
        return
    save_config(get_config_path(ctx), updated)
    typer.echo(f"Set {key} = {value}")
