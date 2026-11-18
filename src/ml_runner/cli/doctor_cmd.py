import typer

from ml_runner.checks import Check, run_checks
from ml_runner.cli.state import State, get_config_path
from ml_runner.errors import EXIT_FAILED

STATUS_LABELS = { "pass": "PASS", "warn": "WARN", "fail": "FAIL" }

def _print_check( check: Check ) -> None:
    typer.echo( f"{STATUS_LABELS[ check.status ]}  {check.title}: {check.detail}" )
    if check.fix and check.status != "pass":
        typer.echo( f"      fix: {check.fix}" )

def doctor( ctx: typer.Context ) -> None:
    state: State = ctx.obj
    checks = run_checks( state.runner, get_config_path( ctx ) )
    for check in checks:
        _print_check( check )

    if any( check.status == "fail" for check in checks ):
        raise typer.Exit( EXIT_FAILED )
