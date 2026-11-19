import importlib.util
from pathlib import Path

from ml_runner import layout

AGENT = Path( __file__ ).parents[ 1 ] / "src" / "ml_runner" / "vm" / "agent.py"

def load_agent():
    spec = importlib.util.spec_from_file_location( "agent_for_drift", AGENT )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec( spec )
    spec.loader.exec_module( module )

    return module

def test_agent_paths_equal_the_layout_paths() -> None:
    agent = load_agent()

    assert agent.source_uri( "b", "s" ) == layout.source( "b", "s" )
    assert agent.run_dir_uri( "b", "e", "r" ) == layout.run_dir( "b", "e", "r" )

def test_agent_run_paths_equal_the_layout_paths() -> None:
    agent = load_agent()
    job = agent.Job(
        job_id = "J",
        bucket = "b",
        max_hours = 1,
        stop_on_failure = True,
        gpu = "t4",
        source = agent.Source( kind = "snapshot" ),
        runs = [],
        env = {}
    )
    run = agent.JobRun( run_id = "r", experiment = "e", command = [] )

    assert agent.record_uri( job, run ) == layout.run_json( "b", "e", "r" )
    assert agent.log_uri( job, run ) == layout.run_log( "b", "e", "r" )
    assert agent.outputs_uri( job, run ) == layout.outputs( "b", "e", "r" )
    assert agent.checkpoints_uri( job, run ) == layout.checkpoints( "b", "e", "r" )
    assert agent.cancel_uri( job ) == layout.job_cancel( "b", "J" )
    assert agent.agent_log_uri( job ) == layout.job_agent_log( "b", "J" )
