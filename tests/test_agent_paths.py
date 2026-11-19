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
