import importlib.util
from pathlib import Path

AGENT = Path(__file__).parents[2] / "src" / "ml_runner" / "vm" / "agent.py"


def test_agent_loads_by_path() -> None:
    spec = importlib.util.spec_from_file_location("agent", AGENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
