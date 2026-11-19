import os
from pathlib import Path
from types import ModuleType

import pytest

from vmsupport import FAKE_NVIDIA_SMI, FakeBin

@pytest.fixture
def project( tmp_path: Path ) -> Path:
    path = tmp_path / "project"
    path.mkdir()

    return path

def test_uv_that_is_already_installed_is_not_installed_again(
    agent: ModuleType,
    project: Path,
    fake_bin: FakeBin
) -> None:
    fake_bin.add_logging( "uv" )
    fake_bin.add_logging( "curl" )
    ( project / "pyproject.toml" ).write_text( "" )

    agent.install_dependencies( project )

    assert fake_bin.lines( "curl" ) == []

def test_uv_is_installed_with_the_installer_script( agent: ModuleType, project: Path, fake_bin: FakeBin ) -> None:
    fake_bin.add_logging( "curl" )

    agent.ensure_uv()

    assert fake_bin.lines( "curl" ) == [ "-LsSf https://astral.sh/uv/install.sh" ]

def test_uv_falls_back_to_pip_when_the_installer_fails(
    agent: ModuleType,
    fake_bin: FakeBin,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bin.add_logging( "curl" )
    fake_bin.add_logging( "python3" )
    monkeypatch.setenv( "FAKE_CURL_EXIT", "1" )

    agent.ensure_uv()

    assert fake_bin.lines( "python3" ) == [ "-m pip install --user uv" ]

def test_uv_failing_both_ways_is_an_install_error(
    agent: ModuleType,
    fake_bin: FakeBin,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bin.add_logging( "curl" )
    fake_bin.add_logging( "python3" )
    monkeypatch.setenv( "FAKE_CURL_EXIT", "1" )
    monkeypatch.setenv( "FAKE_PYTHON3_EXIT", "1" )

    with pytest.raises( agent.InstallError, match = "dependency install failed" ):
        agent.ensure_uv()

def test_the_local_bin_is_put_on_the_path_after_an_install( agent: ModuleType, fake_bin: FakeBin ) -> None:
    fake_bin.add_logging( "curl" )

    agent.ensure_uv()

    assert str( Path.home() / ".local" / "bin" ) in os.environ[ "PATH" ].split( os.pathsep )

def test_a_lock_file_syncs_frozen( agent: ModuleType, project: Path, fake_bin: FakeBin ) -> None:
    fake_bin.add_logging( "uv" )
    ( project / "pyproject.toml" ).write_text( "" )
    ( project / "uv.lock" ).write_text( "" )

    environment = agent.install_dependencies( project )

    assert fake_bin.lines( "uv" ) == [ "sync --frozen" ]
    assert environment.prefix == [ "uv", "run", "--no-sync", "--" ]
    assert environment.path_prepend is None

def test_a_pyproject_without_a_lock_syncs( agent: ModuleType, project: Path, fake_bin: FakeBin ) -> None:
    fake_bin.add_logging( "uv" )
    ( project / "pyproject.toml" ).write_text( "" )

    agent.install_dependencies( project )

    assert fake_bin.lines( "uv" ) == [ "sync" ]

def test_requirements_get_a_venv_and_their_bin_first_on_the_path(
    agent: ModuleType,
    project: Path,
    fake_bin: FakeBin
) -> None:
    fake_bin.add_logging( "uv" )
    ( project / "requirements.txt" ).write_text( "" )

    environment = agent.install_dependencies( project )

    assert fake_bin.lines( "uv" ) == [ "venv", "pip install -r requirements.txt" ]
    assert environment.prefix == []
    assert environment.path_prepend == str( project / ".venv" / "bin" )

def test_no_dependency_file_is_an_install_error( agent: ModuleType, project: Path, fake_bin: FakeBin ) -> None:
    fake_bin.add_logging( "uv" )

    with pytest.raises( agent.InstallError, match = "pyproject.toml or requirements.txt" ):
        agent.install_dependencies( project )

def test_a_failing_sync_is_an_install_error(
    agent: ModuleType,
    project: Path,
    fake_bin: FakeBin,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bin.add_logging( "uv" )
    ( project / "pyproject.toml" ).write_text( "" )
    monkeypatch.setenv( "FAKE_UV_EXIT", "1" )

    with pytest.raises( agent.InstallError, match = "dependency install failed" ):
        agent.install_dependencies( project )

@pytest.fixture
def quick_gpu_wait( monkeypatch: pytest.MonkeyPatch ) -> None:
    monkeypatch.setenv( "MLR_GPU_POLL_SECONDS", "0.01" )
    monkeypatch.setenv( "MLR_GPU_WAIT_SECONDS", "0.2" )

def test_the_gpu_wait_polls_until_the_driver_answers(
    agent: ModuleType,
    fake_bin: FakeBin,
    quick_gpu_wait: None,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bin.add( "nvidia-smi", FAKE_NVIDIA_SMI )
    monkeypatch.setenv( "FAKE_NVIDIA_FAILS", "2" )
    # A generous limit: the driver answers on the third poll, however slow the machine is.
    monkeypatch.setenv( "MLR_GPU_WAIT_SECONDS", "30" )

    agent.wait_for_gpu( "t4" )

    assert len( fake_bin.lines( "nvidia-smi" ) ) == 3

def test_a_driver_that_never_comes_up_is_a_gpu_error(
    agent: ModuleType,
    fake_bin: FakeBin,
    quick_gpu_wait: None,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bin.add( "nvidia-smi", FAKE_NVIDIA_SMI )
    monkeypatch.setenv( "FAKE_NVIDIA_FAILS", "999" )

    with pytest.raises( agent.GpuError, match = "GPU driver not ready" ):
        agent.wait_for_gpu( "t4" )

def test_a_cpu_job_does_not_wait_for_a_gpu( agent: ModuleType, fake_bin: FakeBin ) -> None:
    fake_bin.add( "nvidia-smi", FAKE_NVIDIA_SMI )

    agent.wait_for_gpu( "none" )

    assert fake_bin.lines( "nvidia-smi" ) == []
