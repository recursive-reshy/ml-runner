from itertools import product
from pathlib import Path

import pytest

from ml_runner.machines import profile_for
from ml_runner.records import Gpu
from ml_runner.vmspec import VmFiles, VmSpec, build_create_args, runtime_cap_minutes

JOB = "20261120-143005-7c21"
ACCOUNT = "ml-runner-vm@my-project.iam.gserviceaccount.com"
FILES = VmFiles(
    startup = Path( "/tmp/job/startup.sh" ),
    shutdown = Path( "/tmp/job/shutdown.sh" ),
    agent = Path( "/tmp/job/agent.py" ),
    job = Path( "/tmp/job/job.json" )
)
FROM_FILES = (
    "--metadata-from-file=startup-script=/tmp/job/startup.sh,shutdown-script=/tmp/job/shutdown.sh,"
    "mlr-agent=/tmp/job/agent.py,mlr-job=/tmp/job/job.json"
)

def make_spec(
    gpu: Gpu = "t4",
    *,
    is_spot: bool = False,
    machine_type: str | None = None,
    disk_gb: int = 100,
    max_hours: float = 6,
    zone: str = "asia-southeast1-a"
) -> VmSpec:
    return VmSpec(
        job_id = JOB,
        zone = zone,
        profile = profile_for( gpu, machine_type ),
        is_spot = is_spot,
        disk_gb = disk_gb,
        max_hours = max_hours,
        service_account = ACCOUNT,
        version = "0.1.0"
    )

def test_a_t4_spot_job_is_exactly_this_command() -> None:
    assert build_create_args( make_spec( "t4", is_spot = True ), FILES ) == [
        "compute",
        "instances",
        "create",
        f"mlr-{JOB}",
        "--zone=asia-southeast1-a",
        "--machine-type=n1-standard-4",
        "--accelerator=type=nvidia-tesla-t4,count=1",
        "--image-family=common-cu129-ubuntu-2404-nvidia-580",
        "--image-project=deeplearning-platform-release",
        "--boot-disk-size=100GB",
        "--boot-disk-type=pd-balanced",
        "--boot-disk-auto-delete",
        "--maintenance-policy=TERMINATE",
        "--provisioning-model=SPOT",
        "--instance-termination-action=DELETE",
        "--max-run-duration=375m",
        f"--service-account={ACCOUNT}",
        "--scopes=cloud-platform",
        f"--labels=ml-runner=1,mlr-job={JOB}",
        "--metadata=install-nvidia-driver=True,mlr-version=0.1.0",
        FROM_FILES
    ]

def test_an_l4_on_demand_job_has_no_accelerator_flag_and_no_spot_flag() -> None:
    assert build_create_args( make_spec( "l4" ), FILES ) == [
        "compute",
        "instances",
        "create",
        f"mlr-{JOB}",
        "--zone=asia-southeast1-a",
        "--machine-type=g2-standard-4",
        "--image-family=common-cu129-ubuntu-2404-nvidia-580",
        "--image-project=deeplearning-platform-release",
        "--boot-disk-size=100GB",
        "--boot-disk-type=pd-balanced",
        "--boot-disk-auto-delete",
        "--maintenance-policy=TERMINATE",
        "--instance-termination-action=DELETE",
        "--max-run-duration=375m",
        f"--service-account={ACCOUNT}",
        "--scopes=cloud-platform",
        f"--labels=ml-runner=1,mlr-job={JOB}",
        "--metadata=install-nvidia-driver=True,mlr-version=0.1.0",
        FROM_FILES
    ]

def test_a_cpu_job_uses_debian_and_does_not_ask_for_a_driver() -> None:
    assert build_create_args( make_spec( "none" ), FILES ) == [
        "compute",
        "instances",
        "create",
        f"mlr-{JOB}",
        "--zone=asia-southeast1-a",
        "--machine-type=e2-standard-4",
        "--image-family=debian-12",
        "--image-project=debian-cloud",
        "--boot-disk-size=100GB",
        "--boot-disk-type=pd-balanced",
        "--boot-disk-auto-delete",
        "--maintenance-policy=TERMINATE",
        "--instance-termination-action=DELETE",
        "--max-run-duration=375m",
        f"--service-account={ACCOUNT}",
        "--scopes=cloud-platform",
        f"--labels=ml-runner=1,mlr-job={JOB}",
        "--metadata=mlr-version=0.1.0",
        FROM_FILES
    ]

COMBINATIONS = list( product( [ "t4", "l4", "none" ], [ False, True ] ) )

@pytest.mark.parametrize( ( "gpu", "is_spot" ), COMBINATIONS )
def test_every_combination_has_the_runtime_cap_the_label_and_self_deletion( gpu: Gpu, is_spot: bool ) -> None:
    args = build_create_args( make_spec( gpu, is_spot = is_spot ), FILES )
    assert "--max-run-duration=375m" in args
    assert "--instance-termination-action=DELETE" in args
    assert f"--labels=ml-runner=1,mlr-job={JOB}" in args
    assert f"mlr-{JOB}" in args

@pytest.mark.parametrize( "gpu", [ "t4", "l4", "none" ] )
def test_spot_adds_exactly_one_flag( gpu: Gpu ) -> None:
    on_demand = build_create_args( make_spec( gpu ), FILES )
    spot = build_create_args( make_spec( gpu, is_spot = True ), FILES )
    assert [ arg for arg in spot if arg not in on_demand ] == [ "--provisioning-model=SPOT" ]
    assert len( spot ) == len( on_demand ) + 1

@pytest.mark.parametrize( ( "gpu", "is_spot" ), COMBINATIONS )
def test_no_network_flags_in_any_combination( gpu: Gpu, is_spot: bool ) -> None:
    args = build_create_args( make_spec( gpu, is_spot = is_spot ), FILES )
    assert not [ arg for arg in args if arg.startswith( ( "--network", "--subnet", "--address" ) ) ]

def test_overrides_change_only_their_flags() -> None:
    base = build_create_args( make_spec( "t4" ), FILES )
    changed = build_create_args(
        make_spec( "t4", machine_type = "n1-highmem-8", disk_gb = 250, max_hours = 2, zone = "asia-southeast1-c" ),
        FILES
    )
    assert [ arg for arg in changed if arg not in base ] == [
        "--zone=asia-southeast1-c",
        "--machine-type=n1-highmem-8",
        "--boot-disk-size=250GB",
        "--max-run-duration=135m"
    ]

@pytest.mark.parametrize(
    ( "max_hours", "minutes" ),
    [ ( 6, 375 ), ( 0.1, 21 ), ( 1.5, 105 ), ( 0.5, 45 ), ( 24, 1455 ), ( 48, 2895 ), ( 6.01, 376 ) ]
)
def test_the_runtime_cap_is_the_hours_rounded_up_plus_fifteen_minutes( max_hours: float, minutes: int ) -> None:
    assert runtime_cap_minutes( max_hours ) == minutes
