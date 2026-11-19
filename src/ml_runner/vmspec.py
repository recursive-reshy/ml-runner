import math
from dataclasses import dataclass
from pathlib import Path

from ml_runner.ids import vm_name
from ml_runner.machines import MachineProfile

EXTRA_CAP_MINUTES = 15
MINUTES_PER_HOUR = 60
# Rounding first keeps 0.1 h from becoming 6.000000000000001 minutes and then 7.
CAP_ROUNDING_DIGITS = 6
BOOT_DISK_TYPE = "pd-balanced"
SERVICE_ACCOUNT_SCOPES = "cloud-platform"
OWNER_LABEL = "ml-runner=1"

@dataclass( frozen = True )
class VmFiles:
    startup: Path
    shutdown: Path
    agent: Path
    job: Path

@dataclass( frozen = True )
class VmSpec:
    job_id: str
    zone: str
    profile: MachineProfile
    is_spot: bool
    disk_gb: int
    max_hours: float
    service_account: str
    version: str

# The cap is the job's own limit plus time for the final sync; Google deletes the VM at that point.
def runtime_cap_minutes( max_hours: float ) -> int:
    minutes = math.ceil( round( max_hours * MINUTES_PER_HOUR, CAP_ROUNDING_DIGITS ) )

    return minutes + EXTRA_CAP_MINUTES

# The arguments of `gcloud compute instances create` (TRD §10.3). `Gcloud` adds the project and `--quiet`.
def build_create_args( spec: VmSpec, files: VmFiles ) -> list[ str ]:
    profile = spec.profile
    metadata = [ "install-nvidia-driver=True" ] if profile.installs_driver else []
    metadata.append( f"mlr-version={spec.version}" )
    from_files = (
        f"startup-script={files.startup},shutdown-script={files.shutdown},"
        f"mlr-agent={files.agent},mlr-job={files.job}"
    )
    accelerator = (
        [ f"--accelerator=type={profile.accelerator_type},count=1" ] if profile.accelerator_type is not None else []
    )
    spot = [ "--provisioning-model=SPOT" ] if spec.is_spot else []

    return [
        "compute",
        "instances",
        "create",
        vm_name( spec.job_id ),
        f"--zone={spec.zone}",
        f"--machine-type={profile.machine_type}",
        *accelerator,
        f"--image-family={profile.image_family}",
        f"--image-project={profile.image_project}",
        f"--boot-disk-size={spec.disk_gb}GB",
        f"--boot-disk-type={BOOT_DISK_TYPE}",
        "--boot-disk-auto-delete",
        "--maintenance-policy=TERMINATE",
        *spot,
        "--instance-termination-action=DELETE",
        f"--max-run-duration={runtime_cap_minutes( spec.max_hours )}m",
        f"--service-account={spec.service_account}",
        f"--scopes={SERVICE_ACCOUNT_SCOPES}",
        f"--labels={OWNER_LABEL},mlr-job={spec.job_id}",
        f"--metadata={','.join( metadata )}",
        f"--metadata-from-file={from_files}"
    ]
