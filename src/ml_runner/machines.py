from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, TypeVar

from ml_runner.errors import MlRunnerError, UsageError
from ml_runner.gcloud import GcloudError
from ml_runner.records import Gpu

Created = TypeVar( "Created" )

# Captured in task 2.1: the DLVM has no CPU-only family, so CPU jobs boot plain Debian 12.
CUDA_IMAGE_PROJECT = "deeplearning-platform-release"
CUDA_IMAGE_FAMILY = "common-cu129-ubuntu-2404-nvidia-580"
CPU_IMAGE_PROJECT = "debian-cloud"
CPU_IMAGE_FAMILY = "debian-12"

@dataclass( frozen = True )
class MachineProfile:
    gpu: Gpu
    machine_type: str
    accelerator_type: str | None
    # The name `compute accelerator-types list` uses, for finding zones; L4 has no flag but still has a name.
    zone_accelerator: str | None
    image_project: str
    image_family: str
    installs_driver: bool
    machine_family_prefix: str | None
    forbidden_prefix: str | None

PROFILES: Mapping[ Gpu, MachineProfile ] = {
    "t4": MachineProfile(
        gpu = "t4",
        machine_type = "n1-standard-4",
        accelerator_type = "nvidia-tesla-t4",
        zone_accelerator = "nvidia-tesla-t4",
        image_project = CUDA_IMAGE_PROJECT,
        image_family = CUDA_IMAGE_FAMILY,
        installs_driver = True,
        machine_family_prefix = "n1-",
        forbidden_prefix = None
    ),
    "l4": MachineProfile(
        gpu = "l4",
        machine_type = "g2-standard-4",
        accelerator_type = None,
        zone_accelerator = "nvidia-l4",
        image_project = CUDA_IMAGE_PROJECT,
        image_family = CUDA_IMAGE_FAMILY,
        installs_driver = True,
        machine_family_prefix = "g2-",
        forbidden_prefix = None
    ),
    "none": MachineProfile(
        gpu = "none",
        machine_type = "e2-standard-4",
        accelerator_type = None,
        zone_accelerator = None,
        image_project = CPU_IMAGE_PROJECT,
        image_family = CPU_IMAGE_FAMILY,
        installs_driver = False,
        machine_family_prefix = None,
        # A g2 machine has an L4 built in, so it would not be a CPU-only job.
        forbidden_prefix = "g2-"
    )
}

_FAMILY_NAMES = { "n1-": "N1", "g2-": "G2" }

# The GPU's profile, with `--machine-type` applied after checking that it fits the GPU.
def profile_for( gpu: Gpu, machine_type: str | None ) -> MachineProfile:
    profile = PROFILES[ gpu ]
    if machine_type is None:
        return profile

    required = profile.machine_family_prefix
    if required is not None and not machine_type.startswith( required ):
        raise UsageError(
            f"--gpu {gpu} needs a {_FAMILY_NAMES[ required ]} machine type ({required}*), got {machine_type}."
        )
    if profile.forbidden_prefix is not None and machine_type.startswith( profile.forbidden_prefix ):
        raise UsageError(
            f"--gpu none cannot use {machine_type}: "
            f"{_FAMILY_NAMES[ profile.forbidden_prefix ]} machines come with a GPU."
        )

    return replace( profile, machine_type = machine_type )

def image_path( profile: MachineProfile ) -> str:
    return f"projects/{profile.image_project}/global/images/family/{profile.image_family}"

# `rows` is the JSON of `compute accelerator-types list`; its `zone` is a URL (.../zones/<zone>).
def zones_offering( rows: Sequence[ Mapping[ str, Any ] ], accelerator_type: str ) -> list[ str ]:
    return sorted( { row[ "zone" ].rsplit( "/", 1 )[ -1 ] for row in rows if row[ "name" ] == accelerator_type } )

# The preferred zone first, then the rest alphabetically (TRD §10.2). `offering` is None for a CPU job.
def zone_candidates( preferred: str, offering: list[ str ] | None, *, is_explicit: bool ) -> list[ str ]:
    if offering is None:
        return [ preferred ]

    region = preferred.rsplit( "-", 1 )[ 0 ]
    if not offering:
        raise UsageError( f"No zone in {region} offers this GPU. Try another --region or --gpu." )
    if preferred in offering:
        return [ preferred, *( zone for zone in offering if zone != preferred ) ]
    if is_explicit:
        raise UsageError( f"Zone {preferred} does not offer this GPU. Zones that do: {', '.join( offering )}." )

    return offering

class NoCapacityError( MlRunnerError ):
    pass

# `zone_exhausted` moves on to the next zone. A quota error (already worded by `GcloudError`) and anything
# else stop at once. Returns the zone that worked and what `create` returned.
def create_in_first_available_zone(
    zones: Sequence[ str ],
    create: Callable[ [ str ], Created ]
) -> tuple[ str, Created ]:
    for zone in zones:
        try:
            return ( zone, create( zone ) )
        except GcloudError as error:
            if error.kind != "zone_exhausted":
                raise

    raise NoCapacityError(
        f"No zone has capacity for this machine right now (tried {', '.join( zones )}). "
        "Try again later, or use another --gpu or --region."
    )
