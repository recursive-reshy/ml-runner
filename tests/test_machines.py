import pytest

from ml_runner.errors import MlRunnerError, UsageError
from ml_runner.gcloud import GcloudError
from ml_runner.machines import (
    CPU_IMAGE_FAMILY,
    CPU_IMAGE_PROJECT,
    CUDA_IMAGE_FAMILY,
    CUDA_IMAGE_PROJECT,
    NoCapacityError,
    create_in_first_available_zone,
    image_path,
    profile_for,
    zone_candidates,
    zones_offering
)
from ml_runner.records import Gpu
from tests.fakes.project import fixture

ZONES = [ "asia-southeast1-a", "asia-southeast1-b", "asia-southeast1-c" ]

def accelerators() -> list[ dict[ str, str ] ]:
    rows = fixture( "accelerator_types" )
    assert isinstance( rows, list )

    return rows

def test_the_gpu_table_matches_the_trd() -> None:
    t4 = profile_for( "t4", None )
    l4 = profile_for( "l4", None )
    cpu = profile_for( "none", None )
    assert ( t4.machine_type, t4.accelerator_type, t4.image_family, t4.image_project ) == (
        "n1-standard-4",
        "nvidia-tesla-t4",
        CUDA_IMAGE_FAMILY,
        CUDA_IMAGE_PROJECT
    )
    assert ( l4.machine_type, l4.accelerator_type, l4.image_family ) == ( "g2-standard-4", None, CUDA_IMAGE_FAMILY )
    assert ( cpu.machine_type, cpu.accelerator_type, cpu.image_family, cpu.image_project ) == (
        "e2-standard-4",
        None,
        CPU_IMAGE_FAMILY,
        CPU_IMAGE_PROJECT
    )

def test_zones_are_looked_up_by_the_accelerator_name_even_when_there_is_no_flag() -> None:
    assert profile_for( "t4", None ).zone_accelerator == "nvidia-tesla-t4"
    assert profile_for( "l4", None ).zone_accelerator == "nvidia-l4"
    assert profile_for( "l4", None ).accelerator_type is None
    assert profile_for( "none", None ).zone_accelerator is None

def test_only_gpu_profiles_install_the_driver() -> None:
    assert profile_for( "t4", None ).installs_driver
    assert profile_for( "l4", None ).installs_driver
    assert not profile_for( "none", None ).installs_driver

def test_the_image_path_names_the_family_and_its_project() -> None:
    assert image_path( profile_for( "none", None ) ) == "projects/debian-cloud/global/images/family/debian-12"
    assert image_path( profile_for( "t4", None ) ).startswith( "projects/deeplearning-platform-release/" )

@pytest.mark.parametrize(
    ( "gpu", "machine_type" ),
    [ ( "t4", "n1-highmem-8" ), ( "l4", "g2-standard-8" ), ( "none", "n2-standard-8" ), ( "none", "e2-medium" ) ]
)
def test_a_machine_type_override_that_fits_the_gpu_is_used( gpu: Gpu, machine_type: str ) -> None:
    assert profile_for( gpu, machine_type ).machine_type == machine_type

@pytest.mark.parametrize(
    ( "gpu", "machine_type", "needle" ),
    [
        ( "t4", "g2-standard-4", "N1" ),
        ( "t4", "e2-standard-4", "N1" ),
        ( "l4", "n1-standard-4", "G2" ),
        ( "none", "g2-standard-4", "G2" )
    ]
)
def test_a_machine_type_that_does_not_fit_the_gpu_is_refused( gpu: Gpu, machine_type: str, needle: str ) -> None:
    with pytest.raises( UsageError, match = needle ):
        profile_for( gpu, machine_type )

def test_zones_offering_a_gpu_come_from_the_accelerator_list() -> None:
    assert zones_offering( accelerators(), "nvidia-tesla-t4" ) == ZONES
    assert zones_offering( accelerators(), "nvidia-l4" ) == ZONES
    assert zones_offering( accelerators(), "nvidia-no-such" ) == []

def test_the_preferred_zone_goes_first_then_the_rest_alphabetically() -> None:
    assert zone_candidates( "asia-southeast1-b", ZONES, is_explicit = False ) == [
        "asia-southeast1-b",
        "asia-southeast1-a",
        "asia-southeast1-c"
    ]

def test_a_configured_zone_that_does_not_offer_the_gpu_is_skipped() -> None:
    offering = [ "asia-southeast1-a", "asia-southeast1-c" ]
    assert zone_candidates( "asia-southeast1-b", offering, is_explicit = False ) == offering

def test_an_explicit_zone_that_does_not_offer_the_gpu_is_refused() -> None:
    offering = [ "asia-southeast1-a", "asia-southeast1-c" ]
    with pytest.raises( UsageError, match = "asia-southeast1-b" ):
        zone_candidates( "asia-southeast1-b", offering, is_explicit = True )

def test_a_region_with_no_zone_for_the_gpu_is_refused() -> None:
    with pytest.raises( UsageError, match = "asia-southeast1" ):
        zone_candidates( "asia-southeast1-b", [], is_explicit = False )

def test_a_cpu_job_tries_only_its_own_zone() -> None:
    assert zone_candidates( "asia-southeast1-b", None, is_explicit = False ) == [ "asia-southeast1-b" ]

def exhausted() -> GcloudError:
    return GcloudError( "zone_exhausted", "no capacity", "ZONE_RESOURCE_POOL_EXHAUSTED" )

def test_the_first_zone_with_capacity_wins() -> None:
    tried: list[ str ] = []

    def create( zone: str ) -> str:
        tried.append( zone )

        return f"vm in {zone}"

    assert create_in_first_available_zone( ZONES, create ) == ( "asia-southeast1-a", "vm in asia-southeast1-a" )
    assert tried == [ "asia-southeast1-a" ]

def test_an_exhausted_zone_moves_on_to_the_next() -> None:
    tried: list[ str ] = []

    def create( zone: str ) -> str:
        tried.append( zone )
        if zone != "asia-southeast1-c":
            raise exhausted()

        return "created"

    assert create_in_first_available_zone( ZONES, create ) == ( "asia-southeast1-c", "created" )
    assert tried == ZONES

def test_a_quota_error_stops_at_once() -> None:
    tried: list[ str ] = []

    def create( zone: str ) -> str:
        tried.append( zone )
        raise GcloudError( "quota_exceeded", "A Google Cloud quota is exceeded. Run: ml-runner doctor" )

    with pytest.raises( GcloudError, match = "ml-runner doctor" ):
        create_in_first_available_zone( ZONES, create )
    assert tried == [ ZONES[ 0 ] ]

def test_other_errors_are_not_swallowed() -> None:
    def create( zone: str ) -> str:
        raise GcloudError( "permission_denied", "denied" )

    with pytest.raises( GcloudError, match = "denied" ):
        create_in_first_available_zone( ZONES, create )

def test_every_zone_exhausted_lists_the_zones_tried() -> None:
    def create( zone: str ) -> str:
        raise exhausted()

    with pytest.raises( NoCapacityError ) as raised:
        create_in_first_available_zone( ZONES, create )
    assert isinstance( raised.value, MlRunnerError )
    for zone in ZONES:
        assert zone in raised.value.message
    assert raised.value.exit_code == 1
