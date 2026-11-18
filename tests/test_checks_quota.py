import pytest

from ml_runner.checks import GLOBAL_METRIC, QUOTA_URL, GpuQuota, read_gpu_quotas, summarise_quotas
from ml_runner.gcloud import Gcloud
from tests.fakes.fake_gcloud import FakeGcloud
from tests.fakes.project import PROJECT, fixture

REGION = "asia-southeast1"

def fake_with_quotas() -> FakeGcloud:
    fake = FakeGcloud()
    fake.register_json( [ "compute", "regions", "describe" ], fixture( "region_describe" ) )
    fake.register_json( [ "compute", "project-info", "describe" ], fixture( "project_info" ) )

    return fake

def test_the_usable_limit_is_the_lower_of_regional_and_global() -> None:
    assert GpuQuota( "T4", "NVIDIA_T4_GPUS", regional_limit = 1, global_limit = 0 ).usable_limit == 0
    assert GpuQuota( "T4", "NVIDIA_T4_GPUS", regional_limit = 1, global_limit = 8 ).usable_limit == 1
    assert GpuQuota( "T4", "NVIDIA_T4_GPUS", regional_limit = 0, global_limit = 8 ).usable_limit == 0

def test_reads_the_four_gpu_metrics_and_the_global_limit_from_the_captured_output() -> None:
    fake = fake_with_quotas()
    quotas = read_gpu_quotas( Gcloud( project = PROJECT, runner = fake ), REGION )
    assert [ quota.metric for quota in quotas ] == [
        "NVIDIA_T4_GPUS",
        "PREEMPTIBLE_NVIDIA_T4_GPUS",
        "NVIDIA_L4_GPUS",
        "PREEMPTIBLE_NVIDIA_L4_GPUS"
    ]
    assert [ quota.regional_limit for quota in quotas ] == [ 1.0, 1.0, 1.0, 1.0 ]
    assert { quota.global_limit for quota in quotas } == { 0.0 }
    assert [ quota.usable_limit for quota in quotas ] == [ 0.0, 0.0, 0.0, 0.0 ]
    assert fake.calls[ 0 ][ :4 ] == [ "gcloud", "compute", "regions", "describe" ]
    assert REGION in fake.calls[ 0 ]

def test_a_metric_missing_from_the_output_counts_as_zero() -> None:
    fake = FakeGcloud()
    fake.register_json( [ "compute", "regions", "describe" ], { "quotas": [] } )
    fake.register_json( [ "compute", "project-info", "describe" ], { "quotas": [] } )
    quotas = read_gpu_quotas( Gcloud( runner = fake ), REGION )
    assert all( quota.regional_limit == 0 and quota.global_limit == 0 for quota in quotas )

def test_summary_names_the_global_quota_and_links_the_quota_page_when_it_is_the_limit() -> None:
    quotas = read_gpu_quotas( Gcloud( runner = fake_with_quotas() ), REGION )
    text = "\n".join( summarise_quotas( quotas ) )
    assert GLOBAL_METRIC in text
    assert QUOTA_URL in text
    assert "T4 (Spot): 0" in text

def test_summary_is_quiet_about_the_global_limit_when_it_is_not_the_limit() -> None:
    quotas = [ GpuQuota( "T4", "NVIDIA_T4_GPUS", regional_limit = 1, global_limit = 8 ) ]
    text = "\n".join( summarise_quotas( quotas ) )
    assert "T4: 1" in text
    assert GLOBAL_METRIC not in text

def test_summary_names_the_regional_metric_when_the_region_is_the_limit() -> None:
    quotas = [ GpuQuota( "L4", "NVIDIA_L4_GPUS", regional_limit = 0, global_limit = 8 ) ]
    text = "\n".join( summarise_quotas( quotas ) )
    assert "NVIDIA_L4_GPUS" in text
    assert QUOTA_URL in text

@pytest.mark.parametrize( "usable", [ 1, 4 ] )
def test_a_positive_usable_limit_has_no_quota_link( usable: int ) -> None:
    quotas = [ GpuQuota( "T4", "NVIDIA_T4_GPUS", regional_limit = usable, global_limit = usable ) ]
    assert QUOTA_URL not in "\n".join( summarise_quotas( quotas ) )
