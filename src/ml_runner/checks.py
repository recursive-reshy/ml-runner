from dataclasses import dataclass
from typing import Any

from ml_runner.gcloud import Gcloud

QUOTA_URL = "https://console.cloud.google.com/iam-admin/quotas"
BUDGETS_URL = "https://console.cloud.google.com/billing/budgets"
GLOBAL_METRIC = "GPUS_ALL_REGIONS"
# (label, regional quota metric); the order is the order of the summary.
GPU_METRICS = (
    ( "T4", "NVIDIA_T4_GPUS" ),
    ( "T4 (Spot)", "PREEMPTIBLE_NVIDIA_T4_GPUS" ),
    ( "L4", "NVIDIA_L4_GPUS" ),
    ( "L4 (Spot)", "PREEMPTIBLE_NVIDIA_L4_GPUS" )
)

# ADR 0013: what a launch can really use is the lower of the regional and the project-wide limit.
@dataclass( frozen = True )
class GpuQuota:
    label: str
    metric: str
    regional_limit: float
    global_limit: float

    @property
    def usable_limit( self ) -> float:
        return min( self.regional_limit, self.global_limit )

def _limits( quotas: Any ) -> dict[ str, float ]:
    return { quota[ "metric" ]: float( quota[ "limit" ] ) for quota in quotas or [] }

# A metric missing from gcloud's output counts as 0.
def read_gpu_quotas( gcloud: Gcloud, region: str ) -> list[ GpuQuota ]:
    region_limits = _limits( gcloud.json( "compute", "regions", "describe", region ).get( "quotas" ) )
    project_limits = _limits( gcloud.json( "compute", "project-info", "describe" ).get( "quotas" ) )
    global_limit = project_limits.get( GLOBAL_METRIC, 0.0 )

    return [
        GpuQuota( label, metric, region_limits.get( metric, 0.0 ), global_limit )
        for label, metric in GPU_METRICS
    ]

def _limiting_metrics( quota: GpuQuota ) -> list[ str ]:
    metrics = []
    if quota.regional_limit == 0:
        metrics.append( quota.metric )
    if quota.global_limit == 0:
        metrics.append( GLOBAL_METRIC )

    return metrics

def summarise_quotas( quotas: list[ GpuQuota ] ) -> list[ str ]:
    lines = [ f"{quota.label}: {int( quota.usable_limit )}" for quota in quotas ]
    for quota in quotas:
        if quota.usable_limit > 0:
            continue

        reasons = " and ".join( _limiting_metrics( quota ) )
        lines.append( f"  {quota.label} cannot launch: {reasons} is 0. Request an increase: {QUOTA_URL}" )

    return lines
