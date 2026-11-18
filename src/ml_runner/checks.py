from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ml_runner.config import Config, load_config
from ml_runner.errors import UsageError
from ml_runner.gcloud import Gcloud, GcloudError, Runner
from ml_runner.setup import (
    BUCKET_ROLE,
    CONDITION_TITLE,
    REQUIRED_APIS,
    SELF_DELETE_ROLE_ID,
    has_binding,
    service_account_email
)
from ml_runner.layout import bucket_uri

QUOTA_URL = "https://console.cloud.google.com/iam-admin/quotas"
BUDGETS_URL = "https://console.cloud.google.com/billing/budgets"
GLOBAL_METRIC = "GPUS_ALL_REGIONS"
# The only gcloud version verified so far (task 2.1); older ones are a warning, not a failure.
MIN_GCLOUD_VERSION = ( 590, 0, 0 )
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

def limiting_metrics( quota: GpuQuota ) -> list[ str ]:
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

        reasons = " and ".join( limiting_metrics( quota ) )
        lines.append( f"  {quota.label} cannot launch: {reasons} is 0. Request an increase: {QUOTA_URL}" )

    return lines

CheckStatus = Literal[ "pass", "warn", "fail" ]

@dataclass( frozen = True )
class Check:
    status: CheckStatus
    title: str
    detail: str
    fix: str = ""

INIT_HINT = "Run: ml-runner init"

def _failed( title: str, error: GcloudError, fix: str ) -> Check:
    return Check( "fail", title, error.message, fix )

def _check_gcloud( gcloud: Gcloud ) -> Check:
    try:
        version = gcloud.version()
    except GcloudError as error:
        return _failed( "gcloud", error, "Install the Google Cloud CLI: https://cloud.google.com/sdk/docs/install" )

    text = ".".join( str( part ) for part in version )
    if version < MIN_GCLOUD_VERSION:
        minimum = ".".join( str( part ) for part in MIN_GCLOUD_VERSION )

        return Check(
            "warn",
            "gcloud",
            f"{text} is older than {minimum}, the version ml-runner was tested with.",
            "Run: gcloud components update"
        )

    return Check( "pass", "gcloud", text )

def _check_signed_in( gcloud: Gcloud ) -> Check:
    try:
        accounts = gcloud.json( "auth", "list", "--filter=status:ACTIVE" ) or []
    except GcloudError as error:
        return _failed( "Signed in", error, "Run: gcloud auth login" )

    if not accounts:
        return Check( "fail", "Signed in", "No active gcloud account.", "Run: gcloud auth login" )

    return Check( "pass", "Signed in", accounts[ 0 ][ "account" ] )

def _skipped( reason: str ) -> Check:
    return Check(
        "warn",
        "Other checks",
        f"Skipped: {reason}.",
        "Fix the failure above and run ml-runner doctor again."
    )

def _check_project( gcloud: Gcloud, config: Config ) -> Check:
    try:
        gcloud.json( "projects", "describe", config.project )
    except GcloudError as error:
        return _failed( "Project", error, f"Check the project in the config, or {INIT_HINT}" )

    return Check( "pass", "Project", config.project )

def _check_apis( gcloud: Gcloud ) -> list[ Check ]:
    try:
        enabled = { service[ "config" ][ "name" ] for service in gcloud.json( "services", "list", "--enabled" ) or [] }
    except GcloudError as error:
        return [ _failed( "APIs", error, "Run: gcloud services list --enabled" ) ]

    return [
        Check( "pass", f"API {api}", "enabled" ) if api in enabled
        else Check( "fail", f"API {api}", "not enabled", INIT_HINT )
        for api in REQUIRED_APIS
    ]

def _check_resource( gcloud: Gcloud, title: str, detail: str, *args: str ) -> Check:
    try:
        gcloud.json( *args )
    except GcloudError as error:
        return _failed( title, error, INIT_HINT )

    return Check( "pass", title, detail )

def _check_bucket_access( gcloud: Gcloud, config: Config ) -> Check:
    member = f"serviceAccount:{service_account_email( config.project )}"
    try:
        policy = gcloud.json( "storage", "buckets", "get-iam-policy", bucket_uri( config.bucket ) )
    except GcloudError as error:
        return _failed( "Bucket access", error, INIT_HINT )

    if not has_binding( policy, BUCKET_ROLE, member ):
        return Check( "fail", "Bucket access", f"{member} lacks {BUCKET_ROLE} on the bucket.", INIT_HINT )

    return Check( "pass", "Bucket access", f"{BUCKET_ROLE} for the runner service account" )

def _check_self_delete( gcloud: Gcloud, config: Config ) -> Check:
    member = f"serviceAccount:{service_account_email( config.project )}"
    role = f"projects/{config.project}/roles/{SELF_DELETE_ROLE_ID}"
    try:
        policy = gcloud.json( "projects", "get-iam-policy", config.project )
    except GcloudError as error:
        return _failed( "Self-delete role", error, INIT_HINT )

    if not has_binding( policy, role, member, CONDITION_TITLE ):
        detail = f"{member} lacks the conditional {SELF_DELETE_ROLE_ID} role."

        return Check( "fail", "Self-delete role", detail, INIT_HINT )

    return Check( "pass", "Self-delete role", f"{SELF_DELETE_ROLE_ID} for mlr-* instances only" )

def _check_network( gcloud: Gcloud ) -> Check:
    try:
        gcloud.json( "compute", "networks", "describe", "default" )
    except GcloudError as error:
        # Users with a custom VPC can pass --network later (KIV), so this is not a failure.
        return Check( "warn", "Network", error.message, "ml-runner v0.1.0 uses the project's default network." )

    return Check( "pass", "Network", "default" )

def _check_gpu_quota( gcloud: Gcloud, config: Config ) -> Check:
    gpu = config.defaults.gpu
    if gpu == "none":
        return Check( "pass", "GPU quota", "No GPU by default." )

    label = gpu.upper() + ( " (Spot)" if config.defaults.spot else "" )
    try:
        quotas = read_gpu_quotas( gcloud, config.region )
    except GcloudError as error:
        return Check( "warn", "GPU quota", f"Could not read quotas: {error.message}", f"Check them at {QUOTA_URL}" )

    quota = next( quota for quota in quotas if quota.label == label )
    if quota.usable_limit > 0:
        return Check( "pass", "GPU quota", f"{label}: {int( quota.usable_limit )} usable" )

    reasons = " and ".join( limiting_metrics( quota ) )

    return Check( "warn", "GPU quota", f"{label} limit is 0 ({reasons}).", f"Request an increase: {QUOTA_URL}" )

def _check_budget() -> Check:
    return Check(
        "warn",
        "Budget",
        "A budget alert cannot be checked without billing permissions.",
        f"Set a budget alert: {BUDGETS_URL}"
    )

def _check_config( config_file: Path ) -> tuple[ Config | None, Check ]:
    try:
        config = load_config( config_file )
    except UsageError as error:
        return None, Check( "fail", "Config", error.message, INIT_HINT )

    return config, Check( "pass", "Config", str( config_file ) )

# Every check of TRD §6. Read-only. A failure that makes later checks meaningless stops the run with a "skipped" note.
def run_checks( runner: Runner | None, config_file: Path ) -> list[ Check ]:
    bare = Gcloud( runner = runner )
    gcloud_check = _check_gcloud( bare )
    if gcloud_check.status == "fail":
        return [ gcloud_check ]

    checks = [ gcloud_check ]
    signed_in = _check_signed_in( bare )
    checks.append( signed_in )
    if signed_in.status == "fail":
        return [ *checks, _skipped( "not signed in" ) ]

    config, config_check = _check_config( config_file )
    checks.append( config_check )
    if config is None:
        return [ *checks, _skipped( "no config" ) ]

    gcloud = Gcloud( project = config.project, runner = runner )

    return [
        *checks,
        _check_project( gcloud, config ),
        *_check_apis( gcloud ),
        _check_resource(
            gcloud,
            "Bucket",
            config.bucket,
            "storage",
            "buckets",
            "describe",
            bucket_uri( config.bucket )
        ),
        _check_resource(
            gcloud,
            "Service account",
            service_account_email( config.project ),
            "iam",
            "service-accounts",
            "describe",
            service_account_email( config.project )
        ),
        _check_bucket_access( gcloud, config ),
        _check_self_delete( gcloud, config ),
        _check_network( gcloud ),
        _check_gpu_quota( gcloud, config ),
        _check_budget()
    ]
