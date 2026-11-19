import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from ml_runner.config import Config, Defaults
from ml_runner.errors import MlRunnerError, UsageError
from ml_runner.gcloud import Gcloud, GcloudError
from ml_runner.layout import bucket_uri
from ml_runner.machines import zones_offering

DEFAULT_REGION = "asia-southeast1"
REQUIRED_APIS = ( "compute.googleapis.com", "storage.googleapis.com" )
SERVICE_ACCOUNT_NAME = "ml-runner-vm"
SERVICE_ACCOUNT_DISPLAY_NAME = "ml-runner VMs"
BUCKET_ROLE = "roles/storage.objectAdmin"
SELF_DELETE_ROLE_ID = "mlRunnerSelfDelete"
SELF_DELETE_PERMISSIONS = "compute.instances.delete,compute.instances.get"
SELF_DELETE_ROLE_TITLE = "ml-runner self delete"
CONDITION_EXPRESSION = 'resource.name.extract("instances/{name}").startsWith("mlr-")'
CONDITION_TITLE = "mlr-instances-only"
UNSET_PROJECT_VALUES = ( "", "(unset)" )
DEFAULT_GPU_ACCELERATOR = "nvidia-tesla-t4"
PROPAGATION_ATTEMPTS = 8
PROPAGATION_DELAY_SECONDS = 5.0
PROPAGATION_MARKER = "does not exist"

StepStatus = Literal[ "exists", "created", "planned" ]

@dataclass( frozen = True )
class InitSettings:
    project: str
    region: str
    bucket: str

@dataclass( frozen = True )
class Step:
    name: str
    is_done: Callable[ [], bool ]
    apply: Callable[ [], None ]

@dataclass( frozen = True )
class StepResult:
    name: str
    status: StepStatus

def default_bucket( project: str ) -> str:
    return f"ml-runner-{project}"

def service_account_email( project: str ) -> str:
    return f"{SERVICE_ACCOUNT_NAME}@{project}.iam.gserviceaccount.com"

def resolve_project( gcloud: Gcloud, flag: str | None ) -> str:
    if flag:
        return flag

    configured = gcloud.run( "config", "get-value", "project", mutating = False ).strip()
    if configured in UNSET_PROJECT_VALUES:
        raise UsageError( "No project given. Pass --project or run: gcloud config set project <id>" )

    return configured

# Not found is an answer ("it does not exist yet"); any other error is a real failure.
def _read_or_none( read: Callable[ [], Any ] ) -> Any | None:
    try:
        return read()
    except GcloudError as error:
        if error.kind == "not_found":
            return None

        raise

def has_binding( policy: Any, role: str, member: str, condition_title: str | None = None ) -> bool:
    if policy is None:
        return False

    for binding in policy.get( "bindings", [] ):
        binding_title = ( binding.get( "condition" ) or {} ).get( "title" )
        if binding[ "role" ] == role and member in binding[ "members" ] and binding_title == condition_title:
            return True

    return False

def _taken_bucket_error( bucket: str ) -> UsageError:
    return UsageError( f"The bucket name {bucket} is taken by another project. Choose another with --bucket." )

# IAM is eventually consistent: a binding made right after the service account is created can
# fail with "does not exist" (seen live, exit 1, kind unknown), so it is retried for a short while.
def _retry_while_propagating( apply: Callable[ [], None ], sleep: Callable[ [ float ], None ] ) -> None:
    for attempt in range( PROPAGATION_ATTEMPTS ):
        try:
            apply()
        except GcloudError as error:
            is_last = attempt == PROPAGATION_ATTEMPTS - 1
            if PROPAGATION_MARKER not in error.stderr or is_last:
                raise

            sleep( PROPAGATION_DELAY_SECONDS )
        else:
            return

def _api_step( gcloud: Gcloud, api: str ) -> Step:
    def is_done() -> bool:
        enabled = gcloud.json( "services", "list", "--enabled" ) or []

        return any( service[ "config" ][ "name" ] == api for service in enabled )

    def apply() -> None:
        gcloud.run( "services", "enable", api, mutating = True )

    return Step( f"Enable {api}", is_done, apply )

def _bucket_step( gcloud: Gcloud, settings: InitSettings ) -> Step:
    uri = bucket_uri( settings.bucket )

    def is_done() -> bool:
        try:
            return _read_or_none( lambda: gcloud.json( "storage", "buckets", "describe", uri ) ) is not None
        except GcloudError as error:
            if error.kind == "permission_denied":
                raise _taken_bucket_error( settings.bucket ) from error

            raise

    def apply() -> None:
        try:
            gcloud.run(
                "storage",
                "buckets",
                "create",
                uri,
                f"--location={settings.region}",
                "--uniform-bucket-level-access",
                "--public-access-prevention",
                mutating = True
            )
        except GcloudError as error:
            if error.kind in ( "already_exists", "permission_denied" ):
                raise _taken_bucket_error( settings.bucket ) from error

            raise

    return Step( f"Bucket {uri}", is_done, apply )

def _service_account_step( gcloud: Gcloud, settings: InitSettings ) -> Step:
    email = service_account_email( settings.project )

    def is_done() -> bool:
        return _read_or_none( lambda: gcloud.json( "iam", "service-accounts", "describe", email ) ) is not None

    def apply() -> None:
        gcloud.run(
            "iam",
            "service-accounts",
            "create",
            SERVICE_ACCOUNT_NAME,
            f"--display-name={SERVICE_ACCOUNT_DISPLAY_NAME}",
            mutating = True
        )

    return Step( f"Service account {SERVICE_ACCOUNT_NAME}", is_done, apply )

def _bucket_binding_step( gcloud: Gcloud, settings: InitSettings, sleep: Callable[ [ float ], None ] ) -> Step:
    uri = bucket_uri( settings.bucket )
    member = f"serviceAccount:{service_account_email( settings.project )}"

    def is_done() -> bool:
        policy = _read_or_none( lambda: gcloud.json( "storage", "buckets", "get-iam-policy", uri ) )

        return has_binding( policy, BUCKET_ROLE, member )

    def add_binding() -> None:
        gcloud.run(
            "storage",
            "buckets",
            "add-iam-policy-binding",
            uri,
            f"--member={member}",
            f"--role={BUCKET_ROLE}",
            mutating = True
        )

    def apply() -> None:
        _retry_while_propagating( add_binding, sleep )

    return Step( f"Grant {BUCKET_ROLE} on {uri}", is_done, apply )

def _role_step( gcloud: Gcloud ) -> Step:
    def is_done() -> bool:
        return _read_or_none( lambda: gcloud.json( "iam", "roles", "describe", SELF_DELETE_ROLE_ID ) ) is not None

    def apply() -> None:
        gcloud.run(
            "iam",
            "roles",
            "create",
            SELF_DELETE_ROLE_ID,
            f"--permissions={SELF_DELETE_PERMISSIONS}",
            "--stage=GA",
            f"--title={SELF_DELETE_ROLE_TITLE}",
            mutating = True
        )

    return Step( f"Custom role {SELF_DELETE_ROLE_ID}", is_done, apply )

def _project_binding_step( gcloud: Gcloud, settings: InitSettings, sleep: Callable[ [ float ], None ] ) -> Step:
    member = f"serviceAccount:{service_account_email( settings.project )}"
    role = f"projects/{settings.project}/roles/{SELF_DELETE_ROLE_ID}"

    def is_done() -> bool:
        policy = _read_or_none( lambda: gcloud.json( "projects", "get-iam-policy", settings.project ) )

        return has_binding( policy, role, member, CONDITION_TITLE )

    def add_binding() -> None:
        gcloud.run(
            "projects",
            "add-iam-policy-binding",
            settings.project,
            f"--member={member}",
            f"--role={role}",
            f"--condition=expression={CONDITION_EXPRESSION},title={CONDITION_TITLE}",
            mutating = True
        )

    def apply() -> None:
        _retry_while_propagating( add_binding, sleep )

    return Step( f"Self-delete binding for mlr-* instances ({SELF_DELETE_ROLE_ID})", is_done, apply )

# Step 8 of TRD §5. `--zone` wins; otherwise the first zone alphabetically that offers the default GPU.
# gcloud reports `zone` here as a URL (.../zones/<zone>), so the filter cannot be anchored with `^`.
def choose_zone( gcloud: Gcloud, region: str, requested: str | None ) -> str:
    if requested:
        if not requested.startswith( f"{region}-" ):
            raise UsageError( f"Zone {requested} is not in region {region}." )

        return requested

    offered = gcloud.json( "compute", "accelerator-types", "list", f"--filter=zone~{region}-" ) or []
    zones = zones_offering( offered, DEFAULT_GPU_ACCELERATOR )
    if not zones:
        raise UsageError( f"No zone in {region} offers a T4 GPU. Try another --region." )

    return zones[ 0 ]

# Step 9 of TRD §5. A re-run keeps the defaults the user already has.
def build_config( settings: InitSettings, zone: str, existing: Config | None ) -> Config:
    return Config(
        project = settings.project,
        region = settings.region,
        zone = zone,
        bucket = settings.bucket,
        service_account = service_account_email( settings.project ),
        defaults = existing.defaults if existing else Defaults()
    )

# Steps 1-7 of TRD §5, in order. Nothing is changed until `run_steps`.
def build_steps(
    gcloud: Gcloud,
    settings: InitSettings,
    sleep: Callable[ [ float ], None ] = time.sleep
) -> list[ Step ]:
    try:
        gcloud.json( "projects", "describe", settings.project )
    except GcloudError as error:
        if error.kind in ( "not_found", "permission_denied" ):
            raise UsageError( f"Project {settings.project} was not found or is not accessible to you." ) from error

        raise

    return [
        *( _api_step( gcloud, api ) for api in REQUIRED_APIS ),
        _bucket_step( gcloud, settings ),
        _service_account_step( gcloud, settings ),
        _bucket_binding_step( gcloud, settings, sleep ),
        _role_step( gcloud ),
        _project_binding_step( gcloud, settings, sleep )
    ]

# Reads everything first, asks once if anything would change, then applies the missing steps.
def run_steps(
    steps: list[ Step ],
    confirm: Callable[ [ list[ str ] ], bool ],
    *,
    is_dry_run: bool
) -> list[ StepResult ]:
    pending = [ step for step in steps if not step.is_done() ]
    if pending and not is_dry_run and not confirm( [ step.name for step in pending ] ):
        raise MlRunnerError( "Cancelled. Nothing was changed." )

    applied_status: StepStatus = "planned" if is_dry_run else "created"
    results: list[ StepResult ] = []
    for step in steps:
        if step not in pending:
            results.append( StepResult( step.name, "exists" ) )
            continue

        step.apply()
        results.append( StepResult( step.name, applied_status ) )

    return results
