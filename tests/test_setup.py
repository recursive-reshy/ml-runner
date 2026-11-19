from collections.abc import Sequence

import pytest

from ml_runner.config import Defaults
from ml_runner.errors import MlRunnerError, UsageError
from ml_runner.gcloud import Completed, Gcloud
from ml_runner.setup import (
    DEFAULT_REGION,
    PROPAGATION_ATTEMPTS,
    InitSettings,
    build_config,
    build_steps,
    choose_zone,
    default_bucket,
    resolve_project,
    run_steps,
    service_account_email
)
from tests.fakes.fake_gcloud import FakeGcloud
from tests.fakes.project import (
    BUCKET,
    CONDITION,
    PROJECT,
    SERVICE_ACCOUNT,
    fixture,
    mutating_calls,
    register_mutations,
    register_project
)

SETTINGS = InitSettings( project = PROJECT, region = DEFAULT_REGION, bucket = BUCKET )

def always_yes( pending: list[ str ] ) -> bool:
    return True

def never( pending: list[ str ] ) -> bool:
    raise AssertionError( f"should not have asked for confirmation: {pending}" )

def make_gcloud( fake: FakeGcloud, *, dry_run: bool = False, lines: list[ str ] | None = None ) -> Gcloud:
    echo = lines.append if lines is not None else print
    return Gcloud( project = PROJECT, dry_run = dry_run, runner = fake, echo = echo )

def test_names_are_derived_from_the_project() -> None:
    assert default_bucket( PROJECT ) == BUCKET
    assert service_account_email( PROJECT ) == SERVICE_ACCOUNT
    assert DEFAULT_REGION == "asia-southeast1"

def test_project_comes_from_the_flag_without_calling_gcloud() -> None:
    fake = FakeGcloud()
    assert resolve_project( Gcloud( runner = fake ), "from-flag" ) == "from-flag"
    assert fake.calls == []

def test_project_falls_back_to_the_gcloud_config() -> None:
    fake = FakeGcloud()
    fake.register( [ "config", "get-value", "project" ], stdout = "my-project\n" )
    assert resolve_project( Gcloud( runner = fake ), None ) == "my-project"
    assert fake.calls == [ [ "gcloud", "config", "get-value", "project", "--quiet" ] ]

def test_an_unset_project_is_a_usage_error() -> None:
    fake = FakeGcloud()
    fake.register( [ "config", "get-value", "project" ], stdout = "" )
    with pytest.raises( UsageError, match = "--project" ):
        resolve_project( Gcloud( runner = fake ), None )

def test_a_fresh_project_runs_every_step_in_order() -> None:
    fake = FakeGcloud()
    register_project(
        fake,
        has_apis = False,
        has_bucket = False,
        has_service_account = False,
        has_role = False,
        has_project_binding = False
    )
    register_mutations( fake )
    steps = build_steps( make_gcloud( fake ), SETTINGS )
    results = run_steps( steps, always_yes, is_dry_run = False )
    assert [ result.status for result in results ] == [ "created" ] * len( results )
    assert mutating_calls( fake ) == [
        [ "gcloud", "services", "enable", "compute.googleapis.com", "--quiet", f"--project={PROJECT}" ],
        [ "gcloud", "services", "enable", "storage.googleapis.com", "--quiet", f"--project={PROJECT}" ],
        [
            "gcloud",
            "storage",
            "buckets",
            "create",
            f"gs://{BUCKET}",
            "--location=asia-southeast1",
            "--uniform-bucket-level-access",
            "--public-access-prevention",
            "--quiet",
            f"--project={PROJECT}"
        ],
        [
            "gcloud",
            "iam",
            "service-accounts",
            "create",
            "ml-runner-vm",
            "--display-name=ml-runner VMs",
            "--quiet",
            f"--project={PROJECT}"
        ],
        [
            "gcloud",
            "storage",
            "buckets",
            "add-iam-policy-binding",
            f"gs://{BUCKET}",
            f"--member=serviceAccount:{SERVICE_ACCOUNT}",
            "--role=roles/storage.objectAdmin",
            "--quiet",
            f"--project={PROJECT}"
        ],
        [
            "gcloud",
            "iam",
            "roles",
            "create",
            "mlRunnerSelfDelete",
            "--permissions=compute.instances.delete,compute.instances.get",
            "--stage=GA",
            "--title=ml-runner self delete",
            "--quiet",
            f"--project={PROJECT}"
        ],
        [
            "gcloud",
            "projects",
            "add-iam-policy-binding",
            PROJECT,
            f"--member=serviceAccount:{SERVICE_ACCOUNT}",
            f"--role=projects/{PROJECT}/roles/mlRunnerSelfDelete",
            f"--condition=expression={CONDITION['expression']},title={CONDITION['title']}",
            "--quiet",
            f"--project={PROJECT}"
        ]
    ]

def test_a_second_run_changes_nothing_and_does_not_ask() -> None:
    fake = FakeGcloud()
    register_project( fake )
    results = run_steps( build_steps( make_gcloud( fake ), SETTINGS ), never, is_dry_run = False )
    assert [ result.status for result in results ] == [ "exists" ] * len( results )
    assert mutating_calls( fake ) == []

def test_only_the_missing_service_account_is_created() -> None:
    fake = FakeGcloud()
    register_project( fake, has_service_account = False )
    register_mutations( fake )
    results = run_steps( build_steps( make_gcloud( fake ), SETTINGS ), always_yes, is_dry_run = False )
    created = [ result.name for result in results if result.status == "created" ]
    assert created == [ "Service account ml-runner-vm" ]
    assert len( mutating_calls( fake ) ) == 1

def test_a_missing_binding_on_an_existing_bucket_is_added() -> None:
    fake = FakeGcloud()
    register_project( fake, has_bucket_binding = False )
    register_mutations( fake )
    run_steps( build_steps( make_gcloud( fake ), SETTINGS ), always_yes, is_dry_run = False )
    assert [ call[ 3 ] for call in mutating_calls( fake ) ] == [ "add-iam-policy-binding" ]

# Real stderr seen right after creating the service account (IAM is eventually consistent).
PROPAGATING_STDERR = (
    f"ERROR: (gcloud.storage.buckets.add-iam-policy-binding) HTTPError 400: Service account {SERVICE_ACCOUNT} "
    "does not exist."
)

class SlowToPropagate:
    def __init__( self, fake: FakeGcloud, failures: int ) -> None:
        self.fake = fake
        self.failures_left = failures

    def __call__( self, argv: Sequence[ str ], *, timeout: float ) -> Completed:
        if "add-iam-policy-binding" in argv and self.failures_left > 0:
            self.failures_left -= 1
            return Completed( 1, "", PROPAGATING_STDERR )

        return self.fake( argv, timeout = timeout )

def test_a_binding_is_retried_while_the_new_service_account_propagates() -> None:
    fake = FakeGcloud()
    register_project( fake, has_bucket_binding = False )
    register_mutations( fake )
    runner = SlowToPropagate( fake, failures = 2 )
    pauses: list[ float ] = []
    gcloud = Gcloud( project = PROJECT, runner = runner, echo = print )
    steps = build_steps( gcloud, SETTINGS, sleep = pauses.append )
    run_steps( steps, always_yes, is_dry_run = False )
    assert len( pauses ) == 2
    assert [ call[ 3 ] for call in mutating_calls( fake ) ] == [ "add-iam-policy-binding" ]

def test_a_binding_gives_up_when_the_service_account_never_appears() -> None:
    fake = FakeGcloud()
    register_project( fake, has_bucket_binding = False )
    register_mutations( fake )
    runner = SlowToPropagate( fake, failures = PROPAGATION_ATTEMPTS + 5 )
    gcloud = Gcloud( project = PROJECT, runner = runner, echo = print )
    steps = build_steps( gcloud, SETTINGS, sleep = lambda seconds: None )
    with pytest.raises( MlRunnerError, match = "does not exist" ):
        run_steps( steps, always_yes, is_dry_run = False )

def test_declining_the_confirmation_changes_nothing() -> None:
    fake = FakeGcloud()
    register_project( fake, has_bucket = False )
    asked: list[ list[ str ] ] = []

    def decline( pending: list[ str ] ) -> bool:
        asked.append( pending )
        return False

    with pytest.raises( MlRunnerError, match = "Nothing was changed" ):
        run_steps( build_steps( make_gcloud( fake ), SETTINGS ), decline, is_dry_run = False )
    assert mutating_calls( fake ) == []
    assert asked and f"Bucket gs://{BUCKET}" in asked[ 0 ]

def test_dry_run_echoes_the_commands_and_runs_none_of_them() -> None:
    fake = FakeGcloud()
    lines: list[ str ] = []
    register_project( fake, has_bucket = False, has_service_account = False )
    steps = build_steps( make_gcloud( fake, dry_run = True, lines = lines ), SETTINGS )
    results = run_steps( steps, never, is_dry_run = True )
    assert mutating_calls( fake ) == []
    assert any( line.startswith( "gcloud storage buckets create" ) for line in lines )
    assert any( line.startswith( "gcloud iam service-accounts create" ) for line in lines )
    assert [ result.status for result in results if result.status != "exists" ] == [ "planned" ] * 3

def test_a_bucket_owned_by_someone_else_suggests_another_name() -> None:
    fake = FakeGcloud()
    register_project( fake, bucket_is_foreign = True )
    with pytest.raises( UsageError, match = "--bucket" ):
        run_steps( build_steps( make_gcloud( fake ), SETTINGS ), always_yes, is_dry_run = False )

def test_a_taken_bucket_name_found_at_create_time_suggests_another_name() -> None:
    fake = FakeGcloud()
    register_project( fake, has_bucket = False )
    fake.register( [ "storage", "buckets", "create" ], stderr = "ERROR: 409 bucket already exists", returncode = 1 )
    with pytest.raises( UsageError, match = "--bucket" ):
        run_steps( build_steps( make_gcloud( fake ), SETTINGS ), always_yes, is_dry_run = False )

def test_an_unknown_project_is_a_clear_error() -> None:
    fake = FakeGcloud()
    fake.register( [ "projects", "describe" ], stderr = "ERROR: NOT_FOUND: project not found", returncode = 1 )
    with pytest.raises( UsageError, match = PROJECT ):
        build_steps( make_gcloud( fake ), SETTINGS )

def zone_fake() -> FakeGcloud:
    fake = FakeGcloud()
    fake.register_json( [ "compute", "accelerator-types", "list" ], fixture( "accelerator_types" ) )

    return fake

def test_the_zone_is_the_first_alphabetical_one_offering_a_t4() -> None:
    fake = zone_fake()
    assert choose_zone( make_gcloud( fake ), DEFAULT_REGION, None ) == "asia-southeast1-a"

def test_the_zone_filter_is_not_anchored_because_gcloud_reports_the_zone_as_a_url() -> None:
    fake = zone_fake()
    choose_zone( make_gcloud( fake ), DEFAULT_REGION, None )
    assert "--filter=zone~asia-southeast1-" in fake.calls[ 0 ]

def test_an_explicit_zone_is_used_without_calling_gcloud() -> None:
    fake = FakeGcloud()
    assert choose_zone( make_gcloud( fake ), DEFAULT_REGION, "asia-southeast1-c" ) == "asia-southeast1-c"
    assert fake.calls == []

def test_an_explicit_zone_outside_the_region_is_a_usage_error() -> None:
    with pytest.raises( UsageError, match = "europe-west4-a" ):
        choose_zone( make_gcloud( FakeGcloud() ), DEFAULT_REGION, "europe-west4-a" )

def test_a_region_with_no_t4_zone_is_a_usage_error() -> None:
    fake = FakeGcloud()
    fake.register_json( [ "compute", "accelerator-types", "list" ], [] )
    with pytest.raises( UsageError, match = "--region" ):
        choose_zone( make_gcloud( fake ), DEFAULT_REGION, None )

def test_the_config_is_built_from_the_settings() -> None:
    config = build_config( SETTINGS, "asia-southeast1-b", None )
    assert ( config.project, config.region, config.zone, config.bucket ) == (
        PROJECT,
        DEFAULT_REGION,
        "asia-southeast1-b",
        BUCKET
    )
    assert config.service_account == SERVICE_ACCOUNT
    assert config.defaults == Defaults()

def test_an_existing_configs_defaults_are_kept() -> None:
    existing = build_config( SETTINGS, "asia-southeast1-c", None ).model_copy(
        update = { "defaults": Defaults( gpu = "l4" ) }
    )
    config = build_config( SETTINGS, "asia-southeast1-b", existing )
    assert config.defaults.gpu == "l4"
    assert config.zone == "asia-southeast1-b"
