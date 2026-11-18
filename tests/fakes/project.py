import json
from pathlib import Path

from tests.fakes.fake_gcloud import FakeGcloud

FIXTURES = Path( __file__ ).parent / "gcloud"
PROJECT = "my-project"
BUCKET = "ml-runner-my-project"
SERVICE_ACCOUNT = "ml-runner-vm@my-project.iam.gserviceaccount.com"
SELF_DELETE_ROLE = "projects/my-project/roles/mlRunnerSelfDelete"
CONDITION = {
    "expression": 'resource.name.extract("instances/{name}").startsWith("mlr-")',
    "title": "mlr-instances-only"
}
MUTATING_WORDS = { "create", "enable", "add-iam-policy-binding", "delete", "set-iam-policy" }

def fixture( name: str ) -> object:
    return json.loads( ( FIXTURES / f"{name}.json" ).read_text() )

def stderr( name: str ) -> str:
    return ( FIXTURES / "stderr" / f"{name}.txt" ).read_text()

def mutating_calls( fake: FakeGcloud ) -> list[ list[ str ] ]:
    return [ call for call in fake.calls if MUTATING_WORDS & set( call ) ]

# Registers read-only answers for a project in a chosen state. Anything not
# registered (a mutating call, for instance) fails the test as an unexpected call.
def register_project(
    fake: FakeGcloud,
    *,
    has_apis: bool = True,
    has_bucket: bool = True,
    has_service_account: bool = True,
    has_bucket_binding: bool = True,
    has_role: bool = True,
    has_project_binding: bool = True,
    bucket_is_foreign: bool = False
) -> None:
    fake.register_json( [ "projects", "describe" ], fixture( "projects_describe" ) )
    fake.register_json( [ "services", "list" ], fixture( "services_enabled" ) if has_apis else [] )

    if bucket_is_foreign:
        fake.register( [ "storage", "buckets", "describe" ], stderr = stderr( "permission_denied" ), returncode = 1 )
    elif has_bucket:
        fake.register_json( [ "storage", "buckets", "describe" ], { "name": BUCKET } )
    else:
        fake.register( [ "storage", "buckets", "describe" ], stderr = stderr( "not_found_bucket" ), returncode = 1 )

    if has_service_account:
        fake.register_json( [ "iam", "service-accounts", "describe" ], { "email": SERVICE_ACCOUNT } )
    else:
        fake.register(
            [ "iam", "service-accounts", "describe" ],
            stderr = stderr( "not_found_service_account" ),
            returncode = 1
        )

    if has_bucket and has_bucket_binding:
        admin = { "role": "roles/storage.objectAdmin", "members": [ f"serviceAccount:{SERVICE_ACCOUNT}" ] }
        bucket_policy = { "bindings": [ admin ] }
        fake.register_json( [ "storage", "buckets", "get-iam-policy" ], bucket_policy )
    elif has_bucket:
        fake.register_json( [ "storage", "buckets", "get-iam-policy" ], { "bindings": [] } )
    else:
        fake.register(
            [ "storage", "buckets", "get-iam-policy" ],
            stderr = stderr( "not_found_bucket" ),
            returncode = 1
        )

    if has_role:
        fake.register_json( [ "iam", "roles", "describe" ], { "name": SELF_DELETE_ROLE } )
    else:
        fake.register( [ "iam", "roles", "describe" ], stderr = stderr( "not_found_role" ), returncode = 1 )

    policy = fixture( "project_policy" )
    assert isinstance( policy, dict )
    if has_project_binding:
        binding = {
            "role": SELF_DELETE_ROLE,
            "members": [ f"serviceAccount:{SERVICE_ACCOUNT}" ],
            "condition": CONDITION
        }
        policy = { **policy, "bindings": [ *policy[ "bindings" ], binding ] }
    fake.register_json( [ "projects", "get-iam-policy" ], policy )

# Accepts every mutating call `init` makes; used by tests that let steps apply.
def register_mutations( fake: FakeGcloud ) -> None:
    for pattern in (
        [ "services", "enable" ],
        [ "storage", "buckets", "create" ],
        [ "iam", "service-accounts", "create" ],
        [ "storage", "buckets", "add-iam-policy-binding" ],
        [ "iam", "roles", "create" ],
        [ "projects", "add-iam-policy-binding" ]
    ):
        fake.register( pattern )

# The read-only answers `init` needs after the steps: zones offering a GPU, and quotas.
def register_zone_and_quotas( fake: FakeGcloud ) -> None:
    fake.register_json( [ "compute", "accelerator-types", "list" ], fixture( "accelerator_types" ) )
    fake.register_json( [ "compute", "regions", "describe" ], fixture( "region_describe" ) )
    fake.register_json( [ "compute", "project-info", "describe" ], fixture( "project_info" ) )

HEALTHY_PROJECT_QUOTAS = { "quotas": [ { "metric": "GPUS_ALL_REGIONS", "limit": 8.0, "usage": 0.0 } ] }

# Everything `doctor` reads, in a healthy state: the project of `register_project`, plus
# gcloud itself, sign-in, the network and quotas whose global limit is above zero.
def register_healthy( fake: FakeGcloud ) -> None:
    fake.register_json( [ "version" ], fixture( "version" ) )
    fake.register_json( [ "auth", "list" ], fixture( "auth_list" ) )
    fake.register_json( [ "compute", "networks", "describe" ], fixture( "network_default" ) )
    fake.register_json( [ "compute", "regions", "describe" ], fixture( "region_describe" ) )
    fake.register_json( [ "compute", "project-info", "describe" ], HEALTHY_PROJECT_QUOTAS )
    register_project( fake )
