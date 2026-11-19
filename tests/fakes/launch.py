from collections.abc import Sequence

from ml_runner.config import Config, Defaults
from ml_runner.gcloud import Completed
from tests.fakes.fake_gcloud import FakeGcloud
from tests.fakes.project import BUCKET, PROJECT, SERVICE_ACCOUNT, fixture, stderr

COMMAND = [ "python", "train.py", "--seed", "42" ]
REPO = "https://github.com/example/nlp-emotion"

def make_config( defaults: Defaults | None = None ) -> Config:
    return Config(
        project = PROJECT,
        region = "asia-southeast1",
        zone = "asia-southeast1-b",
        bucket = BUCKET,
        service_account = SERVICE_ACCOUNT,
        defaults = defaults or Defaults()
    )

# What `launch` reads and writes: the accelerator zones, the snapshot check, uploads and the create.
def make_launch_fake() -> FakeGcloud:
    fake = FakeGcloud()
    fake.register_json( [ "compute", "accelerator-types", "list" ], fixture( "accelerator_types" ) )
    fake.register( [ "storage", "objects", "describe" ], stderr = stderr( "not_found_object" ), returncode = 1 )
    fake.register( [ "storage", "cp" ] )
    fake.register( [ "compute", "instances", "create" ] )

    return fake

def found_ref( url: str, ref: str ) -> str:
    return f"0123456789abcdef0123456789abcdef01234567\trefs/heads/{ref}\n"

def zone_of( argv: Sequence[ str ] ) -> str:
    return [ arg for arg in argv if arg.startswith( "--zone=" ) ][ 0 ].removeprefix( "--zone=" )

# Fails `instances create` in the given zones with a canned stderr (`zone_exhausted` or `quota_exceeded`).
class CapacityRunner:
    def __init__( self, fake: FakeGcloud, exhausted_zones: Sequence[ str ], kind: str = "zone_exhausted" ) -> None:
        self.fake = fake
        self.exhausted_zones = exhausted_zones
        self.kind = kind

    def __call__( self, argv: Sequence[ str ], *, timeout: float ) -> Completed:
        creating = argv[ 1:4 ] == [ "compute", "instances", "create" ]
        if creating and zone_of( argv ) in self.exhausted_zones:
            return Completed( 1, "", stderr( self.kind ) )

        return self.fake( argv, timeout = timeout )
