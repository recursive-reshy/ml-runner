import random
import re
from datetime import UTC, datetime
from urllib.parse import urlsplit

SLUG_MAX = 30
FALLBACK_SLUG = "run"
VM_NAME_MAX = 63
RAND_DIGITS = 4
_VM_NAME = re.compile( r"[a-z]([-a-z0-9]*[a-z0-9])?" )

# Pure: the clock and the RNG are passed in so tests can fix them.
def timestamp( now: datetime ) -> str:
    return now.astimezone( UTC ).strftime( "%Y%m%d-%H%M%S" )

def _rand( rng: random.Random | None ) -> str:
    source = rng or random.SystemRandom()
    value = source.randrange( 16**RAND_DIGITS )

    return f"{value:0{RAND_DIGITS}x}"

def slugify( text: str, max_len: int = SLUG_MAX ) -> str:
    slug = re.sub( r"[^a-z0-9]+", "-", text.lower() ).strip( "-" )
    slug = slug[ : max_len ].strip( "-" )

    return slug or FALLBACK_SLUG

def job_id( now: datetime | None = None, rng: random.Random | None = None ) -> str:
    return f"{timestamp(now or datetime.now(UTC))}-{_rand(rng)}"

def vm_name( job: str ) -> str:
    name = f"mlr-{job}"
    if len( name ) > VM_NAME_MAX or not _VM_NAME.fullmatch( name ):
        raise ValueError( f"not a valid VM name: {name!r}" )

    return name

# The slug comes from `--name` or the command.
def run_id( name_or_command: str, now: datetime | None = None, rng: random.Random | None = None ) -> str:
    stamp = timestamp( now or datetime.now( UTC ) )

    return f"{stamp}-{slugify(name_or_command)}-{_rand(rng)}"

def _repo_name( url: str ) -> str:
    segment = urlsplit( url ).path.rstrip( "/" ).rsplit( "/", 1 )[ -1 ]

    return segment.removesuffix( ".git" )

# `--experiment`, else the repo name, else the snapshot folder name.
def experiment_name( explicit: str | None, repo_url: str | None, folder_name: str ) -> str:
    if explicit:
        return slugify( explicit )
    if repo_url:
        return slugify( _repo_name( repo_url ) )

    return slugify( folder_name )
