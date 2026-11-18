import json
import re
import shlex
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ml_runner.errors import MlRunnerError

# The only module that runs `gcloud` (ADR 0003, TRD §4).
DEFAULT_TIMEOUT = 120
COPY_TIMEOUT = 30 * 60
COPY_COMMANDS = ( "cp", "rsync" )

ErrorKind = Literal[
    "not_installed",
    "not_authenticated",
    "api_disabled",
    "permission_denied",
    "not_found",
    "already_exists",
    "quota_exceeded",
    "zone_exhausted",
    "unknown"
]

# The result of one gcloud process.
@dataclass( frozen = True )
class Completed:
    returncode: int
    stdout: str
    stderr: str

# Runs an argv and returns its result; `subprocess.run` in production.
class Runner( Protocol ):
    def __call__( self, argv: Sequence[ str ], *, timeout: float ) -> Completed: ...

def subprocess_runner( argv: Sequence[ str ], *, timeout: float ) -> Completed:
    done = subprocess.run( list( argv ), capture_output = True, text = True, timeout = timeout, check = False )

    return Completed( done.returncode, done.stdout, done.stderr )

# A failed gcloud call, classified by `kind`.
class GcloudError( MlRunnerError ):
    def __init__( self, kind: ErrorKind, message: str, stderr: str = "" ) -> None:
        super().__init__( message, detail = stderr )
        self.kind: ErrorKind = kind
        self.stderr = stderr

# Order matters: the first match wins (api_disabled also says PERMISSION_DENIED).
# Patterns are pinned by tests/fakes/gcloud/stderr/*.txt (provisional until 2.1).
_PATTERNS: tuple[ tuple[ ErrorKind, re.Pattern[ str ] ], ... ] = (
    ( "not_authenticated", re.compile( r"do not currently have an active account|gcloud auth login", re.IGNORECASE ) ),
    (
        "api_disabled",
        re.compile( r"has not been used in project|api .*is disabled|SERVICE_DISABLED", re.IGNORECASE )
    ),
    ( "quota_exceeded", re.compile( r"quota .*exceeded|QUOTA_EXCEEDED", re.IGNORECASE ) ),
    ( "zone_exhausted", re.compile( r"ZONE_RESOURCE_POOL_EXHAUSTED|does not have enough resources", re.IGNORECASE ) ),
    ( "already_exists", re.compile( r"already exists|alreadyExists", re.IGNORECASE ) ),
    ( "not_found", re.compile( r"was not found|not found|notFound", re.IGNORECASE ) ),
    ( "permission_denied", re.compile( r"PERMISSION_DENIED|permission .*denied|forbidden|403", re.IGNORECASE ) )
)

_MESSAGES: dict[ ErrorKind, str ] = {
    "not_installed": "gcloud is not installed or not on PATH. Install the Google Cloud CLI.",
    "not_authenticated": "You are not signed in to gcloud. Run: gcloud auth login",
    "api_disabled": "A required Google Cloud API is not enabled. Run: ml-runner init",
    "permission_denied": "Permission denied by Google Cloud. Check your IAM roles.",
    "not_found": "Google Cloud could not find the resource.",
    "already_exists": "The resource already exists.",
    "quota_exceeded": "A Google Cloud quota is exceeded. Run: ml-runner doctor",
    "zone_exhausted": "The zone has no capacity for this machine right now."
}

def classify_error( returncode: int, stderr: str ) -> ErrorKind:
    for kind, pattern in _PATTERNS:
        if pattern.search( stderr ):
            return kind

    return "unknown"

def _error_for( returncode: int, stderr: str ) -> GcloudError:
    kind = classify_error( returncode, stderr )
    if kind != "unknown":
        return GcloudError( kind, _MESSAGES[ kind ], stderr )

    last_line = stderr.strip().splitlines()[ -1 ] if stderr.strip() else "no output"

    return GcloudError( kind, f"gcloud failed (exit {returncode}): {last_line}", stderr )

def _timeout_for( args: Sequence[ str ] ) -> int:
    if len( args ) >= 2 and args[ 0 ] == "storage" and args[ 1 ] in COPY_COMMANDS:
        return COPY_TIMEOUT

    return DEFAULT_TIMEOUT

# Runs the gcloud CLI. In dry-run mode, mutating calls are echoed, not run.
class Gcloud:
    def __init__(
        self,
        *,
        project: str | None = None,
        dry_run: bool = False,
        runner: Runner | None = None,
        echo: Callable[ [ str ], None ] | None = None
    ) -> None:
        self._project = project
        self._dry_run = dry_run
        self._runner: Runner = runner or subprocess_runner
        self._echo: Callable[ [ str ], None ] = echo or print

    def _argv( self, args: Sequence[ str ], *, json_output: bool, with_project: bool = True ) -> list[ str ]:
        argv = [ "gcloud", *args, "--quiet" ]
        if json_output:
            argv.append( "--format=json" )
        if with_project and self._project:
            argv.append( f"--project={self._project}" )

        return argv

    def _exec( self, argv: list[ str ], args: Sequence[ str ] ) -> str:
        try:
            done = self._runner( argv, timeout = _timeout_for( args ) )
        except FileNotFoundError as error:
            raise GcloudError( "not_installed", _MESSAGES[ "not_installed" ], str( error ) ) from error
        except subprocess.TimeoutExpired as error:
            raise GcloudError( "unknown", f"gcloud timed out: {shlex.join(argv)}", "" ) from error

        if done.returncode != 0:
            raise _error_for( done.returncode, done.stderr )

        return done.stdout

    # Returns stdout. Mutating calls are echoed, not run, in dry-run.
    def run( self, *args: str, mutating: bool ) -> str:
        argv = self._argv( args, json_output = False )
        if mutating and self._dry_run:
            self._echo( shlex.join( argv ) )
            return ""

        return self._exec( argv, args )

    # A read-only call with --format=json; stdout is parsed.
    def json( self, *args: str ) -> Any:
        stdout = self._exec( self._argv( args, json_output = True ), args )

        return json.loads( stdout ) if stdout.strip() else None

    def version( self ) -> tuple[ int, int, int ]:
        args = ( "version", )
        argv = self._argv( args, json_output = True, with_project = False )
        data = json.loads( self._exec( argv, args ) )
        text = str( data[ "Google Cloud SDK" ] )
        major, minor, patch = ( int( part ) for part in text.split( "." )[ :3 ] )

        return ( major, minor, patch )
