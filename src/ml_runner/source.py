import gzip
import hashlib
import os
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

from ml_runner import layout
from ml_runner.errors import UsageError
from ml_runner.gcloud import Gcloud, GcloudError

# Outside git, these folders are skipped at any depth (TRD §9.1).
EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "mlr-runs"
    }
)
PROJECT_FILES = ( "pyproject.toml", "requirements.txt" )
LARGEST_FILES_SHOWN = 10
BYTES_PER_MB = 1024 * 1024
ARCHIVE_NAME = "snapshot.tar.gz"
GZIP_LEVEL = 6
READ_CHUNK_BYTES = BYTES_PER_MB

@dataclass( frozen = True )
class Snapshot:
    archive: Path
    sha256: str
    size_bytes: int
    file_count: int
    git_commit: str | None
    git_dirty: bool | None

def _git( directory: Path, *args: str ) -> subprocess.CompletedProcess[ bytes ] | None:
    try:
        return subprocess.run( [ "git", *args ], cwd = directory, capture_output = True, check = False )
    except FileNotFoundError:
        return None

def _is_git_work_tree( src: Path ) -> bool:
    done = _git( src, "rev-parse", "--is-inside-work-tree" )

    return done is not None and done.returncode == 0 and done.stdout.strip() == b"true"

# Tracked plus untracked-but-not-ignored, relative to `src`; files deleted from disk are dropped.
def _git_files( src: Path ) -> list[ str ]:
    done = _git( src, "ls-files", "-z", "--cached", "--others", "--exclude-standard" )
    if done is None or done.returncode != 0:
        raise UsageError( f"Could not list the files of {src} with git." )

    names = { name for name in done.stdout.decode().split( "\0" ) if name }

    return sorted( name for name in names if ( src / name ).is_symlink() or ( src / name ).is_file() )

def _walk_files( src: Path ) -> list[ str ]:
    names: list[ str ] = []
    for directory, subdirectories, files in os.walk( src ):
        subdirectories[ : ] = [ name for name in subdirectories if name not in EXCLUDED_DIRS ]
        folder = Path( directory ).relative_to( src )
        names.extend( ( folder / name ).as_posix() for name in files )

    return sorted( names )

def _git_state( src: Path ) -> tuple[ str | None, bool | None ]:
    head = _git( src, "rev-parse", "HEAD" )
    status = _git( src, "status", "--porcelain" )
    commit = head.stdout.decode().strip() if head is not None and head.returncode == 0 else None
    is_dirty = bool( status.stdout.strip() ) if status is not None and status.returncode == 0 else None

    return ( commit or None, is_dirty )

def _check_size( src: Path, files: list[ str ], max_source_mb: int ) -> None:
    sizes = { name: ( src / name ).lstat().st_size for name in files }
    total = sum( sizes.values() )
    if total <= max_source_mb * BYTES_PER_MB:
        return

    largest = sorted( sizes.items(), key = lambda item: item[ 1 ], reverse = True )[ :LARGEST_FILES_SHOWN ]
    lines = [ f"  {name} ({size / BYTES_PER_MB:.1f} MB)" for name, size in largest ]
    raise UsageError(
        "\n".join(
            [
                f"The source is {total / BYTES_PER_MB:.1f} MB, over the {max_source_mb} MB limit. Largest files:",
                *lines,
                "Download datasets in the training code, or raise the limit with --max-source-mb."
            ]
        )
    )

# Entries are sorted and stripped of owner and time, so identical trees give identical bytes.
def _write_archive( src: Path, files: list[ str ], archive: Path ) -> int:
    count = 0
    with (
        archive.open( "wb" ) as raw,
        gzip.GzipFile( filename = "", mode = "wb", fileobj = raw, compresslevel = GZIP_LEVEL, mtime = 0 ) as zipped,
        tarfile.open( fileobj = zipped, mode = "w" ) as tar
    ):
        for name in files:
            info = tar.gettarinfo( str( src / name ), arcname = name )
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            if info.isreg():
                with ( src / name ).open( "rb" ) as handle:
                    tar.addfile( info, handle )
            elif info.issym():
                tar.addfile( info )
            else:
                continue

            count += 1

    return count

def _sha256( path: Path ) -> str:
    digest = hashlib.sha256()
    with path.open( "rb" ) as handle:
        while chunk := handle.read( READ_CHUNK_BYTES ):
            digest.update( chunk )

    return digest.hexdigest()

# Refuses before building anything when `src` is not a project or is too big.
def build_snapshot( src: Path, max_source_mb: int, work_dir: Path ) -> Snapshot:
    if not src.is_dir():
        raise UsageError( f"The source folder {src} does not exist." )
    if not any( ( src / name ).is_file() for name in PROJECT_FILES ):
        raise UsageError( f"{src} has no {' or '.join( PROJECT_FILES )}, so there is nothing to install on the VM." )

    is_git = _is_git_work_tree( src )
    files = _git_files( src ) if is_git else _walk_files( src )
    _check_size( src, files, max_source_mb )
    commit, is_dirty = _git_state( src ) if is_git else ( None, None )
    archive = work_dir / ARCHIVE_NAME
    count = _write_archive( src, files, archive )

    return Snapshot( archive, _sha256( archive ), archive.stat().st_size, count, commit, is_dirty )

# Returns True when it uploaded; an identical snapshot already in the bucket is left alone.
def upload_snapshot( gcloud: Gcloud, bucket: str, snapshot: Snapshot ) -> bool:
    uri = layout.source( bucket, snapshot.sha256 )
    try:
        gcloud.json( "storage", "objects", "describe", uri )
        return False
    except GcloudError as error:
        if error.kind != "not_found":
            raise

    gcloud.run( "storage", "cp", str( snapshot.archive ), uri, mutating = True )

    return True
