import hashlib
import subprocess
import tarfile
from pathlib import Path

import pytest

from ml_runner.errors import UsageError
from ml_runner.gcloud import Gcloud
from ml_runner.source import LARGEST_FILES_SHOWN, Snapshot, build_snapshot, upload_snapshot
from tests.fakes.fake_gcloud import FakeGcloud
from tests.fakes.project import BUCKET, PROJECT, mutating_calls, stderr

BIG_LIMIT_MB = 200
MEGABYTE = 1024 * 1024

def write( root: Path, relative: str, text: str = "x" ) -> Path:
    path = root / relative
    path.parent.mkdir( parents = True, exist_ok = True )
    path.write_text( text )

    return path

def git( cwd: Path, *args: str ) -> None:
    subprocess.run(
        [ "git", "-c", "user.name=tester", "-c", "user.email=tester@example.com", "-c", "commit.gpgsign=false", *args ],
        cwd = cwd,
        check = True,
        capture_output = True
    )

def make_repo( root: Path ) -> Path:
    git( root, "init", "-q" )
    write( root, "pyproject.toml" )
    write( root, "train.py", "print( 1 )" )
    write( root, ".gitignore", "ignored.txt\n" )
    git( root, "add", "-A" )
    git( root, "commit", "-q", "-m", "first" )

    return root

def members( snapshot: Snapshot ) -> list[ str ]:
    with tarfile.open( snapshot.archive ) as tar:
        return tar.getnames()

def build( src: Path, work: Path, limit: int = BIG_LIMIT_MB ) -> Snapshot:
    work.mkdir( exist_ok = True )

    return build_snapshot( src, limit, work )

def test_a_git_snapshot_has_tracked_and_untracked_files_but_not_ignored_ones_or_git( tmp_path: Path ) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    make_repo( repo )
    write( repo, "new.py" )
    write( repo, "ignored.txt" )
    write( repo, "pkg/mod.py" )
    snapshot = build( repo, tmp_path / "work" )
    assert members( snapshot ) == [ ".gitignore", "new.py", "pkg/mod.py", "pyproject.toml", "train.py" ]

def test_a_tracked_file_deleted_from_disk_is_skipped( tmp_path: Path ) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    make_repo( repo )
    ( repo / "train.py" ).unlink()
    assert "train.py" not in members( build( repo, tmp_path / "work" ) )

def test_a_git_snapshot_records_the_commit_and_whether_the_tree_is_dirty( tmp_path: Path ) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    make_repo( repo )
    head = subprocess.run(
        [ "git", "rev-parse", "HEAD" ],
        cwd = repo,
        check = True,
        capture_output = True,
        text = True
    ).stdout.strip()
    clean = build( repo, tmp_path / "work1" )
    assert ( clean.git_commit, clean.git_dirty ) == ( head, False )
    write( repo, "train.py", "print( 2 )" )
    dirty = build( repo, tmp_path / "work2" )
    assert ( dirty.git_commit, dirty.git_dirty ) == ( head, True )

def test_a_source_folder_below_the_repo_root_is_archived_relative_to_itself( tmp_path: Path ) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    make_repo( repo )
    write( repo, "sub/pyproject.toml" )
    write( repo, "sub/run.py" )
    write( repo, "other/skipped.py" )
    git( repo, "add", "-A" )
    assert members( build( repo / "sub", tmp_path / "work" ) ) == [ "pyproject.toml", "run.py" ]

def test_a_folder_outside_git_skips_the_default_excludes( tmp_path: Path ) -> None:
    src = tmp_path / "plain"
    write( src, "requirements.txt" )
    write( src, "main.py" )
    for junk in ( ".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache", "mlr-runs", "a/__pycache__" ):
        write( src, f"{junk}/junk.txt" )
    snapshot = build( src, tmp_path / "work" )
    assert members( snapshot ) == [ "main.py", "requirements.txt" ]
    assert ( snapshot.git_commit, snapshot.git_dirty ) == ( None, None )

def test_a_source_without_a_project_file_is_refused_before_anything_is_built( tmp_path: Path ) -> None:
    src = tmp_path / "bare"
    write( src, "main.py" )
    work = tmp_path / "work"
    with pytest.raises( UsageError, match = "pyproject.toml" ):
        build( src, work )
    assert list( work.iterdir() ) == []

def test_a_missing_source_folder_is_a_usage_error( tmp_path: Path ) -> None:
    with pytest.raises( UsageError, match = "nowhere" ):
        build( tmp_path / "nowhere", tmp_path / "work" )

def test_identical_trees_give_identical_archives_and_hashes( tmp_path: Path ) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    for root in ( first, second ):
        write( root, "pyproject.toml" )
        write( root, "a/b.py", "same" )
    first_snapshot = build( first, tmp_path / "work1" )
    second_snapshot = build( second, tmp_path / "work2" )
    assert first_snapshot.sha256 == second_snapshot.sha256
    assert first_snapshot.archive.read_bytes() == second_snapshot.archive.read_bytes()

def test_the_hash_is_the_sha256_of_the_archive_and_changes_with_content( tmp_path: Path ) -> None:
    src = tmp_path / "src"
    write( src, "pyproject.toml" )
    write( src, "a.py", "one" )
    before = build( src, tmp_path / "work1" )
    assert before.sha256 == hashlib.sha256( before.archive.read_bytes() ).hexdigest()
    assert before.size_bytes == before.archive.stat().st_size
    write( src, "a.py", "two" )
    assert build( src, tmp_path / "work2" ).sha256 != before.sha256

def test_archive_entries_are_normalised_but_keep_the_mode( tmp_path: Path ) -> None:
    src = tmp_path / "src"
    write( src, "pyproject.toml" )
    script = write( src, "run.sh", "#!/bin/sh" )
    script.chmod( 0o755 )
    snapshot = build( src, tmp_path / "work" )
    with tarfile.open( snapshot.archive ) as tar:
        info = tar.getmember( "run.sh" )
    assert ( info.mtime, info.uid, info.gid, info.uname, info.gname ) == ( 0, 0, 0, "", "" )
    assert info.mode & 0o777 == 0o755

def test_a_symlink_is_archived_as_a_symlink( tmp_path: Path ) -> None:
    src = tmp_path / "src"
    write( src, "pyproject.toml" )
    ( src / "link.toml" ).symlink_to( "pyproject.toml" )
    with tarfile.open( build( src, tmp_path / "work" ).archive ) as tar:
        assert tar.getmember( "link.toml" ).issym()

def test_a_snapshot_over_the_limit_is_refused_with_the_largest_files( tmp_path: Path ) -> None:
    src = tmp_path / "src"
    write( src, "pyproject.toml" )
    for index in range( LARGEST_FILES_SHOWN + 2 ):
        write( src, f"data/file{index:02d}.bin", "x" * ( MEGABYTE // 4 + index ) )
    write( src, "data/huge.bin", "x" * ( 2 * MEGABYTE ) )
    with pytest.raises( UsageError ) as raised:
        build( src, tmp_path / "work", limit = 1 )
    message = raised.value.message
    assert "--max-source-mb" in message
    assert message.splitlines()[ 1 ].startswith( "  data/huge.bin" )
    assert len( [ line for line in message.splitlines() if line.startswith( "  data/" ) ] ) == LARGEST_FILES_SHOWN
    assert not ( tmp_path / "work" / "snapshot.tar.gz" ).exists()

def make_snapshot( tmp_path: Path ) -> Snapshot:
    src = tmp_path / "src"
    write( src, "pyproject.toml" )

    return build( src, tmp_path / "work" )

def make_gcloud( fake: FakeGcloud, lines: list[ str ] | None = None, *, dry_run: bool = False ) -> Gcloud:
    echo = lines.append if lines is not None else print

    return Gcloud( project = PROJECT, dry_run = dry_run, runner = fake, echo = echo )

def test_an_existing_snapshot_is_not_uploaded_again( tmp_path: Path ) -> None:
    snapshot = make_snapshot( tmp_path )
    fake = FakeGcloud()
    fake.register_json( [ "storage", "objects", "describe" ], { "name": "sources/x" } )
    assert upload_snapshot( make_gcloud( fake ), BUCKET, snapshot ) is False
    assert mutating_calls( fake ) == []

def test_a_missing_snapshot_is_uploaded_once_to_its_hash_path( tmp_path: Path ) -> None:
    snapshot = make_snapshot( tmp_path )
    fake = FakeGcloud()
    fake.register( [ "storage", "objects", "describe" ], stderr = stderr( "not_found_object" ), returncode = 1 )
    fake.register( [ "storage", "cp" ] )
    assert upload_snapshot( make_gcloud( fake ), BUCKET, snapshot ) is True
    assert mutating_calls( fake ) == [
        [
            "gcloud",
            "storage",
            "cp",
            str( snapshot.archive ),
            f"gs://{BUCKET}/sources/{snapshot.sha256}.tar.gz",
            "--quiet",
            f"--project={PROJECT}"
        ]
    ]

def test_other_errors_while_checking_for_the_snapshot_are_raised( tmp_path: Path ) -> None:
    snapshot = make_snapshot( tmp_path )
    fake = FakeGcloud()
    fake.register( [ "storage", "objects", "describe" ], stderr = stderr( "permission_denied" ), returncode = 1 )
    with pytest.raises( Exception, match = "Permission denied" ):
        upload_snapshot( make_gcloud( fake ), BUCKET, snapshot )

def test_a_dry_run_upload_echoes_the_command_and_runs_nothing( tmp_path: Path ) -> None:
    snapshot = make_snapshot( tmp_path )
    fake = FakeGcloud()
    fake.register( [ "storage", "objects", "describe" ], stderr = stderr( "not_found_object" ), returncode = 1 )
    lines: list[ str ] = []
    assert upload_snapshot( make_gcloud( fake, lines, dry_run = True ), BUCKET, snapshot ) is True
    assert mutating_calls( fake ) == []
    assert len( lines ) == 1 and lines[ 0 ].startswith( "gcloud storage cp " )
