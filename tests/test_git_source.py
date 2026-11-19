from pathlib import Path

import pytest

from ml_runner.errors import UsageError
from ml_runner.source import GitSource, check_git_source, ls_remote, ref_exists, validate_source_flags
from tests.test_source import git, make_repo

URL = "https://github.com/example/project"
SHA = "0123456789abcdef0123456789abcdef01234567"

class RecordingRunner:
    def __init__( self, answer: str ) -> None:
        self.answer = answer
        self.calls: list[ tuple[ str, str ] ] = []

    def __call__( self, url: str, ref: str ) -> str:
        self.calls.append( ( url, ref ) )

        return self.answer

def make_remote( tmp_path: Path ) -> Path:
    repo = tmp_path / "remote"
    repo.mkdir()
    make_repo( repo )
    git( repo, "branch", "-M", "main" )
    git( repo, "tag", "v1" )
    git( repo, "tag", "-a", "v2", "-m", "annotated" )

    return repo

def test_a_branch_a_tag_and_an_annotated_tag_exist_in_a_local_repo( tmp_path: Path ) -> None:
    remote = str( make_remote( tmp_path ) )
    assert ref_exists( remote, "main" )
    assert ref_exists( remote, "v1" )
    assert ref_exists( remote, "v2" )

def test_a_missing_ref_does_not_exist( tmp_path: Path ) -> None:
    assert not ref_exists( str( make_remote( tmp_path ) ), "no-such-branch" )

def test_an_unreadable_repository_is_a_usage_error_not_a_hang( tmp_path: Path ) -> None:
    with pytest.raises( UsageError, match = "Could not read" ):
        ls_remote( str( tmp_path / "nowhere" ), "main" )

def test_a_found_ref_gives_a_git_source() -> None:
    runner = RecordingRunner( f"{SHA}\trefs/heads/main\n" )
    assert check_git_source( URL, "main", runner = runner ) == GitSource( repo = URL, ref = "main" )
    assert runner.calls == [ ( URL, "main" ) ]

def test_a_missing_ref_is_refused_and_named() -> None:
    with pytest.raises( UsageError, match = "no-such-ref" ):
        check_git_source( URL, "no-such-ref", runner = RecordingRunner( "" ) )

def test_a_full_sha_is_accepted_without_asking_git() -> None:
    runner = RecordingRunner( "" )
    assert check_git_source( URL, SHA, runner = runner ) == GitSource( repo = URL, ref = SHA )
    assert runner.calls == []

def test_an_abbreviated_sha_is_checked_like_any_ref() -> None:
    runner = RecordingRunner( "" )
    with pytest.raises( UsageError ):
        check_git_source( URL, SHA[ :7 ], runner = runner )
    assert runner.calls == [ ( URL, SHA[ :7 ] ) ]

@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/example/project",
        "git@github.com:example/project.git",
        "ssh://git@github.com/example/project",
        "file:///tmp/project",
        "/tmp/project",
        "github.com/example/project"
    ]
)
def test_only_https_urls_are_accepted( url: str ) -> None:
    runner = RecordingRunner( f"{SHA}\trefs/heads/main\n" )
    with pytest.raises( UsageError, match = "https" ):
        check_git_source( url, "main", runner = runner )
    assert runner.calls == []

def test_repo_and_src_together_are_refused() -> None:
    with pytest.raises( UsageError, match = "--repo" ):
        validate_source_flags( src = Path( "." ), repo = URL, ref = "main" )

def test_a_repo_needs_a_ref() -> None:
    with pytest.raises( UsageError, match = "--ref" ):
        validate_source_flags( src = None, repo = URL, ref = None )

def test_a_ref_needs_a_repo() -> None:
    with pytest.raises( UsageError, match = "--repo" ):
        validate_source_flags( src = None, repo = None, ref = "main" )

def test_valid_flag_combinations_pass() -> None:
    validate_source_flags( src = None, repo = None, ref = None )
    validate_source_flags( src = Path( "." ), repo = None, ref = None )
    validate_source_flags( src = None, repo = URL, ref = "main" )
