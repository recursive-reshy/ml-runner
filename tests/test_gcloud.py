import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from ml_runner.gcloud import Completed, Gcloud, GcloudError, classify_error
from tests.fakes.fake_gcloud import FakeGcloud

STDERR = Path(__file__).parent / "fakes" / "gcloud" / "stderr"


def test_run_adds_quiet_and_project() -> None:
    fake = FakeGcloud()
    fake.register(["compute", "instances", "list"], stdout="NAME\n")
    gcloud = Gcloud(project="my-proj", runner=fake)
    assert gcloud.run("compute", "instances", "list", mutating=False) == "NAME\n"
    assert fake.calls == [
        ["gcloud", "compute", "instances", "list", "--quiet", "--project=my-proj"]
    ]


def test_no_project_flag_when_not_configured() -> None:
    fake = FakeGcloud()
    fake.register(["config"], stdout="p")
    Gcloud(runner=fake).run("config", "get-value", "project", mutating=False)
    assert fake.calls == [["gcloud", "config", "get-value", "project", "--quiet"]]


def test_json_adds_format_and_parses_stdout() -> None:
    fake = FakeGcloud()
    fake.register_json(["projects", "describe"], {"projectId": "p"})
    gcloud = Gcloud(project="p", runner=fake)
    assert gcloud.json("projects", "describe", "p") == {"projectId": "p"}
    assert "--format=json" in fake.calls[0]


def test_json_with_empty_stdout_is_none() -> None:
    fake = FakeGcloud()
    fake.register(["compute"], stdout="")
    assert Gcloud(runner=fake).json("compute", "x") is None


def test_dry_run_echoes_mutating_calls_and_does_not_run_them() -> None:
    fake = FakeGcloud()
    lines: list[str] = []
    gcloud = Gcloud(project="p", dry_run=True, runner=fake, echo=lines.append)
    out = gcloud.run("compute", "instances", "delete", "mlr-1", mutating=True)
    assert out == ""
    assert fake.calls == []
    assert lines == ["gcloud compute instances delete mlr-1 --quiet --project=p"]


def test_dry_run_echo_is_shell_quoted() -> None:
    lines: list[str] = []
    gcloud = Gcloud(dry_run=True, runner=FakeGcloud(), echo=lines.append)
    gcloud.run("x", "--metadata=a=b c", mutating=True)
    assert lines == ["gcloud x '--metadata=a=b c' --quiet"]


def test_dry_run_still_runs_read_only_calls() -> None:
    fake = FakeGcloud()
    fake.register(["compute", "regions"], stdout="[]")
    gcloud = Gcloud(dry_run=True, runner=fake, echo=lambda _: None)
    assert gcloud.json("compute", "regions", "list") == []
    assert len(fake.calls) == 1


def test_default_timeout_and_storage_copy_timeout() -> None:
    seen: list[float] = []

    def runner(argv: Sequence[str], *, timeout: float) -> Completed:
        seen.append(timeout)
        return Completed(0, "", "")

    gcloud = Gcloud(runner=runner)
    gcloud.run("compute", "instances", "list", mutating=False)
    gcloud.run("storage", "cp", "a", "gs://b/a", mutating=True)
    gcloud.run("storage", "rsync", "-r", "a", "gs://b/a", mutating=True)
    gcloud.run("storage", "ls", "gs://b", mutating=False)
    assert seen == [120, 1800, 1800, 120]


def test_version_parses_sdk_version() -> None:
    fake = FakeGcloud()
    fake.register_json(["version"], {"Google Cloud SDK": "512.3.0", "core": "x"})
    assert Gcloud(project="p", runner=fake).version() == (512, 3, 0)
    assert "--project=p" not in fake.calls[0]


def test_nonzero_exit_raises_classified_error() -> None:
    fake = FakeGcloud()
    fake.register(["compute"], stderr="ERROR: weird", returncode=1)
    with pytest.raises(GcloudError) as info:
        Gcloud(runner=fake).run("compute", "x", mutating=False)
    assert info.value.kind == "unknown"
    assert info.value.stderr == "ERROR: weird"
    assert "weird" in info.value.message


def test_gcloud_not_installed() -> None:
    def runner(argv: Sequence[str], *, timeout: float) -> Completed:
        raise FileNotFoundError("gcloud")

    with pytest.raises(GcloudError) as info:
        Gcloud(runner=runner).run("x", mutating=False)
    assert info.value.kind == "not_installed"


def test_timeout_is_reported_as_unknown_error() -> None:
    def runner(argv: Sequence[str], *, timeout: float) -> Completed:
        raise subprocess.TimeoutExpired(list(argv), timeout)

    with pytest.raises(GcloudError) as info:
        Gcloud(runner=runner).run("x", mutating=False)
    assert info.value.kind == "unknown"
    assert "timed out" in info.value.message


@pytest.mark.parametrize(
    "kind",
    [
        "not_authenticated",
        "api_disabled",
        "permission_denied",
        "not_found",
        "already_exists",
        "quota_exceeded",
        "zone_exhausted",
    ],
)
def test_stderr_fixtures_are_classified(kind: str) -> None:
    stderr = (STDERR / f"{kind}.txt").read_text()
    assert classify_error(1, stderr) == kind


def test_unrecognised_stderr_is_unknown() -> None:
    assert classify_error(1, "ERROR: something new") == "unknown"


def test_error_message_is_human_readable_per_kind() -> None:
    fake = FakeGcloud()
    fake.register(
        ["compute"],
        stderr=(STDERR / "not_authenticated.txt").read_text(),
        returncode=1,
    )
    with pytest.raises(GcloudError) as info:
        Gcloud(runner=fake).run("compute", "x", mutating=False)
    assert "gcloud auth login" in info.value.message


def test_json_output_is_valid_for_dict_fixture() -> None:
    # Guards the fake: register_json output round-trips through json.loads.
    fake = FakeGcloud()
    fake.register_json(["a"], [1, 2])
    assert json.loads(fake(["gcloud", "a"], timeout=1).stdout) == [1, 2]
