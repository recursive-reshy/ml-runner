import pytest

from ml_runner.gcloud import Completed
from tests.fakes.fake_gcloud import FakeGcloud


def test_matches_by_argv_prefix_and_records_calls() -> None:
    fake = FakeGcloud()
    fake.register(["compute", "instances", "list"], stdout="[]")
    result = fake(["gcloud", "compute", "instances", "list", "--quiet"], timeout=1)
    assert result == Completed(0, "[]", "")
    assert fake.calls == [["gcloud", "compute", "instances", "list", "--quiet"]]


def test_wildcard_matches_one_argument() -> None:
    fake = FakeGcloud()
    fake.register(["compute", "instances", "describe", "*"], stdout="x")
    assert (
        fake(["gcloud", "compute", "instances", "describe", "mlr-1"], timeout=1).stdout
        == "x"
    )


def test_first_matching_rule_wins() -> None:
    fake = FakeGcloud()
    fake.register(["storage", "ls", "gs://b"], stdout="specific")
    fake.register(["storage", "ls"], stdout="general")
    assert fake(["gcloud", "storage", "ls", "gs://b"], timeout=1).stdout == "specific"


def test_register_json_and_exit_code() -> None:
    fake = FakeGcloud()
    fake.register_json(["projects", "describe"], {"projectId": "p"})
    fake.register(["boom"], stderr="bad", returncode=1)
    assert (
        fake(["gcloud", "projects", "describe", "p"], timeout=1).stdout
        == '{"projectId": "p"}'
    )
    assert fake(["gcloud", "boom"], timeout=1) == Completed(1, "", "bad")


def test_unmatched_call_fails_the_test_with_the_argv() -> None:
    fake = FakeGcloud()
    with pytest.raises(pytest.fail.Exception, match="compute instances delete mlr-1"):
        fake(["gcloud", "compute", "instances", "delete", "mlr-1"], timeout=1)
