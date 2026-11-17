import random
import re
from datetime import UTC, datetime, timedelta, timezone

import pytest

from ml_runner.ids import (
    experiment_name,
    job_id,
    run_id,
    slugify,
    timestamp,
    vm_name,
)

NOW = datetime(2026, 11, 17, 9, 5, 3, tzinfo=UTC)
RFC1035 = re.compile(r"^[a-z]([-a-z0-9]*[a-z0-9])?$")


def test_timestamp_is_utc_compact() -> None:
    assert timestamp(NOW) == "20261117-090503"


def test_timestamp_converts_other_zones_to_utc() -> None:
    plus8 = datetime(2026, 11, 17, 17, 5, 3, tzinfo=timezone(timedelta(hours=8)))
    assert timestamp(plus8) == "20261117-090503"


def test_job_id_is_timestamp_and_four_hex() -> None:
    value = job_id(NOW, random.Random(1))
    assert re.fullmatch(r"20261117-090503-[0-9a-f]{4}", value)


def test_job_id_is_deterministic_for_a_seeded_rng() -> None:
    assert job_id(NOW, random.Random(7)) == job_id(NOW, random.Random(7))


def test_job_id_default_clock_and_rng_work() -> None:
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{4}", job_id())


def test_vm_name_prefix_length_and_rfc1035() -> None:
    name = vm_name(job_id(NOW, random.Random(1)))
    assert name.startswith("mlr-20261117-090503-")
    assert len(name) <= 63
    assert RFC1035.match(name)


def test_vm_name_rejects_ids_that_would_break_the_limit() -> None:
    with pytest.raises(ValueError):
        vm_name("x" * 60)
    with pytest.raises(ValueError):
        vm_name("Bad_ID")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("My Experiment", "my-experiment"),
        ("python train.py --seed 42", "python-train-py-seed-42"),
        ("  --weird__name!!  ", "weird-name"),
        ("ÄÖ-résumé", "r-sum"),
        ("a" * 50, "a" * 30),
        ("a" * 29 + "-b", "a" * 29),  # truncation must not leave a trailing '-'
        ("", "run"),
        ("!!!", "run"),
    ],
)
def test_slugify(text: str, expected: str) -> None:
    assert slugify(text) == expected


def test_slugify_output_is_always_valid() -> None:
    for text in ["A B", "x--y", "-lead", "trail-", "a/b\\c", "1 2 3"]:
        slug = slugify(text)
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", slug)
        assert len(slug) <= 30


def test_run_id_uses_name_else_command() -> None:
    rng = random.Random(1)
    named = run_id("baseline", NOW, rng)
    assert re.fullmatch(r"20261117-090503-baseline-[0-9a-f]{4}", named)
    cmd = run_id("python train.py --seed 42", NOW, random.Random(1))
    assert re.fullmatch(r"20261117-090503-python-train-py-seed-42-[0-9a-f]{4}", cmd)


def test_run_id_is_bounded() -> None:
    assert len(run_id("x" * 200, NOW, random.Random(1))) <= 15 + 1 + 30 + 1 + 4


def test_experiment_prefers_explicit_name() -> None:
    assert experiment_name("My Exp", "https://github.com/o/r.git", "folder") == "my-exp"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/owner/my-repo.git", "my-repo"),
        ("https://github.com/owner/my-repo", "my-repo"),
        ("https://github.com/owner/my-repo/", "my-repo"),
        ("https://github.com/owner/My_Repo.git?x=1#frag", "my-repo"),
    ],
)
def test_experiment_from_repo_url(url: str, expected: str) -> None:
    assert experiment_name(None, url, "ignored") == expected


def test_experiment_from_snapshot_folder() -> None:
    assert experiment_name(None, None, "My Project") == "my-project"


def test_experiment_falls_back_when_nothing_usable() -> None:
    assert experiment_name(None, None, "") == "run"
