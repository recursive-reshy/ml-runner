import re
from pathlib import Path

from ml_runner import layout

B = "my-bucket"
SHA = "a" * 64


def test_every_path_of_the_trd_layout() -> None:
    assert layout.source(B, SHA) == f"gs://my-bucket/sources/{SHA}.tar.gz"
    assert layout.job_json(B, "J") == "gs://my-bucket/jobs/J/job.json"
    assert layout.job_cancel(B, "J") == "gs://my-bucket/jobs/J/cancel"
    assert layout.run_dir(B, "exp", "R") == "gs://my-bucket/runs/exp/R"
    assert layout.run_json(B, "exp", "R") == "gs://my-bucket/runs/exp/R/run.json"
    assert layout.run_log(B, "exp", "R") == "gs://my-bucket/runs/exp/R/run.log"
    assert layout.outputs(B, "exp", "R") == "gs://my-bucket/runs/exp/R/outputs"
    assert layout.checkpoints(B, "exp", "R") == "gs://my-bucket/runs/exp/R/checkpoints"


def test_listing_prefixes() -> None:
    assert layout.bucket_uri(B) == "gs://my-bucket"
    assert layout.runs_root(B) == "gs://my-bucket/runs"
    assert layout.experiment_dir(B, "exp") == "gs://my-bucket/runs/exp"
    assert layout.jobs_root(B) == "gs://my-bucket/jobs"


def test_no_trailing_slashes() -> None:
    paths = [
        layout.run_dir(B, "e", "r"),
        layout.outputs(B, "e", "r"),
        layout.runs_root(B),
        layout.bucket_uri(B),
    ]
    assert not any(p.endswith("/") for p in paths)


def test_gs_paths_are_built_only_in_layout_py() -> None:
    src = Path(layout.__file__).parent
    offenders = [
        str(path.relative_to(src))
        for path in src.rglob("*")
        if path.is_file()
        and path.name != "layout.py"
        and path.suffix in {".py", ".sh"}
        and re.search(r"gs://", path.read_text())
    ]
    assert offenders == []
