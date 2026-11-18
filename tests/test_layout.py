import re
from pathlib import Path

from ml_runner import layout

BUCKET = "my-bucket"
SHA = "a" * 64

def test_every_path_of_the_trd_layout() -> None:
    assert layout.source( BUCKET, SHA ) == f"gs://my-bucket/sources/{SHA}.tar.gz"
    assert layout.job_json( BUCKET, "J" ) == "gs://my-bucket/jobs/J/job.json"
    assert layout.job_cancel( BUCKET, "J" ) == "gs://my-bucket/jobs/J/cancel"
    assert layout.run_dir( BUCKET, "exp", "R" ) == "gs://my-bucket/runs/exp/R"
    assert layout.run_json( BUCKET, "exp", "R" ) == "gs://my-bucket/runs/exp/R/run.json"
    assert layout.run_log( BUCKET, "exp", "R" ) == "gs://my-bucket/runs/exp/R/run.log"
    assert layout.outputs( BUCKET, "exp", "R" ) == "gs://my-bucket/runs/exp/R/outputs"
    assert layout.checkpoints( BUCKET, "exp", "R" ) == "gs://my-bucket/runs/exp/R/checkpoints"

def test_listing_prefixes() -> None:
    assert layout.bucket_uri( BUCKET ) == "gs://my-bucket"
    assert layout.runs_root( BUCKET ) == "gs://my-bucket/runs"
    assert layout.experiment_dir( BUCKET, "exp" ) == "gs://my-bucket/runs/exp"
    assert layout.jobs_root( BUCKET ) == "gs://my-bucket/jobs"

def test_no_trailing_slashes() -> None:
    paths = [
        layout.run_dir( BUCKET, "e", "r" ),
        layout.outputs( BUCKET, "e", "r" ),
        layout.runs_root( BUCKET ),
        layout.bucket_uri( BUCKET )
    ]
    assert not any( path.endswith( "/" ) for path in paths )

def test_gs_paths_are_built_only_in_layout_py() -> None:
    src = Path( layout.__file__ ).parent
    offenders = [
        str( path.relative_to( src ) )
        for path in src.rglob( "*" )
        if path.is_file()
        and path.name != "layout.py"
        and path.suffix in { ".py", ".sh" }
        and re.search( r"gs://", path.read_text() )
    ]
    assert offenders == []
