"""Bucket paths (TRD §7.2). The only module that builds a `gs://` path."""

_SCHEME = "gs://"


def bucket_uri(bucket: str) -> str:
    return f"{_SCHEME}{bucket}"


def source(bucket: str, sha256: str) -> str:
    return f"{bucket_uri(bucket)}/sources/{sha256}.tar.gz"


def jobs_root(bucket: str) -> str:
    return f"{bucket_uri(bucket)}/jobs"


def job_json(bucket: str, job: str) -> str:
    return f"{jobs_root(bucket)}/{job}/job.json"


def job_cancel(bucket: str, job: str) -> str:
    return f"{jobs_root(bucket)}/{job}/cancel"


def runs_root(bucket: str) -> str:
    return f"{bucket_uri(bucket)}/runs"


def experiment_dir(bucket: str, experiment: str) -> str:
    return f"{runs_root(bucket)}/{experiment}"


def run_dir(bucket: str, experiment: str, run: str) -> str:
    return f"{experiment_dir(bucket, experiment)}/{run}"


def run_json(bucket: str, experiment: str, run: str) -> str:
    return f"{run_dir(bucket, experiment, run)}/run.json"


def run_log(bucket: str, experiment: str, run: str) -> str:
    return f"{run_dir(bucket, experiment, run)}/run.log"


def outputs(bucket: str, experiment: str, run: str) -> str:
    return f"{run_dir(bucket, experiment, run)}/outputs"


def checkpoints(bucket: str, experiment: str, run: str) -> str:
    return f"{run_dir(bucket, experiment, run)}/checkpoints"
