import json
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1
JSON_INDENT = 2

Gpu = Literal[ "t4", "l4", "none" ]

# `lost` is shown by `status` only (PRD §6.8): it is never stored.
class RunStatus( StrEnum ):
    PROVISIONING = "provisioning"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PREEMPTED = "preempted"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

class SourceKind( StrEnum ):
    SNAPSHOT = "snapshot"
    GIT = "git"

class _Record( BaseModel ):
    model_config = ConfigDict( extra = "forbid", populate_by_name = True )

    # `schema` would shadow a BaseModel attribute, so the field has another name and a JSON alias.
    schema_version: Literal[ 1 ] = Field( default = SCHEMA_VERSION, alias = "schema" )

    def to_json( self ) -> str:
        return self.model_dump_json( by_alias = True, indent = JSON_INDENT )

    @classmethod
    def from_json( cls, text: str ) -> Self:
        return cls.model_validate_json( text )

class Source( BaseModel ):
    model_config = ConfigDict( extra = "forbid" )

    kind: SourceKind
    path: str | None = None
    sha256: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    repo: str | None = None
    ref: str | None = None

class Machine( BaseModel ):
    model_config = ConfigDict( extra = "forbid" )

    vm: str
    zone: str
    machine_type: str
    gpu: Gpu
    spot: bool
    image: str

class Times( BaseModel ):
    model_config = ConfigDict( extra = "forbid" )

    created: datetime
    started: datetime | None = None
    finished: datetime | None = None

# `run.json` (TRD §8.1). It keeps env names only, never values.
class RunRecord( _Record ):
    run_id: str
    job_id: str
    experiment: str
    name: str
    command: list[ str ]
    env_names: list[ str ]
    status: RunStatus
    exit_code: int | None = None
    source: Source
    machine: Machine
    ml_runner_version: str
    times: Times
    duration_seconds: float | None = None
    message: str | None = None

class JobRun( BaseModel ):
    model_config = ConfigDict( extra = "forbid" )

    run_id: str
    experiment: str
    command: list[ str ]

# `job.json` (TRD §8.2). `env` is excluded from `to_json`, so values reach only instance metadata.
class JobSpec( _Record ):
    job_id: str
    bucket: str
    max_hours: float
    stop_on_failure: bool
    source: Source
    runs: list[ JobRun ]
    gpu: Gpu
    env: dict[ str, str ] = Field( default_factory = dict, exclude = True )

    def metadata_json( self ) -> str:
        return json.dumps( { **json.loads( self.to_json() ), "env": self.env }, indent = JSON_INDENT )
