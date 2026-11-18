import json
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from ml_runner.gcloud import Completed

WILDCARD = "*"

@dataclass( frozen = True )
class _Rule:
    pattern: tuple[ str, ... ]
    result: Completed

# A `Runner` that matches argv patterns to canned results (ADR 0009). Register
# `(argv pattern -> stdout, stderr, exit code)`; an unmatched call fails the test.
# A pattern matches when it equals the start of the argv after `gcloud`; `"*"`
# matches any one argument; the first matching rule wins. Every call is recorded
# in `calls` (full argv, including `gcloud`).
class FakeGcloud:
    def __init__( self ) -> None:
        self._rules: list[ _Rule ] = []
        self.calls: list[ list[ str ] ] = []

    def register( self, pattern: Sequence[ str ], *, stdout: str = "", stderr: str = "", returncode: int = 0 ) -> None:
        self._rules.append( _Rule( tuple( pattern ), Completed( returncode, stdout, stderr ) ) )

    def register_json( self, pattern: Sequence[ str ], data: Any ) -> None:
        self.register( pattern, stdout = json.dumps( data ) )

    def __call__( self, argv: Sequence[ str ], *, timeout: float ) -> Completed:
        self.calls.append( list( argv ) )
        args = list( argv[ 1: ] )
        for rule in self._rules:
            if _matches( rule.pattern, args ):
                return rule.result

        pytest.fail( f"FakeGcloud: unexpected call: {shlex.join(argv)}" )

def _matches( pattern: tuple[ str, ... ], args: list[ str ] ) -> bool:
    if len( args ) < len( pattern ):
        return False

    return all( expected in ( WILDCARD, actual ) for expected, actual in zip( pattern, args, strict = False ) )
