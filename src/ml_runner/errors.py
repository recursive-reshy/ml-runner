EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

# The CLI prints `message` and exits with `exit_code`; `detail` (raw gcloud stderr) shows only with --verbose.
class MlRunnerError( Exception ):
    exit_code = EXIT_FAILED

    def __init__( self, message: str, *, detail: str = "" ) -> None:
        super().__init__( message )
        self.message = message
        self.detail = detail

# Bad usage or configuration: exit 2 instead of 1.
class UsageError( MlRunnerError ):
    exit_code = EXIT_USAGE
