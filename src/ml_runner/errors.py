"""Exit codes and the two error types the CLI turns into them (TRD §14)."""

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


class MlRunnerError(Exception):
    """An operation failed. The CLI prints `message` and exits 1.

    `detail` (for example raw gcloud stderr) is printed only with --verbose.
    """

    exit_code = EXIT_FAILED

    def __init__(self, message: str, *, detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class UsageError(MlRunnerError):
    """Bad usage or configuration. The CLI prints `message` and exits 2."""

    exit_code = EXIT_USAGE
