# Checkpoint A plan — Kick-off and foundations (tasks 1.1–1.5)

## Context
Repo state: only the `uv init` stub (`src/ml_runner/__init__.py` prints hello, `requires-python >=3.15`, no deps, empty README). `docs/` and `CLAUDE.md` exist but are **git-ignored** (`.gitignore` lists `docs`, `CLAUDE.md`), `.claude/` is untracked. Branch is `main`; work goes on `release/0.1.0`. Goal of A: installable CLI skeleton, wrapper + fake, config, IDs/layout, and a six-step gate. Precedence: TRD §2–4, 3, 7; ADRs 0001, 0002, 0003, 0007, 0009.

On approval, before 1.1: create `release/0.1.0` from `main`; remove `docs` and `CLAUDE.md` from `.gitignore` (they are ignored today, so the plan could not be committed); save this plan to `docs/v0.1.0/plans/checkpoint-a.md`; tick the Plan cell; commit `docs: add checkpoint A plan` (staging `.gitignore`, the plan file and `tasks.md` by name) and push. The rest of `docs/`, `CLAUDE.md` and `.claude/commands/` are committed in 1.1 as the task says.

## 1.1 Tooling, gate, packaging (no TDD target; verified by running)
- `.gitignore`: remove `docs`, `CLAUDE.md`; add `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.pyright/`, `dist/`, `mlr-runs/`, capture output folder (`fixtures-capture/`, name fixed in 2.1).
- `pyproject.toml`: `requires-python >=3.11`; description from README About line; `license = "MIT"`, `[project.urls]`; scripts `ml-runner = "ml_runner.cli:app"`; runtime deps `typer`, `pydantic>=2`, `platformdirs`, `tomli-w` (named in TRD §2.2); dev group `pytest`, `ruff`, `pyright`; `[tool.ruff]` (defaults + `I` import sorting), `[tool.pyright]` (standard, `src`, `tests`), `[tool.pytest.ini_options]` (`testpaths=["tests"]`). `.python-version` left alone.
- `LICENSE` (MIT, 2026, Naresh), minimal `README.md` About line (full README is 6.1).
- `scripts/gate.sh`: six steps from TRD §2.3, one-line result each, stop at first failure, `--only <n…>`. Verify the uv flags for steps 5–6 locally (needs Python 3.10/3.11 via uv; uv may download interpreters — that's tooling, not cloud) and note the working spelling in the script header and the task log line. Steps 4–6 need at least one test: add `tests/test_smoke.py` placeholder only if pytest exits 5 on no tests (gate treats exit 5 as fail); replaced in 1.2. Add `tests/vm/` with a trivial test that loads `src/ml_runner/vm/agent.py` by path, plus a stub `agent.py` (docstring only) so step 6 runs; real agent is 4.1.
- Package data: `uv build`, inspect wheel for `vm/*.sh` and `agent.py` (create empty stub `startup.sh`/`shutdown.sh` placeholders); configure `uv_build` if missing. Update `requires` in `[build-system]` untouched.
- Commit tracked: `docs/`, `CLAUDE.md`, `.claude/commands/`. Never the capture folder.

## 1.2 Typer app (TRD §14)
Tests first (`tests/test_cli.py`, Typer `CliRunner`): `--help` exit 0 and lists name; `--version` prints `ml-runner 0.1.0` from `__version__`; `--verbose`/`--config` accepted and stored in a context object (dataclass); unknown option exit 2; `python -m ml_runner --version` via subprocess in-process-safe (`runpy`/`subprocess` of the local interpreter, no network). Exit-code constants (0/1/2) and an `MlRunnerError` → exit 1, `UsageError` → exit 2 mapping tested.
Code: `src/ml_runner/__init__.py` (`__version__` only, replace stub `main`), `__main__.py`, `cli/__init__.py` (`app`, global callback), `cli/` one module per group later.

## 1.3 Gcloud wrapper + FakeGcloud (TRD §4, §15; ADR 0003, 0009)
Tests first (`tests/test_gcloud.py`): argv built as `gcloud … --quiet [--format=json] --project=<id>`; `json()` parses stdout; `run(mutating=True)` in dry-run echoes `shlex.join` line and returns `""` without calling the runner; read-only calls still run in dry-run; timeouts 120 s default, 30 min for `storage cp/rsync`; `version()` parses `Google Cloud SDK 5xx.y.z`; error classification table (exit status + stderr → `GcloudError.kind` for all nine kinds) using **provisional** stderr strings in `tests/fakes/gcloud/stderr/*.txt`, marked provisional until 2.1; `FileNotFoundError` → `not_installed`.
Code: `src/ml_runner/gcloud.py` (`Runner` Protocol over `subprocess.run`-shaped call returning a small `Completed` dataclass, `GcloudError`, `Gcloud`); `tests/fakes/fake_gcloud.py` (`FakeGcloud`: register `(argv pattern → stdout, stderr, code)`, unmatched call fails the test printing argv, `calls` recorded) with its own tests. `gcloud.py` stays the only module running `gcloud`.

## 1.4 Config (TRD §3)
Tests first (`tests/test_config.py`): path = `$ML_RUNNER_CONFIG` else `platformdirs.user_config_dir("ml-runner")/config.toml` (monkeypatched); round-trip load/save with `tomllib`/`tomli-w`; `extra="forbid"` rejects unknown keys; defaults (`gpu=t4`, `spot=false`, `max_hours` 0.1–24 float, `disk_gb=100`, `max_source_mb=200`); missing file → usage error exit 2 "Run `ml-runner init` first." for every command except `init`, `doctor`, `--version` (placeholder commands registered for test); `config show` prints path + values; `config set` edits `defaults.*` and `zone` with validation and rejects other keys; `--config` flag overrides env.
Code: `config.py` (pure model + load/save, no cloud), `cli/config_cmd.py`. Note: `init`/`doctor` commands themselves arrive in B; only the exemption mechanism is built now.

## 1.5 IDs, slugs, layout (TRD §7; ADR 0007)
Tests first (`tests/test_ids.py`, `tests/test_layout.py`): `ts` format `YYYYMMDD-HHMMSS` UTC from an injected clock; `rand` 4 lowercase hex from an injected RNG; job ID, VM name `mlr-<job id>` ≤63 chars and RFC 1035; run ID `<ts>-<slug>-<rand>`; slug lowercase `[a-z0-9-]`, ≤30, no leading/trailing `-`, empty/all-symbol input fallback; experiment from `--experiment`, repo URL name (strip `.git`, query), or snapshot folder name; layout functions return every path of §7.2 (`sources/<sha>.tar.gz`, `jobs/<id>/job.json`, `jobs/<id>/cancel`, `runs/<exp>/<run>/{run.json,run.log,outputs/,checkpoints/}`); a test greps `src/` and asserts `gs://` appears only in `layout.py` (and gcloud-free docs strings excluded).
Code: `ids.py`, `layout.py`, pure, no I/O, no global state.

## Per-task loop
Red (show failing for the right reason) → least code → refactor → `scripts/gate.sh` green → tick `tasks.md` + log line → one commit (`feat:`/`chore:`, subject names the change, Co-Authored-By trailer), staged by name → push. Report at end of A.

## Verification (end of A)
`scripts/gate.sh` all six steps green; `uv tool install .` then `ml-runner --version` prints 0.1.0; `uv build` wheel contains `vm/` files; `grep -rn 'gs://' src` hits only `layout.py`; no real gcloud invoked anywhere.

## Open points (assumptions, flag on approval)
1. `.gitignore` currently ignores `docs` and `CLAUDE.md`, contradicting task 1.1; I will remove those entries.
2. Stub `agent.py`, `startup.sh`, `shutdown.sh` and a placeholder test are added in 1.1 only so gate steps 6 and the package-data check are meaningful; real content is 4.x.
3. Runtime deps (typer, pydantic, platformdirs, tomli-w) are TRD §2.2-named, so no new-dependency ADR is needed.
