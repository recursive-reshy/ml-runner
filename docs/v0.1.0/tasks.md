# Tasks — ml-runner v0.1.0

Built from `prd.md` and `trd.md` in this folder. Tasks run strictly in order. Each one is test-first (ADR 0009) and ends in its own commit. This folder, `CLAUDE.md` and `.claude/` are git-ignored (local only): ticks and log lines are edited on disk and never committed.

**Branch:** `release/0.1.0`. Never commit to `main`, never merge, never tag.

**Status:** ⬜ Not started · 🟨 In progress · ✅ Done (gate green, committed, pushed) · ⏸️ Deferred · 🔒 Blocked (waiting on a decision)
**Plan:** ⬜ Not approved · ✅ Approved (the link points to the saved plan)

## How this file works
- **Start of a checkpoint:** in plan mode, write one plan for every task in it (files to touch, tests to write first, expected red → green path), then stop. When Naresh approves it, save the plan to the linked file and tick the Plan cell (no commit; the folder is git-ignored).
- **During a checkpoint:** work through its tasks without stopping. For each task:
  1. failing tests first → green → refactor;
  2. `scripts/gate.sh` green;
  3. tick Status and add a log line at the bottom (local file);
  4. commit the code and tests, then push.
- **Commits** are conventional, with no scope: `feat: …`, `fix: …`, `refactor: …`, `docs: …`, `test: …`, `chore: …`. The subject names the change; the task id lives here.
- **End of a checkpoint:** run `scripts/gate.sh`, then stop and report:
  - what was built, with commit hashes;
  - the gate's output;
  - anything deferred;
  - any ADR that changes the TRD.

  Don't start the next checkpoint until Naresh says so.
- Never start a task whose checkpoint Plan cell is ⬜.
- Not covered by the PRD, TRD or an ADR (a new dependency, command, flag, file in the bucket, status)? Stop and ask. Record the decision as the next ADR (from 0011), add it to `docs/README.md` and the task's ADR column (local files).
- **Verify vendor facts** where the TRD says so (§17): use the output Naresh pastes, never a guess, and note any difference from the TRD in the log line.
- **No test touches the network, Google Cloud or a real bucket.**
- **GCP is Naresh's.** Claude Code never runs `gcloud`, `ml-runner` against a real project, or anything that creates or reads cloud resources. Tasks marked *With Naresh* give him exact commands; he runs them and pastes the output back.
- **No secrets** in the repo, fixtures, logs or docs. Fixture JSON from real output has project IDs, numbers and emails replaced.
- Update `README.md`, the architecture diagram and `CLAUDE.md` in the same step as the change they describe; only `README.md` is tracked and committed.

---

## Checkpoint A: Kick-off and foundations
**Plan:** ✅ [`plans/checkpoint-a.md`](plans/checkpoint-a.md)

| ID | Task | ADRs | Status |
| --- | --- | --- | --- |
| 1.1 | pyproject fixes (TRD §2.4), dev group (`pytest`, `ruff`, `pyright`), Ruff/Pyright/pytest config, `LICENSE` (MIT); `.gitignore` (keeps `docs`, `CLAUDE.md`, `.claude`; adds `.venv/`, caches, `dist/`, `mlr-runs/`, capture output); `scripts/gate.sh` with `--only`, uv flags for steps 5–6 verified locally and noted (TRD §2.3); package data for `vm/` checked with `uv build` | 0001, 0002, 0009 | ⬜ |
| 1.2 | Typer app: `ml-runner --help`, `--version`, `--verbose`, `--config`, `python -m ml_runner`, exit codes 0/1/2 (TRD §14) | 0002 | ⬜ |
| 1.3 | `Gcloud` wrapper with `Runner` protocol, `--project`/`--quiet`/`--format=json`, dry-run echo, timeouts; `GcloudError` kinds with provisional stderr fixtures; `FakeGcloud` in `tests/fakes/` (TRD §4, §15) | 0003, 0009 | ⬜ |
| 1.4 | `Config` model, path resolution (`$ML_RUNNER_CONFIG`, platformdirs), load/save, "run init first" exit 2, `config show` and `config set` (TRD §3) | 0002 | ⬜ |
| 1.5 | IDs, slugs, VM names and `layout.py` bucket paths, with length and character tests (TRD §7) | 0007 | ⬜ |

**Checkpoint A done when:** `ml-runner --version` works from `uv tool install .`, the gate runs all six steps green, and every `gs://` path comes from `layout.py`.

---

## Checkpoint B: Setup, doctor and vendor facts
**Plan:** ⬜ [`plans/checkpoint-b.md`](plans/checkpoint-b.md)

| ID | Task | ADRs | Status |
| --- | --- | --- | --- |
| 2.1 | `scripts/capture-gcloud-fixtures.sh` (read-only commands of TRD §17, output to a git-ignored folder); **with Naresh:** he runs it and pastes results; Claude Code trims and anonymises them into `tests/fakes/gcloud/`; image families, quota metric names, stderr patterns and minimum gcloud version filled in; differences from the TRD noted (ADR if a decision changes) | 0003, 0009 | ⬜ |
| 2.2 | `init` steps 1–7: project, APIs, bucket, service account, bucket binding, custom role, conditional binding; idempotent "exists" handling; `--dry-run`, `--yes` (TRD §5) | 0003, 0010 | ⬜ |
| 2.3 | `init` steps 8–10: zone choice from accelerator types, config write, quota summary and budgets link (TRD §5) | 0010 | ⬜ |
| 2.4 | `doctor`: every check of TRD §6 with pass/warn/fail, fixes, exit code | 0003, 0010 | ⬜ |
| 2.5 | **With Naresh:** `init --dry-run`, `init`, `init` again, `doctor` on his project; then a throwaway check of TRD §17's runtime cap and self-delete condition (commands given in the plan: an `e2-micro` `mlr-verify-*` VM with the runner service account, `--max-run-duration=10m`, a startup script that deletes itself); outputs in the log; fixes found here | 0004, 0010 | ⬜ |

**Checkpoint B done when:** `init` and `doctor` are green on Naresh's project, a runner-identity VM has deleted itself, and every vendor fact in TRD §17 is either confirmed or recorded in an ADR.

---

## Checkpoint C: Launch (the laptop side)
**Plan:** ⬜ [`plans/checkpoint-c.md`](plans/checkpoint-c.md)

| ID | Task | ADRs | Status |
| --- | --- | --- | --- |
| 3.1 | `RunRecord` and `JobSpec` models, status values, JSON round-trip; env values never serialised to `job.json`/`run.json` (TRD §8) | 0005, 0007 | ⬜ |
| 3.2 | Snapshot source: git file list, non-git walk with excludes, reproducible tar.gz and hash, size refusal with largest files, project-file check, upload-if-missing (TRD §9.1) | 0006 | ⬜ |
| 3.3 | Git source: `https` only, `git ls-remote` ref check against a local temp repo, `--repo`/`--src` exclusivity (TRD §9.2) | 0006 | ⬜ |
| 3.4 | `machines.py`: GPU mapping, `--machine-type` validation, zones offering a GPU, fallback order, exhausted vs quota handling (TRD §10.1–10.2) | 0004 | ⬜ |
| 3.5 | `vmspec.py`: full create argv for t4/l4/none × Spot on/off × overrides, runtime cap minutes, labels, metadata files (TRD §10.3) | 0004, 0010 | ⬜ |
| 3.6 | `launch` command: flag merge and validation, reserved env names, uploads, `job.json` and `run.json` (`provisioning`), create with fallback, failure marking, summary; `--dry-run` creates nothing (TRD §12.1) | 0004–0008 | ⬜ |

**Checkpoint C done when:** `launch --dry-run` prints the full plan and every `gcloud` command for a snapshot and a git source, and the argv for every GPU/Spot combination is pinned by a test.

---

## Checkpoint D: The VM agent and first live run
**Plan:** ⬜ [`plans/checkpoint-d.md`](plans/checkpoint-d.md)

| ID | Task | ADRs | Status |
| --- | --- | --- | --- |
| 4.1 | `agent.py` foundation: metadata reader (with `MLR_METADATA_FILE` for tests), `state.json`, source fetch for snapshot and git; fake executables harness in `tests/vm/` (TRD §11.2 steps 1–4, §15) | 0006, 0009 | ⬜ |
| 4.2 | uv install with pip fallback, `uv sync` / requirements paths, install failure handling, GPU driver wait (TRD §11.2 steps 5–6) | 0006 | ⬜ |
| 4.3 | Run loop: env, argv execution, log tee, sync threads, final sync with retries, `stop_on_failure`, cancel marker, job timeout, self-delete with shutdown fallback (TRD §11.2 steps 7–8, §11.4) | 0004, 0005, 0007, 0008 | ⬜ |
| 4.4 | `finalize`, `startup.sh`, `shutdown.sh`, reboot guard, 25 s budget; drift test validating agent-written JSON with `RunRecord` (TRD §11.1, §11.3, §8.1) | 0004, 0005 | ⬜ |
| 4.5 | `examples/hello-gpu/` (GPU check, output file, checkpoint file, `pyproject.toml`); **with Naresh:** first live `launch --gpu t4 --spot` from it, checked with `gcloud storage ls/cat` and the instance list; outputs in the log; fixes found here | 0005 | ⬜ |

**Checkpoint D done when:** a live T4 Spot run of `hello-gpu` succeeded, its outputs, checkpoint, log and `run.json` are in the bucket, and its VM deleted itself.

---

## Checkpoint E: Observe, cancel and sweep
**Plan:** ⬜ [`plans/checkpoint-e.md`](plans/checkpoint-e.md)

| ID | Task | ADRs | Status |
| --- | --- | --- | --- |
| 5.1 | `status`: list with filters and limit, detail by unique prefix, `lost` reconciliation from labelled instances, `--json` (TRD §13) | 0007 | ⬜ |
| 5.2 | `logs` with `--follow` by byte offset, stopping on a final status (TRD §13) | 0007 | ⬜ |
| 5.3 | `fetch` for a run or an experiment, `--dest` (TRD §13) | 0007 | ⬜ |
| 5.4 | `cancel` for a run or job: confirmation, marker, delete, status writes, VM-already-gone case (TRD §13) | 0004, 0007 | ⬜ |
| 5.5 | Sweep file model: defaults, matrix product, template placeholders, extra runs, validation errors naming the key (TRD §12.2) | 0008 | ⬜ |
| 5.6 | `sweep` command: round-robin into `parallel` jobs, one snapshot upload, partial-failure summary, `--dry-run` (TRD §12.2) | 0008 | ⬜ |

**Checkpoint E done when:** every command in PRD §6 exists, is tested against `FakeGcloud`, and has a dry-run where it mutates.

---

## Checkpoint F: Docs, live checks and release (with Naresh)
**Plan:** ⬜ [`plans/checkpoint-f.md`](plans/checkpoint-f.md)

| ID | Task | ADRs | Status |
| --- | --- | --- | --- |
| 6.1 | `README.md` (PRD §6.9: about, install, quick start, contract, costs and safety, security model, commands, sweep reference, troubleshooting); `docs/architecture.py` updated to jobs, runs and the agent, PNG regenerated; `docs/README.md` current | 0005, 0010 | ⬜ |
| 6.2 | **With Naresh:** live checks 1–8 of TRD §16 following only the README; outputs and total cost in the log; fixes to code or README found here, each with a test where possible | 0004, 0008 | ⬜ |
| 6.3 | `docs/v0.1.0/release-notes.md` from TRD §18, linked from `docs/README.md`; version `0.1.0` confirmed in `pyproject.toml` and `__version__`; KIV list carried into the README's roadmap | | ⬜ |

**Checkpoint F done when:** all live checks passed, no `mlr-*` instance is left in the project, and Naresh can tag `v0.1.0` from `main`.

---

## Log

