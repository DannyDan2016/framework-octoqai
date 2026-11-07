# Architecture

Octoqai is organised **by feature** and follows a **hexagonal** style: the core never performs I/O
itself. Adapters implement **ports** (`typing.Protocol`), and only the CLI decides which concrete
adapter to use. This document is the source of truth for where code and tests go; the import rules
below are enforced by `lint-imports` and the test layout by `tests/quality/test_layout.py`.

## Package map (`src/octoqai/`)

| Package | Responsibility | May depend on |
|---|---|---|
| `errors.py` | `ConfigError` and other errors shared by every package | nothing |
| `experiment/` | Pydantic model, YAML loader, `${VAR}` expansion from `.env` files (`envfiles.py`), per-environment data sets (`datasets.py`) and `{{ }}` templates | `errors` |
| `stats/` | Pure estimators: Wilson interval, paired bootstrap, pass@k | `errors` |
| `agents/` | `port.py` (`AgentPort`, `AgentReply`), `sse_events.py` (SSE parser), adapters such as `http_json.py` and `sse.py`, and `factory.py` (`adapter_for` picks the adapter from the config) | `experiment`, `errors` |
| `db/` | Database port (`DatabasePort`, `QueryResult`) and the SQLAlchemy adapter `SqlDatabase`, with `connection_secrets` to list the connection values the report must mask | `errors` |
| `hooks/` | Lifecycle hooks: `LifecyclePort`, `HookExecutor` for http, sql and command (argv list, never a shell) | `db.port`, `experiment`, `errors` |
| `expectations/` | Registry of deterministic checks applied to an agent reply, plus the evaluation of database checks and forbidden changes | `agents.port`, `db.port`, `experiment` |
| `security/` | Detectors for canaries, secrets and personal data, leak scanning and the single report redactor | `agents.port`, `experiment`, `errors` |
| `running/` | Serial runner; knows only the ports `AgentPort`, `DatabasePort` and `LifecyclePort`, never a concrete adapter | lower layers |
| `verdict/` | Go/no-go decision, hard gates and A/B recommendation | `running`, `stats` |
| `reports/` | Report writers (JSON today) | `verdict`, `running` |
| `cli/` | `argparse` entry point; wires concrete adapters | everything |

Planned package: `providers/` (LLM provider port and adapters). It gets a row here, an import
contract and a mirror test folder when it is created.

## Layers

From top to bottom; a package may import only from packages below it:

1. `octoqai.cli`
2. `octoqai.reports`
3. `octoqai.verdict`
4. `octoqai.running`
5. `octoqai.expectations` and `octoqai.security` (independent of each other)
6. `octoqai.agents` and `octoqai.hooks` (independent of each other)
7. `octoqai.db`
8. `octoqai.stats` and `octoqai.experiment` (independent of each other)
9. `octoqai.errors`

Additional contracts in `[tool.importlinter]`:

- `httpx` is imported only inside `octoqai.agents`, `octoqai.hooks` and `octoqai.cli`.
- `sqlalchemy` is imported only inside `octoqai.db`.
- Only `octoqai.hooks` starts processes (`tests/quality/test_architecture.py`), always with an argument list and never a shell.
- The core (`running`, `security`, `expectations`, `verdict`, `reports`) never imports `octoqai.agents.http_json`, `octoqai.agents.sse`, `octoqai.agents.factory`, `octoqai.db.sql` or `octoqai.hooks.executor`; it imports the ports (`octoqai.agents.port`, `octoqai.db.port`, `octoqai.hooks.port`).
- `octoqai.stats` is pure: it imports nothing from Octoqai except `errors`, and never imports `octoqai.db` or `octoqai.hooks`.

## Ports and adapters

A **port** is a `Protocol` that describes what the core needs, for example `AgentPort.send()`
returning an `AgentReply`. An **adapter** is a class that fulfils a port using real I/O, for
example `HttpAgentAdapter`. The runner receives a port, so a new transport never changes the core,
and tests can pass any object with the same methods.

## Time and size limits of an agent call

Both adapters share these rules (`agents/_delivery.py`):

- `wait.total_s` is a wall-clock cap on one attempt, from sending the request to the last byte
  read. It is enforced twice. Every chunk read checks the deadlines (`first_fragment_s`,
  `idle_s`, `total_s`) against the injected clock before the chunk is buffered, so a body that
  never sends a line break cannot outlive them. A watchdog, a daemon `threading.Timer` by
  default and injectable as `AdapterKwargs["watchdog"]`, closes the HTTP client when `total_s`
  elapses, so a read blocked inside httpx (for example headers sent one byte at a time) also ends
  with the usual total timeout. It is armed for each attempt, cancelled in `finally`, and a firing
  while connecting is never retried.
- Backoff between retries is outside `total_s`: each attempt gets its own `total_s`, so one turn
  can last up to `(retries + 1) * total_s` plus the backoff sleeps, at most
  `backoff_s * (2**retries - 1)` (jitter only shortens them). Latency and TTFT never include
  backoff.
- A response is capped at 16 MiB and fails with `stream exceeded 16777216 bytes`. Error bodies and
  bodies with the wrong content type are read under the same deadlines and kept up to 64 KiB so
  they can be scanned; the HTTP status stays the error.
- Data received before a deadline, the cap or a read failure is kept: the SSE adapter turns the
  pending event, including an unterminated line, into a fragment, so it is scanned for leaks.

## Public API

Every package exports its public names in `__init__.py` and lists them in `__all__`. Other packages
and tests import from the package (`from octoqai.experiment import load_experiment`), not from its
modules. A module that is not public starts with `_`. The only exceptions are the ports imported
directly (`octoqai.agents.port`, `octoqai.db.port`, `octoqai.hooks.port`) to keep the core away from concrete adapters.

## Tests

`tests/` mirrors `src/octoqai/`:

```
tests/
  conftest.py        only: pytest_plugins = ["support.fixtures"]
  support/           test kit: builders.py, agents.py, databases.py, hooks.py, assertions.py, paths.py, samples.py, fixtures.py
  <package>/         test_<topic>.py and data/<name>.yaml with the tabular cases of that package
  quality/           style, hygiene, architecture and layout checks
  e2e/               real agent process on 127.0.0.1
```

- **No helpers in `conftest.py`.** Builders, simulated agents, reusable assertions, paths and
  fixtures live in `tests/support/`.
- **Paths:** never write relative paths by hand. Use `support.paths.cases(__file__, "name")`,
  `data_file(__file__, "name")` or the `datadir` fixture. `REPO_ROOT`, `SRC` and `EXAMPLES` come from
  `support.paths`.
- **Suites:** every module declares `pytestmark` with one of `unit`, `integration`, `e2e` or
  `quality`. Behavioural modules also carry `regression`. Key tests add `smoke` (one to three happy
  paths per package) or `security` (secrets, hygiene, headers). Pytest runs with `--strict-markers`.
- **Tabular data** goes in the package's `data/*.yaml`, one row per case with an `id`. Secret and
  token samples are never written in YAML: they are built by concatenation in Python so the
  hygiene checks do not flag the repository.
- **Simulated agents:** use `FakeJsonAgent` (`answers`, `replies`, `returns`) or `FakeSseAgent`
  (`streams`, `emits`, `script`, `heartbeats`, `leaks`, `cuts_after`, `stalls_after`,
  `wrong_content_type`); both also offer `fails`, `sends`, `stalls`, `times_out`, `refuses`,
  `refuses_first`, `requests` and `headers`. Never use an ad hoc `httpx.MockTransport` inside a
  test. Time comes from `FakeClock`, so deadlines and backoff never sleep for real; simulated
  agents fix `jitter` to 0 (change it with `fake.jitter`) and inject a `FakeWatchdog` that the
  agent fires by hand (`stalls`), so no test depends on wall-clock time.
- **End-to-end:** `tests/e2e/` starts `toy_sse_agent.py` (standard library only, `127.0.0.1` only)
  through the `toy_sse_agent_url` fixture. The fixture waits for the `PORT <n>` line the agent prints
  (a signal, never a sleep) and fails at once with the agent's stderr if it does not arrive. To try
  it by hand: `python tests/e2e/toy_sse_agent.py --port 8000`, then
  `octoqai run examples/streaming/experiment.yaml`.
- **Database end-to-end:** `toy_db_agent` starts the same agent with `--db <tmp_path>/app.db` and
  returns its URL and the database path. With `--db`, a message containing `crea un ajuste` inserts
  `('A', '001', 10)` into `ajustes` and one containing `otro tenant` inserts `('B', '900', 99)`
  (the forbidden CRUD of the example); the agent commits and closes the connection before it replies.
  `examples/database/reset_db.py` is the real `before_trial` hook. To try it by hand, from the repo
  root: `python tests/e2e/toy_sse_agent.py --port 8000 --db examples/database/conciliaciones.db`;
  in another terminal, inside `examples/database`: `copy .env.example .env`, then
  `octoqai run experiment.yaml`.
- **Hooks and databases:** `FakeCommands` stands in for the process runner (`exits`, `times_out`, `missing`) and never starts a process; `RecordingHooks` is a `LifecyclePort` that records stages and fails the ones it is told to; `FakeDatabase` is a `DatabasePort` that records the statements and queries; `sqlite_db` and `sql_database` give a real SQLite file in `tmp_path`. Http hooks are tested against `FakeJsonAgent` as the endpoint.
- **Experiments:** build them with `ExperimentBuilder` and `turn(...)`, or the `experiment` fixture.
- **CLI:** use the `run_cli` fixture; it returns the exit code, captured output and report.
- Tests are written first and seen failing before the implementation.

Useful selections:

```
pytest -q                     full suite
pytest -m smoke -q            fast happy paths
pytest -m "smoke or quality"  what the pre-commit hook runs
pytest -m security -q         secrets, hygiene and headers
```

## Recipes

### Add an agent adapter

1. Create `src/octoqai/agents/<name>.py` with a class that satisfies `AgentPort`.
2. Export it in `agents/__init__.py` and `__all__`.
3. Register it in `agents/factory.py`, which `cli/` uses through `adapter_for`.
4. Add `tests/agents/test_<name>.py` (`unit`, `regression`, one `smoke`) and a fake agent in
   `tests/support/agents.py`.

### Add an expectation

1. Add the literal to `ExpectationKind` and its value validator in `experiment/model.py`.
2. Write `_<kind>(value, reply)` in `expectations/checks.py` and register it.
3. When the data needed is missing, return `None` (not evaluable), never a pass.
4. Add rows to `tests/expectations/data/checks.yaml` and error rows to
   `tests/experiment/data/errors.yaml`.

### Add a detector

1. Add the pattern in `security/_patterns.py` and wire it in `security/detectors.py`.
2. Build a positive and a negative sample by concatenation in `tests/support/samples.py`.
3. Add known false positives, if any, to `tests/security/data/false_positives.yaml`.
4. Mark the tests `security`.

### Add a report format

1. Create `reports/<format>.py` and export its writer.
2. Always pass the content through the redactor (`Redactor.for_report`), which runs every
   detector whatever `security.detectors` says, ignores `security.allow` (both only decide what
   counts as a leak, never what the report may show) and masks the union of overlapping findings.
   `redact_tree` redacts values, not dict keys; no report key is controlled by the agent, so
   a new key must never be built from agent output. JSON text with a finding in a decoded
   string is decoded, redacted and written again without escapes, so unicode escapes or
   escaped slashes cannot hide a value; leak scanning decodes JSON the same way.
3. Never include headers or environment values.
4. Add `tests/reports/test_<format>.py`, with a `security` test that no secret leaks.

### Add a database check

1. Add the literal to `DbExpectKind` and its value validator in `experiment/model.py`.
2. Write the function in `expectations/db_checks.py`. When the data needed is missing, return
   `None` (not evaluable), never a pass.
3. Never put a value read from the database in the detail: only counts, deltas and match/differs.
4. Add rows to `tests/expectations/data/db_checks.yaml` and error rows to
   `tests/experiment/data/db_errors.yaml`.

### Add a hook kind

1. Add the field to `Hook` in `experiment/model.py` and its validation.
2. Add the method to `HookExecutor`. Commands run without a shell and their output is discarded;
   never capture it.
3. Test it with a fake (`FakeCommands`, `FakeJsonAgent`, `FakeDatabase`), never with a real process.

### Add a CLI option

1. Add the argument in `cli/main.py`.
2. Put the logic in the package that owns it, not in `cli/`.
3. Add a test in `tests/cli/` using the `run_cli` fixture.

## Database checks in practice

- Use two database users: a read-only one in `database.url` for the checks, and a separate one
  with write access that only the hooks use. `sql` hooks run through `database.url` as well, so
  with a read-only user the reset belongs in a `command` hook that reads its own variable. The adapter rolls every query back and rejects queries that do not start with
  `SELECT` or `WITH`, but only the database can guarantee that nothing is written.
- Set a statement timeout in the database or in the driver URL, for example PostgreSQL
  `statement_timeout`. The adapter has no timeout of its own and `timeout_s` is not accepted by
  `sql` hooks.
- Queries run through SQLAlchemy `text()`: a literal `:` in the SQL is written `\:`, and values
  go through `params` (`:name`). A query never accepts `{{ }}` or `${VAR}`, and neither do expected
  values or `sql` hooks (`{{ }}` only); a placeholder there would be sent as literal text.
- `postgresql://` uses psycopg 3. Octoqai does not install any driver: install the one your
  URL needs.
- A relative SQLite path resolves against the directory `octoqai` runs from, while hooks run in
  the experiment's folder; run from that folder or use an absolute path.
- A failed `sql` hook or query reports only its exception class, because the driver message can
  repeat stored values, parameters or connection details; to debug it, run the statement by
  hand. Database values and connection details never reach an error.
- Report masking uses values taken from the environment and from the database connection, and it
  applies to the whole report, so a value such as `pass` also masks a field that holds that word.
- Database numbers are compared exactly, never rounded. A `delta` must equal the expected value;
  only when a database value is a float is an absolute difference of `1e-9` accepted.

## Conventions

- Functions are small and named by intent; cyclomatic complexity is at most 8.
- Everything is typed and checked with `mypy --strict`.
- No comments in `src/`; public API uses Google style docstrings.
- Code, identifiers, messages and documentation are in English.
- Example hosts are `127.0.0.1`, `localhost`, `*.local` or `*.example`.
- Every dependency has a major-version ceiling and a row in `DEPENDENCIES.md`.

## Quality gates

All of them pass before every commit:

```
ruff check .
ruff format --check .
mypy
lint-imports
pytest -q
```
