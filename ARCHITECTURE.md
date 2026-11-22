# Architecture

Octoqai is organised **by feature** and follows a **hexagonal** style: the core never performs I/O
itself. Adapters implement **ports** (`typing.Protocol`), and only the CLI decides which concrete
adapter to use. This document is the source of truth for where code and tests go; the import rules
below are enforced by `lint-imports` and the test layout by `tests/quality/test_layout.py`.

## Package map (`src/octoqai/`)

| Package | Responsibility | May depend on |
|---|---|---|
| `errors.py` | `ConfigError` and other errors shared by every package | nothing |
| `experiment/` | Pydantic model, YAML loader, `${VAR}` expansion from `.env` files (`envfiles.py`), per-environment data sets (`datasets.py`), `{{ }}` templates, `SourceFloat` (`sourcefloat.py`, a YAML float that keeps the text it was written with) and `is_sensitive_key` (`urlkeys.py`, which connection-string keys hold a secret) | `errors` |
| `stats/` | Pure estimators: Wilson interval, paired bootstrap, pass@k | `errors` |
| `agents/` | `port.py` (`AgentPort`, `AgentReply`), `sse_events.py` (SSE parser), adapters such as `http_json.py` and `sse.py`, and `factory.py` (`adapter_for` picks the adapter from the config) | `experiment`, `errors` |
| `providers/` | Language model port (`port.py`: `LlmPort`, `LlmRequest`, `LlmReply`), the call cap `BudgetedLlm` (`budget.py`), the factory `provider_for` (`factory.py`) and the adapters `AnthropicProvider` (`anthropic_api.py`) and `OllamaProvider` (`ollama_api.py`, over `httpx`, local and free). The provider settings (`ProviderSettings`, `JudgeConfig`, `SimulatedUserConfig`) live in `experiment/` | `experiment`, `errors` |
| `db/` | Database port (`DatabasePort`, `QueryResult`) and the SQLAlchemy adapter `SqlDatabase`, with `connection_secrets` to list the connection values the report must mask | `experiment`, `errors` |
| `hooks/` | Lifecycle hooks: `LifecyclePort`, `HookExecutor` for http, sql and command (argv list, never a shell) | `db.port`, `experiment`, `errors` |
| `expectations/` | Registry of deterministic checks applied to an agent reply, plus the evaluation of database checks and forbidden changes | `agents.port`, `db.port`, `experiment` |
| `security/` | Detectors for canaries, secrets and personal data, leak scanning and the single report redactor | `agents.port`, `experiment`, `errors` |
| `running/` | Serial runner; knows only the ports `AgentPort`, `DatabasePort` and `LifecyclePort`, never a concrete adapter | lower layers |
| `verdict/` | Go/no-go decision, hard gates and A/B recommendation | `running`, `stats` |
| `reports/` | Report writers (JSON today) | `verdict`, `running` |
| `cli/` | `argparse` entry point; wires concrete adapters | everything |

## Layers

From top to bottom; a package may import only from packages below it:

1. `octoqai.cli`
2. `octoqai.reports`
3. `octoqai.verdict`
4. `octoqai.running`
5. `octoqai.expectations` and `octoqai.security` (independent of each other)
6. `octoqai.agents`, `octoqai.hooks` and `octoqai.providers` (independent of each other)
7. `octoqai.db`
8. `octoqai.stats` and `octoqai.experiment` (independent of each other)
9. `octoqai.errors`

Additional contracts in `[tool.importlinter]`:

- `httpx` is imported only inside `octoqai.agents`, `octoqai.hooks`, `octoqai.providers` and `octoqai.cli`.
- `anthropic` and `httpx2` are imported only inside `octoqai.providers`. The Anthropic SDK is built on `httpx2`, a separate package from `httpx`, so the `httpx` rules and contract do not cover it (an `httpx` client is even rejected by the SDK); the "anthropic and httpx2 only inside providers" contract guards it, and `octoqai.cli` only calls `provider_for`.
- `sqlalchemy` is imported only inside `octoqai.db`; `octoqai.providers` is in its list of forbidden importers.
- Only `octoqai.hooks` starts processes (`tests/quality/test_architecture.py`), always with an argument list and never a shell.
- The core (`running`, `security`, `expectations`, `verdict`, `reports`) never imports `octoqai.agents.http_json`, `octoqai.agents.sse`, `octoqai.agents.factory`, `octoqai.db.sql` or `octoqai.hooks.executor`; it imports the ports (`octoqai.agents.port`, `octoqai.db.port`, `octoqai.hooks.port`).
- `octoqai.stats` is pure: it imports nothing from Octoqai except `errors`, and never imports `octoqai.db`, `octoqai.hooks` or `octoqai.providers`.

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
directly (`octoqai.agents.port`, `octoqai.db.port`, `octoqai.hooks.port`, `octoqai.providers.port`, `octoqai.providers.budget`) to keep the core away from concrete adapters.

## Tests

`tests/` mirrors `src/octoqai/`:

```
tests/
  conftest.py        only: pytest_plugins = ["support.fixtures"]
  support/           test kit: builders.py, agents.py, databases.py, hooks.py, providers.py, netguard.py, assertions.py, paths.py, samples.py, fixtures.py
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
- **Language models:** `FakeLlm` is an `LlmPort` that answers from a queue (`judges`, `returns_text`, `script`, `fails`, `times_out`, `refuses`, `truncates`, `serves_model`, `uses`) and records `requests`; an exhausted script gives the fixed error `fake script exhausted`, never a text that could read as a pass. `ScriptedUser` adds `says(*messages, done_after=None)`. No test makes a real call.
- **Hermetic network:** an autouse fixture (`support/netguard.py`) lets a test connect only to loopback (`127.0.0.1`, `::1`, `localhost`) and fails it with a clear message otherwise; it covers `socket.connect`, `connect_ex`, `getaddrinfo` and the asyncio loops, and it clears the `ANTHROPIC_*` variables from the environment. Subprocesses such as the toy agent are not affected.
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
3. Test it with a fake (`FakeCommands`, `FakeJsonAgent`, `FakeDatabase`). Real processes belong
   only in `tests/hooks/test_commands.py`, for what a fake cannot show (process trees, path
   resolution), and wait with `support.hooks.wait_until`, never with a sleep.

### Add an LLM provider

1. Add the name to `ProviderName` in `experiment/model.py` and its credential rules to
   `ProviderSettings._fit_provider` (a key only through a whole `${VAR}`, a `base_url` only where
   the provider has one).
2. Create `providers/<name>_api.py` with a class that satisfies `LlmPort`. `complete` never
   raises: a failure comes back as an `LlmReply` whose `error` is a class name or a fixed message,
   never an exception text, and the reply reports the model that served it.
3. Register it in `_BUILDERS` in `providers/factory.py`. Import an optional SDK lazily and report
   a missing install as a `ConfigError`; add the extra to `pyproject.toml` and a row to
   `DEPENDENCIES.md`.
4. Add `tests/providers/test_<name>_api.py` with a fake transport. No test makes a real call.

### Add a judge check

1. In an experiment, add a `judge` list to a turn (that reply), to a case (the whole
   conversation) or `each_turn_judge` to a simulated case. One rubric states one criterion;
   `hard: true` only for a criterion a wrong `pass` of which you could not live with, because a
   `fail` then vetoes the verdict.
2. Raise `max_calls` to the new worst case (`worst_case_calls`); the run refuses to start
   otherwise.
3. To change what the judge sees or how it answers, edit `expectations/judge.py` and bump
   `JUDGE_PROMPT_VERSION`: reports record it next to `rubric_sha256`, so two reports are only
   comparable when both match.
4. Test it with `FakeLlm.judges(...)` in `tests/expectations/`; add the case to
   `examples/judge/experiment.yaml` only if the example needs it.

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
  values or `sql` hooks. In a `sql` hook a `${VAR}` is rejected when the file loads, before it is
  expanded, because its value would be pasted into the statement unescaped; keep those statements
  fixed or move the logic to a `command` hook.
- A query returns at most `MAX_ROWS` (10,000) rows. More is an error with no rows, so a flooded
  table cannot exhaust memory: it fails a `row_count` or `rows` check that expects fewer rows, and
  an `unchanged` check or a forbidden change when the other snapshot was readable; anything else
  is not evaluable. The cap counts rows, not bytes, so a single huge value is still read whole,
  and a driver that buffers the whole result on the client (psycopg 2 without a server-side
  cursor) reads it before the cap applies. Bound watched queries with `count(*)`, `WHERE` or
  `LIMIT`.
- `postgresql://` uses psycopg 3. Octoqai does not install any driver: install the one your
  URL needs.
- A relative SQLite path resolves against the directory `octoqai` runs from, while hooks run in
  the experiment's folder; run from that folder or use an absolute path.
- The statements of a `sql` hook run in one transaction, so a statement that cannot run inside
  one (`VACUUM` and similar) belongs in a `command` hook.
- The `timeout_s` of an `http` hook applies to each phase of the request (connect, write, read
  and pool), not to the whole call.
- A `command` hook with a non-empty `env` replaces the child's environment instead of extending
  it; the CLI passes the process environment plus the `.env` files, so a command sees both.
- A `command` program with a folder in it (`scripts/reset`, `./reset`, `..\bin\reset.exe`)
  runs from the experiment's folder on every platform; one relative to another Windows drive
  (`D:reset.exe`) never starts. A bare name (`python`) is resolved by `octoqai`, not the operating
  system: the first absolute folder of `PATH` (from the hook environment, or the process one when
  that is empty; any key casing on Windows) that holds it as a file wins, executable on POSIX and
  with `.exe` appended on Windows, where `.bat` and `.cmd` names are rejected. Empty, `.` and
  relative entries are skipped, so the folder `octoqai` runs from is never searched.
- When a `command` hook times out or the run is interrupted, its whole process tree is killed
  (a new session and `killpg` on POSIX, `taskkill /T` on Windows), so helper processes a reset
  script started do not outlive it. A descendant that detaches into its own session escapes.
- A query that starts with `SELECT` or `WITH` can still hold a second statement
  (`SELECT ...; DROP ...`) or a data-modifying `WITH`. Only the rollback and a read-only database
  user stop them, so use that user.
- A failed `sql` hook or query reports only its exception class, because the driver message can
  repeat stored values, parameters or connection details; to debug it, run the statement by
  hand. Database values and connection details never reach an error.
- Report masking uses values taken from the environment and from the database connection, and it
  applies to the whole report, so a value such as `pass` also masks a name written by the user,
  such as a variant. The fields with a closed set of values (`outcome` and each `status`) are not
  masked, because they can never hold a secret.
- Database numbers are compared exactly, never rounded, and a float written in the experiment is
  read as the text the user wrote, not as the nearest double. A `delta` must equal the expected
  value; only when a database value is a float is an absolute difference of `1e-9` accepted.
- `delta` over `FLOAT` or `REAL` columns allows an absolute tolerance of `1e-9`, so a 32-bit float
  can fail; cast to `NUMERIC` or `ROUND` in the query.

## Judges and simulated users in practice

`examples/judge/` is a worked experiment; `tests/e2e/test_judge_agent.py` runs it against the toy
agent with fake models. The rules behind it:

- What is sent. A judge request carries the redacted transcript, the rubric and the reference; a
  simulated user request carries the redacted goal, persona and conversation. Texts are redacted
  (canaries, secrets, personal data, `${VAR}` values) before they leave, and a sensitive value
  split across two texts is not sent. See `SECURITY.md` for what the redactor cannot know.
- The cap is checked before anything runs. `worst_case_calls` multiplies variants, trials and the
  judge checks or simulated turns of the selected cases (after `--tags`); above `max_calls` the run
  exits `3` before the agent, a hook or a provider is reached. A cap is also enforced during the
  run (`BudgetedLlm`): a call over it is not made and counts as not evaluable.
- Unsure is not a pass. `unsure`, a refusal, a truncated or unparsable answer, a different served
  model or a provider error make the check not evaluable, never a pass or a fail. If every call of
  a role failed at the provider, the CLI ends with exit `3` (a leak or forbidden change still
  gives `1`).
- No fallbacks. A failed call is retried against the same model up to `max_retries`; Octoqai never
  switches model, provider or endpoint, and never answers a failed call by itself.
- Sampling is not deterministic with Anthropic: the API rejects `temperature`, `top_p` and
  `top_k`, so the same rubric can give different verdicts. Use `trials` of 3 or more, and read
  `judge calls` and the token lines as the cost of the run.
- A simulated user adds a second source of variation: each trial plays a different conversation,
  so a paired A/B comparison is noisier. Use more trials and check `stop_reason` in the report.
- The rubric is versioned. Each verdict records `rubric_sha256` and `prompt_version`; change
  neither silently between the runs you compare.
- Cost. Every judge check is one call per trial and every simulated turn one more, and a simulated
  case with `each_turn_judge` multiplies them by `max_turns`. Keep `trials`, cases and `max_calls`
  low until the experiment is stable; Ollama on loopback costs nothing.

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
