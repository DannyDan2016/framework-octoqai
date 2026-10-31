# Architecture

Octoqai is organised **by feature** and follows a **hexagonal** style: the core never performs I/O
itself. Adapters implement **ports** (`typing.Protocol`), and only the CLI decides which concrete
adapter to use. This document is the source of truth for where code and tests go; the import rules
below are enforced by `lint-imports` and the test layout by `tests/quality/test_layout.py`.

## Package map (`src/octoqai/`)

| Package | Responsibility | May depend on |
|---|---|---|
| `errors.py` | `ConfigError` and other errors shared by every package | nothing |
| `experiment/` | Pydantic model, YAML loader, `${VAR}` expansion and `{{ }}` templates | `errors` |
| `stats/` | Pure estimators: Wilson interval, paired bootstrap, pass@k | `errors` |
| `agents/` | `port.py` (`AgentPort`, `AgentReply`) and adapters such as `http_json.py` | `experiment`, `errors` |
| `expectations/` | Registry of deterministic checks applied to an agent reply | `agents.port`, `experiment` |
| `running/` | Serial runner; knows only `AgentPort`, never a concrete adapter | lower layers |
| `verdict/` | Go/no-go decision, hard gates and A/B recommendation | `running`, `stats` |
| `reports/` | Report writers (JSON today) | `verdict`, `running` |
| `cli/` | `argparse` entry point; wires concrete adapters | everything |

Planned packages: `security/` (detectors, leak checks, redactor), `db/` (database port and adapter)
and `providers/` (LLM provider port and adapters). Each one gets a row here, an import contract and
a mirror test folder when it is created.

## Layers

From top to bottom; a package may import only from packages below it:

1. `octoqai.cli`
2. `octoqai.reports`
3. `octoqai.verdict`
4. `octoqai.running`
5. `octoqai.expectations`
6. `octoqai.agents`
7. `octoqai.stats` and `octoqai.experiment` (independent of each other)
8. `octoqai.errors`

Additional contracts in `[tool.importlinter]`:

- `httpx` is imported only inside `octoqai.agents` and `octoqai.cli`.
- `octoqai.running` never imports `octoqai.agents.http_json`; it imports `octoqai.agents.port`.
- `octoqai.stats` is pure: it imports nothing from Octoqai except `errors`.

## Ports and adapters

A **port** is a `Protocol` that describes what the core needs, for example `AgentPort.send()`
returning an `AgentReply`. An **adapter** is a class that fulfils a port using real I/O, for
example `HttpAgentAdapter`. The runner receives a port, so a new transport never changes the core,
and tests can pass any object with the same methods.

## Public API

Every package exports its public names in `__init__.py` and lists them in `__all__`. Other packages
and tests import from the package (`from octoqai.experiment import load_experiment`), not from its
modules. A module that is not public starts with `_`. The only exception is a port imported
directly (`octoqai.agents.port`) to keep the runner away from concrete adapters.

## Tests

`tests/` mirrors `src/octoqai/`:

```
tests/
  conftest.py        only: pytest_plugins = ["support.fixtures"]
  support/           test kit: builders.py, agents.py, assertions.py, paths.py, fixtures.py
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
- **Simulated agents:** use `FakeJsonAgent` (`answers`, `replies`, `returns`, `fails`, `times_out`,
  `refuses`, `requests`, `headers`), never an ad hoc `httpx.MockTransport` inside a test.
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
3. Wire it in `cli/`.
4. Add `tests/agents/test_<name>.py` (`unit`, `regression`, one `smoke`) and a fake agent in
   `tests/support/agents.py`.

### Add an expectation

1. Add the literal to `ExpectationKind` and its value validator in `experiment/model.py`.
2. Write `_<kind>(value, reply)` in `expectations/checks.py` and register it.
3. When the data needed is missing, return `None` (not evaluable), never a pass.
4. Add rows to `tests/expectations/data/checks.yaml` and error rows to
   `tests/experiment/data/errors.yaml`.

### Add a detector

1. Add the pattern in `security/` and export it.
2. Build a positive and a negative sample by concatenation in `tests/support/`.
3. Add known false positives, if any, to `tests/security/data/`.
4. Mark the tests `security`.

### Add a report format

1. Create `reports/<format>.py` and export its writer.
2. Always pass the content through the redactor.
3. Never include headers or environment values.
4. Add `tests/reports/test_<format>.py`, with a `security` test that no secret leaks.

### Add a CLI option

1. Add the argument in `cli/main.py`.
2. Put the logic in the package that owns it, not in `cli/`.
3. Add a test in `tests/cli/` using the `run_cli` fixture.

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
