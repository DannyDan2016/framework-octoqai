# Dependencies

Every dependency is pinned below its next major version. Before raising a ceiling, check the impact column and run the full suite.

Last review: 2026-10-30.

| Dependency | Range | Latest stable | Next major known | Impact on Octoqai | Action |
|---|---|---|---|---|---|
| httpx | `>=0.28,<1` | 0.28.1 (2024-12-06) | **1.0 in development** (1.0.dev6, 2026-08-31) | `octoqai/agents/http_json.py` and `octoqai/cli/` (client, transports, exceptions) | When 1.0 is released: read its migration guide, update `agents/http_json.py` and its tests, then raise the ceiling to `<2` |
| pydantic | `>=2.14,<3` | 2.14.2 (2026-10-29) | none announced | `octoqai/experiment/` | Watch for a v3 announcement |
| PyYAML | `>=6.0.3,<7` | 6.0.3 (2025-09-25) | none announced | `octoqai/experiment/loader.py` (`yaml.safe_load` only) | None |
| jsonpath-ng | `>=1.15,<2` | 1.15.0 (2026-10-29) | none announced | `octoqai/agents/http_json.py` (response mapping) | None |
| python-dotenv | `>=1.2,<2` | 1.2.4 | none announced | `octoqai/experiment/envfiles.py` (`dotenv_values` only, without interpolation) | None |
| hatchling (build) | `>=1.33,<2` | 1.33.0 (2026-10-28) | none announced | Packaging only | None |
| pytest (dev) | `>=9.2,<10` | 9.2.0 (2026-10-27) | none announced | Test suite | None |
| ruff (dev) | `>=0.17.3,<0.18` | 0.17.3 | still 0.x: any minor may change rules | Lint and format | Raise the minor deliberately and fix new findings |
| mypy (dev) | `>=2.4,<3` | 2.4.1 | unknown | Strict type checking only; no runtime impact | Read the release notes and fix new findings before raising the ceiling |
| types-PyYAML (dev) | `>=6.0,<7` | 6.0.12.20261030 | follows PyYAML | Type stubs for `yaml`; no runtime impact | Raise together with PyYAML |
| import-linter (dev) | `>=2.15,<3` | 2.15 | none announced | Enforces package layers in `pyproject.toml`; no runtime impact | None |

Rejected: `typer` (still 0.x, minor releases may break the CLI). The CLI uses the standard library `argparse`.
