# Security Policy

## Supported versions

Octoqai is pre-1.0. Only the latest release on `main` receives security fixes.

## Reporting a vulnerability

Please do not open a public issue for security problems.

Report it privately through
[GitHub Security Advisories](https://github.com/DannyDan2016/framework-octoqai/security/advisories/new).
Include the affected version, steps to reproduce and the impact you observed.

You will get an acknowledgement within 7 days. Once a fix is available the advisory is
published and the reporter is credited unless they prefer otherwise.

## Scope

Octoqai is a client: it sends requests to the agent you configure and never opens
listening ports. Treat experiment files as code; secrets belong in environment variables
referenced as `${VAR}` in `url` or `headers`, which are never written to reports.
