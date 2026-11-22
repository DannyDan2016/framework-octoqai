# Octoqai

Octoqai (pronounced "octo-kái") is an open source QA framework that evaluates AI agents with business-driven experiments and statistically sound A/B comparisons.

## Running an experiment

`octoqai run experiment.yaml [--out DIR] [--env NAME] [--tags a,b]` plays the experiment and writes `report.json`.

Exit codes: `0` go, `1` no-go, `2` insufficient evidence, `3` error.

When the experiment configures a `judge` or a `simulated_user`, the worst case number of calls is computed before anything runs (variants x trials x the judge checks or simulated turns of the selected cases). If it exceeds the role's `max_calls`, the run stops with `Config error: worst case needs <n> judge call(s) but judge.max_calls is <m>` and exit code `3`, before the agent, a hook or the provider is reached. After the run, `judge calls: <used>/<max>` and the token usage are printed for each role. If a role made calls and every one failed at the provider, the report is written and the run ends with `Error: judge unreachable in every call (<Class>).` and exit code `3`; leaks or forbidden database changes still exit `1`.

### Judge and simulated user

The Anthropic provider is an optional extra: `pip install 'octoqai[anthropic]'`. Ollama needs nothing extra. `examples/judge/` is a complete experiment; copy `.env.example` to `.env` and fill in the key.

**Cost warning:** a real judge or simulated user makes paid calls on every trial (`trials` x judge checks, plus one call per simulated turn). Start with few cases, `trials: 3` and a low `max_calls`; the run refuses to start when the worst case exceeds it, and prints the calls and tokens used afterwards.

To use a local model instead, replace the provider section (here and in `simulated_user`):

```yaml
# judge:
#   provider: ollama
#   model: <local model>
#   max_calls: 6
```

How the data is redacted, the cap, what "unsure" means and the cost of a simulated user are in [ARCHITECTURE.md](ARCHITECTURE.md#judges-and-simulated-users-in-practice); what leaves your machine, in [SECURITY.md](SECURITY.md).
