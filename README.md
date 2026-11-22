# Octoqai

Octoqai (pronounced "octo-kái") is an open source QA framework that evaluates AI agents with business-driven experiments and statistically sound A/B comparisons.

## Running an experiment

`octoqai run experiment.yaml [--out DIR] [--env NAME] [--tags a,b]` plays the experiment and writes `report.json`.

Exit codes: `0` go, `1` no-go, `2` insufficient evidence, `3` error.

When the experiment configures a `judge` or a `simulated_user`, the worst case number of calls is computed before anything runs (variants x trials x the judge checks or simulated turns of the selected cases). If it exceeds the role's `max_calls`, the run stops with `Config error: worst case needs <n> judge call(s) but judge.max_calls is <m>` and exit code `3`, before the agent, a hook or the provider is reached. Before the run, one line per role gives the worst case in calls and output tokens (`judge worst case: <n> call(s) of max_calls <m>, up to <n x max_tokens> output tokens`); no token counting call is made. After the run, `judge calls: <used>/<max>` and the token usage are printed for each role (the label is `simulated user` for that role). If a role made calls and every one failed at the provider, the report is written and the run ends with `Error: judge unreachable in every call (<Class>).` and exit code `3`; leaks or forbidden database changes still exit `1`.

### Judge and simulated user

The Anthropic provider is an optional extra: `pip install 'octoqai[anthropic]'`. Ollama needs nothing extra. `examples/judge/` is a complete experiment; copy `.env.example` to `.env` and fill in the key.

**Cost warning:** a real judge or simulated user makes paid calls on every trial (`trials` x judge checks, plus one call per simulated turn). Start with few cases, `trials: 3` and a low `max_calls`; the run refuses to start when the worst case exceeds it, and prints the calls and tokens used afterwards.

Fields of `judge` and `simulated_user` (defaults come from the code):

| Field | Default | Meaning |
| --- | --- | --- |
| `provider` | required | `anthropic` or `ollama` |
| `model` | required | exact model id; may be `${VAR}`, which is not treated as a secret |
| `api_key` | required for `anthropic` | only a whole `${VAR}`; not accepted for `ollama` |
| `base_url` | `http://127.0.0.1:11434` | `ollama` only |
| `allow_remote` | `false` | allow a non loopback `base_url` |
| `effort` | `low` | reasoning effort |
| `max_tokens` | `4096` | output tokens per call (256 to 32000) |
| `timeout_s` | `60` | seconds per attempt (up to 600) |
| `max_retries` | `2` | retries per call (0 to 5); Ollama never retries |
| `max_calls` | required | cap on calls; the run refuses to start above it |
| `max_input_chars` | `32000` | most characters of rubric, reference or transcript (judge) or of goal, persona and transcript (simulated user) sent in one call, 1000 to 200000; above it the check or trial is not evaluable (`transcript too long`) and no call is made |

The wall-clock bound per call is `timeout_s x (max_retries + 1)` (`timeout_s` for Ollama).

To use a local model instead, replace the provider section (here and in `simulated_user`):

```yaml
# judge:
#   provider: ollama
#   model: <local model>
#   max_calls: 6
```

With `trials: 3` the example exits `2` (insufficient evidence) even when every check passes, because three trials are too few for a decision; raise `trials` to get `0` or `1`.

How the data is redacted, the cap, what "unsure" means and the cost of a simulated user are in [ARCHITECTURE.md](ARCHITECTURE.md#judges-and-simulated-users-in-practice); what leaves your machine, in [SECURITY.md](SECURITY.md).
