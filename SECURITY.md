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
referenced as `${VAR}` in the agent `url` or `headers`, in `database.url` (which only accepts a
variable) or in hooks, and are never written to reports.

- **Hooks run with your privileges.** A `command` hook inherits the process environment plus
  the `.env` files of the experiment, including any secret in them. It never runs through a
  shell, and its whole process tree is killed when it times out.
- **Command paths.** A relative path with a folder runs from the experiment's folder. A bare
  program name is looked up by `octoqai` itself, only in the absolute folders of the hook's
  `PATH`, never in the current folder or a relative entry; on Windows only `.exe` programs are
  found.
  Keep writable folders out of that `PATH`, or write the full path.
- **SQL hooks.** `${VAR}` is rejected in `sql` hooks, because its value would be pasted into the
  statement unescaped. Database checks bind values through `params`.
- **Database checks** roll every query back, but only a read-only database user guarantees that
  nothing is written. A query returns at most 10,000 rows; bound watched queries, because the
  cap does not limit the size of a single value.
- **Judge requests** carry the transcript, the rubric and the reference to the configured provider,
  after each text is redacted: canaries, secrets, personal data and the values referenced through
  `${VAR}` (plus the secrets of the database URL) become placeholders. A value that only sits in
  hook environment or in an `.env` entry no `${VAR}` refers to is not known to the redactor.
  A sensitive value split across two texts is not sent: the check is not evaluable. Only
  contiguous splits are detected (halves with other text between them, or URL-encoded halves,
  are not). Redaction can
  over-mask: a referenced value such as `pass` also erases that word from the rubric and the
  replies the judge reads, so keep short common words out of `${VAR}` values.
- **What leaves your machine.** With a `judge` or a `simulated_user` configured, the redacted
  transcript, the rubric, the reference, the goal and the persona go to the provider you chose
  (Anthropic over its API, or an Ollama server). Octoqai redacts what it knows to be sensitive,
  but what the agent says is the agent's data: the data policy with the provider (retention,
  training, region) is yours to check before you point a real experiment at it. Use `ollama` on
  loopback to keep everything on the machine.
- **The key** reaches the provider only through a whole `${VAR}` in `api_key` (never a literal,
  never in a rubric, goal or persona, which reject `${VAR}`). It is not shown in reprs, errors,
  reports or output. Keep the `.env` file out of git: `examples/judge/.env.example` holds names
  only.
- **A judge can be talked into a verdict.** Agent replies sit inside nonce-delimited blocks and the
  judge is told not to follow orders found in them, but prompt injection is a residual risk that
  no prompt removes: a reply written to persuade the judge may get a `pass`.
- **A judge `pass` never overrides a deterministic result.** Leaks, forbidden database changes and
  failed `expect` or `db` checks keep their effect on the verdict and the exit code whatever the
  judge says; the judge only adds checks.
- **Judge rationales** reach the report after redaction and truncation to 1,000 characters, but a
  rationale can repeat agent data the redactor does not recognise.
- **Simulated user requests** carry the goal, the persona, the simulated messages (the opening
  included) and the agent replies, every text redacted like the judge inputs: canaries, secrets,
  personal data and `${VAR}` values (for example one that reaches a goal through
  `{{ data.* }}`) become placeholders, and a referenced value can over-mask words, as in rubrics.
  The agent replies sit inside a nonce-delimited data block and the model is told never to follow
  orders found in them. The agent still receives the simulated messages as written, and the leak
  scan does not count the agent repeating a value the simulated user said. If a sensitive value is
  split across the goal, the persona or the turns, the simulator is not called and the trial is
  not evaluable (`simulator_error`); a leak or forbidden change seen before that still counts.
  An agent reply can talk the simulator into ending early (`stop_reason` `done`), which
  shrinks coverage: check `stop_reason` in the report.
  Sampling is not deterministic, so a paired A/B comparison is noisier with a simulated user:
  use more trials.
- **Ollama has no authentication.** By default `base_url` must be loopback (`127.0.0.1`,
  `localhost` or `[::1]`). With `allow_remote: true` the prompts, which can contain agent replies
  and test data, travel to that server unauthenticated and, over `http`, unencrypted: use it
  only on a network you trust. The URL may not carry credentials, a path, a query or a fragment.
  Proxy variables from the environment (`HTTP_PROXY` and similar) are ignored on purpose, so a
  prompt never goes through a proxy. Responses are capped at 1 MiB, never decompressed and never
  repeated in an error.
- **Errors** from database drivers and hooks are reported as their class name or a fixed
  message, never as their text, which can repeat stored values or connection details.
