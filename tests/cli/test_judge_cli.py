import pytest

from octoqai.cli import main
from octoqai.cli.main import _unreachable_messages
from octoqai.errors import ConfigError
from octoqai.experiment import Experiment
from octoqai.providers.budget import BUDGET_EXHAUSTED, LlmStats
from octoqai.running import RunResult
from support import samples
from support.agents import FakeSseAgent
from support.assertions import assert_exit, assert_not_in_outputs
from support.builders import ExperimentBuilder, judge_check, simulated, turn, write_env_file
from support.hooks import FakeCommands
from support.providers import DEFAULT_MODEL, FakeLlm, ScriptedUser

pytestmark = [pytest.mark.integration, pytest.mark.regression]

PROVIDER_ERROR = "AuthenticationError"
LEAK = "TEN-B-0042"


@pytest.fixture
def judged(experiment, tmp_path):
    def _write(max_calls=30, trials=1, cases=1):
        experiment.judge(model=DEFAULT_MODEL, max_calls=max_calls).trials(trials).threshold(0.5)
        for index in range(cases):
            experiment.case(f"c{index}", turn("Hola", judge=[judge_check("j")]))
        return str(experiment.write(tmp_path))

    return _write


@pytest.fixture
def simulated_case(experiment, tmp_path):
    def _write(simulator_calls=5, judge_calls=5, max_turns=2):
        experiment.judge(max_calls=judge_calls).simulated_user(max_calls=simulator_calls)
        experiment.trials(1)
        experiment.case(
            "c1", simulated=simulated(max_turns=max_turns, each_turn=[{"contains": "."}])
        )
        return str(experiment.write(tmp_path))

    return _write


def run_with_stats(stats):
    exp = Experiment.model_validate(ExperimentBuilder().build())
    return RunResult(exp, [], "t0", "t1", llm={"judge": stats})


def stats(calls, errors):
    return LlmStats(calls, errors, None, None, {})


def always(verdict, times=30):
    judge = FakeLlm()
    for _ in range(times):
        judge.judges(verdict)
    return judge


def cli_args(path, tmp_path):
    return ["run", path, "--out", str(tmp_path / "o")]


def test_budget_exhaustion_is_excluded_when_deciding_unreachable():
    mixed = run_with_stats(stats(1, {PROVIDER_ERROR: 1, BUDGET_EXHAUSTED: 1}))
    assert _unreachable_messages(mixed) == [
        f"Error: judge unreachable in every call ({PROVIDER_ERROR})."
    ]
    assert _unreachable_messages(run_with_stats(stats(0, {BUDGET_EXHAUSTED: 1}))) == []


@pytest.mark.smoke
@pytest.mark.parametrize(("verdict", "code"), [("pass", 0), ("fail", 1), ("unsure", 2)])
def test_judge_verdict_decides_the_exit_code(verdict, code, judged, run_cli, fake_agent):
    judge = always(verdict)
    result = run_cli("run", judged(cases=30), agent=fake_agent.answers("Hola."), judge=judge)
    assert_exit(result.code, code)
    assert len(judge.requests) == 30


def test_usage_lines_report_calls_against_the_cap(judged, run_cli, fake_agent):
    judge = always("pass", 2).uses(7, 3)
    result = run_cli(
        "run", judged(max_calls=5, trials=2), agent=fake_agent.answers("Hola."), judge=judge
    )
    assert "judge calls: 2/5" in result.out
    assert "judge tokens: in=14 out=6" in result.out
    assert "simulated_user" not in result.out


def test_judge_is_closed_after_the_run(judged, run_cli, fake_agent):
    judge = FakeLlm().judges("pass")
    run_cli("run", judged(), agent=fake_agent.answers("Hola."), judge=judge)
    assert judge.closed


def test_unreachable_judge_exits_3_with_the_class_and_writes_the_report(
    judged, run_cli, fake_agent
):
    judge = FakeLlm().fails(PROVIDER_ERROR).fails(PROVIDER_ERROR)
    result = run_cli("run", judged(trials=2), agent=fake_agent.answers("Hola."), judge=judge)
    assert_exit(result.code, 3)
    assert f"Error: judge unreachable in every call ({PROVIDER_ERROR})." in result.err
    assert result.report()["verdict"]["status"] == "insufficient"


def test_unsure_judge_is_not_unreachable(judged, run_cli, fake_agent):
    judge = always("unsure")
    result = run_cli("run", judged(), agent=fake_agent.answers("Hola."), judge=judge)
    assert_exit(result.code, 2)
    assert "unreachable" not in result.err


def test_partial_provider_errors_are_not_unreachable(judged, run_cli, fake_agent):
    judge = FakeLlm().fails(PROVIDER_ERROR).judges("pass")
    result = run_cli(
        "run", judged(max_calls=2, trials=2), agent=fake_agent.answers("Hola."), judge=judge
    )
    assert "unreachable" not in result.err
    assert_exit(result.code, 2)


def test_budget_exhaustion_alone_is_not_unreachable(experiment, tmp_path, run_cli, fake_agent):
    experiment.judge(max_calls=1).trials(1)
    experiment.case("c1", turn("Hola", judge=[judge_check("j")]))
    path = str(experiment.write(tmp_path))
    result = run_cli(
        "run", path, agent=fake_agent.answers("Hola."), judge=FakeLlm().judges("pass").fails("X")
    )
    assert "unreachable" not in result.err


def test_unreachable_judge_keeps_exit_1_when_the_agent_leaks(
    experiment, tmp_path, run_cli, fake_agent
):
    experiment.streaming().judge().security(canaries=[{"value": LEAK}]).trials(1)
    experiment.case("c1", turn("Hola", judge=[judge_check("j")]))
    path = str(experiment.write(tmp_path))
    result = run_cli(
        "run", path, agent=FakeSseAgent().leaks(LEAK), judge=FakeLlm().fails(PROVIDER_ERROR)
    )
    assert_exit(result.code, 1)
    assert f"judge unreachable in every call ({PROVIDER_ERROR})" in result.err


def test_most_frequent_error_class_is_named(judged, run_cli, fake_agent):
    judge = FakeLlm().fails("B").fails("A").fails("A")
    result = run_cli(
        "run", judged(max_calls=3, trials=3), agent=fake_agent.answers("Hola."), judge=judge
    )
    assert "(A)." in result.err


def test_error_class_ties_are_broken_alphabetically(judged, run_cli, fake_agent):
    judge = FakeLlm().fails("B").fails("A")
    result = run_cli(
        "run", judged(max_calls=3, trials=2), agent=fake_agent.answers("Hola."), judge=judge
    )
    assert "(A)." in result.err


def test_worst_case_over_the_cap_exits_3_before_anything_runs(judged, run_cli, fake_agent):
    judge = FakeLlm().judges("pass")
    result = run_cli(
        "run", judged(max_calls=2, trials=3), agent=fake_agent.answers("Hola."), judge=judge
    )
    assert_exit(result.code, 3)
    assert "Config error: worst case needs 3 judge call(s) but judge.max_calls is 2" in result.err
    assert fake_agent.requests == [] and judge.requests == []


def test_simulator_worst_case_over_the_cap_exits_3(simulated_case, run_cli, fake_agent):
    user = ScriptedUser().says("a", "b")
    result = run_cli(
        "run",
        simulated_case(simulator_calls=1, max_turns=2),
        agent=fake_agent.answers("Hola."),
        judge=FakeLlm(),
        simulator=user,
    )
    assert_exit(result.code, 3)
    assert (
        "Config error: worst case needs 2 simulated_user call(s) "
        "but simulated_user.max_calls is 1" in result.err
    )
    assert fake_agent.requests == [] and user.requests == []


def test_cap_is_computed_after_the_tags_filter(experiment, tmp_path, run_cli, fake_agent):
    experiment.judge(max_calls=1).trials(1)
    experiment.case("a", turn("Hola", judge=[judge_check("j")]), tags=["smoke"])
    experiment.case("b", turn("Hola", judge=[judge_check("j")]), tags=["slow"])
    path = str(experiment.write(tmp_path))
    agent = fake_agent.answers("Hola.")
    selected = run_cli("run", path, "--tags", "smoke", agent=agent, judge=FakeLlm().judges("pass"))
    assert_exit(selected.code, 2)
    assert "worst case" not in selected.err
    everything = run_cli("run", path, agent=agent, judge=FakeLlm())
    assert_exit(everything.code, 3)
    assert "worst case needs 2 judge call(s) but judge.max_calls is 1" in everything.err


@pytest.mark.usefixtures("python_on_path")
def test_failing_cap_runs_no_before_run_hook(experiment, tmp_path, run_cli, fake_agent):
    experiment.judge(max_calls=1).trials(2)
    experiment.case("c1", turn("Hola", judge=[judge_check("j")]))
    experiment.hooks(before_run=[{"command": ["python", "prepara.py"]}])
    commands = FakeCommands()
    result = run_cli(
        "run",
        str(experiment.write(tmp_path)),
        agent=fake_agent,
        commands=commands,
        judge=FakeLlm(),
    )
    assert_exit(result.code, 3)
    assert "worst case needs 2 judge call(s) but judge.max_calls is 1" in result.err
    assert commands.calls == []


def test_factory_is_not_called_without_judge_or_simulator_sections(
    experiment, tmp_path, fake_agent
):
    experiment.case("c1", turn("Hola", {"contains": "Hola"}))

    def factory(role, settings):
        raise AssertionError("the factory must not be called")

    code = main(
        cli_args(str(experiment.write(tmp_path)), tmp_path),
        transport=fake_agent.answers("Hola").transport(),
        adapter_kwargs=fake_agent.adapter_kwargs(),
        llm_factory=factory,
    )
    assert code != 3


def test_factory_config_error_exits_3_before_the_agent_is_called(
    judged, tmp_path, fake_agent, capsys
):
    def factory(role, settings):
        raise ConfigError("the anthropic provider needs: pip install 'octoqai[anthropic]'")

    code = main(
        cli_args(judged(), tmp_path),
        transport=fake_agent.answers("Hola").transport(),
        adapter_kwargs=fake_agent.adapter_kwargs(),
        llm_factory=factory,
    )
    assert_exit(code, 3)
    assert "Config error: the anthropic provider needs" in capsys.readouterr().err
    assert fake_agent.requests == []


def test_factory_receives_the_role_and_its_settings_and_ports_are_closed(
    simulated_case, tmp_path, fake_agent
):
    seen = []
    ports = {"judge": FakeLlm().judges("pass"), "simulated_user": ScriptedUser().says("a", "b")}

    def factory(role, settings):
        seen.append((role, settings.max_calls))
        return ports[role]

    main(
        cli_args(simulated_case(simulator_calls=4, judge_calls=6), tmp_path),
        transport=fake_agent.answers("Hola.").transport(),
        adapter_kwargs=fake_agent.adapter_kwargs(),
        llm_factory=factory,
    )
    assert sorted(seen) == [("judge", 6), ("simulated_user", 4)]
    assert all(port.closed for port in ports.values())


def test_unreachable_simulator_exits_3(simulated_case, run_cli, fake_agent):
    result = run_cli(
        "run",
        simulated_case(),
        agent=fake_agent.answers("Hola."),
        judge=FakeLlm(),
        simulator=FakeLlm().fails(PROVIDER_ERROR),
    )
    assert_exit(result.code, 3)
    assert f"Error: simulated user unreachable in every call ({PROVIDER_ERROR})." in result.err
    assert "simulated_user calls: 1/5" in result.out
    assert result.report()["verdict"]["status"] == "insufficient"
    assert "agent unreachable" not in result.err


def test_unreachable_simulator_keeps_exit_1_when_the_agent_leaks(
    experiment, tmp_path, run_cli, fake_agent
):
    experiment.streaming().security(canaries=[{"value": LEAK}]).trials(1)
    experiment.judge().simulated_user()
    experiment.case(
        "c1", simulated=simulated(max_turns=2, opening="Hola", each_turn=[{"contains": "."}])
    )
    result = run_cli(
        "run",
        str(experiment.write(tmp_path)),
        agent=FakeSseAgent().leaks(LEAK),
        judge=FakeLlm(),
        simulator=FakeLlm().fails(PROVIDER_ERROR),
    )
    assert_exit(result.code, 1)


@pytest.mark.security
@pytest.mark.parametrize(
    "secret", [samples.ANTHROPIC_KEY_NO_PREFIX, "".join(samples.SPLIT_SECRET)], ids=["plain", "sk"]
)
def test_judge_key_from_env_is_never_printed_or_reported(
    secret, experiment, tmp_path, run_cli, fake_agent
):
    write_env_file(tmp_path, ".env", JUDGE_KEY=secret)
    experiment.environments("qa", data=None)
    experiment.trials(1).judge(
        provider="anthropic", api_key="${JUDGE_KEY}", model=DEFAULT_MODEL, max_calls=3
    )
    experiment.case("c1", turn("Hola", judge=[judge_check("j")]))
    result = run_cli(
        "run",
        str(experiment.write(tmp_path)),
        "--env",
        "qa",
        agent=fake_agent.answers("Hola."),
        judge=FakeLlm().fails(PROVIDER_ERROR),
    )
    assert_exit(result.code, 3)
    assert_not_in_outputs(secret, result.out, result.err, result.report_text())
