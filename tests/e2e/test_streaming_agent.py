import pytest
import yaml

from support.assertions import assert_exit, assert_not_in_outputs
from support.paths import EXAMPLES

pytestmark = [pytest.mark.e2e, pytest.mark.regression]

CANARY = "CANARY-7f3a9c2e"


@pytest.fixture
def streaming_example(tmp_path, toy_sse_agent_url):
    data = yaml.safe_load((EXAMPLES / "streaming" / "experiment.yaml").read_text(encoding="utf-8"))
    data["agent"]["url"] = toy_sse_agent_url
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return str(path)


@pytest.mark.security
def test_real_sse_leak_is_no_go_with_masked_report(streaming_example, run_cli):
    result = run_cli("run", streaming_example)
    assert_exit(result.code, 1)
    assert_not_in_outputs(CANARY, result.out, result.err, result.report_text())
    leaks = [
        leak
        for trial in result.report()["trials"]
        for turn in trial["turns"]
        for leak in turn["leaks"]
    ]
    assert leaks and {leak["masked"] for leak in leaks} == {"CAN…c2e"}
    assert {leak["location"] for leak in leaks} == {"fragment 3"}


@pytest.mark.smoke
def test_real_sse_agent_measures_time_to_first_fragment(streaming_example, run_cli):
    result = run_cli("run", streaming_example, "--tags", "smoke")
    assert_exit(result.code, 2)
    report = result.report()
    assert report["verdict"]["status"] == "insufficient"
    assert all(trial["leaks"] == 0 for trial in report["trials"])
    turn = report["trials"][0]["turns"][0]
    assert turn["error"] is None and turn["ttft_ms"] is not None and turn["fragments"] == 3
