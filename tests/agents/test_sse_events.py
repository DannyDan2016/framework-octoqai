import pytest

from octoqai.agents import SseEvent, parse_sse
from support.paths import cases

pytestmark = [pytest.mark.unit, pytest.mark.regression]


@pytest.mark.parametrize("row", cases(__file__, "sse_events"))
def test_parses_sse_lines(row):
    expected = [SseEvent(event, data) for event, data in row["events"]]
    assert list(parse_sse(row["lines"])) == expected


@pytest.mark.smoke
def test_is_lazy_so_partial_streams_can_be_scanned():
    events = parse_sse(iter(["data: a", "", "data: b"]))
    assert next(events) == SseEvent("message", "a")
    assert next(events) == SseEvent("message", "b")
