import io
import tokenize

import pytest

from support.paths import SRC, TESTS

pytestmark = pytest.mark.quality

SLEEPS = ("time." + "sleep(", "asyncio." + "sleep(")


def test_source_has_no_comments():
    offenders = []
    for path in SRC.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                offenders.append(f"{path.relative_to(SRC)}:{token.start[0]}")
    assert offenders == []


def test_tests_never_sleep():
    offenders = [
        str(path.relative_to(TESTS))
        for path in TESTS.rglob("*.py")
        if any(needle in path.read_text(encoding="utf-8") for needle in SLEEPS)
    ]
    assert offenders == []
