import io
import tokenize

import pytest

from support.paths import SRC

pytestmark = pytest.mark.quality


def test_source_has_no_comments():
    offenders = []
    for path in SRC.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                offenders.append(f"{path.relative_to(SRC)}:{token.start[0]}")
    assert offenders == []
