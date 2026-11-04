import importlib
import json

import pytest

from support import samples
from support.assertions import assert_linear_steps

pytestmark = [pytest.mark.unit, pytest.mark.regression, pytest.mark.security]

JSON_TEXT = importlib.import_module("octoqai.security._json_text")
unescape = JSON_TEXT.unescape


def test_unescape_matches_json_on_every_valid_escape():
    text = json.dumps('TEN-B-0042 ñ \U0001f600 / " \\ \b\f\n\r\t').replace("/", "\\/")
    assert unescape(text[1:-1]) == json.loads(text)


def test_unescape_undoes_the_escapes_of_text_that_is_not_json():
    assert (
        unescape(samples.CLEAN_ESCAPED) == 'linea\nuna ñandú /ruta/ "citado" \\ \U0001f600 fin \\u'
    )


@pytest.mark.parametrize("text", ["\\u", "\\ud800", "\\uZZZZ", "fin\\", "\\x41", "\\u12"])
def test_unescape_never_raises(text):
    assert isinstance(unescape(text), str)


@pytest.mark.parametrize("text", ["\\u", "\\uZZZZ", "fin\\", "\\x41", "\\u12 x"])
def test_unescape_leaves_invalid_escapes_as_they_are(text):
    assert unescape(text) == text


@pytest.mark.parametrize(
    ("text", "plain"),
    [("\\ud800", "�"), ("\\udc00x", "�x"), ("\\ud800\\u0041", "�A")],
)
def test_unescape_turns_a_lone_surrogate_into_the_replacement_character(text, plain):
    assert unescape(text) == plain


def test_unescape_does_not_reuse_an_escaped_backslash():
    assert unescape("\\\\u0041 \\\\\\/") == "\\u0041 \\/"


@pytest.mark.parametrize("unit", ["\\u0041", "\\", "\\ud83d\\ude00"])
def test_unescape_runs_in_linear_steps(unit):
    assert_linear_steps(unescape, lambda n: unit * n, 300, JSON_TEXT.__file__)


def test_linear_steps_check_fails_loudly_on_a_module_that_never_runs():
    with pytest.raises(AssertionError, match="not_a_module.py"):
        assert_linear_steps(unescape, lambda n: "x" * n, 300, "not_a_module.py")
