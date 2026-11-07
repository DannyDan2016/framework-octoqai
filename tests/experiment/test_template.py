import re

import pytest

from octoqai.errors import ConfigError
from octoqai.experiment import render, render_text

pytestmark = [pytest.mark.unit, pytest.mark.regression]

CTX = {
    "message": "Hola ñandú",
    "conversation_id": "c1",
    "turn_index": 0,
    "history": [],
    "variant": {"temperature": 0, "model": "llama3.1"},
}


@pytest.mark.smoke
def test_whole_placeholder_preserves_type_including_zero():
    assert render({"t": "{{ variant.temperature }}"}, CTX) == {"t": 0}


def test_inline_placeholder_interpolates():
    assert render("model={{ variant.model }}", CTX) == "model=llama3.1"


def test_nested_lists_and_history_passthrough():
    assert render({"m": ["{{ message }}"], "h": "{{ history }}"}, CTX) == {
        "m": ["Hola ñandú"],
        "h": [],
    }


def test_unknown_variable_raises():
    with pytest.raises(ConfigError, match="variant.top_p"):
        render("{{ variant.top_p }}", CTX)


def test_non_string_scalars_untouched():
    assert render({"n": 3, "b": False, "x": None}, CTX) == {"n": 3, "b": False, "x": None}


def test_template_is_not_mutated():
    template = {"m": ["{{ message }}"]}
    render(template, CTX)
    assert template == {"m": ["{{ message }}"]}


def test_render_text_escapes_inserted_values_only():
    text = render_text("ref {{ data.ref }}.", {"data": {"ref": "(01)"}}, escape=re.escape)
    assert text == r"ref \(01\)."


def test_render_text_unknown_variable_raises():
    with pytest.raises(ConfigError, match="data.x"):
        render_text("{{ data.x }}", {"data": {}})


def test_literal_applies_to_text_around_placeholders_only():
    template = {"a": "x-{{ message }}-y", "b": ["plain"], "c": "{{ message }}"}
    result = render(template, CTX, literal=str.upper)
    assert result == {"a": "X-Hola ñandú-Y", "b": ["PLAIN"], "c": "Hola ñandú"}


def test_literal_is_not_applied_to_a_string_that_is_one_placeholder():
    assert render(" {{ message }} ", {"message": "a b"}, literal=str.upper) == "a b"


def test_render_without_literal_keeps_text_as_is():
    assert render("a-{{ message }}-b", CTX) == "a-Hola ñandú-b"
