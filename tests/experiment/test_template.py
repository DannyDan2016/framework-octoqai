import pytest

from octoqai.errors import ConfigError
from octoqai.experiment import render

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
