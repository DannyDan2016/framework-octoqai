from pathlib import Path

import pytest
import yaml


def base_experiment(**overrides):
    data = {
        "name": "demo",
        "agent": {
            "url": "http://agent.local/chat",
            "headers": overrides.pop("headers", {}),
            "request": {"message": "{{ message }}"},
            "response": {"answer": "$.answer"},
        },
        "variants": overrides.pop("variants", [{"name": "base", "params": {"temperature": 0}}]),
        "thresholds": {"pass_rate": 0.8},
        "cases": [
            {
                "id": "c1",
                "turns": [{"user": "Hola", "expect": overrides.pop("expect", [])}],
            }
        ],
    }
    data.update(overrides)
    return data


@pytest.fixture
def write_yaml(tmp_path: Path):
    def _write(**overrides) -> Path:
        path = tmp_path / "experiment.yaml"
        path.write_text(
            yaml.safe_dump(base_experiment(**overrides), allow_unicode=True), encoding="utf-8"
        )
        return path

    return _write
