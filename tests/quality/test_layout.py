import ast
import pkgutil
import tomllib

import pytest

import octoqai
from support.paths import REPO_ROOT

pytestmark = pytest.mark.quality

TESTS = REPO_ROOT / "tests"
SUITES = {"unit", "integration", "e2e", "quality"}
MARKERS = SUITES | {"smoke", "regression", "security"}


def _feature_packages() -> list[str]:
    return [m.name for m in pkgutil.iter_modules(octoqai.__path__) if m.ispkg]


def _test_modules() -> list:
    return sorted(TESTS.rglob("test_*.py"))


def _declared_suites(path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
        ):
            return {
                n.attr
                for n in ast.walk(node.value)
                if isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Attribute)
                and n.value.attr == "mark"
            } & SUITES
    return set()


def test_every_package_has_a_mirror_test_folder():
    missing = [p for p in _feature_packages() if not list((TESTS / p).glob("test_*.py"))]
    assert missing == []


def test_no_test_modules_at_tests_root():
    assert [p.name for p in TESTS.glob("test_*.py")] == []


def test_every_test_module_declares_a_suite():
    offenders = [str(p.relative_to(TESTS)) for p in _test_modules() if not _declared_suites(p)]
    assert offenders == []


def test_every_data_file_is_used():
    unused = []
    for data in sorted(TESTS.rglob("data/*.yaml")):
        folder = data.parent.parent
        sources = " ".join(p.read_text(encoding="utf-8") for p in folder.glob("test_*.py"))
        if f'"{data.stem}"' not in sources:
            unused.append(str(data.relative_to(TESTS)))
    assert unused == []


def test_conftest_only_wires_plugins():
    tree = ast.parse((TESTS / "conftest.py").read_text(encoding="utf-8"))
    assert len(tree.body) == 1
    node = tree.body[0]
    assert isinstance(node, ast.Assign)
    assert [t.id for t in node.targets if isinstance(t, ast.Name)] == ["pytest_plugins"]


def test_markers_are_registered():
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    options = config["tool"]["pytest"]["ini_options"]
    registered = {line.split(":", 1)[0].strip() for line in options["markers"]}
    assert registered == MARKERS
    assert "--strict-markers" in options["addopts"]


def test_architecture_doc_lists_every_package():
    doc = (REPO_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    missing = [p for p in _feature_packages() if f"`{p}/`" not in doc]
    assert missing == []
    assert "`errors.py`" in doc
