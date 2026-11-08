import importlib
import pkgutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

import octoqai
from support.paths import REPO_ROOT as ROOT
from support.paths import SRC

pytestmark = pytest.mark.quality


def _subpackages() -> list[str]:
    return [m.name for m in pkgutil.iter_modules(octoqai.__path__, "octoqai.") if m.ispkg]


def _installed_script(name: str) -> Path | None:
    folders = (Path(sysconfig.get_path("scripts")), Path(sys.executable).parent)
    suffix = ".exe" if sys.platform == "win32" else ""
    found = [folder / f"{name}{suffix}" for folder in folders]
    return next((path for path in found if path.is_file()), None)


def test_import_contracts_hold():
    exe = _installed_script("lint-imports")
    assert exe is not None, "lint-imports is not installed"
    result = subprocess.run(
        [exe], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_package_exports_public_api():
    packages = _subpackages()
    assert packages, "octoqai has no feature packages"
    for name in packages:
        module = importlib.import_module(name)
        exported = getattr(module, "__all__", None)
        assert exported, f"{name} does not declare __all__"
        missing = [item for item in exported if not hasattr(module, item)]
        assert missing == [], f"{name}.__all__ lists missing names: {missing}"


@pytest.mark.security
def test_only_hooks_start_processes():
    offenders = [
        str(p.relative_to(SRC))
        for p in SRC.rglob("*.py")
        if "subprocess" in p.read_text(encoding="utf-8") and p.parent.name != "hooks"
    ]
    assert offenders == []


@pytest.mark.security
def test_no_shell_in_source():
    needle = "shell" + "=True"
    assert [p.name for p in SRC.rglob("*.py") if needle in p.read_text(encoding="utf-8")] == []
