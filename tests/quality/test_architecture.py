import importlib
import pkgutil
import shutil
import subprocess
import sys
from pathlib import Path

import octoqai

ROOT = Path(__file__).resolve().parents[2]


def _subpackages() -> list[str]:
    return [m.name for m in pkgutil.iter_modules(octoqai.__path__, "octoqai.") if m.ispkg]


def test_import_contracts_hold():
    exe = shutil.which("lint-imports", path=str(Path(sys.executable).parent))
    assert exe is not None, "lint-imports is not installed"
    result = subprocess.run([exe], cwd=ROOT, capture_output=True, text=True)
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
