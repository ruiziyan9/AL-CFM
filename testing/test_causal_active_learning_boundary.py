"""The core and evaluation modules of causal_active_learning never import
CITNP (only integrations/citnp may)."""

import ast
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "causal_active_learning"


def _core_modules() -> list[Path]:
    return [
        p for p in PACKAGE.rglob("*.py")
        if "integrations" not in p.relative_to(PACKAGE).parts
    ]  # fmt: skip


def _module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def test_core_source_never_imports_citnp():
    offenders = []
    for path in _core_modules():
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(n == "CITNP" or n.startswith("CITNP.") for n in names):
                offenders.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
    assert not offenders, offenders


def test_importing_the_core_leaves_citnp_unloaded():
    modules = sorted(_module_name(p) for p in _core_modules())
    code = (
        "import importlib, sys\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        "assert not any(m == 'CITNP' or m.startswith('CITNP.') "
        "for m in sys.modules), sorted(m for m in sys.modules if 'CITNP' in m)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=PACKAGE.parent,
        capture_output=True, text=True,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert len(modules) >= 15
