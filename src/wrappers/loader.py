"""Load an asset wrapper by file path.

Asset folders are named for the registry (`system/st-connectivity`), not for Python,
so wrappers are loaded by path rather than imported by module name. Two consequences
make this worth having in one place rather than repeated per asset:

- Every asset's wrapper is called `model.py`. After a `sys.path` insert,
  `from model import X` resolves to whichever asset was loaded first. That is
  invisible with one asset and silently wrong with two, and it fails as a confusing
  ImportError rather than as a name clash.
- A wrapper using `@dataclass` with `from __future__ import annotations` cannot be
  imported at all unless its module is in `sys.modules` first, because dataclasses
  resolves string annotations through `sys.modules[cls.__module__]`. The failure
  surfaces as an AttributeError on NoneType, nowhere near the cause.

Both the conformance harness and each asset's tests go through here, so the two
cannot drift apart.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def module_name_for(model_path: Path) -> str:
    """Unique module name derived from the asset folder, e.g. `dra_asset_gmpe`."""
    return "dra_asset_" + model_path.parent.name.replace("-", "_")


def load_asset_module(model_path: str | Path) -> ModuleType:
    """Import an asset's wrapper file under a name unique to its folder."""
    path = Path(model_path).resolve()
    name = module_name_for(path)
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]  # do not leave a half-initialised module registered
        raise
    return module


def load_asset_class(impl: str, asset_dir: str | Path) -> type:
    """Resolve a `cap:implementedBy` string, "file.py:ClassName", to the class."""
    rel, _, class_name = impl.partition(":")
    if not rel or not class_name:
        raise ValueError(f'expected "file.py:ClassName", got {impl!r}')
    module = load_asset_module(Path(asset_dir) / rel)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"{rel} defines no {class_name}") from exc
