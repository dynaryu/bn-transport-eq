"""Single-file assets: one ``<name>.asset.json`` per knowledge asset.

A folder asset spreads its contract, provenance and numbers over ``asset.ttl``,
``SOURCE.md`` and ``params/``. A single-file asset carries all three in one JSON
object, keyed by the ``cap:`` vocabulary, so that it can be contributed, reviewed
and withdrawn on its own without touching its neighbours. Several of them may share
one wrapper: ``implementedBy`` resolves against the folder the file sits in.

The suffix is the discovery rule. A bare ``*.json`` under ``assets/`` is somebody's
parameter table or test fixture, not a declaration, and must not be picked up as one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ASSET_FILE_SUFFIX = ".asset.json"


def is_asset_file(path: str | Path) -> bool:
    return Path(path).name.endswith(ASSET_FILE_SUFFIX)


def asset_name(path: str | Path) -> str:
    """The file's name without the suffix, e.g. ``HWB17``. The last segment of ``assetId``."""
    name = Path(path).name
    if not name.endswith(ASSET_FILE_SUFFIX):
        raise ValueError(f"{name} is not a single-file asset (*{ASSET_FILE_SUFFIX})")
    return name[: -len(ASSET_FILE_SUFFIX)]


def asset_files(folder: str | Path) -> list[Path]:
    """Every single-file asset directly inside ``folder``, in name order."""
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and is_asset_file(p))


def read_asset_file(path: str | Path) -> dict[str, Any]:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"{Path(path).name} must hold one JSON object")
    return record


def expected_asset_id(path: str | Path) -> str:
    """``<category>/<folder>/<name>``: the id a file's location commits it to.

    Deriving the id from the path, and checking the declared one against it, is what
    keeps two contributed files from claiming the same identity.
    """
    path = Path(path).resolve()
    return f"{path.parent.parent.name}/{path.parent.name}/{asset_name(path)}"
