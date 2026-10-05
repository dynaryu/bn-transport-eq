"""Check every asset's wrapper against the capabilities its declaration makes.

Run:  python -m src.conformance.check [assets_dir]

A declaration is either an ``asset.ttl`` in the asset's own folder, or a single-file
asset, ``<name>.asset.json``, which must first observe the schema derived from the
ontology (src/conformance/schema.py) and carry the id its location commits it to.

Three failure modes, in descending order of how badly they mislead you:

  OVERCLAIM   the declaration names a capability the wrapper does not implement, or
              implements but which raises when called. The compiler will select an
              operator that cannot run.
  UNDERCLAIM  the wrapper implements something the declaration does not name.
              Harmless to correctness, but the compiler will never choose it.
  BROKEN      the asset could not be loaded at all, or its file breaks the schema.

One state that is not a failure:

  DATA        the wrapper refuses to construct because the numbers are not in the
              repository (DataUnavailable), and the declaration claims no capability.
              Claiming one in that state is an OVERCLAIM like any other.
"""

from __future__ import annotations

import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

try:
    from rdflib import Graph, Namespace
except ImportError:  # pragma: no cover
    sys.exit(
        "rdflib is required.\n"
        "  conda env create -f environment.yml   (then: conda activate dra)\n"
        "or install it into the env you are already using."
    )

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.conformance.schema import validation_errors  # noqa: E402
from src.wrappers.base import CAPABILITY_METHOD, DataUnavailable  # noqa: E402
from src.wrappers.loader import load_asset_class  # noqa: E402
from src.wrappers.record import expected_asset_id, is_asset_file, read_asset_file  # noqa: E402

CAP = Namespace("https://example.org/dra/cap#")


@dataclass
class Result:
    path: Path
    asset_id: str = "?"
    overclaimed: set[str] = field(default_factory=set)
    underclaimed: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)
    unavailable: str = ""  # why the wrapper refused to construct, if it did

    @property
    def ok(self) -> bool:
        return not (self.overclaimed or self.errors)


def load_declaration(path: Path) -> tuple[str, set[str], str]:
    """(asset id, declared capabilities, wrapper location) from either kind of declaration."""
    if is_asset_file(path):
        return load_record_declaration(path)
    g = Graph().parse(path, format="turtle")
    subj = next(g.subjects(CAP.implementedBy, None), None)
    if subj is None:
        raise ValueError("no subject carries cap:implementedBy")
    asset_id = str(next(g.objects(subj, CAP.assetId), subj))
    impl = str(next(g.objects(subj, CAP.implementedBy)))
    declared = {str(o).rsplit("#", 1)[-1] for o in g.objects(subj, CAP.hasCapability)}
    return asset_id, declared, impl


def load_record_declaration(path: Path) -> tuple[str, set[str], str]:
    """Read a single-file asset, refusing one that departs from the derived schema.

    Validation comes before anything is loaded or run: a file with a misspelt key
    would otherwise pass as an asset that merely declares less than it meant to.
    """
    record = read_asset_file(path)
    problems = validation_errors(record)
    if problems:
        raise ValueError(
            "does not observe ontology/asset.schema.json:\n"
            + "\n".join(f"           - {p}" for p in problems)
        )
    expected = expected_asset_id(path)
    if record["assetId"] != expected:
        raise ValueError(
            f"assetId {record['assetId']!r} does not match its location; expected {expected!r}"
        )
    return record["assetId"], set(record.get("hasCapability", [])), record["implementedBy"]


def instantiate(impl: str, declaration: Path):
    """Load "file.py:ClassName" relative to the folder holding the declaration.

    Loaded by path, not module name, so asset directories are free to use any
    naming the registry wants rather than being constrained to Python identifiers.
    The loading itself lives in src/wrappers/loader.py, shared with the assets'
    own tests so the two cannot disagree about how a wrapper is imported.

    A single-file asset shares its wrapper with its neighbours, so the wrapper is
    bound to that file: the smoke test then exercises the asset declared, not
    whichever one the wrapper happens to default to.
    """
    cls = load_asset_class(impl, declaration.parent)
    return cls.from_asset_file(declaration) if is_asset_file(declaration) else cls()


def check_asset(path: Path) -> Result:
    path = Path(path)
    res = Result(path=path if is_asset_file(path) else path.parent)
    try:
        res.asset_id, declared, impl = load_declaration(path)
    except Exception as exc:
        res.errors.append(f"could not read declaration: {exc}")
        return res

    try:
        asset = instantiate(impl, path)
    except DataUnavailable as exc:
        # Nothing can run, so nothing may be claimed. An empty declaration conforms.
        res.unavailable = str(exc).strip().splitlines()[0]
        res.overclaimed = set(declared)
        return res
    except Exception as exc:
        res.errors.append(f"could not instantiate {impl}: {exc}")
        return res

    implemented = asset.implemented_capabilities()
    res.overclaimed = declared - implemented
    res.underclaimed = implemented - declared

    # A method can exist and still not work. Call each declared capability once.
    try:
        smoke = asset.smoke_inputs()
    except NotImplementedError as exc:
        res.errors.append(str(exc))
        return res
    except Exception as exc:
        res.errors.append(f"smoke_inputs() failed: {exc}")
        return res

    for cap in sorted(declared & implemented):
        method = getattr(asset, CAPABILITY_METHOD[cap])
        try:
            kwargs = {k: v for k, v in smoke.items() if k in method.__code__.co_varnames}
            method(**kwargs)
        except Exception:
            res.overclaimed.add(cap)
            res.errors.append(
                f"{cap} declared and defined but raised:\n"
                + "    ".join(traceback.format_exc().splitlines(keepends=True)[-3:])
            )
    return res


def _natural(path: Path) -> list:
    """Sort key putting HWB2 before HWB10."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.as_posix())]


def find_declarations(base: Path) -> list[Path]:
    found = [p for p in Path(base).rglob("*") if p.name == "asset.ttl" or is_asset_file(p)]
    return sorted((p for p in found if "_template" not in p.parts), key=_natural)


def main(root: str = "assets") -> int:
    base = Path(root)
    found = find_declarations(base)
    if not found:
        print(f"no asset.ttl or *.asset.json under {base.resolve()}")
        return 0

    declared_dirs = {p.parent for p in found}
    pending = sorted(
        p.parent
        for p in base.rglob("SOURCE.md")
        if "_template" not in p.parts and p.parent not in declared_dirs
    )

    failures = awaiting = 0
    for declaration in found:
        res = check_asset(declaration)
        waiting = res.ok and bool(res.unavailable)
        status = "FAIL" if not res.ok else "data" if waiting else "ok  "
        print(f"[{status}] {res.asset_id:<42} {res.path}")
        for cap in sorted(res.overclaimed):
            print(f"         OVERCLAIM  {cap}")
        for cap in sorted(res.underclaimed):
            print(f"         underclaim {cap} (implemented, not declared)")
        for err in res.errors:
            print(f"         {err}")
        if res.unavailable:
            print(f"         {'declares no capability' if waiting else 'wrapper refuses'}: {res.unavailable}")
        failures += not res.ok
        awaiting += waiting

    for d in pending:
        print(f"[pend] {str(d):<42} awaiting source - see {d}/SOURCE.md")

    print()
    summary = f'{len(found) - failures}/{len(found)} assets conform'
    if awaiting:
        summary += f' ({awaiting} of them awaiting data)'
    if pending:
        summary += f', {len(pending)} pending'
    print(summary)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
