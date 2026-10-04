"""Check every asset's wrapper against the capabilities its asset.ttl declares.

Run:  python -m src.conformance.check [assets_dir]

Three failure modes, in descending order of how badly they mislead you:

  OVERCLAIM   asset.ttl declares a capability the wrapper does not implement, or
              implements but which raises when called. The compiler will select an
              operator that cannot run.
  UNDERCLAIM  the wrapper implements something asset.ttl does not declare. Harmless
              to correctness, but the compiler will never choose it.
  BROKEN      the asset could not be loaded at all.
"""

from __future__ import annotations

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

from src.wrappers.base import CAPABILITY_METHOD  # noqa: E402
from src.wrappers.loader import load_asset_class  # noqa: E402

CAP = Namespace("https://example.org/dra/cap#")


@dataclass
class Result:
    path: Path
    asset_id: str = "?"
    overclaimed: set[str] = field(default_factory=set)
    underclaimed: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.overclaimed or self.errors)


def load_declaration(ttl: Path) -> tuple[str, set[str], str]:
    g = Graph().parse(ttl, format="turtle")
    subj = next(g.subjects(CAP.implementedBy, None), None)
    if subj is None:
        raise ValueError("no subject carries cap:implementedBy")
    asset_id = str(next(g.objects(subj, CAP.assetId), subj))
    impl = str(next(g.objects(subj, CAP.implementedBy)))
    declared = {str(o).rsplit("#", 1)[-1] for o in g.objects(subj, CAP.hasCapability)}
    return asset_id, declared, impl


def instantiate(impl: str, asset_dir: Path):
    """Load "file.py:ClassName" relative to the asset folder.

    Loaded by path, not module name, so asset directories are free to use any
    naming the registry wants rather than being constrained to Python identifiers.
    The loading itself lives in src/wrappers/loader.py, shared with the assets'
    own tests so the two cannot disagree about how a wrapper is imported.
    """
    return load_asset_class(impl, asset_dir)()


def check_asset(ttl: Path) -> Result:
    res = Result(path=ttl.parent)
    try:
        res.asset_id, declared, impl = load_declaration(ttl)
    except Exception as exc:
        res.errors.append(f"could not read declaration: {exc}")
        return res

    try:
        asset = instantiate(impl, ttl.parent)
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


def main(root: str = "assets") -> int:
    base = Path(root)
    found = sorted(base.rglob("asset.ttl"))
    found = [p for p in found if "_template" not in p.parts]
    if not found:
        print(f"no asset.ttl under {base.resolve()}")
        return 0

    pending = sorted(
        p.parent
        for p in base.rglob("SOURCE.md")
        if "_template" not in p.parts and not (p.parent / "asset.ttl").exists()
    )

    failures = 0
    for ttl in found:
        res = check_asset(ttl)
        status = "ok  " if res.ok else "FAIL"
        print(f"[{status}] {res.asset_id:<34} {res.path}")
        for cap in sorted(res.overclaimed):
            print(f"         OVERCLAIM  {cap}")
        for cap in sorted(res.underclaimed):
            print(f"         underclaim {cap} (implemented, not declared)")
        for err in res.errors:
            print(f"         {err}")
        failures += not res.ok

    for d in pending:
        print(f"[pend] {str(d):<34} awaiting source - see {d}/SOURCE.md")

    print()
    summary = f'{len(found) - failures}/{len(found)} assets conform'
    if pending:
        summary += f', {len(pending)} pending'
    print(summary)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
