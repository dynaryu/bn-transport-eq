"""Fresh copies of source tables so a caller cannot mutate another evaluation.

Each Hazus class is its own asset file, ``assets/fragility/<folder>/<class>.asset.json``.
The engine reasons over whole tables -- classification walks every Table 7-1 row in
order -- so they are put back together here, in the shape the importer extracts.
"""
import json
import re
from pathlib import Path

from src.wrappers.record import asset_files, asset_name, read_asset_file

ROOT = Path(__file__).resolve().parents[3]
FOLDERS = {"roadway": "hazus-roadway", "highway_bridge": "hazus-bridge"}


def folder(family):
    if family not in FOLDERS:
        raise ValueError("family must be roadway or highway_bridge")
    return ROOT / "assets/fragility" / FOLDERS[family]


def _number(code):
    return int(re.sub(r"\D", "", code))


def records(family):
    """Every asset file of the family, by class code, in class order."""
    found = {}
    for path in asset_files(folder(family)):
        record = read_asset_file(path)
        parameters = record["parameters"]
        code = parameters["asset_class"]
        if code != asset_name(path) or parameters["asset_family"] != family:
            raise ValueError(f"{path.name} declares {parameters['asset_family']}/{code}, which is not what its location says")
        found[code] = record
    if not found:
        raise ValueError(f"no asset files in {folder(family)}")
    return dict(sorted(found.items(), key=lambda item: _number(item[0])))


_ASSEMBLED = {}  # family -> (what the files looked like, the tables as JSON text)


def database(family):
    """The family's tables. Reassembled only when a file has changed; always a new copy."""
    seen = tuple((path.name, stat.st_mtime_ns, stat.st_size)
                 for path in asset_files(folder(family)) for stat in [path.stat()])
    if _ASSEMBLED.get(family, (None,))[0] != seen:
        _ASSEMBLED[family] = (seen, json.dumps(_assemble(family)))
    return json.loads(_ASSEMBLED[family][1])


def _assemble(family):
    found = records(family)
    sources = {repr(sorted(r["source"].items())) for r in found.values()}
    if len(sources) != 1:
        raise ValueError(f"{family} asset files disagree on their source record; they are one transcription")
    source = next(iter(found.values()))["source"]
    data = {"source": source, "curves": []}
    if family == "highway_bridge":
        data.update(classification_rules=[], k3d={}, pgd_modifiers={})
    for code, record in found.items():
        parameters = record["parameters"]
        common = {key: parameters[key] for key in ("model_family", "asset_family", "asset_class")}
        data["curves"] += [{**common, **curve, "source": {**source, **curve["source"]}} for curve in parameters["curves"]]
        if family != "highway_bridge":
            continue
        data["classification_rules"] += parameters["classification_rules"]
        data["pgd_modifiers"][code] = parameters["pgd_modifiers"]
        if parameters["k3d"] is not None:
            equation = {k: v for k, v in parameters["k3d"].items() if k != "equation"}
            name = parameters["k3d"]["equation"]
            if data["k3d"].setdefault(name, equation) != equation:
                raise ValueError(f"{code} states a different {name} from another class; Table 7-2 has one")
    if family == "highway_bridge":
        # Table order is the matching precedence, so it is restored rather than assumed.
        data["classification_rules"].sort(key=lambda rule: rule["priority"])
        data["k3d"] = dict(sorted(data["k3d"].items()))
    return data


def curves(family, asset_class, hazard_mode):
    selected = [r for r in database(family)["curves"] if r["asset_class"] == asset_class and r["hazard_mode"] == hazard_mode]
    if not selected:
        raise ValueError(f"no curves for {family}/{asset_class}/{hazard_mode}")
    return selected
