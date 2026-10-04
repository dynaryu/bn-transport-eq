"""Evaluate an auditable Hazus query: python -m src.fragility.hazus61 query.json."""
import argparse
import json
from pathlib import Path

from .engine import evaluate, evaluate_bridge_modes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    query = json.loads(args.query.read_text(encoding="utf-8"))
    family = query["family"]
    options = query.get("options", {})
    if query.get("hazard_mode") == "both":
        if family != "highway_bridge":
            raise ValueError("both hazard modes are supported only for highway_bridge")
        result = evaluate_bridge_modes(query["given"], **options)
    else:
        result = evaluate(family, query["given"], hazard_mode=query["hazard_mode"], **options)
    output = {"query": query, "result": result}
    text = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["match_status", "asset_class", "states", "probabilities", "missing_fields"] if k in result}, indent=2))


if __name__ == "__main__":
    main()
