"""Generated tbnpy model. Regenerate with: python -m src build toy-ca/demo.

Run from any directory: python path/to/model.py --samples 2000 --device cpu
Or import define_variables() and define_probs(varis, device="cpu").
"""
from pathlib import Path
import argparse
import json
import sys

BASE = Path(__file__).resolve().parent
ROOT = next(p for p in BASE.parents if (p / "src/builder/bn_builder.py").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.builder.bn_builder import define_tbn_variables, define_tbn_probabilities


def read_data():
    return json.loads((BASE / "model.json").read_text(encoding="utf-8"))


def define_variables():
    return define_tbn_variables(read_data())


def define_probs(varis, device="cpu"):
    return define_tbn_probabilities(read_data(), varis, device=device, directory=BASE)


if __name__ == "__main__":
    import torch
    from tbnpy import inference

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    varis = define_variables()
    probs = define_probs(varis, device=args.device)
    sampled = inference.sample(probs, list(probs), n_sample=args.samples)
    summary = {}
    for prob in sampled.values():
        for j, var in enumerate(prob.childs):
            x = prob.Cs[:, j]
            if isinstance(var.values, list):
                summary[var.name] = {"P": {str(s): (x == k).double().mean().item()
                                          for k, s in enumerate(var.values)}}
            else:
                finite = x[torch.isfinite(x)]
                summary[var.name] = {"mean_finite": finite.mean().item() if len(finite) else None,
                                     "P_inf": torch.isinf(x).double().mean().item()}
    result = {"samples": args.samples, "seed": args.seed, "device": args.device, "variables": summary}
    output = BASE / "sample_summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Sampled {len(varis)} variables in {len(probs)} probability objects; wrote {output}")
