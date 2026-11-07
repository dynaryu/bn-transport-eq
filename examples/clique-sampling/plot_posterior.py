"""Plot the posterior distributions in a compare.py results file, one row per case: P(edge closed) and P(system Fail)
as bars, prior (the case without evidence) beside posterior (the clique tree), with the MCS estimate
and 2 SE on each posterior bar, for the TOP edges and systems with the largest prior or posterior
probability; and the posterior density of ln SA, clique tree against MCS, over the prior, for the
bridge whose SAObs is observed, or else the one the evidence moved most. Writes posterior_range<b>km.png
beside results_range<b>km.json.

    python examples/clique-sampling/plot_posterior.py examples/<example>/clique-sampling/results_range<b>km.json
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SOURCE = Path(sys.argv[1])
EXAMPLE = SOURCE.resolve().parents[1].name
TOP = 15
results = json.loads(SOURCE.read_text())
PRIOR, CLIQUE, MCS = "#c3c2b7", "#2a78d6", "#eb6834"       # neutral reference; categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

cases = results["cases"]
prior = cases["none"]
fig, axes = plt.subplots(len(cases), 3, figsize=(15, 3.3 * len(cases)), gridspec_kw={"width_ratios": [10, 6, 7]})


def bars(ax, case, keys, labels):
    """Prior and posterior bars for each key, with the MCS posterior and 2 SE as a marker."""
    x, w = np.arange(len(keys)), 0.38
    est = case["estimates"]
    ax.bar(x - w / 2, [prior["estimates"][k]["clique"] for k in keys], w * 0.92, color=PRIOR, label="prior")
    ax.bar(x + w / 2, [est[k]["clique"] for k in keys], w * 0.92, color=CLIQUE, label="posterior, clique tree")
    ax.errorbar(x + w / 2, [est[k]["mcs"] for k in keys], yerr=[2 * est[k]["mcs_se"] for k in keys], fmt="D",
                ms=4, color=MCS, mec="white", mew=0.6, elinewidth=1, label="posterior, MCS (2 SE)")
    for j, k in enumerate(keys):                                  # an observed variable is labelled, not estimated
        if any(k.startswith(f"P({v} ") or k == f"P({v} closed)" for v in case["evidence"]):
            ax.text(x[j] + w / 2, est[k]["clique"] + 0.03, "obs", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, labels, fontsize=8, color=MUTED, rotation=60 if len(keys) > 10 else 0)
    ax.set_ylim(0, 1.08)


def top(est, prefix):
    """The TOP keys with the given prefix whose prior or posterior probability is largest, in their own order."""
    keys = [k for k in est if k.startswith(prefix)]
    size = lambda k: max(prior["estimates"][k]["clique"], est[k]["clique"], est[k]["mcs"])
    kept = set(sorted(keys, key=size, reverse=True)[:TOP])
    return [k for k in keys if k in kept]


for row, (name, case) in enumerate(cases.items()):
    est = case["estimates"]
    edges, systems = top(est, "P(EdgeOpen"), top(est, "P(SystemState")
    bars(axes[row, 0], case, edges, [k.split("::")[1].split()[0] for k in edges])
    bars(axes[row, 1], case, systems, [k.split("::")[1].split()[0] for k in systems])
    short = lambda q: re.sub(r"\.[A-Za-z].*$", "", q)              # DamageState.HazusBridge -> DamageState
    evidence = "\n".join(f"{v.split('::')[1]} {short(v.split('::')[0])} = {val}" for v, val in case["evidence"].items())
    axes[row, 0].set_ylabel(f"{name}\n{evidence or 'no evidence'}", fontsize=9, color=INK)
    axes[row, 0].set_title("P(edge closed)" if row == 0 else "", fontsize=10, color=INK)
    axes[row, 1].set_title("P(system Fail)" if row == 0 else "", fontsize=10, color=INK)

    # ln SA: the bridge whose SAObs is observed, or else the one whose posterior mean moved most from the prior,
    # or, without evidence, the one shaken most
    dens = case["ln_sa_density"]
    centres = np.diff(dens["bins"]) / 2 + np.array(dens["bins"][:-1])
    sa = [k for k in dens if k != "bins"]
    measured = [k for k in sa if k.replace("SA(", "SAObs(") in case["evidence"]]
    moved = measured[0] if measured else max(
        sa, key=lambda k: (abs(est[f"E[ln {k}]"]["clique"] - prior["estimates"][f"E[ln {k}]"]["clique"]) if case is not prior
                else est[f"E[ln {k}]"]["clique"]))
    ax = axes[row, 2]
    ax.fill_between(centres, prior["ln_sa_density"][moved]["clique"], color=PRIOR, lw=0, label="prior")
    ax.plot(centres, dens[moved]["clique"], color=CLIQUE, lw=2, label="posterior, clique tree")
    ax.plot(centres, dens[moved]["mcs"], color=MCS, lw=1.2, ls=(0, (3, 2)), label="posterior, MCS")
    ax.set_title(f"density of ln {moved}", fontsize=10, color=INK)
    ax.set_xlabel("ln SA (g)" if row == len(cases) - 1 else "", fontsize=9, color=MUTED)
    for a in axes[row]:
        a.grid(axis="y", color=GRID, lw=0.6)
        a.set_axisbelow(True)
        a.tick_params(labelsize=8, colors=MUTED)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10)
fig.suptitle(f"Posterior distributions on {EXAMPLE} (ln SA correlated, range {results['range_km']:g} km): "
             f"clique tree ({results['sampler']['function']}) against "
             f"{results['mcs']['n']:.0e} MCS", fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
out = SOURCE.with_name(SOURCE.stem.replace("results", "posterior") + ".png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
