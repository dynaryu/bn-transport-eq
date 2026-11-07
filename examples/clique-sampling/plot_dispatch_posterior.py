"""Plot how the posterior moves through a dispatch.py session, one row per observation: P(edge closed),
P(system Fail) and, per edge, the population do(open) gives back, each as the step before (pre)
beside this step (the session), with the MCS estimate and 2 SE where the step was checked.
Quantities the evidence fixes are labelled known. The TOP edges and systems with the largest values
are shown. Writes dispatch_posterior.png beside dispatch.json.

    python examples/clique-sampling/plot_dispatch_posterior.py examples/<example>/clique-sampling/dispatch.json
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SOURCE = Path(sys.argv[1])
results = json.loads(SOURCE.read_text())
network = json.loads((SOURCE.resolve().parents[1] / "network.json").read_text())["tables"]
EDGE_OF = {b["id"]: b["edge"] for b in network["Bridge"]}
TOP = 15
PRE, NOW, MCS = "#c3c2b7", "#2a78d6", "#eb6834"            # neutral reference; categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def known(evidence):
    """The quantities the evidence fixes: an observed system's Fail, an edge observed or fixed by its bridge's damage."""
    out = set()
    for v in evidence:
        kind, name = v.split("::")
        if kind == "SystemState":
            out.add(f"P({v} Fail)")
        elif kind == "EdgeOpen":
            out.add(f"P({v} closed)")
        elif kind.startswith("DamageState"):
            out.add(f"P(EdgeOpen::{EDGE_OF[name]} closed)")
    return out


def top(keys, pre, now):
    size = lambda k: max(pre[k][0], now[k][0])
    kept = set(sorted(keys, key=size, reverse=True)[:TOP])
    return [k for k in keys if k in kept]


def bars(ax, keys, labels, pre, now, ref, fixed, ylim=None):
    x, w = np.arange(len(keys)), 0.38
    ax.bar(x - w / 2, [pre[k][0] for k in keys], w * 0.92, color=PRE, label="before this observation")
    ax.bar(x + w / 2, [now[k][0] for k in keys], w * 0.92, color=NOW, label="after, session")
    if ref is not None:
        ax.errorbar(x + w / 2, [ref[k][0] for k in keys], yerr=[2 * ref[k][1] for k in keys], fmt="D", ms=4,
                    color=MCS, mec="white", mew=0.6, elinewidth=1, label="after, MCS (2 SE)")
    top_y = ylim[1] if ylim else ax.get_ylim()[1]
    for j, k in enumerate(keys):
        if k in fixed:
            ax.text(x[j] + w / 2, now[k][0] + 0.03 * top_y, "known", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, labels, fontsize=8, color=MUTED, rotation=60 if len(keys) > 10 else 0)
    if ylim:
        ax.set_ylim(*ylim)


steps = results["steps"]
fig, axes = plt.subplots(len(steps) - 1, 3, figsize=(16, 3.4 * (len(steps) - 1)), squeeze=False,
                         gridspec_kw={"width_ratios": [10, 6, 10]})
evidence = []
for row, step in enumerate(steps[1:]):
    pre, now, ref = steps[row]["estimates"], step["estimates"], step.get("mcs")
    evidence.append(step["variable"])
    fixed = known(evidence)
    edges = top([k for k in now if k.startswith("P(EdgeOpen")], pre, now)
    systems = top([k for k in now if k.startswith("P(SystemState")], pre, now)
    back = top([k for k in now if k.startswith("restored by do(")], pre, now)
    name = lambda k: k.split("::")[1].split()[0]
    bars(axes[row, 0], edges, [name(k) for k in edges], pre, now, ref, fixed, (0, 1.1))
    bars(axes[row, 1], systems, [name(k) for k in systems], pre, now, ref, fixed, (0, 1.1))
    bars(axes[row, 2], back, [name(k) for k in back], pre, now, ref, fixed)
    how = f"{step['update']}, ESS {step['ess']:.0%}" + (", checked" if ref is not None else "")
    axes[row, 0].set_ylabel(f"{step['label']}\n{step['variable'].split('::')[1]} = {step['value']}\n({how})",
                            fontsize=9, color=INK)
    if row == 0:
        axes[row, 0].set_title("P(edge closed)", fontsize=10, color=INK)
        axes[row, 1].set_title("P(system Fail)", fontsize=10, color=INK)
        axes[row, 2].set_title(f"{results['unit']} given back by do(edge open)", fontsize=10, color=INK)
    for a in axes[row]:
        a.grid(axis="y", color=GRID, lw=0.6)
        a.set_axisbelow(True)
        a.tick_params(labelsize=8, colors=MUTED)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

handles, labels = axes[[i for i, s in enumerate(steps[1:]) if "mcs" in s][0] if any("mcs" in s for s in steps)
                        else 0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10)
fig.suptitle(f"How the posterior moves on {results['example']} as evidence arrives: each step against the one before",
             fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
out = SOURCE.with_name("dispatch_posterior.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
