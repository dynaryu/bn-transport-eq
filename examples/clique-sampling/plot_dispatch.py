"""Plot a dispatch.py results file: at each step checked against MCS, the session's estimate against
MCS for every quantity the dispatch reads, by kind. Rows: probabilities (P(S Fail), P(E closed));
the population cut off, E[L], and per edge what do(open) gives back; z = difference / combined SE.
Observed quantities (known exactly in both) are left out. Writes dispatch.png beside dispatch.json.

    python examples/clique-sampling/plot_dispatch.py examples/<example>/clique-sampling/dispatch.json
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

SOURCE = Path(sys.argv[1])
results = json.loads(SOURCE.read_text())
unit = results["unit"]
KINDS = {   # name prefix: label, colour (categorical slots), marker, row
    "P(SystemState": ("P(S Fail)", "#eda100", "D", 0),
    "P(EdgeOpen": ("P(E closed)", "#1baf7a", "^", 0),
    "E[L]": (f"E[L], {unit} cut off", "#2a78d6", "o", 1),
    "restored by do(": (f"{unit} given back by do(E open)", "#eb6834", "s", 1),
}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

checked = [s for s in results["steps"] if "mcs" in s]
fig, axes = plt.subplots(3, len(checked), figsize=(4.6 * len(checked), 11), squeeze=False)
for col, step in enumerate(checked):
    est, ref = step["estimates"], step["mcs"]
    axes[0, col].set_title(f"{step['label']}\n{step['update']} (ESS {step['ess']:.1%} of the rows before)",
                           fontsize=10, color=INK)
    z_x = 0
    for prefix, (label, colour, marker, row) in KINDS.items():
        keys = [k for k in est if k.startswith(prefix) and (est[k][1] > 0 or ref[k][1] > 0)]   # observed: exact
        if not keys:
            continue
        ax = axes[row, col]
        ax.errorbar([ref[k][0] for k in keys], [est[k][0] for k in keys],
                    xerr=[2 * ref[k][1] for k in keys], yerr=[2 * est[k][1] for k in keys],
                    fmt=marker, ms=6, color=colour, mec="white", mew=0.8, elinewidth=1, label=label)
        z = [(est[k][0] - ref[k][0]) / (est[k][1] ** 2 + ref[k][1] ** 2) ** 0.5 for k in keys]
        axes[2, col].plot(range(z_x, z_x + len(z)), z, marker, ms=6, color=colour, mec="white", mew=0.8)
        z_x += len(z)
    for ax, lo_hi in ((axes[0, col], (0, 1)), (axes[1, col], None)):
        lo, hi = lo_hi or (min(0, ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1]))
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, zorder=0)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"MCS ({results['n_mcs']:.0e} forward samples, effective {step['mcs_ess']:.0f})",
                      fontsize=9, color=MUTED)
        ax.set_ylabel("session (clique tree)", fontsize=9, color=MUTED)
    axes[2, col].axhspan(-3, 3, color=GRID, zorder=0)
    axes[2, col].axhline(0, color=MUTED, lw=1, zorder=0)
    axes[2, col].set_xlabel("quantity", fontsize=9, color=MUTED)
    axes[2, col].set_ylabel("z = (session - MCS) / SE", fontsize=9, color=MUTED)
    zs = [(est[k][0] - ref[k][0]) / (est[k][1] ** 2 + ref[k][1] ** 2) ** 0.5 for k in est
          if est[k][1] > 0 or ref[k][1] > 0]
    axes[2, col].set_title(f"mean z² {sum(z * z for z in zs) / len(zs):.2f}, |z| > 3: {sum(abs(z) > 3 for z in zs)} "
                           f"of {len(zs)}", fontsize=9, color=INK)
    for ax in axes[:, col]:
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(labelsize=8, colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

handles = [plt.Line2D([], [], ls="", marker=m, ms=7, color=c, mec="white") for _, c, m, _ in KINDS.values()]
fig.legend(handles, [k[0] for k in KINDS.values()], loc="lower center", ncol=len(KINDS), frameon=False, fontsize=10,
           title="error bars: 2 SE; grey band: |z| < 3", title_fontsize=9)
s = results["sampler"]
fig.suptitle(f"Dispatch session against MCS on {results['example']}, evidence arriving one observation at a time\n"
             f"{s['n_chain']} chains x {s['n_keep']} kept sweeps per sampling", fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0.06, 1, 0.96))
out = SOURCE.with_suffix(".png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
