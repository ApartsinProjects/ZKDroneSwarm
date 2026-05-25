"""Regenerate the physical-regime crossover figure WITHOUT the nbr variant, from the saved
emergent_swarmfix JSON (no recompute). Panels: sensing range, modality count S, broadcast density rho;
lines: kNN-CF, SwarmCF (base), SwarmCF+bias.  ->  docs/figures/F_emergent_regime.png
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures", "F_emergent_regime.png")
PDF = os.path.join(ROOT, "docs", "figures", "pdf", "F_emergent_regime.pdf")

METHODS = ["knn_cf", "swarm_cf", "swarm_cf_bias"]            # nbr dropped
PRETTY = {"knn_cf": "kNN-CF", "swarm_cf": "SwarmCF", "swarm_cf_bias": "SwarmCF+bias"}
COL = {"knn_cf": "C0", "swarm_cf": "0.55", "swarm_cf_bias": "C2"}

f = sorted(glob.glob(os.path.join(ROOT, "results", "pilots", "emergent_swarmfix_*.json")))[-1]
d = json.load(open(f))


def panel(ax, sweep, xlabel, title, logx=False):
    keys = sorted(sweep, key=float); xs = [float(k) for k in keys]
    for a in METHODS:
        mu = np.array([float(np.mean(sweep[k][a])) for k in keys])
        se = np.array([float(np.std(sweep[k][a]) / np.sqrt(len(sweep[k][a]))) for k in keys])
        ax.plot(xs, mu, marker="o", ms=4, color=COL[a], ls=("--" if a == "swarm_cf" else "-"), label=PRETTY[a])
        ax.fill_between(xs, mu - se, mu + se, color=COL[a], alpha=0.15)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_title(title, fontsize=10); ax.grid(alpha=0.25)


fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
panel(ax[0], d["sweep_range"], "sensing range  $r/$field   (small = strong decay)", "(a) Sensing range")
panel(ax[1], d["sweep_S"], "number of sensing modalities $S$", "(b) Modality count (rank)")
panel(ax[2], d["sweep_rho"], "broadcast density  $\\rho$", "(c) Observation density")
ax[0].set_ylabel("unseen-pair skill (held-out)")
ax[0].legend(fontsize=9, loc="best")
fig.suptitle("Emergent physical sensing reward: SwarmCF+bias matches/beats model-free kNN-CF across "
             "physical regimes (weak decay, low rank, sparse density)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs(os.path.dirname(PDF), exist_ok=True)
plt.savefig(FIG, dpi=150, bbox_inches="tight"); plt.savefig(PDF, bbox_inches="tight"); plt.close()
print("replotted (no nbr) from", os.path.basename(f), "->", FIG)
