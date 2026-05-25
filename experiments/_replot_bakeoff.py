"""Regenerate the emergent-reward bake-off figure with NEUTRAL panel labels (no 'shine'), from the
saved emergent_shine JSON (no recompute). Two disclosed operating points; our method (bias-augmented
SwarmCF) highlighted.  ->  docs/figures/F_emergent_bakeoff.png
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures", "F_emergent_bakeoff.png")
PDF = os.path.join(ROOT, "docs", "figures", "pdf", "F_emergent_bakeoff.pdf")

METHODS = ["random", "ucb_indep", "tabular", "mf_sgd", "bias_model", "soft_impute",
           "club", "knn_cf", "swarm_cf", "swarm_cf_bias"]
PRETTY = {"random": "Random", "ucb_indep": "Indep-UCB", "tabular": "Tabular", "mf_sgd": "MF-SGD",
          "bias_model": "BiasModel", "soft_impute": "SoftImpute", "club": "CLUB", "knn_cf": "kNN-CF",
          "swarm_cf": "SwarmCF", "swarm_cf_bias": "SwarmCF (bias-augmented)"}
OURS = {"swarm_cf", "swarm_cf_bias"}
PANEL = {"default": "(a) Headline operating point:  $\\rho$=0.25, $S$=4, range=2.0",
         "shine":   "(b) Sparse, low-rank regime:  $\\rho$=0.15, $S$=3, range=4.0"}

f = sorted(glob.glob(os.path.join(ROOT, "results", "pilots", "emergent_shine_*.json")))[-1]
d = json.load(open(f))["results"]

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for axi, name in zip(ax, ["default", "shine"]):
    b = d[name]
    xs = np.arange(len(METHODS))
    mu = [b[a][0] for a in METHODS]
    lo = [b[a][0] - b[a][1] for a in METHODS]; hi = [b[a][2] - b[a][0] for a in METHODS]
    cols = ["C2" if a == "swarm_cf_bias" else ("0.45" if a in OURS else "0.72") for a in METHODS]
    axi.bar(xs, mu, yerr=[lo, hi], capsize=3, color=cols, edgecolor="black", linewidth=0.6)
    axi.set_xticks(xs); axi.set_xticklabels([PRETTY[a] for a in METHODS], rotation=35, ha="right", fontsize=8)
    axi.axhline(0, color="k", lw=0.6); axi.grid(alpha=0.25, axis="y")
    axi.set_title(PANEL[name], fontsize=10)
ax[0].set_ylabel("held-out unseen-pair skill")
fig.suptitle("Bake-off on the emergent physical sensing reward (16 seeds, bootstrap 95% CIs)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs(os.path.dirname(PDF), exist_ok=True)
plt.savefig(FIG, dpi=150, bbox_inches="tight"); plt.savefig(PDF, bbox_inches="tight"); plt.close()
print("replotted (neutral labels) from", os.path.basename(f), "->", FIG)
