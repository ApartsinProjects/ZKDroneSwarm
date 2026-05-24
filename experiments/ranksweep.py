"""Graceful-degradation ablation (robustness to the guessed rank).

Sweeps the guessed rank d-hat from d/2 (under-ranking: the model is mis-specified and can only
fit a rank-d-hat approximation) through 3d (over-ranking: surplus directions are regularized
away), with the true rank d marked. Uses the latentswarm package and reports SwarmCF (and MF-SGD)
earned + unseen-pair skill at each d-hat, with bootstrap 95% CIs. Saves data + figure F21.

Run from the repo root:  python experiments/ranksweep.py
"""
import os, sys, io, json, contextlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latentswarm import RunConfig
from latentswarm.run import run

D = 5
DHATS = [2, 3, 4, 5, 6, 8, 10, 12, 15]      # d/2 (=2.5 -> 2) ... 3d (=15)
SEEDS = list(range(int(os.environ.get("ZK_SEEDS", "16"))))   # 8->16 for Fig 7 seed consistency (referee)

data = []
for dh in DHATS:
    cfg = RunConfig(rank_guess=dh, seeds=SEEDS, scenario="uniform_cosine",
                    algorithms=["random", "swarm_cf", "mf_sgd", "club"])
    with contextlib.redirect_stdout(io.StringIO()):
        out = run(cfg)
    sw, mf, cl = out["summary"]["swarm_cf"], out["summary"]["mf_sgd"], out["summary"]["club"]
    data.append({"dhat": dh, "swarm": sw, "mf": mf, "club": cl})
    print("dhat=%2d  SwarmCF earned=%.3f unseen=%.3f  |  MF-SGD unseen=%.3f  |  CLUB unseen=%.3f"
          % (dh, sw["earned"][0], sw["unseen"][0], mf["unseen"][0], cl["unseen"][0]))

os.makedirs("results/pilots", exist_ok=True)
json.dump({"d": D, "dhats": DHATS, "seeds": SEEDS, "data": data},
          open("results/pilots/latentswarm_ranksweep.json", "w"), indent=0)


def _band(ax, x, tris, color, label):
    m = [t[0] for t in tris]; lo = [t[1] for t in tris]; hi = [t[2] for t in tris]
    ax.plot(x, m, "-o", color=color, lw=2, ms=4, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18)


fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, (key, ttl) in enumerate([("earned", "Earned (anytime) skill"),
                                ("unseen", "Unseen-pair skill")]):
    _band(ax[k], DHATS, [d["swarm"][key] for d in data], "C2", "SwarmCF")
    _band(ax[k], DHATS, [d["club"][key] for d in data], "C1", "CLUB")
    _band(ax[k], DHATS, [d["mf"][key] for d in data], "C3", "MF-SGD")
    ax[k].axvline(D, color="k", ls="--", lw=1, alpha=0.7)
    ax[k].text(D + 0.15, ax[k].get_ylim()[0], "true rank d=%d" % D, fontsize=7, rotation=90,
               va="bottom", ha="left", color="k", alpha=0.7)
    ax[k].axhline(0, color="k", lw=0.5)
    ax[k].set_xlabel("guessed rank  $\\hat d$"); ax[k].set_ylabel(ttl)
    ax[k].set_title("(%s)" % "ab"[k], loc="left", fontsize=9)
    ax[k].grid(alpha=0.25); ax[k].legend(fontsize=8)
fig.tight_layout()
plt.savefig("docs/figures/F21_ranksweep.png", dpi=150, bbox_inches="tight")
os.makedirs("docs/figures/pdf", exist_ok=True)
plt.savefig("docs/figures/pdf/F21_ranksweep.pdf", bbox_inches="tight")
plt.close("all")
print("wrote docs/figures/F21_ranksweep.png + pdf")
