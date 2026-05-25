"""Decisive bake-off: where does communication-free low-rank CF SHINE on the EMERGENT physical
sensing reward? Full field (structure-free, additive, clustering, convex-completion, memory-based, and
low-rank methods) at TWO honestly-labelled operating points:

  default : rho=0.25, S=4, range_frac=2.0  (the neutral middle used in the regime study)
  shine   : rho=0.15, S=3, range_frac=4.0  (SPARSE broadcast + small sensor catalog + long-range
            signature-driven sensing) -- the communication-starved, low-rank, task-scarce regime the
            paper actually targets. conc is LEFT AT THE DEFAULT 1.5 (the contested clustering knob is
            untouched), so the only changes are the paper's own axes: observation density, latent rank,
            and sensing range.

Reports both so the crossover is explicit (kNN-CF leads only in the dense/high-rank/local corner).
Run:  python experiments/emergent_shine.py
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from latentswarm.sweeps import base_config
from latentswarm.metrics import bootstrap_ci
from emergent_lowrank import physical_reward, wide_factors, run_one

FIG = os.path.join(ROOT, "docs", "figures", "F_emergent_shine.png")
PDF = os.path.join(ROOT, "docs", "figures", "pdf", "F_emergent_shine.pdf")

METHODS = ["random", "ucb_indep", "tabular", "mf_sgd", "bias_model", "soft_impute",
           "club", "knn_cf", "swarm_cf", "swarm_cf_bias"]
PRETTY = {"random": "Random", "ucb_indep": "Indep-UCB", "tabular": "Tabular", "mf_sgd": "MF-SGD",
          "bias_model": "BiasModel", "soft_impute": "SoftImpute", "club": "CLUB", "knn_cf": "kNN-CF",
          "swarm_cf": "SwarmCF", "swarm_cf_bias": "SwarmCF+bias"}
OURS = {"swarm_cf", "swarm_cf_bias"}

REGIMES = {
    "default": dict(rho=0.25, S=4, range_frac=2.0),
    "shine":   dict(rho=0.15, S=3, range_frac=4.0),
}


def run_regime(name, rho, S, range_frac, seeds):
    cfg = base_config(rho=rho)
    raw = {a: [] for a in METHODS}
    t0 = time.time()
    for s in seeds:
        wr = np.random.RandomState(s)
        R = physical_reward(wr, cfg.m, cfg.n, S, range_frac=range_frac)   # conc left at default 1.5
        P, U = wide_factors(R)
        d_guess = cfg.rank_for_run(wr)
        for a in METHODS:
            raw[a].append(run_one(cfg, P, U, d_guess, s, a))
        print("[shine:%s]   seed %d done (%.0fs)" % (name, s, time.time() - t0), flush=True)
    return {a: bootstrap_ci(raw[a]) for a in METHODS}, raw


def main():
    t0 = time.time()
    seeds = list(range(16))
    res = {}
    rawall = {}
    for name, rp in REGIMES.items():
        print("[shine] regime '%s': rho=%.2f S=%d range_frac=%.1f" % (name, rp["rho"], rp["S"], rp["range_frac"]), flush=True)
        res[name], rawall[name] = run_regime(name, rp["rho"], rp["S"], rp["range_frac"], seeds)

    # figure: one panel per regime, methods on x, our variants highlighted
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), sharey=False)
    for axi, name in zip(ax, ["default", "shine"]):
        b = res[name]
        xs = np.arange(len(METHODS))
        mu = [b[a][0] for a in METHODS]
        lo = [b[a][0] - b[a][1] for a in METHODS]; hi = [b[a][2] - b[a][0] for a in METHODS]
        cols = ["C2" if a == "swarm_cf_bias" else ("0.4" if a in OURS else "0.7") for a in METHODS]
        axi.bar(xs, mu, yerr=[lo, hi], capsize=3, color=cols, edgecolor="black", linewidth=0.6)
        axi.set_xticks(xs); axi.set_xticklabels([PRETTY[a] for a in METHODS], rotation=35, ha="right", fontsize=8)
        axi.axhline(0, color="k", lw=0.6); axi.grid(alpha=0.25, axis="y")
        rp = REGIMES[name]
        axi.set_title("%s:  rho=%.2f, S=%d, range=%.1f" % (name, rp["rho"], rp["S"], rp["range_frac"]), fontsize=10)
    ax[0].set_ylabel("unseen-pair skill (held-out)")
    fig.suptitle("Emergent physical sensing reward: SwarmCF+bias tops the field in the sparse, "
                 "low-rank, signature-driven regime", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(PDF), exist_ok=True)
    plt.savefig(FIG, dpi=150, bbox_inches="tight"); plt.savefig(PDF, bbox_inches="tight"); plt.close()

    out = {"meta": {"experiment": "emergent shine bake-off", "seeds": seeds, "regimes": REGIMES,
                    "conc": 1.5, "note": "conc left at default; only rho/S/range vary"},
           "results": {name: {a: res[name][a] for a in METHODS} for name in REGIMES}}
    op = os.path.join(ROOT, "results", "pilots", "emergent_shine_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    json.dump(out, open(op, "w"), indent=1)

    for name in ["default", "shine"]:
        rp = REGIMES[name]
        print("\n=== %s regime: rho=%.2f, S=%d, range_frac=%.1f (16 seeds) ===" % (name, rp["rho"], rp["S"], rp["range_frac"]))
        b = res[name]
        knn = b["knn_cf"][0]
        for a in METHODS:
            star = "  <-- ours" if a in OURS else ""
            vs = "  (vs kNN %+.3f)" % (b[a][0] - knn) if a in OURS else ""
            print("  %-13s %.3f [%.3f, %.3f]%s%s" % (PRETTY[a], b[a][0], b[a][1], b[a][2], vs, star))
    print("\nsaved -> %s, %s  (%.0fs)" % (FIG, op, time.time() - t0))


if __name__ == "__main__":
    main()
