"""Tier C (appendix): trait-distribution sensitivity of the categorical separation.

Runs the SAME body bake-off harness (latentswarm.sweeps.sweep_bakeoff, the c14_compare schema that
feeds Table 3) on three trait-generating distributions, varying ONLY the scenario so the comparison
is apples-to-apples (identical methods, config, seeds, metrics):

  block_cosine     discrete latent types (clustered traits): the regime most favorable to clustering.
  uniform_cosine   i.i.d. unit sphere, no types (the body's headline world).
  approx_lowrank   continuous but NOT exactly low-rank (full-rank Gaussian perturbation, eps=0.5,
                   so the low-rank energy fraction is 1/(1+eps^2)=0.80): stresses the assumption.

For each scenario we report, at the masked broadcast rho=0.25, every method's held-out unseen-pair
skill and overall skill (mean with bootstrap 95% CI over 16 seeds). The story: SwarmCF leads on the
unseen everywhere; its margin over the discrete-clustering CLUB is largest in the no-types uniform
world and shrinks toward the block world (clustering has a natural target there); under approximate
low-rank the separation degrades only gracefully.

Single process (no bash respawn); the ProcessPoolExecutor workers inside sweep_bakeoff are children
of this process. The `if __name__ == "__main__"` guard is REQUIRED on Windows (spawn re-imports).

Run from repo root:  python experiments/tierc_distribution.py
"""
import os
import sys
import json
import time
import contextlib
import io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# scenario label -> extra config overrides (beyond scenario name)
SCENARIOS = [
    ("block_cosine", {}),
    ("uniform_cosine", {}),
    ("approx_lowrank", {"approx_eps": 0.5}),
]
APPROX_EPS = 0.5
RHO = 0.25                     # masked broadcast regime (Table 3 headline column)


def main():
    from latentswarm.sweeps import base_config, sweep_bakeoff, _BAKEOFF_METHODS
    from latentswarm.metrics import bootstrap_ci

    out_dir = os.path.join(ROOT, "results", "pilots")
    os.makedirs(out_dir, exist_ok=True)
    methods = list(_BAKEOFF_METHODS)          # 12 methods incl. club, knn_cf, swarm_cf
    t0 = time.time()
    results = {}
    print("[TierC] START 3-scenario bake-off (rho=%.2f, 16 seeds, %d methods)" % (RHO, len(methods)), flush=True)

    for scen, extra in SCENARIOS:
        cfg = base_config(scenario=scen, **extra)
        with contextlib.redirect_stdout(io.StringIO()):
            payload, _name = sweep_bakeoff(cfg, methods=methods, rhos=[RHO])
        raw = payload["raw"][str(RHO)]        # {KEY: {"overall":[...], "unseen":[...], "uniq":[...]}}
        scen_tag = scen if scen != "approx_lowrank" else ("approx_lowrank(eps=%.2f)" % APPROX_EPS)
        results[scen_tag] = {}
        for key, cell in raw.items():
            results[scen_tag][key] = {
                "unseen": bootstrap_ci(cell["unseen"]),
                "overall": bootstrap_ci(cell["overall"]),
            }
        print("[TierC] %-22s done (%.0fs elapsed)" % (scen_tag, time.time() - t0), flush=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, "tierc_distribution_%s.json" % stamp)
    json.dump({"meta": {"experiment": "Tier C trait-distribution sensitivity",
                        "rho": RHO, "seeds": 16, "methods": methods,
                        "scenarios": list(results.keys()), "approx_eps": APPROX_EPS},
               "results": results}, open(out_path, "w"), indent=1)

    # readable console table: unseen-pair skill (the categorical headline) per method per scenario
    scen_tags = list(results.keys())
    show = ["random", "ucb_indep", "tabular", "mf_sgd", "estr", "bpmf", "soft_impute",
            "bias_model", "knn_cf", "club", "swarm_cf"]
    PKEY = {"random": "Random", "ucb_indep": "Indep-UCB", "tabular": "Tabular", "mf_sgd": "MF-SGD",
            "estr": "ESTR", "bpmf": "BPMF", "soft_impute": "SoftImpute", "bias_model": "BiasModel",
            "knn_cf": "KNNCF", "club": "CLUB", "swarm_cf": "SwarmCF"}
    DKEY = {"random": "Random", "ucb_indep": "UCBIndep", "tabular": "Tabular", "mf_sgd": "MFSGD",
            "estr": "ESTR", "bpmf": "BPMF", "soft_impute": "SoftImpute", "bias_model": "BiasModel",
            "knn_cf": "KNNCF", "club": "CLUB", "swarm_cf": "RewardCF"}

    def cell_unseen(scen_tag, pkg):
        key = DKEY[pkg]
        m = results[scen_tag].get(key, {}).get("unseen", (None, None, None))
        return m[0]

    print("\n=== Tier C: UNSEEN-PAIR SKILL (rho=%.2f, 16 seeds) ===" % RHO)
    print("%-11s | %s" % ("method", " | ".join("%-22s" % s for s in scen_tags)))
    print("-" * (13 + 25 * len(scen_tags)))
    for pkg in show:
        row = []
        for s in scen_tags:
            v = cell_unseen(s, pkg)
            row.append("%-22s" % ("%.3f" % v if v is not None else "n/a"))
        print("%-11s | %s" % (PKEY[pkg], " | ".join(row)))

    print("\n=== SwarmCF lead over CLUB / KNNCF on unseen-pair skill ===")
    for s in scen_tags:
        sc = cell_unseen(s, "swarm_cf"); cl = cell_unseen(s, "club"); kn = cell_unseen(s, "knn_cf")
        if sc is not None and cl is not None and kn is not None:
            print("  %-22s SwarmCF=%.3f  CLUB=%.3f (lead +%.3f)  KNNCF=%.3f (lead +%.3f)"
                  % (s, sc, cl, sc - cl, kn, sc - kn))
    print("\nsaved -> %s  (%.0fs total)" % (out_path, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
