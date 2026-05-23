"""RAS GROUNDING: physically-motivated observability. Instead of injecting an abstract mask
rate rho and noise sigma_obs, we PLACE m drones and n targets in a 2-D arena and DERIVE
observability from sensing geometry: drone i senses a teammate's engagement at target j only if
the target lies within i's sensing radius R_sense (limited range / line-of-sight), and the reward
read-off noise GROWS with distance, sigma(d) = sigma0 * (1 + d / R_sense). The persistent
per-drone, per-target visibility V[i,j] = [dist(i,j) <= R_sense] is exactly the structured,
non-uniform 'persistent masking' regime, now EMERGENT from physics rather than a free parameter.
Sweeping R_sense traces effective coverage; we show the categorical CF win (generalize to UNSEEN
targets, beat structure-free) survives sensing-grounded observability. Writes docs/SENSING.md."""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
from core import make_world
from _results_io import save_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R_SENSES = [0.20, 0.35, 0.50, 0.80]    # sensing radius in a unit arena (small = sparse sensing)
SEEDS = list(range(8))
ORDER = ["RewardCF", "KNNCF", "Tabular", "UCBIndep"]
RNG = np.random.RandomState(0)

REG = {
    "RewardCF": pc.REGISTRY["RewardCF"],
    "KNNCF":    pc.REGISTRY["KNNCF"],
    "Tabular":  pc.REGISTRY["Tabular"],
    "UCBIndep": pc.REGISTRY["UCBIndep"],
}


def run_sensing(Cls, hp, world, seed, R_sense, dim=2):
    """Geometry-derived observability: returns (overall_skill, unseen_skill, eff_coverage)."""
    P, U, R = world[:3]; m, n = R.shape
    T, cand, so, sb0 = pc.T, pc.CAND, pc.SO, pc.SB
    rng = np.random.RandomState(seed + 999)
    dpos = rng.rand(m, dim)                 # drone positions in [0,1]^dim
    tpos = rng.rand(n, dim)                 # target positions
    D = np.sqrt(((dpos[:, None, :] - tpos[None, :, :]) ** 2).sum(-1))   # (m,n) drone-target distance
    V = D <= R_sense                        # persistent per-(drone,target) visibility
    eff_cov = float(V.mean())               # fraction of (drone,target) engagements that are sensible
    learners = [Cls(m, n, pc.D_HAT, i, seed + 7 * i + 1, **hp) for i in range(m)]
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        for i in range(m):
            revealed = np.full(m, np.nan); rvar = np.full(m, np.inf)
            revealed[i] = true_r[i] + rng.normal(0, so); rvar[i] = so ** 2     # own outcome, clean
            for k in range(m):
                if k == i:
                    continue
                j = int(choices[k])
                if V[i, j]:                                                    # target in i's sensing range
                    sig = sb0 * (1.0 + D[i, j] / R_sense)                      # noise grows with distance
                    revealed[k] = true_r[k] + rng.normal(0, sig); rvar[k] = sig ** 2
            learners[i].observe(t, choices, revealed, cand_sets, rvar)
    preds = [learners[i].predict_scores() for i in range(m)]
    pulled = [learners[i].pulled_mask() for i in range(m)]
    g = np.random.RandomState(seed + 555)
    ovg, ovo, ovr, ung, uno, unr = [], [], [], [], [], []
    for _ in range(120):
        for k in range(m):
            off = g.choice(n, size=cand, replace=False)
            ovg.append(R[k, off[int(np.argmax(preds[k][off]))]]); ovo.append(R[k, off].max()); ovr.append(R[k, off].mean())
            unseen = np.where(~pulled[k])[0]
            if len(unseen) >= cand:
                offu = g.choice(unseen, size=cand, replace=False)
                ung.append(R[k, offu[int(np.argmax(preds[k][offu]))]]); uno.append(R[k, offu].max()); unr.append(R[k, offu].mean())

    def sk(gg, oo, rr):
        gm, om, rm = np.mean(gg), np.mean(oo), np.mean(rr)
        return (gm - rm) / max(om - rm, 1e-6)
    return sk(ovg, ovo, ovr), sk(ung, uno, unr), eff_cov


def _job(args):
    nm, R_sense, seed = args
    Cls, hp = REG[nm]
    w = make_world(pc.M, pc.N, pc.D, pc.K, pc.K, within=0.15, seed=seed, signed=True)
    o, u, cov = run_sensing(Cls, hp, w, seed, R_sense)
    return nm, R_sense, seed, float(o), float(u), float(cov)


def ci(vals, B=10000):
    a = np.asarray([v for v in vals if v is not None], float)
    idx = RNG.randint(0, len(a), (B, len(a))); mb = a[idx].mean(1)
    return a.mean(), np.percentile(mb, 2.5), np.percentile(mb, 97.5)


def cell(vals):
    mu, lo, hi = ci(vals); return "%.3f [%.3f, %.3f]" % (mu, lo, hi)


def main():
    jobs = [(nm, r, s) for r in R_SENSES for nm in ORDER for s in SEEDS]
    raw = {str(r): {nm: {"overall": [None] * len(SEEDS), "unseen": [None] * len(SEEDS)}
                    for nm in ORDER} for r in R_SENSES}
    cov = {str(r): [None] * len(SEEDS) for r in R_SENSES}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            nm, r, s, o, u, c = fut.result()
            raw[str(r)][nm]["overall"][s] = o
            raw[str(r)][nm]["unseen"][s] = u
            cov[str(r)][s] = c
            done += 1
            if done % 16 == 0 or done == len(jobs):
                print("  ... %d/%d" % (done, len(jobs)))

    save_results("sensing", {
        "meta": {"experiment": "sensing-grounded observability (geometry-derived masking + distance noise)",
                 "methods": ORDER, "R_senses": R_SENSES, "seeds": SEEDS, "m": pc.M, "n": pc.N, "d": pc.D,
                 "d_hat": pc.D_HAT, "T": pc.T, "cand": pc.CAND, "sigma_own": pc.SO, "sigma0": pc.SB,
                 "metric": "overall + unseen-pair skill under geometry-derived persistent masking, per sensing radius",
                 "coverage": {r: float(np.mean([v for v in cov[r] if v is not None])) for r in cov}},
        "raw": raw}, results_dir=os.path.join(ROOT, "results", "pilots"))

    L = ["# Sensing-grounded observability: does the categorical win survive PHYSICAL masking? (RAS)\n",
         "Masking + noise are DERIVED from sensing geometry (drone senses a target's engagement iff "
         "within radius R_sense; read-off noise grows with distance), not injected. 8 seeds, "
         "bootstrap 95%% CI. Effective coverage = mean fraction of (drone,target) engagements sensible.\n",
         "| R_sense | mean coverage |", "|---|---|"]
    for r in R_SENSES:
        L.append("| %.2f | %.3f |" % (r, float(np.mean([v for v in cov[str(r)] if v is not None]))))
    L.append("")
    for metric, title in [("unseen", "Unseen-pair skill (the categorical claim)"),
                          ("overall", "Overall skill")]:
        L.append("## %s\n" % title)
        L.append("| method | " + " | ".join("R=%.2f" % r for r in R_SENSES) + " |")
        L.append("|" + "---|" * (len(R_SENSES) + 1))
        for nm in ORDER:
            cells = [cell(raw[str(r)][nm][metric]) for r in R_SENSES]
            lab = "**%s**" % nm if nm == "RewardCF" else nm
            L.append("| %s | %s |" % (lab, " | ".join(cells)))
        L.append("")
    L.append("Read: if RewardCF keeps a high unseen-pair skill above the structure-free floor (~0) even "
             "at SMALL sensing radius (sparse, distance-noisy observation), the categorical generalization "
             "result is not an artifact of an abstract mask, it survives physically-grounded, "
             "geometry-limited sensing, the regime a real drone swarm operates in.\n")
    out_md = os.path.join(ROOT, "docs", "SENSING.md")
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L)); print("wrote %s (raw saved earlier)" % out_md)


if __name__ == "__main__":
    main()
