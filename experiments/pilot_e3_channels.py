"""
E3 (CHANNEL GRID): the two observability channels CROSSED. Masking rho (action
channel) x reward-noise sigma_obs (reward channel). Both channels masked
CONSISTENTLY: if teammate k's broadcast event is masked for observer i, drone i
sees NEITHER k's choice (-1) NOR k's reward (nan). Methods: Tabular (own only),
RewardCF (reward channel), ChoiceCF (action channel, clean), BothCF (fuse), PTF
(reward-based batch-SVD competitor). Own reward always clean (sigma_own).
Hypothesis: RewardCF best at low noise; as sigma_obs rises the clean CHOICE
channel (ChoiceCF) overtakes; BothCF dominates; masking hurts both; PTF degrades.
"""
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
from core import make_world
from _results_io import save_results

RHOS = [1.0, 0.5, 0.25]
SIGS = [0.0, 0.3, 0.6, 1.0, 2.0]
METHODS = ["Tabular", "RewardCF", "ChoiceCF", "BothCF", "PTF"]
SEEDS = list(range(8))


def run_2ch(args):
    name, rho, sig, seed = args
    Cls, hp = pc.REGISTRY[name]
    w = make_world(pc.M, pc.N, pc.D, pc.K, pc.K, within=0.15, seed=seed, signed=True)
    P, U, R = w[:3]; m, n = R.shape
    T, cand, so = pc.T, pc.CAND, pc.SO
    rng = np.random.RandomState(seed + 999)
    Mask = rng.rand(m, m) < rho; np.fill_diagonal(Mask, True)
    learners = [Cls(m, n, pc.D_HAT, i, seed + 7 * i + 1, **hp) for i in range(m)]
    real = np.zeros(T); orac = np.zeros(T); rnd = np.zeros(T)
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        real[t] = true_r.sum()
        orac[t] = sum(R[i, cand_sets[i]].max() for i in range(m))
        rnd[t] = sum(R[i, cand_sets[i]].mean() for i in range(m))
        for i in range(m):
            revealed = np.full(m, np.nan); rvar = np.full(m, np.inf); ch_i = choices.copy()
            revealed[i] = true_r[i] + rng.normal(0, so); rvar[i] = so ** 2
            for k in range(m):
                if k != i:
                    if Mask[i, k]:
                        revealed[k] = true_r[k] + rng.normal(0, sig); rvar[k] = sig ** 2
                    else:
                        ch_i[k] = -1                    # masked: choice unobserved too
            learners[i].observe(t, ch_i, revealed, cand_sets, rvar)
    preds = [learners[i].predict_scores() for i in range(m)]
    pulled = [learners[i].pulled_mask() for i in range(m)]
    g = np.random.RandomState(seed + 555)
    ovg, ovo, ovr, ung, uno, unr = [], [], [], [], [], []
    for _ in range(80):
        for k in range(m):
            off = g.choice(n, size=cand, replace=False)
            ovg.append(R[k, off[int(np.argmax(preds[k][off]))]]); ovo.append(R[k, off].max()); ovr.append(R[k, off].mean())
            unseen = np.where(~pulled[k])[0]
            if len(unseen) >= cand:
                offu = g.choice(unseen, size=cand, replace=False)
                ung.append(R[k, offu[int(np.argmax(preds[k][offu]))]]); uno.append(R[k, offu].max()); unr.append(R[k, offu].mean())

    def sk(gg, oo, rr):
        gm, om, rm = np.mean(gg), np.mean(oo), np.mean(rr); return (gm - rm) / max(om - rm, 1e-6)
    cr, co, cd = np.cumsum(real), np.cumsum(orac), np.cumsum(rnd)
    anytime = float((cr[-1] - cd[-1]) / max(co[-1] - cd[-1], 1e-9))
    return name, rho, sig, seed, float(sk(ovg, ovo, ovr)), float(sk(ung, uno, unr)), anytime


def main():
    print("=" * 92)
    print("E3 CHANNEL GRID: rho (action mask) x sigma_obs (reward noise). "
          "m=%d n=%d d_hat=%d T=%d %d seeds" % (pc.M, pc.N, pc.D_HAT, pc.T, len(SEEDS)))
    print("=" * 92)
    jobs = [(nm, rho, sig, s) for rho in RHOS for sig in SIGS for nm in METHODS for s in SEEDS]
    raw = {str(rho): {str(sig): {nm: {"overall": [], "unseen": [], "anytime": []}
                                 for nm in METHODS} for sig in SIGS} for rho in RHOS}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_2ch, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            nm, rho, sig, s, ov, un, an = fut.result()
            c = raw[str(rho)][str(sig)][nm]; c["overall"].append(ov); c["unseen"].append(un); c["anytime"].append(an)
            done += 1
            if done % 40 == 0 or done == len(jobs):
                print("  ... %d/%d cells done" % (done, len(jobs)))

    for rho in RHOS:
        print("\n" + "-" * 92)
        print("rho=%.2f  OVERALL skill vs sigma_obs (mean/%d seeds); headline = ChoiceCF - RewardCF"
              % (rho, len(SEEDS)))
        print("%9s | " % "sigma_obs" + " ".join("%7.2f" % s for s in SIGS) + " | gap@hi")
        print("-" * 92)
        for nm in METHODS:
            vals = [np.mean(raw[str(rho)][str(sig)][nm]["overall"]) for sig in SIGS]
            print("%9s | " % nm + " ".join("%7.3f" % v for v in vals))
        gap = [np.mean(raw[str(rho)][str(sig)]["ChoiceCF"]["overall"])
               - np.mean(raw[str(rho)][str(sig)]["RewardCF"]["overall"]) for sig in SIGS]
        print("%9s | " % "Ch - Rew" + " ".join("%+7.3f" % v for v in gap))
    print("-" * 92)
    print("READ: gap (ChoiceCF - RewardCF) should rise with sigma_obs (clean choices beat")
    print("noisy rewards); BothCF >= both; masking (lower rho) shrinks both channels.")

    path = save_results("e3_channels", {
        "meta": {"experiment": "E3 channel grid rho x sigma_obs",
                 "methods": METHODS, "rhos": RHOS, "sigmas": SIGS, "seeds": SEEDS,
                 "m": pc.M, "n": pc.N, "d": pc.D, "K": pc.K, "d_hat": pc.D_HAT,
                 "T": pc.T, "cand": pc.CAND, "sigma_own": pc.SO,
                 "metric": "per-seed overall/unseen/anytime skill; both channels masked consistently"},
        "raw": raw})
    print("complete data saved ->", path)


if __name__ == "__main__":
    main()
