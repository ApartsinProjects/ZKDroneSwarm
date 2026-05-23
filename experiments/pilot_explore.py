"""
H1 (backlog C10): collective info-directed exploration. Is POSTERIOR-directed exploration
(probe where the factor posterior is most uncertain) more SAMPLE-EFFICIENT than the
count-based bonus or eps-greedy? All three share the broadcast (so a probe helps everyone),
so this isolates the EXPLORATION rule:
  eps-greedy   (RewardCFconv)  -- random exploration
  count-bonus  (ActiveCFconv)  -- ActiveCF latent-UCB with a broadcast COUNT proxy
  posterior-UCB (EMCF)         -- Bayesian factorization, UCB on the predictive-variance
                                  (D-optimal / IDS-style info gain on the factor posterior)
Metric: the anytime cumulative-skill TRAJECTORY at rho=0.25 (masked, sample-starved), read
at early/mid/final rounds + rounds-to-half-of-final (rise speed). 8 seeds, bootstrap 95% CI.
Writes docs/EXPLORE.md. Saves raw before formatting.
"""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
import pilot_anytime as pa
from pilot_noise import RewardCF, ActiveCF, EMCF, CoordCF
from _results_io import save_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CONV = dict(eps0=0.5, eps_min=0.05, eps_decay=0.97, als_sweeps=12, refit_every=2)
_EM = dict(em_beta=1.0, em_sweeps=6, refit_every=4, eps0=0.5, eps_min=0.05, eps_decay=0.97)
REG = {
    "eps-greedy (RewardCF)":   (RewardCF, dict(**_CONV)),
    "count-bonus (ActiveCF)":  (ActiveCF, dict(c_active=0.5, **_CONV)),
    "posterior-UCB b=1 (EMCF)": (EMCF,    dict(**_EM)),                      # full predictive var (over-explores)
    "collective-UCB b=1 (EMCF)": (EMCF,   dict(coll=True, **_EM)),           # collective uncertainty, big bonus
    "collective b=0.3 (EMCF)":  (EMCF,    dict(**{**_EM, "coll": True, "em_beta": 0.3})),  # small exploit-bias
    "neg-corr (CoordCF)":       (CoordCF, dict(c_explore=0.5, **_CONV)),    # explicit division-of-labor
}
ORDER = ["eps-greedy (RewardCF)", "count-bonus (ActiveCF)", "posterior-UCB b=1 (EMCF)",
         "collective-UCB b=1 (EMCF)", "collective b=0.3 (EMCF)", "neg-corr (CoordCF)"]
RHO = 0.25
SEEDS = list(range(8))
RNG = np.random.RandomState(0)


def _job(args):
    name, seed = args
    Cls, hp = REG[name]
    traj = pa.run_anytime_clshp(Cls, hp, RHO, seed, d_hat=pc.D_HAT)   # cumulative skill per round
    return name, seed, traj


def ci(vals, B=10000):
    a = np.asarray(vals, float); idx = RNG.randint(0, len(a), (B, len(a))); m = a[idx].mean(1)
    return a.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def cell(vals):
    mu, lo, hi = ci(vals); return "%.3f [%.3f, %.3f]" % (mu, lo, hi)


def main():
    T = pc.T
    jobs = [(nm, s) for nm in ORDER for s in SEEDS]
    raw = {nm: [None] * len(SEEDS) for nm in ORDER}    # raw[nm][seed] = traj[T]
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            nm, s, traj = fut.result(); raw[nm][s] = traj
            done += 1
            if done % 6 == 0 or done == len(jobs):
                print("  ... %d/%d" % (done, len(jobs)))

    save_results("explore8", {"meta": {"experiment": "info-directed exploration sample efficiency",
                 "methods": ORDER, "rho": RHO, "T": T, "seeds": SEEDS,
                 "metric": "anytime cumulative-skill trajectory"},
                 "raw": raw}, results_dir=os.path.join(ROOT, "results", "pilots"))

    checkpoints = [max(1, T // 5), max(1, T // 2), T]   # early / mid / final round
    L = ["# Info-directed exploration: sample efficiency (rho=%.2f, T=%d)\n" % (RHO, T),
         "Anytime cumulative-skill at early/mid/final rounds (higher + earlier = more "
         "sample-efficient). 8 seeds, bootstrap 95%% CI.\n",
         "| exploration | round %d | round %d | round %d (final) | rounds to 1/2-final |"
         % tuple(checkpoints),
         "|---|---|---|---|---|"]
    for nm in ORDER:
        A = np.array(raw[nm])                       # (seeds, T)
        cks = [cell(A[:, c - 1]) for c in checkpoints]
        # rounds to reach half the final cumulative skill (rise speed), per seed
        half = 0.5 * A[:, -1]
        rr = []
        for s in range(len(SEEDS)):
            idx = np.where(A[s] >= half[s])[0]
            rr.append(int(idx[0]) + 1 if len(idx) else T)
        lab = "**%s**" % nm if "EMCF" in nm else nm
        L.append("| %s | %s | %s | %s | %.1f |" % (lab, cks[0], cks[1], cks[2], float(np.mean(rr))))
    L.append("")
    L.append("Read: if posterior-directed exploration is more sample-efficient, EMCF should show the "
             "highest early-round cumulative skill and the fewest rounds-to-half-final (it probes where "
             "the factor posterior is most uncertain, the most informative engagements). count-bonus "
             "(ActiveCF) is the cheap proxy; eps-greedy is uninformed.\n")

    out_md = os.path.join(ROOT, "docs", "EXPLORE.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote %s (raw saved earlier)" % out_md)


if __name__ == "__main__":
    main()
