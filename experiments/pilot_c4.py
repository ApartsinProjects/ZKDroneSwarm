"""E-C4: anisotropy. Does the low-rank / unseen-pair win survive a SKEWED factor spectrum
(some latent directions far stronger than others), vs the isotropic worlds used elsewhere? Build
worlds with a geometric singular-value spectrum s_r = decay^r and sweep decay from 1.0 (isotropic)
toward 0 (-> dominant rank-1 = pure popularity). HYP: CF's unseen win degrades GRACEFULLY and
shrinks toward 0 as decay->0 (Thm 5: at rank-1 there is no personalization to transfer), while
structure-free stays at the floor. rho=1.0 (isolate anisotropy from masking). Writes docs/C4.md."""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
from pilot_c11_masking import run_masked
from core import make_world
from _results_io import save_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECAYS = [1.0, 0.7, 0.5, 0.3]      # 1.0 = isotropic; smaller = more skewed (-> rank-1)
RHO = 1.0
SEEDS = list(range(8))
ORDER = ["RewardCF", "BiasModel", "KNNCF", "Tabular", "UCBIndep"]
RNG = np.random.RandomState(0)

REG = {
    "RewardCF":  pc.REGISTRY["RewardCF"],
    "BiasModel": pc.REGISTRY["BiasModel"],     # popularity/additive (rank<=2): the Thm-5 reference
    "KNNCF":     pc.REGISTRY["KNNCF"],
    "Tabular":   pc.REGISTRY["Tabular"],
    "UCBIndep":  pc.REGISTRY["UCBIndep"],
}


def make_world_aniso(seed, decay):
    """Isotropic base world, then scale latent dim r by decay^r and renormalize R to [-1,1]."""
    P, U, R, meta = make_world(pc.M, pc.N, pc.D, pc.K, pc.K, within=0.15, seed=seed, signed=True)
    s = decay ** np.arange(pc.D)
    Ps = P * s; Us = U * s
    Ra = Ps @ Us.T
    mx = float(np.abs(Ra).max()) + 1e-9
    Ra = Ra / mx                                  # keep rewards in [-1,1] for comparable skill scale
    return (Ps, Us, Ra, meta)


def _job(args):
    nm, decay, seed = args
    Cls, hp = REG[nm]
    w = make_world_aniso(seed, decay)
    o, u, q = run_masked(Cls, hp, w, pc.T, seed, pc.SO, pc.SB, pc.CAND, pc.D_HAT, RHO)
    return nm, decay, seed, float(o), float(u)


def ci(vals, B=10000):
    a = np.asarray([v for v in vals if v is not None], float)
    idx = RNG.randint(0, len(a), (B, len(a))); mb = a[idx].mean(1)
    return a.mean(), np.percentile(mb, 2.5), np.percentile(mb, 97.5)


def cell(vals):
    mu, lo, hi = ci(vals); return "%.3f [%.3f, %.3f]" % (mu, lo, hi)


def main():
    jobs = [(nm, dc, s) for dc in DECAYS for nm in ORDER for s in SEEDS]
    raw = {str(dc): {nm: {"overall": [None] * len(SEEDS), "unseen": [None] * len(SEEDS)}
                     for nm in ORDER} for dc in DECAYS}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            nm, dc, s, o, u = fut.result()
            raw[str(dc)][nm]["overall"][s] = o
            raw[str(dc)][nm]["unseen"][s] = u
            done += 1
            if done % 16 == 0 or done == len(jobs):
                print("  ... %d/%d" % (done, len(jobs)))

    save_results("c4_aniso", {
        "meta": {"experiment": "anisotropy: unseen-pair win vs skewed factor spectrum (decay sweep)",
                 "methods": ORDER, "decays": DECAYS, "rho": RHO, "seeds": SEEDS,
                 "m": pc.M, "n": pc.N, "d": pc.D, "d_hat": pc.D_HAT,
                 "metric": "overall + unseen-pair skill vs spectrum decay (s_r = decay^r)"},
        "raw": raw}, results_dir=os.path.join(ROOT, "results", "pilots"))

    L = ["# Anisotropy: does the unseen win survive a skewed factor spectrum? (E-C4)\n",
         "Worlds with singular values s_r = decay^r (decay=1.0 isotropic; smaller = more skewed, "
         "approaching dominant rank-1). rho=%.1f, 8 seeds, bootstrap 95%% CI.\n" % RHO]
    for metric, title in [("unseen", "Unseen-pair skill (the categorical claim)"),
                          ("overall", "Overall skill")]:
        L.append("## %s\n" % title)
        L.append("| method | " + " | ".join("decay=%.1f" % d for d in DECAYS) + " |")
        L.append("|" + "---|" * (len(DECAYS) + 1))
        for nm in ORDER:
            cells = [cell(raw[str(d)][nm][metric]) for d in DECAYS]
            lab = "**%s**" % nm if nm == "RewardCF" else nm
            L.append("| %s | %s |" % (lab, " | ".join(cells)))
        L.append("")
    L.append("Read: if RewardCF's unseen skill stays well above the structure-free floor across decays "
             "and only SHRINKS as decay->0 (the spectrum collapses toward rank-1 = popularity, where "
             "Thm 5 says there is no personalization left to transfer), the low-rank win is robust to "
             "anisotropy and degrades exactly where the theory predicts.\n")
    out_md = os.path.join(ROOT, "docs", "C4.md")
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L)); print("wrote %s (raw saved earlier)" % out_md)


if __name__ == "__main__":
    main()
