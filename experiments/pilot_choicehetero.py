"""
Choice-channel under HETEROGENEOUS teammate competence (Section 5.6 follow-up), with a
built-in SANITY check.

Setting (the user's): the CHOICE-ONLY channel -- each drone keeps its OWN reward but of
teammates sees only their CHOICES (which target they engaged), partial (rho-masked), and
NO teammate rewards. Question: can we quantify, PER TEAMMATE, when it is choosing on solid
knowledge, and fold its choices in weighted by that? That is exactly ChoiceEM's per-
teammate informativeness gamma_k (joint EM over gamma_k AND the latent factors).

Why this experiment: our HOMOGENEOUS test found learned gamma_k merely TIES the fixed
competence ramp -- because every teammate explores/exploits on the SAME schedule, so
there is no per-teammate heterogeneity to exploit. Here we CREATE heterogeneity: a
fraction of the swarm are SPECIAL teammates of a fixed kind, and we ask whether the
learned gamma_k correctly tells them apart and whether that helps the GOOD (method) drones.

Teammate kinds (the SANITY contrast -- obvious expected results):
  random : choices are uniform-RANDOM every round (faulty / off-objective). gamma SHOULD
           stay LOW; trusting them should NOT poison the good drones.
  oracle : choices are the TRUE best-in-offer (maximally informative). gamma SHOULD go
           HIGH; trusting them should HELP the good drones a lot.
  If a method's gamma cannot separate oracle (high) from random (low), it is broken.

Methods (own reward + teammates' masked choices; RewardCF is a reward-channel yardstick):
  RewardCF       -- uses teammate REWARDS (a random teammate's reward on a random target is
                    still a valid obs), so naturally robust; NOT admissible in the choice-only
                    setting, only a reference.
  ChoiceCF       -- choice channel, fixed temporal ramp x consistency heuristic.
  ChoiceEM       -- choice channel, LEARNED gamma_k, IN-SAMPLE responsibility (current).
  ChoiceEM-pred  -- choice channel, LEARNED gamma_k, HELD-OUT (predictive) responsibility:
                    each choice scored once vs the model BEFORE it is incorporated, so a
                    random teammate cannot be overfit into looking informative.

Metrics (8 seeds, bootstrap 95% CI): GOOD-drone unseen-pair skill, GOOD-drone anytime
earned skill, and the gamma SEPARATION diagnostic (mean learned gamma_k for good vs special
teammates). rho=1.0 to isolate the effect. Writes docs/CHOICEHETERO.md. Saves raw first.
"""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
from pilot_noise import RewardCF, ChoiceCF, ChoiceEM
from core import make_world
from _results_io import save_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CONV = dict(eps0=0.5, eps_min=0.05, eps_decay=0.97, als_sweeps=12, refit_every=2)
REG = {
    "RewardCF":      (RewardCF, dict(**_CONV)),
    "ChoiceCF":      (ChoiceCF, dict(comp=True, s2c=0.2, n_neg=1, within=True,
                                     warm_frac=0.3, T_total=pc.T, **_CONV)),
    "ChoiceEM":      (ChoiceEM, dict(tau=0.3, s2c=0.2, n_neg=1, within=True, T_total=pc.T,
                                     gamma_init=0.1, warm_em=0.3, predictive=False, **_CONV)),
    "ChoiceEM-pred": (ChoiceEM, dict(tau=0.3, s2c=0.2, n_neg=1, within=True, T_total=pc.T,
                                     gamma_init=0.1, warm_em=0.3, predictive=True, **_CONV)),
}
ORDER = ["RewardCF", "ChoiceCF", "ChoiceEM", "ChoiceEM-pred"]
KINDS = ["random", "oracle"]
FRACS = [0.0, 0.33, 0.5]          # fraction of the swarm that are SPECIAL teammates
RHO = 1.0
SEEDS = list(range(8))
RNG = np.random.RandomState(0)


class Distractor:
    """A teammate whose CHOICES are uniform-random every round (faulty / off-objective):
    it does not learn and injects uninformative choices into the public broadcast."""
    def __init__(self, m, n, d, idx, seed, Rrow=None):
        self.idx = idx; self.n = n
        self.rng = np.random.RandomState(seed)
        self.pulled = np.zeros(n, bool)

    def select(self, t, cand):
        a = int(cand[self.rng.randint(len(cand))]); self.pulled[a] = True; return a

    def observe(self, t, choices, revealed, cand_sets, rvar):
        pass

    def predict_scores(self):
        return np.zeros(self.n)

    def pulled_mask(self):
        return self.pulled


class OracleMate:
    """A teammate that always engages its TRUE best-in-offer target (maximally informative
    choices). The sanity-check 'expert': its gamma MUST be driven high by a working estimator."""
    def __init__(self, m, n, d, idx, seed, Rrow=None):
        # Rrow is ENVIRONMENT-ONLY (this is a special teammate / env actor, not a method under
        # test): it shapes only the public broadcast others observe, is never read by any learner,
        # and OracleMate is excluded from learning (spec[i] skip) and from all metrics (good_idx).
        self.idx = idx; self.n = n; self.Rrow = Rrow
        self.pulled = np.zeros(n, bool)

    def select(self, t, cand):
        cand = np.asarray(cand)
        a = int(cand[int(np.argmax(self.Rrow[cand]))]); self.pulled[a] = True; return a

    def observe(self, t, choices, revealed, cand_sets, rvar):
        pass

    def predict_scores(self):
        return np.zeros(self.n)

    def pulled_mask(self):
        return self.pulled


SPECIAL = {"random": Distractor, "oracle": OracleMate}


def _run(Cls, hp, frac, kind, seed):
    """Train with a `frac` fraction of SPECIAL teammates of the given kind; return
    (unseen, anytime, gamma_good, gamma_special) over the GOOD (method) drones."""
    w = make_world(pc.M, pc.N, pc.D, pc.K, pc.K, within=0.15, seed=seed, signed=True)
    P, U, R = w[:3]; m, n = R.shape
    T, cand, so, sb = pc.T, pc.CAND, pc.SO, pc.SB
    rng = np.random.RandomState(seed + 999)
    n_spec = int(round(frac * m))
    spec = np.zeros(m, bool)
    if n_spec > 0:
        spec[m - n_spec:] = True                     # last n_spec drones are special
    good_idx = np.where(~spec)[0]
    SpecCls = SPECIAL[kind]
    Mask = rng.rand(m, m) < RHO; np.fill_diagonal(Mask, True)
    learners = [(SpecCls(m, n, pc.D_HAT, i, seed + 7 * i + 1, Rrow=R[i]) if spec[i]
                 else Cls(m, n, pc.D_HAT, i, seed + 7 * i + 1, **hp)) for i in range(m)]

    real = np.zeros(T); orac = np.zeros(T); rnd = np.zeros(T)
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        real[t] = float(sum(true_r[i] for i in good_idx))
        orac[t] = float(sum(R[i, cand_sets[i]].max() for i in good_idx))
        rnd[t] = float(sum(R[i, cand_sets[i]].mean() for i in good_idx))
        for i in range(m):
            if spec[i]:
                continue                              # special teammates do not learn
            revealed = np.full(m, np.nan); rvar = np.full(m, np.inf)
            revealed[i] = true_r[i] + rng.normal(0, so); rvar[i] = so ** 2
            for k in range(m):
                if k != i and Mask[i, k]:
                    revealed[k] = true_r[k] + rng.normal(0, sb); rvar[k] = sb ** 2
            learners[i].observe(t, choices, revealed, cand_sets, rvar)

    cr, co, cd = real.cumsum(), orac.cumsum(), rnd.cumsum()
    anytime = float((cr[-1] - cd[-1]) / max(co[-1] - cd[-1], 1e-9))

    g = np.random.RandomState(seed + 555)
    ung, uno, unr = [], [], []
    for k in good_idx:
        preds = learners[k].predict_scores()
        unseen = np.where(~learners[k].pulled_mask())[0]
        if len(unseen) < cand:
            continue
        for _ in range(120):
            off = g.choice(unseen, size=cand, replace=False)
            ung.append(R[k, off[int(np.argmax(preds[off]))]])
            uno.append(R[k, off].max()); unr.append(R[k, off].mean())
    unseen = float((np.mean(ung) - np.mean(unr)) / max(np.mean(uno) - np.mean(unr), 1e-6))

    g_good = g_spec = float("nan")
    if n_spec > 0 and hasattr(learners[int(good_idx[0])], "gamma"):
        gg, gs = [], []
        for k in good_idx:
            gam = learners[k].gamma
            for j in range(m):
                if j == k:
                    continue
                (gs if spec[j] else gg).append(float(gam[j]))
        g_good = float(np.mean(gg)) if gg else float("nan")
        g_spec = float(np.mean(gs)) if gs else float("nan")
    return unseen, anytime, g_good, g_spec


def _job(args):
    name, kind, frac, seed = args
    Cls, hp = REG[name]
    u, a, gg, gs = _run(Cls, hp, frac, kind, seed)
    return name, kind, frac, seed, u, a, gg, gs


def ci(vals, B=10000):
    a = np.asarray([v for v in vals if v == v], float)
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = RNG.randint(0, len(a), (B, len(a))); mb = a[idx].mean(1)
    return a.mean(), np.percentile(mb, 2.5), np.percentile(mb, 97.5)


def cell(vals):
    mu, lo, hi = ci(vals)
    return "n/a" if mu != mu else "%.3f [%.3f, %.3f]" % (mu, lo, hi)


def main():
    jobs = [(nm, kd, f, s) for kd in KINDS for f in FRACS for nm in ORDER for s in SEEDS]
    raw = {kd: {("%.2f" % f): {nm: {m: [None] * len(SEEDS) for m in ("unseen", "anytime", "g_good", "g_spec")}
                               for nm in ORDER} for f in FRACS} for kd in KINDS}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            nm, kd, f, s, u, a, gg, gs = fut.result()
            key = "%.2f" % f
            raw[kd][key][nm]["unseen"][s] = u; raw[kd][key][nm]["anytime"][s] = a
            raw[kd][key][nm]["g_good"][s] = gg; raw[kd][key][nm]["g_spec"][s] = gs
            done += 1
            if done % 16 == 0 or done == len(jobs):
                print("  ... %d/%d" % (done, len(jobs)))

    save_results("choicehetero8", {
        "meta": {"experiment": "choice channel under heterogeneous teammate competence + sanity",
                 "methods": ORDER, "kinds": KINDS, "fracs": FRACS, "rho": RHO, "seeds": SEEDS,
                 "m": pc.M, "n": pc.N, "d": pc.D, "d_hat": pc.D_HAT, "T": pc.T,
                 "sigma_own": pc.SO, "sigma_obs": pc.SB,
                 "metric": "good-drone unseen + anytime + gamma separation"},
        "raw": raw}, results_dir=os.path.join(ROOT, "results", "pilots"))

    L = ["# Choice channel under heterogeneous teammate competence (+ sanity check)\n",
         "Own reward + teammates' masked CHOICES only (no teammate rewards). A `frac` fraction of the "
         "%d-drone swarm are SPECIAL teammates: either RANDOM choosers (uninformative) or ORACLE choosers "
         "(true best-in-offer, maximally informative). We report the GOOD (method) drones. Does learned "
         "per-teammate informativeness (gamma_k) tell them apart and help? rho=%.1f, 8 seeds, bootstrap "
         "95%% CI.\n" % (pc.M, RHO)]
    for kd in KINDS:
        L.append("## Teammate kind = %s\n" % kd.upper())
        for metric, title in (("unseen", "Unseen-pair skill (good drones)"),
                              ("anytime", "Anytime earned skill (good drones)")):
            L.append("### %s\n" % title)
            L.append("| method | " + " | ".join("special=%d%%" % int(100 * f) for f in FRACS) + " |")
            L.append("|" + "---|" * (len(FRACS) + 1))
            for nm in ORDER:
                cells = [cell(raw[kd]["%.2f" % f][nm][metric]) for f in FRACS]
                lab = "**%s**" % nm if nm.startswith("ChoiceEM") else nm
                L.append("| %s | %s |" % (lab, " | ".join(cells)))
            L.append("")
        L.append("### Learned gamma separation (good vs %s teammates)\n" % kd)
        L.append("| method / teammate | " + " | ".join("special=%d%%" % int(100 * f) for f in FRACS) + " |")
        L.append("|" + "---|" * (len(FRACS) + 1))
        for nm in ("ChoiceEM", "ChoiceEM-pred"):
            for kind_lab, key in (("good", "g_good"), (kd, "g_spec")):
                cells = [cell(raw[kd]["%.2f" % f][nm][key]) for f in FRACS]
                L.append("| %s : gamma(%s) | %s |" % (nm, kind_lab, " | ".join(cells)))
        L.append("")
    L.append("Sanity: a working estimator must drive gamma(oracle) HIGH and gamma(random) LOW. "
             "Win condition: ChoiceEM-pred separates them more sharply than in-sample ChoiceEM and keeps "
             "the good drones' skill robust as RANDOM teammates grow, while leveraging ORACLE teammates.\n")

    out_md = os.path.join(ROOT, "docs", "CHOICEHETERO.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote %s (raw saved earlier)" % out_md)


if __name__ == "__main__":
    main()
