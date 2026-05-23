"""
Choice-channel under HETEROGENEOUS teammate competence (Section 5.6 follow-up).

Setting (the user's): the CHOICE-ONLY channel -- each drone keeps its OWN reward but of
teammates sees only their CHOICES (which target they engaged), partial (rho-masked), and
NO teammate rewards. Question: can we quantify, PER TEAMMATE, when it starts choosing on
solid knowledge, and fold its choices in weighted by that? That is exactly ChoiceEM's
per-teammate informativeness gamma_k (joint EM over gamma_k AND the latent factors).

Why this experiment: our earlier HOMOGENEOUS test found learned gamma_k merely TIES the
fixed competence ramp -- because every teammate explores/exploits on the SAME schedule, so
there is no per-teammate heterogeneity to exploit. Here we CREATE that heterogeneity: a
fraction of the swarm are DISTRACTORS whose choices are uniform-RANDOM every round
(faulty / jamming / off-objective teammates whose actions carry no preference). A drone
that trusts all teammates' choices on a shared schedule is then POISONED by the
distractors; ChoiceEM should learn LOW gamma for distractors and HIGH gamma for real
learners, staying robust.

Method spectrum (own reward + teammates' masked choices, rho=1.0 to isolate the effect):
  RewardCF    -- reference: uses teammate REWARDS (a distractor's reward on a random target
                 is still a valid (target,reward) obs, so the reward channel is naturally
                 robust). The user's setting has NO teammate rewards, so this is only a
                 yardstick, not an admissible choice-only method.
  ChoiceNaive -- choice channel, trust ALL choices equally (comp=False): most poisoned.
  ChoiceCF    -- choice channel, fixed temporal ramp x temporal-consistency heuristic.
  ChoiceEM    -- choice channel, LEARNED per-teammate gamma_k (the model in question).

Sweep the distractor fraction. Metrics (8 seeds, bootstrap 95% CI): unseen-pair skill of
the GOOD (non-distractor) drones' learned models, the anytime earned skill of the good
drones, and the gamma SEPARATION diagnostic (mean learned gamma_k for good vs distractor
teammates) showing ChoiceEM identifies WHO KNOWS. Writes docs/CHOICEHETERO.md. Saves raw
BEFORE formatting.
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
    "RewardCF":    (RewardCF, dict(**_CONV)),
    "ChoiceNaive": (ChoiceCF, dict(comp=False, s2c=0.2, n_neg=1, within=True,
                                   warm_frac=0.3, T_total=pc.T, **_CONV)),
    "ChoiceCF":    (ChoiceCF, dict(comp=True, s2c=0.2, n_neg=1, within=True,
                                   warm_frac=0.3, T_total=pc.T, **_CONV)),
    "ChoiceEM":    (ChoiceEM, dict(tau=0.3, s2c=0.2, n_neg=1, within=True, T_total=pc.T,
                                   gamma_init=0.1, warm_em=0.3, **_CONV)),
}
ORDER = ["RewardCF", "ChoiceNaive", "ChoiceCF", "ChoiceEM"]
FRACS = [0.0, 0.33, 0.5]          # fraction of the swarm that are random-choice distractors
RHO = 1.0                         # full detection: ISOLATE heterogeneity from masking
SEEDS = list(range(8))
RNG = np.random.RandomState(0)


class Distractor:
    """A teammate whose CHOICES are uninformative (uniform random) every round: a faulty,
    jamming, or off-objective drone. It does not learn (its actions carry no preference),
    it only injects random choices into the public broadcast for others to (mis)read."""
    def __init__(self, m, n, d, idx, seed, **hp):
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


def _run(Cls, hp, frac, seed):
    """Train under a swarm with a `frac` fraction of random-choice distractors; return
    (unseen_skill, anytime_skill, gamma_good, gamma_dist) over the GOOD drones."""
    w = make_world(pc.M, pc.N, pc.D, pc.K, pc.K, within=0.15, seed=seed, signed=True)
    P, U, R = w[:3]; m, n = R.shape
    T, cand, so, sb = pc.T, pc.CAND, pc.SO, pc.SB
    rng = np.random.RandomState(seed + 999)
    n_dist = int(round(frac * m))
    dist = np.zeros(m, bool)
    if n_dist > 0:
        dist[m - n_dist:] = True            # last n_dist drones are distractors
    good_idx = np.where(~dist)[0]
    Mask = rng.rand(m, m) < RHO; np.fill_diagonal(Mask, True)
    learners = [(Distractor(m, n, pc.D_HAT, i, seed + 7 * i + 1) if dist[i]
                 else Cls(m, n, pc.D_HAT, i, seed + 7 * i + 1, **hp)) for i in range(m)]

    real = np.zeros(T); orac = np.zeros(T); rnd = np.zeros(T)
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        real[t] = float(sum(true_r[i] for i in good_idx))            # anytime over GOOD drones
        orac[t] = float(sum(R[i, cand_sets[i]].max() for i in good_idx))
        rnd[t] = float(sum(R[i, cand_sets[i]].mean() for i in good_idx))
        for i in range(m):
            if dist[i]:
                continue                                             # distractors do not learn
            revealed = np.full(m, np.nan); rvar = np.full(m, np.inf)
            revealed[i] = true_r[i] + rng.normal(0, so); rvar[i] = so ** 2
            for k in range(m):
                if k != i and Mask[i, k]:
                    revealed[k] = true_r[k] + rng.normal(0, sb); rvar[k] = sb ** 2
            learners[i].observe(t, choices, revealed, cand_sets, rvar)

    cr, co, cd = real.cumsum(), orac.cumsum(), rnd.cumsum()
    anytime = float((cr[-1] - cd[-1]) / max(co[-1] - cd[-1], 1e-9))

    g = np.random.RandomState(seed + 555)                            # unseen-pair skill (good drones)
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

    g_good = g_dist = float("nan")                                   # gamma separation (ChoiceEM only)
    if n_dist > 0 and hasattr(learners[int(good_idx[0])], "gamma"):
        gg, gd = [], []
        for k in good_idx:
            gam = learners[k].gamma
            for j in range(m):
                if j == k:
                    continue
                (gd if dist[j] else gg).append(float(gam[j]))
        g_good = float(np.mean(gg)) if gg else float("nan")
        g_dist = float(np.mean(gd)) if gd else float("nan")
    return unseen, anytime, g_good, g_dist


def _job(args):
    name, frac, seed = args
    Cls, hp = REG[name]
    u, a, gg, gd = _run(Cls, hp, frac, seed)
    return name, frac, seed, u, a, gg, gd


def ci(vals, B=10000):
    a = np.asarray([v for v in vals if v == v], float)              # drop NaNs
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = RNG.randint(0, len(a), (B, len(a))); mb = a[idx].mean(1)
    return a.mean(), np.percentile(mb, 2.5), np.percentile(mb, 97.5)


def cell(vals):
    mu, lo, hi = ci(vals)
    if mu != mu:
        return "n/a"
    return "%.3f [%.3f, %.3f]" % (mu, lo, hi)


def main():
    jobs = [(nm, f, s) for f in FRACS for nm in ORDER for s in SEEDS]
    raw = {("%.2f" % f): {nm: {"unseen": [None] * len(SEEDS), "anytime": [None] * len(SEEDS),
                              "g_good": [None] * len(SEEDS), "g_dist": [None] * len(SEEDS)}
                          for nm in ORDER} for f in FRACS}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            nm, f, s, u, a, gg, gd = fut.result()
            key = "%.2f" % f
            raw[key][nm]["unseen"][s] = u; raw[key][nm]["anytime"][s] = a
            raw[key][nm]["g_good"][s] = gg; raw[key][nm]["g_dist"][s] = gd
            done += 1
            if done % 16 == 0 or done == len(jobs):
                print("  ... %d/%d" % (done, len(jobs)))

    save_results("choicehetero8", {
        "meta": {"experiment": "choice channel under heterogeneous teammate competence",
                 "methods": ORDER, "fracs": FRACS, "rho": RHO, "seeds": SEEDS,
                 "m": pc.M, "n": pc.N, "d": pc.D, "d_hat": pc.D_HAT, "T": pc.T,
                 "sigma_own": pc.SO, "sigma_obs": pc.SB,
                 "metric": "good-drone unseen + anytime + gamma separation"},
        "raw": raw}, results_dir=os.path.join(ROOT, "results", "pilots"))

    L = ["# Choice channel under heterogeneous teammate competence\n",
         "Own reward + teammates' masked CHOICES only (no teammate rewards). A `frac` fraction of the "
         "%d-drone swarm are DISTRACTORS whose choices are uniform-random every round (faulty / "
         "off-objective). We report the GOOD drones' learned-model quality. Does learned per-teammate "
         "informativeness (ChoiceEM gamma_k) beat the fixed competence ramp when teammates differ? "
         "rho=%.1f, 8 seeds, bootstrap 95%% CI.\n" % (pc.M, RHO)]
    for metric, title in (("unseen", "Unseen-pair skill (good drones)"),
                          ("anytime", "Anytime earned skill (good drones)")):
        L.append("## %s\n" % title)
        L.append("| method | " + " | ".join("distractors=%d%%" % int(100 * f) for f in FRACS) + " |")
        L.append("|" + "---|" * (len(FRACS) + 1))
        for nm in ORDER:
            cells = [cell(raw["%.2f" % f][nm][metric]) for f in FRACS]
            lab = "**%s**" % nm if nm == "ChoiceEM" else nm
            L.append("| %s | %s |" % (lab, " | ".join(cells)))
        L.append("")
    # gamma separation diagnostic (ChoiceEM): does it tell good teammates from distractors?
    L.append("## ChoiceEM gamma separation (mean learned gamma_k; + gap = identifies who knows)\n")
    L.append("| teammate kind | " + " | ".join("distractors=%d%%" % int(100 * f) for f in FRACS) + " |")
    L.append("|" + "---|" * (len(FRACS) + 1))
    for kind, lab in (("g_good", "good teammates"), ("g_dist", "distractor teammates")):
        cells = [cell(raw["%.2f" % f]["ChoiceEM"][kind]) for f in FRACS]
        L.append("| %s | %s |" % (lab, " | ".join(cells)))
    L.append("")
    L.append("Read: if learned gamma_k works, ChoiceEM's gamma for GOOD teammates should stay high "
             "while gamma for DISTRACTORS collapses toward 0 (it spots the uninformative drones), and "
             "ChoiceEM's good-drone skill should degrade LESS than ChoiceCF/ChoiceNaive as distractors "
             "grow (it is not poisoned by random choices).\n")

    out_md = os.path.join(ROOT, "docs", "CHOICEHETERO.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote %s (raw saved earlier)" % out_md)


if __name__ == "__main__":
    main()
