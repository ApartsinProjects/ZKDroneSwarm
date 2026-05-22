"""
WHICH teammates to trust: per-teammate, per-channel reliability inferred online.

Setting: heterogeneous teammates. A fraction are FAULTY (novice/malfunctioning):
they choose RANDOMLY and their reward signal (as broadcast) is GARBAGE (uniform
noise). Reliable drones must figure out, with no labels, which teammates to
trust.

Channels (each with its own reliability latent variable):
  - REWARD: RewardCF (uniform, trusts all) vs RewardCFRobust (EM infers
    per-teammate precision tau_k from residuals; faulty -> huge residuals ->
    tau_k -> 0). Student-t / IRLS robust factorisation.
  - CHOICE: ChoiceCF naive (trusts all choices) vs comp (per-teammate competence
    gamma_k from behavioural consistency; faulty random choosers -> low
    consistency -> gamma_k -> 0).

Claim (the user's idea): inferring per-teammate trust robustly beats uniform
pooling as the faulty fraction rises, for BOTH channels. Measured on the
RELIABLE drones only.
"""
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pilot_noise import Tabular, RewardCF, ChoiceCF, make_world_struct, oracle_rand

# Solo baseline (no collaboration): each drone uses ONLY its own reward.
class Solo(Tabular):
    pass


class RandomAgent:
    def __init__(self, m, n, d, idx, seed, **hp):
        self.n = n; self.rng = np.random.RandomState(seed); self.pulled = np.zeros(n, bool)
    def select(self, t, cand):
        a = cand[self.rng.randint(len(cand))]; self.pulled[a] = True; return a
    def observe(self, t, choices, revealed, cand_sets, rvar):
        pass
    def predict_scores(self):
        return np.zeros(self.n)
    def pulled_mask(self):
        return self.pulled


class RewardCFRobust(RewardCF):
    """EM robust factorisation: infer per-teammate precision tau_k from residuals."""
    def __init__(self, *a, em_iters=3, **hp):
        super().__init__(*a, **hp)
        self.em_iters = em_iters
        self.trust = np.ones(self.m)

    def _refit(self):
        if not self.rk:
            return
        K = np.array(self.rk); J = np.array(self.rj); V = np.array(self.rv); base = np.array(self.rw)
        for _ in range(self.em_iters):
            W = base * self.trust[K]
            self._als(K, J, V, W)
            resid = V - np.einsum('nd,nd->n', self.P[K], self.U[J])
            sse = np.zeros(self.m); cnt = np.zeros(self.m)
            np.add.at(sse, K, resid ** 2); np.add.at(cnt, K, 1.0)
            var_k = sse / np.maximum(cnt, 1.0)
            med = np.median(var_k[cnt > 0]) if np.any(cnt > 0) else 1.0
            self.trust = np.where(cnt > 0, np.clip(med / (var_k + 1e-6), 0.02, 1.0), 1.0)


def run_episode_faulty(Cls, hp, world, T, seed, sigma_own, sigma_obs, cand, faulty):
    P, U, R = world; m, n = R.shape; d = P.shape[1]
    lo, hi = float(R.min()), float(R.max())
    rng = np.random.RandomState(seed + 999)
    faulty = set(faulty)
    reliable_set = [i for i in range(m) if i not in faulty]
    online_r = []
    learners = []
    for i in range(m):
        if i in faulty:
            learners.append(RandomAgent(m, n, d, i, seed + 7 * i + 1))
        else:
            learners.append(Cls(m, n, d, i, seed + 7 * i + 1, **hp))
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        online_r.append(float(np.mean([true_r[i] for i in reliable_set])))
        for i in range(m):
            if i in faulty:
                continue
            revealed = np.full(m, np.nan); rvar = np.full(m, np.inf)
            revealed[i] = true_r[i] + rng.normal(0, sigma_own); rvar[i] = sigma_own ** 2
            for k in range(m):
                if k == i:
                    continue
                if k in faulty:
                    revealed[k] = rng.uniform(lo, hi)          # garbage broadcast
                else:
                    revealed[k] = true_r[k] + rng.normal(0, sigma_obs)
                rvar[k] = sigma_obs ** 2                        # drone does NOT know who is faulty
            learners[i].observe(t, choices, revealed, cand_sets, rvar)
    reliable = [i for i in range(m) if i not in faulty]
    preds = {i: learners[i].predict_scores() for i in reliable}
    g = np.random.RandomState(seed + 555); got, oc, rd = [], [], []
    for _ in range(100):
        for k in reliable:
            off = g.choice(n, size=cand, replace=False)
            got.append(R[k, off[int(np.argmax(preds[k][off]))]])
            oc.append(R[k, off].max()); rd.append(R[k, off].mean())
    gm, ocm, rdm = np.mean(got), np.mean(oc), np.mean(rd)
    denom = max(ocm - rdm, 1e-6)
    greedy_skill = (gm - rdm) / denom                       # converged model quality
    auc_skill = (np.mean(online_r) - rdm) / denom           # anytime / cumulative
    return greedy_skill, auc_skill


def skill_ms(Cls, hp, cfg, m, n, d, T, cand, so, sb, frac, seeds):
    gs, aucs = [], []
    for s in seeds:
        w = make_world_struct(m, n, d, cfg["structure"], cfg["eps"], cfg["n_clusters"], cfg["sharp"], s)
        nf = int(round(frac * m))
        faulty = np.random.RandomState(1000 + s).choice(m, size=nf, replace=False) if nf > 0 else []
        g, a = run_episode_faulty(Cls, hp, w, T, s, so, sb, cand, faulty)
        gs.append(g); aucs.append(a)
    return float(np.mean(gs)), float(np.mean(aucs))


def main():
    m, n, d, T, cand = 30, 240, 5, 50, 20
    sigma_own, sigma_obs = 0.10, 0.30
    seeds = list(range(5))
    cfg = dict(structure="cluster_gauss", eps=0.10, n_clusters=15, sharp=1.0)
    base = dict(eps0=0.5, eps_min=0.05, eps_decay=0.93, als_sweeps=5, refit_every=3)
    tab = dict(eps0=0.5, eps_min=0.05, eps_decay=0.93)
    rew = dict(**base)
    rewR = dict(em_iters=3, **base)
    ch_n = dict(comp=False, s2c=0.2, n_neg=1, within=True, T_total=T, **base)
    ch_c = dict(comp=True, s2c=0.2, n_neg=1, within=True, warm_frac=0.3, T_total=T, **base)

    print("=" * 118)
    print("P1: COLLABORATION-HARM THRESHOLD (faulty to 80%). naive vs trust-aware vs SOLO.")
    print("m=%d n=%d d=%d T=%d cand=%d sigma_own=%.2f sigma_obs=%.2f %d seeds. clustG nc15"
          % (m, n, d, T, cand, sigma_own, sigma_obs, len(seeds)))
    print("skill of RELIABLE drones. faulty = random choices + garbage rewards. '<' = BELOW solo (harmful).")
    print("=" * 118)
    rows = {}
    for frac in [0.0, 0.3, 0.5, 0.65, 0.8]:
        rows[frac] = {
            "solo": skill_ms(Solo, tab, cfg, m, n, d, T, cand, sigma_own, sigma_obs, frac, seeds),
            "RewardCF": skill_ms(RewardCF, rew, cfg, m, n, d, T, cand, sigma_own, sigma_obs, frac, seeds),
            "RewRobust": skill_ms(RewardCFRobust, rewR, cfg, m, n, d, T, cand, sigma_own, sigma_obs, frac, seeds),
            "Ch_naive": skill_ms(ChoiceCF, ch_n, cfg, m, n, d, T, cand, sigma_own, sigma_obs, frac, seeds),
            "Ch_comp": skill_ms(ChoiceCF, ch_c, cfg, m, n, d, T, cand, sigma_own, sigma_obs, frac, seeds),
        }
    cols = ["RewardCF", "RewRobust", "Ch_naive", "Ch_comp"]
    for mi, mlabel in [(0, "GREEDY (converged model quality)"), (1, "AUC (anytime / cumulative)")]:
        print(f"\n--- {mlabel} ---")
        print(f"{'faulty%':>7s} | {'SOLO':>7s} | " + " ".join(f"{c:>10s}" for c in cols))
        print("-" * 70)
        for frac in [0.0, 0.3, 0.5, 0.65, 0.8]:
            so = rows[frac]["solo"][mi]
            cells = []
            for c in cols:
                x = rows[frac][c][mi]
                cells.append(f"{x:6.3f}{'<' if x < so - 1e-9 else ' '}")
            print(f"{frac*100:6.0f}% | {so:7.3f} | " + " ".join(f"{c:>10s}" for c in cells))
    print("-" * 70)
    print("GROUNDBREAKING if Ch_comp stays >= solo at ALL faulty% (incl 80%) while naive drops")
    print("below ('<'). Then competence-weighted DECISION pooling is majority-robust -- RANSAC-level")
    print("robustness via per-teammate consistency, no consensus search needed (simple framing).")


if __name__ == "__main__":
    main()
