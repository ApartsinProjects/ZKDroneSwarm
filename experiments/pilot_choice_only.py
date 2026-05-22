"""
Phase A online pilot v3: conjugate / RLS core (reuses bayesian_pmf_policy.py's
closed-form update) + dual-likelihood choices.

WHY v3: pilots v1/v2 used SGD-MF, which failed to learn even with full rewards
(RewardMF lost to Tabular at p=1). The fix is the closed-form conjugate update
(recursive least squares): per observation, an additive rank-1 update to the
precision Lambda and precision-weighted mean eta, no learning rate, no
recomputation. This is exactly bayesian_pmf_policy.py's mechanism.

Regime: sample-STARVED changing candidate subsets (T pulls/drone < n, a random
subset offered each step, so camping is impossible and generalisation is
required). Repeated assignment, maximise reward/step.

Policies:
  Random, Tabular (own-row bandit), RewardMF_SGD (old, for contrast),
  RewardMF_RLS (conjugate), DualMF_RLS (conjugate rewards + preference choices).

Key diagnostics:
  - reward/step at p=1 (rewards shared) and p=0 (decision-only).
  - GATE 2a: RewardMF_RLS should beat Tabular at p=1 (method fixed, CF helps
    when starved). RewardMF_RLS should collapse toward Tabular at p=0.
  - GATE 2b: DualMF_RLS should stay high at p=0 (choices recover U).
"""
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")


def make_world(m, n, d, mode_eps, seed):
    rng = np.random.RandomState(seed)
    drone_modes = rng.randint(0, d, m)
    target_modes = np.concatenate([np.arange(d), rng.randint(0, d, n - d)])
    rng.shuffle(target_modes)
    P = np.zeros((m, d)); U = np.zeros((n, d))
    for i in range(m):
        v = np.zeros(d); v[drone_modes[i]] = 1.0
        v += mode_eps * rng.randn(d); P[i] = v / np.linalg.norm(v)
    for j in range(n):
        v = np.zeros(d); v[target_modes[j]] = 1.0
        v += mode_eps * rng.randn(d); U[j] = v / np.linalg.norm(v)
    return P, U, P @ U.T


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 1e-12 else 0.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# --------------------------------------------------------------------------- #
class Random:
    def __init__(self, m, n, d, idx, seed, **hp):
        self.n = n; self.idx = idx; self.rng = np.random.RandomState(seed)
        self.pulled = np.zeros(n, bool)
    def select(self, t, cand):
        a = cand[self.rng.randint(len(cand))]; self.pulled[a] = True; return a
    def observe(self, t, choices, revealed, cand_sets): pass
    def predict_scores(self): return np.zeros(self.n)
    def pulled_mask(self): return self.pulled


class Tabular:
    def __init__(self, m, n, d, idx, seed, eps=0.15, **hp):
        self.n = n; self.idx = idx; self.rng = np.random.RandomState(seed); self.eps = eps
        self.counts = np.zeros(n); self.sums = np.zeros(n)
        self.tot_n = 0; self.tot_s = 0.0; self.pulled = np.zeros(n, bool)
    def _est(self):
        default = (self.tot_s / self.tot_n) if self.tot_n > 0 else 0.5
        e = np.full(self.n, default); m = self.counts > 0
        e[m] = self.sums[m] / self.counts[m]; return e
    def select(self, t, cand):
        if self.rng.rand() < self.eps:
            a = cand[self.rng.randint(len(cand))]
        else:
            a = cand[int(np.argmax(self._est()[cand]))]
        self.pulled[a] = True; return a
    def observe(self, t, choices, revealed, cand_sets):
        a = choices[self.idx]; r = revealed[self.idx]
        if not np.isnan(r):
            self.counts[a] += 1; self.sums[a] += r; self.tot_n += 1; self.tot_s += r
    def predict_scores(self): return self._est()
    def pulled_mask(self): return self.pulled


class RewardMF_SGD:
    def __init__(self, m, n, d, idx, seed, lr=0.2, reg=1e-3, eps=0.15, init_scale=0.3, **hp):
        self.m = m; self.n = n; self.idx = idx; self.rng = np.random.RandomState(seed)
        self.lr = lr; self.reg = reg; self.eps = eps
        self.P = self.rng.normal(0, init_scale, (m, d)); self.U = self.rng.normal(0, init_scale, (n, d))
        self.pulled = np.zeros(n, bool)
    def _g(self, k, j, r):
        err = self.P[k] @ self.U[j] - r
        self.P[k] -= self.lr * (err * self.U[j] + self.reg * self.P[k])
        self.U[j] -= self.lr * (err * self.P[k] + self.reg * self.U[j])
    def select(self, t, cand):
        if self.rng.rand() < self.eps:
            a = cand[self.rng.randint(len(cand))]
        else:
            a = cand[int(np.argmax(self.U[cand] @ self.P[self.idx]))]
        self.pulled[a] = True; return a
    def observe(self, t, choices, revealed, cand_sets):
        for k in range(self.m):
            r = revealed[k]
            if not np.isnan(r): self._g(k, int(choices[k]), float(r))
    def predict_scores(self): return self.U @ self.P[self.idx]
    def pulled_mask(self): return self.pulled


class RewardMF_RLS:
    """Conjugate / recursive-least-squares MF (the bayesian_pmf_policy core).
    Per obs: additive rank-1 to Lambda and eta. No learning rate."""
    def __init__(self, m, n, d, idx, seed, s2=0.02, prior_prec=1.0, eps=0.15,
                 init_scale=0.1, **hp):
        self.m = m; self.n = n; self.d = d; self.idx = idx; self.s2 = s2; self.eps = eps
        self.rng = np.random.RandomState(seed)
        self.LP = np.tile(prior_prec * np.eye(d), (m, 1, 1)).copy()
        self.LU = np.tile(prior_prec * np.eye(d), (n, 1, 1)).copy()
        muP0 = self.rng.normal(0, init_scale, (m, d)); muU0 = self.rng.normal(0, init_scale, (n, d))
        self.eP = np.einsum('idj,ij->id', self.LP, muP0)
        self.eU = np.einsum('jdk,jk->jd', self.LU, muU0)
        self.muP = muP0.copy(); self.muU = muU0.copy()
        self.dirty = False; self.pulled = np.zeros(n, bool)
    def _refresh(self):
        self.muP = np.linalg.solve(self.LP, self.eP[..., None])[..., 0]
        self.muU = np.linalg.solve(self.LU, self.eU[..., None])[..., 0]
        self.dirty = False
    def _gauss(self, k, j, r):
        u = self.muU[j]; p = self.muP[k]
        self.LP[k] += np.outer(u, u) / self.s2; self.eP[k] += r * u / self.s2
        self.LU[j] += np.outer(p, p) / self.s2; self.eU[j] += r * p / self.s2
    def select(self, t, cand):
        if self.dirty: self._refresh()
        if self.rng.rand() < self.eps:
            a = cand[self.rng.randint(len(cand))]
        else:
            a = cand[int(np.argmax(self.muU[cand] @ self.muP[self.idx]))]
        self.pulled[a] = True; return a
    def observe(self, t, choices, revealed, cand_sets):
        if self.dirty: self._refresh()
        for k in range(self.m):
            r = revealed[k]
            if not np.isnan(r): self._gauss(k, int(choices[k]), float(r))
        self.dirty = True
    def predict_scores(self):
        if self.dirty: self._refresh()
        return self.muU @ self.muP[self.idx]
    def pulled_mask(self): return self.pulled


class DualMF_RLS(RewardMF_RLS):
    """Conjugate rewards + preference (choice) evidence as Gaussian pseudo-obs
    on the score difference (TrueSkill-style ranking in the conjugate frame)."""
    def __init__(self, *a, margin=0.2, s2c=0.5, n_neg=2, within=True, **hp):
        super().__init__(*a, **hp)
        self.margin = margin; self.s2c = s2c; self.n_neg = n_neg; self.within = within
    def _choice(self, k, c, offered):
        p = self.muP[k]
        for _ in range(self.n_neg):
            if self.within and offered is not None and len(offered) > 1:
                o = int(self.rng.choice(offered))
            else:
                o = self.rng.randint(self.n)
            if o == c:
                continue
            uc = self.muU[c]; uo = self.muU[o]; w = uc - uo
            self.LP[k] += np.outer(w, w) / self.s2c; self.eP[k] += self.margin * w / self.s2c
            self.LU[c] += np.outer(p, p) / self.s2c; self.eU[c] += (self.margin + p @ uo) * p / self.s2c
            self.LU[o] += np.outer(p, p) / self.s2c; self.eU[o] += (p @ uc - self.margin) * p / self.s2c
    def observe(self, t, choices, revealed, cand_sets):
        if self.dirty: self._refresh()
        for k in range(self.m):
            r = revealed[k]
            if not np.isnan(r): self._gauss(k, int(choices[k]), float(r))
        for k in range(self.m):
            self._choice(k, int(choices[k]), cand_sets[k] if cand_sets is not None else None)
        self.dirty = True


# --------------------------------------------------------------------------- #
def run_episode(Cls, hp, world, T, p_share, seed, sigma=0.1, cand=12):
    P, U, R = world; m, n = R.shape; d = P.shape[1]
    rng = np.random.RandomState(seed + 999)
    learners = [Cls(m, n, d, i, seed + 7 * i + 1, **hp) for i in range(m)]
    step_reward = np.zeros(T)
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        step_reward[t] = true_r.mean()
        for i in range(m):
            revealed = np.full(m, np.nan)
            revealed[i] = true_r[i] + rng.normal(0, sigma)
            for k in range(m):
                if k != i and rng.rand() < p_share:
                    revealed[k] = true_r[k] + rng.normal(0, sigma)
            learners[i].observe(t, choices, revealed, cand_sets)
    recs = [spearman(learners[i].predict_scores(), R[i]) for i in range(m)]
    return step_reward.mean(), float(np.mean(recs))


def avg_seeds(Cls, hp, T, p_share, seeds, wp):
    rs, rec = [], []
    for s in seeds:
        w = make_world(wp["m"], wp["n"], wp["d"], wp["mode_eps"], s)
        r, u = run_episode(Cls, hp, w, T, p_share, s, wp["sigma"], wp["cand"])
        rs.append(r); rec.append(u)
    return np.mean(rs), np.std(rs), np.mean(rec)


def main():
    wp = dict(m=30, n=120, d=5, mode_eps=0.25, sigma=0.10, cand=12)
    T = 50; seeds = list(range(5))
    policies = {
        "Random":       (Random, {}),
        "Tabular":      (Tabular, dict(eps=0.15)),
        "RewardMF_SGD": (RewardMF_SGD, dict(lr=0.2, reg=1e-3, eps=0.15, init_scale=0.3)),
        "RewardMF_RLS": (RewardMF_RLS, dict(s2=0.02, eps=0.15, init_scale=0.1)),
        "DualMF_RLS":   (DualMF_RLS, dict(s2=0.02, eps=0.15, init_scale=0.1,
                                          margin=0.2, s2c=0.5, n_neg=2, within=True)),
    }
    print("=" * 84)
    print("PILOT v3: conjugate/RLS core. m=%d n=%d d=%d T=%d cand=%d sigma=%.2f %d seeds"
          % (wp['m'], wp['n'], wp['d'], T, wp['cand'], wp['sigma'], len(seeds)))
    print("changing subsets (starved). oracle~0.77 random~0.15")
    print("=" * 84)
    print(f"{'policy':14s}  {'reward p=0':>12s}  {'reward p=1':>12s}  {'Urec p=0':>9s}")
    print("-" * 56)
    for name, (Cls, hp) in policies.items():
        r0, s0, u0 = avg_seeds(Cls, hp, T, 0.0, seeds, wp)
        r1, s1, u1 = avg_seeds(Cls, hp, T, 1.0, seeds, wp)
        print(f"{name:14s}  {r0:7.3f}+-{s0:4.3f}  {r1:7.3f}+-{s1:4.3f}  {u0:9.3f}")
    print("\nGATE 2a: RewardMF_RLS @ p=1 should BEAT Tabular (method fixed).")
    print("GATE 2b: DualMF_RLS @ p=0 should stay high (choices recover U).")

    print("\n--- PHASE TRANSITION reward/step vs p ---")
    names = ["Tabular", "RewardMF_RLS", "DualMF_RLS"]
    print(f"{'p':>6s}  " + "  ".join(f"{nm:>13s}" for nm in names))
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        row = f"{p:6.2f}  "
        for nm in names:
            Cls, hp = policies[nm]; r, _, _ = avg_seeds(Cls, hp, T, p, seeds, wp)
            row += f"{r:13.3f}  "
        print(row)


if __name__ == "__main__":
    main()
