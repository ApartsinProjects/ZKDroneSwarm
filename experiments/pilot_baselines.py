"""
Competitor baselines ported to the pilot harness (select(t,cand) / observe(t,
choices, revealed, cand_sets, rvar) / predict_scores() / pulled_mask()), so they
run in the same block-model + masking setting as our methods.

NO-STRUCTURE (tabular):
  UCBIndep -- per-(drone,target) UCB1 (strong independent baseline; no
              generalization across arms -> floor on unseen).
  UCBHomo  -- single shared target arm table (assumes homogeneity; pools across
              drones -> confounded on heterogeneous worlds; generalizes but WRONG).
LOW-RANK:
  MFSGD    -- SGD matrix factorization (MF-CF).
  ESTR     -- Explore-Then-Spectral-Refit (centralized low-rank bandit: random
              explore, SVD of R_hat, then exploit). Kang-Hsieh-Lee 2022 style.
Our methods (RewardCF online-ALS, BothCF) live in pilot_noise.
"""
import numpy as np


class _Base:
    def __init__(self, m, n, d, idx, seed, **hp):
        self.m = m; self.n = n; self.d = d; self.idx = idx
        self.rng = np.random.RandomState(seed)
        self.pulled = np.zeros(n, bool)
    def pulled_mask(self):
        return self.pulled


class Random(_Base):
    """Uniform-random selection; learns nothing. Sanity floor: skill ~= 0,
    unseen skill ~= 0 (predict_scores is noise -> argmax is a random offer)."""
    def select(self, t, cand):
        a = int(cand[self.rng.randint(len(cand))]); self.pulled[a] = True; return a
    def observe(self, t, choices, revealed, cand_sets, rvar):
        pass
    def predict_scores(self):
        return self.rng.rand(self.n)


class UCBIndep(_Base):
    """Per-(drone,target) UCB1; updates all arms from the observed broadcast,
    selects on its OWN row. No cross-arm generalization -> floor on unseen."""
    def __init__(self, *a, c=2.0, **hp):
        super().__init__(*a, **hp)
        self.c = c
        self.counts = np.zeros((self.m, self.n))
        self.sums = np.zeros((self.m, self.n))
        self.tot = np.zeros(self.m)

    def select(self, t, cand):
        i = self.idx; cnt = self.counts[i]; tot = max(self.tot[i], 1)
        sc = np.full(len(cand), -np.inf)
        for q, j in enumerate(cand):
            if cnt[j] == 0:
                sc[q] = np.inf
            else:
                sc[q] = self.sums[i, j] / cnt[j] + self.c * np.sqrt(np.log(tot) / cnt[j])
        mx = sc.max()
        if np.isinf(mx):
            a = int(self.rng.choice([cand[q] for q in range(len(cand)) if np.isinf(sc[q])]))
        else:
            a = int(cand[int(np.argmax(sc))])
        self.pulled[a] = True; return a

    def observe(self, t, choices, revealed, cand_sets, rvar):
        for k in range(self.m):
            r = revealed[k]
            if not np.isnan(r):
                j = int(choices[k]); self.counts[k, j] += 1; self.sums[k, j] += r; self.tot[k] += 1

    def predict_scores(self):
        i = self.idx; c = self.counts[i]
        return np.where(c > 0, self.sums[i] / np.maximum(c, 1), 0.0)   # 0 default -> floor on unseen


class UCBHomo(_Base):
    """Single shared target arm table (assumes all drones identical)."""
    def __init__(self, *a, c=2.0, **hp):
        super().__init__(*a, **hp)
        self.c = c; self.counts = np.zeros(self.n); self.sums = np.zeros(self.n); self.tot = 0

    def select(self, t, cand):
        tot = max(self.tot, 1)
        sc = np.full(len(cand), -np.inf)
        for q, j in enumerate(cand):
            if self.counts[j] == 0:
                sc[q] = np.inf
            else:
                sc[q] = self.sums[j] / self.counts[j] + self.c * np.sqrt(np.log(tot) / self.counts[j])
        mx = sc.max()
        if np.isinf(mx):
            a = int(self.rng.choice([cand[q] for q in range(len(cand)) if np.isinf(sc[q])]))
        else:
            a = int(cand[int(np.argmax(sc))])
        self.pulled[a] = True; return a

    def observe(self, t, choices, revealed, cand_sets, rvar):
        for k in range(self.m):
            r = revealed[k]
            if not np.isnan(r):
                j = int(choices[k]); self.counts[j] += 1; self.sums[j] += r; self.tot += 1

    def predict_scores(self):
        return np.where(self.counts > 0, self.sums / np.maximum(self.counts, 1), 0.0)


class MFSGD(_Base):
    """SGD matrix factorization (MF-CF). Low-rank; updates from observed rewards."""
    def __init__(self, *a, lr=0.2, reg=1e-3, eps=0.15, init_scale=0.3, **hp):
        super().__init__(*a, **hp)
        self.lr = lr; self.reg = reg; self.eps = eps
        self.P = self.rng.normal(0, init_scale, (self.m, self.d))
        self.U = self.rng.normal(0, init_scale, (self.n, self.d))

    def select(self, t, cand):
        if self.rng.rand() < self.eps:
            a = int(cand[self.rng.randint(len(cand))])
        else:
            a = int(cand[int(np.argmax(self.U[cand] @ self.P[self.idx]))])
        self.pulled[a] = True; return a

    def observe(self, t, choices, revealed, cand_sets, rvar):
        for k in range(self.m):
            r = revealed[k]
            if not np.isnan(r):
                j = int(choices[k]); err = self.P[k] @ self.U[j] - float(r)
                gP = err * self.U[j] + self.reg * self.P[k]
                gU = err * self.P[k] + self.reg * self.U[j]
                self.P[k] -= self.lr * gP; self.U[j] -= self.lr * gU

    def predict_scores(self):
        return self.U @ self.P[self.idx]


class ESTR(_Base):
    """Explore-Then-Spectral-Refit (centralized low-rank bandit): random explore,
    then SVD of the empirical R_hat (rank d) from the observed broadcast, then
    exploit. Generalizes to unseen via low-rank completion (like CF) but is
    explore-then-COMMIT (wastes the explore phase, no online adaptation)."""
    def __init__(self, *a, T_total=50, explore_frac=0.4, eps_floor=0.05, **hp):
        super().__init__(*a, **hp)
        self.explore_steps = int(explore_frac * T_total); self.eps_floor = eps_floor
        self.sum = np.zeros((self.m, self.n)); self.cnt = np.zeros((self.m, self.n))
        self.P_hat = None; self.U_hat = None; self.step = 0; self.trans = False

    def select(self, t, cand):
        if self.step < self.explore_steps or not self.trans or self.rng.rand() < self.eps_floor:
            a = int(cand[self.rng.randint(len(cand))])
        else:
            a = int(cand[int(np.argmax(self.U_hat[cand] @ self.P_hat[self.idx]))])
        self.pulled[a] = True; return a

    def observe(self, t, choices, revealed, cand_sets, rvar):
        for k in range(self.m):
            r = revealed[k]
            if not np.isnan(r):
                j = int(choices[k]); self.sum[k, j] += r; self.cnt[k, j] += 1
        self.step += 1
        if (not self.trans) and self.step >= self.explore_steps:
            self._svd(); self.trans = True

    def _svd(self):
        R_hat = np.where(self.cnt > 0, self.sum / np.maximum(self.cnt, 1), 0.0)
        try:
            Us, S, Vt = np.linalg.svd(R_hat, full_matrices=False)
            r = min(self.d, len(S)); sq = np.sqrt(np.maximum(S[:r], 0))
            self.P_hat = Us[:, :r] * sq[None, :]
            self.U_hat = (Vt[:r, :].T * sq[None, :])
            if r < self.d:
                self.P_hat = np.hstack([self.P_hat, np.zeros((self.m, self.d - r))])
                self.U_hat = np.hstack([self.U_hat, np.zeros((self.n, self.d - r))])
        except np.linalg.LinAlgError:
            self.P_hat = np.zeros((self.m, self.d)); self.U_hat = np.zeros((self.n, self.d))

    def predict_scores(self):
        if self.U_hat is None:
            return np.zeros(self.n)
        return self.U_hat @ self.P_hat[self.idx]


class PTF(_Base):
    """Probe-Then-Fit hybrid: probe with per-(drone,target) UCB to fill an
    empirical R_hat, SVD warm-start rank-d factors, then ONLINE SGD-MF fine-tune.
    Stronger low-rank baseline than MFSGD (warm start) and than ESTR (UCB-targeted
    probe + online adaptation rather than uniform-explore-then-COMMIT). NOTE: UCB
    probing exploits high-reward arms, so R_hat coverage is BIASED vs ESTR's
    uniform random probe -- a genuine matrix-completion trade-off."""
    def __init__(self, *a, T_total=50, probe_frac=0.4, c=2.0, lr=0.2, reg=1e-3,
                 eps=0.15, init_scale=0.3, **hp):
        super().__init__(*a, **hp)
        self.probe_steps = int(probe_frac * T_total); self.c = c
        self.lr = lr; self.reg = reg; self.eps = eps; self.init_scale = init_scale
        self.counts = np.zeros((self.m, self.n)); self.sums = np.zeros((self.m, self.n))
        self.tot = np.zeros(self.m); self.step = 0; self.trans = False
        self.P = None; self.U = None

    def select(self, t, cand):
        if not self.trans:                       # PROBE phase: own-row UCB1
            i = self.idx; cnt = self.counts[i]; tot = max(self.tot[i], 1)
            sc = np.full(len(cand), -np.inf)
            for q, j in enumerate(cand):
                if cnt[j] == 0:
                    sc[q] = np.inf
                else:
                    sc[q] = self.sums[i, j] / cnt[j] + self.c * np.sqrt(np.log(tot) / cnt[j])
            if np.isinf(sc.max()):
                a = int(self.rng.choice([cand[q] for q in range(len(cand)) if np.isinf(sc[q])]))
            else:
                a = int(cand[int(np.argmax(sc))])
        else:                                    # FIT phase: eps-greedy on low-rank
            if self.rng.rand() < self.eps:
                a = int(cand[self.rng.randint(len(cand))])
            else:
                a = int(cand[int(np.argmax(self.U[cand] @ self.P[self.idx]))])
        self.pulled[a] = True; return a

    def observe(self, t, choices, revealed, cand_sets, rvar):
        for k in range(self.m):
            r = revealed[k]
            if not np.isnan(r):
                j = int(choices[k]); self.counts[k, j] += 1; self.sums[k, j] += r; self.tot[k] += 1
        self.step += 1
        if (not self.trans) and self.step >= self.probe_steps:
            self._warm(); self.trans = True
        elif self.trans:                          # online SGD fine-tune
            for k in range(self.m):
                r = revealed[k]
                if not np.isnan(r):
                    j = int(choices[k]); rj = float(np.clip(r, -1.0, 1.0))   # rewards live in [-1,1]
                    err = float(np.clip(self.P[k] @ self.U[j] - rj, -5.0, 5.0))
                    gP = err * self.U[j] + self.reg * self.P[k]
                    gU = err * self.P[k] + self.reg * self.U[j]
                    self.P[k] -= self.lr * gP; self.U[j] -= self.lr * gU

    def _warm(self):
        R_hat = np.where(self.counts > 0, self.sums / np.maximum(self.counts, 1), 0.0)
        R_hat = np.clip(R_hat, -1.0, 1.0)            # true means in [-1,1]; clip noise -> stable SVD
        try:
            Us, S, Vt = np.linalg.svd(R_hat, full_matrices=False)
            r = min(self.d, len(S)); sq = np.sqrt(np.maximum(S[:r], 0))
            self.P = Us[:, :r] * sq[None, :]; self.U = (Vt[:r, :].T * sq[None, :])
            if r < self.d:
                self.P = np.hstack([self.P, self.rng.normal(0, 0.01, (self.m, self.d - r))])
                self.U = np.hstack([self.U, self.rng.normal(0, 0.01, (self.n, self.d - r))])
        except np.linalg.LinAlgError:
            self.P = self.rng.normal(0, self.init_scale, (self.m, self.d))
            self.U = self.rng.normal(0, self.init_scale, (self.n, self.d))

    def predict_scores(self):
        if (not self.trans) or self.U is None:
            i = self.idx; c = self.counts[i]
            return np.where(c > 0, self.sums[i] / np.maximum(c, 1), 0.0)   # floor on unseen
        return self.U @ self.P[self.idx]


class BPMF(_Base):
    """Bayesian Probabilistic Matrix Factorization (Salakhutdinov-Mnih 2008 style):
    per-drone private posterior over rank-d factors P (m,d) and U (n,d), in natural
    parameters (precision Lambda + precision-weighted mean eta). Closed-form
    conjugate (MAP plug-in) updates that consume the per-observation noise rvar as
    the likelihood variance (principled, like our RewardCF). Thompson-sampling
    selection: precision shrinks the posterior over time -> exploration without
    epsilon. Generalizes to unseen pairs via the low-rank posterior MEAN."""
    def __init__(self, *a, prior_var=1.0, init_scale=0.1, **hp):
        super().__init__(*a, **hp)
        d = self.d
        self.LP = np.tile(np.eye(d) / prior_var, (self.m, 1, 1)).astype(float)
        self.LU = np.tile(np.eye(d) / prior_var, (self.n, 1, 1)).astype(float)
        ir = np.random.RandomState((self.idx + 1) * 7919 + 13)
        muP0 = ir.normal(0, init_scale, (self.m, d)); muU0 = ir.normal(0, init_scale, (self.n, d))
        self.eP = np.einsum('idk,ik->id', self.LP, muP0)
        self.eU = np.einsum('jdk,jk->jd', self.LU, muU0)

    def select(self, t, cand):
        d = self.d; i = self.idx
        muPi = np.linalg.solve(self.LP[i], self.eP[i]); SPi = np.linalg.inv(self.LP[i])
        try:
            p_t = muPi + np.linalg.cholesky(SPi + 1e-9 * np.eye(d)) @ self.rng.randn(d)
        except np.linalg.LinAlgError:
            p_t = muPi + np.sqrt(np.abs(np.diag(SPi))) * self.rng.randn(d)
        Lc = self.LU[cand]; ec = self.eU[cand]
        muUc = np.linalg.solve(Lc, ec[..., None])[..., 0]; SUc = np.linalg.inv(Lc)
        z = self.rng.randn(len(cand), d)
        try:
            Lch = np.linalg.cholesky(SUc + 1e-9 * np.eye(d)[None, :, :])
            u_t = muUc + np.einsum('qde,qe->qd', Lch, z)
        except np.linalg.LinAlgError:
            u_t = muUc + np.sqrt(np.abs(np.diagonal(SUc, axis1=1, axis2=2))) * z
        a = int(cand[int(np.argmax(u_t @ p_t))])
        self.pulled[a] = True; return a

    def observe(self, t, choices, revealed, cand_sets, rvar):
        muP = np.linalg.solve(self.LP, self.eP[..., None])[..., 0]   # (m,d) pre-update snapshot
        muU = np.linalg.solve(self.LU, self.eU[..., None])[..., 0]   # (n,d)
        for k in range(self.m):
            r = revealed[k]
            if np.isnan(r):
                continue
            j = int(choices[k]); s2 = max(rvar[k], 1e-6)
            mk = muP[k]; uj = muU[j]
            self.LP[k] += np.outer(uj, uj) / s2; self.eP[k] += (r * uj) / s2
            self.LU[j] += np.outer(mk, mk) / s2; self.eU[j] += (r * mk) / s2

    def predict_scores(self):
        muP = np.linalg.solve(self.LP[self.idx], self.eP[self.idx])
        muU = np.linalg.solve(self.LU, self.eU[..., None])[..., 0]
        return muU @ muP
