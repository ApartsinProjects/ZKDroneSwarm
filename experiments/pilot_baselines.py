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
