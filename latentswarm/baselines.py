"""Competitor baselines as @algorithm drop-ins for the LatentSwarm Policy interface.

These are faithful ports of the analytical-harness baselines (experiments/pilot_baselines.py and
experiments/pilot_noise.py), re-expressed in the package's decentralized Policy contract:

    act(obs)        -> int[m]          action per robot (a task index in its offer, or NO_OP)
    observe(obs)    -> None            update from the post-step broadcast
    predict_rows()  -> [m, n] | None   each robot's predicted reward row (None => structure-free)

Each policy manages all m robots internally (one independent per-robot estimator), exactly as the
native algorithms.py policies do. The pilot harness fed each per-drone learner a `(choices,
revealed, rvar)` triple; here that triple is reconstructed per robot i from its own observation
dict obs[i]: a teammate k is "observed" iff obs[i]["sel"][k] != NO_OP, the reading is
obs[i]["rew"][k], and the per-observer variance is sigma_own^2 for k==i else sigma_obs^2 (the
persistent / per-round masking case the parity check and the headline sweeps use; under
line_of_sight the env's distance-dependent noise is approximated by sigma_obs^2 for the
precision-weighted methods).

Ported policies:
  tabular        own-row eps-greedy empirical means (no cross-agent transfer); predict_rows None.
  ucb_homo       single shared arm table (assumes homogeneity); predict_rows None.
  estr           explore-then-spectral-commit (uniform explore, one SVD of R_hat, then exploit).
  swarmcf_batch  = PTF: probe (own-row UCB) then SVD warm-start + online SGD-MF finetune.
  bpmf           Bayesian PMF (per-robot conjugate posterior, Thompson-sampled selection).

The low-rank ones (estr, swarmcf_batch, bpmf) return a completed [m, n] from predict_rows; the
structure-free ones (tabular, ucb_homo) return None (the unseen-pair floor by construction).
"""
import numpy as np

from .registry import algorithm
from .algorithms import Policy, _LowRankPolicy, NO_OP


def _rvar_row(cfg, i, m):
    """Per-observer reward variance for robot i's view of each teammate k (own = sigma_own^2,
    others = sigma_obs^2). Floored away from 0 so 1/var is finite (matches the pilots' max(., 1e-6))."""
    v = np.full(m, max(cfg.sigma_obs ** 2, 1e-6))
    v[i] = max(cfg.sigma_own ** 2, 1e-6)
    return v


@algorithm("tabular")
class Tabular(Policy):
    """Own-reward-only empirical means with an eps-greedy (decaying) policy. The cross-agent
    channel is IGNORED: robot i learns each task purely from its OWN engagements, so it has zero
    information on tasks it never engaged (predict_rows -> None, the unseen-pair floor). Mirrors
    pilot_noise.Tabular."""
    name = "tabular"

    def __init__(self, cfg, m, n, d_guess, seed=0):
        super().__init__(cfg, m, n, d_guess, seed)
        self.eps = cfg.epsilon
        self.counts = np.zeros((m, n))
        self.sums = np.zeros((m, n))
        self.tot_n = np.zeros(m)
        self.tot_s = np.zeros(m)

    def _est(self, i):
        default = (self.tot_s[i] / self.tot_n[i]) if self.tot_n[i] > 0 else 0.5
        e = np.full(self.n, default)
        seen = self.counts[i] > 0
        e[seen] = self.sums[i][seen] / self.counts[i][seen]
        return e

    def act(self, obs):
        a = np.full(self.m, NO_OP, dtype=int)
        for i in range(self.m):
            off = self._offered(obs[i])
            if not off.size:
                continue
            if self.rng.random() < self.eps:
                a[i] = int(self.rng.choice(off))
            else:
                a[i] = int(off[int(np.argmax(self._est(i)[off]))])
        self.eps = max(self.cfg.epsilon_min, self.eps * self.cfg.epsilon_decay)
        return a

    def observe(self, obs):
        for i in range(self.m):
            j = obs[i]["sel"][i]                     # OWN choice only
            if j != NO_OP:
                r = float(obs[i]["rew"][i])
                self.counts[i, j] += 1; self.sums[i, j] += r
                self.tot_n[i] += 1; self.tot_s[i] += r

    def predict_rows(self):
        return None                                 # structure-free: unseen floor


@algorithm("ucb_homo")
class UCBHomo(Policy):
    """Single shared per-task arm table pooled across ALL robots (assumes homogeneity). It
    generalizes across robots but is WRONG on a heterogeneous world (it cannot personalize); it
    has no per-(robot,task) low-rank model, so predict_rows is None. Mirrors pilot_baselines.UCBHomo
    (one table; here each robot reads it via its own observed broadcast)."""
    name = "ucb_homo"

    def __init__(self, cfg, m, n, d_guess, seed=0):
        super().__init__(cfg, m, n, d_guess, seed)
        self.counts = np.zeros((m, n))              # per-robot view of the shared table
        self.sums = np.zeros((m, n))
        self.tot = np.zeros(m)

    def act(self, obs):
        a = np.full(self.m, NO_OP, dtype=int)
        for i in range(self.m):
            off = self._offered(obs[i])
            if not off.size:
                continue
            cnt = self.counts[i, off]
            tot = max(self.tot[i], 1)
            mean = np.where(cnt > 0, self.sums[i, off] / np.maximum(cnt, 1), 0.0)
            bonus = np.where(cnt > 0, self.cfg.ucb_c * np.sqrt(np.log(tot) / np.maximum(cnt, 1)), np.inf)
            sc = mean + bonus
            if np.isinf(sc.max()):
                a[i] = int(self.rng.choice(off[np.isinf(sc)]))
            else:
                a[i] = int(off[int(np.argmax(sc))])
        return a

    def observe(self, obs):
        for i in range(self.m):
            sel, rew = obs[i]["sel"], obs[i]["rew"]
            for k in range(self.m):
                j = sel[k]
                if j != NO_OP:                       # pool every visible engagement into i's table
                    self.counts[i, j] += 1; self.sums[i, j] += float(rew[k]); self.tot[i] += 1

    def predict_rows(self):
        return None


@algorithm("estr")
class ESTR(_LowRankPolicy):
    """Explore-Then-Spectral-Refit (centralized low-rank bandit, Kang-Hsieh-Lee 2022 style): each
    robot explores uniformly for an explore-fraction of the horizon, builds an empirical R_hat from
    its observed broadcast (unobserved entries imputed 0), takes ONE rank-d SVD, then exploits. It
    generalizes to unseen pairs via low-rank completion but is explore-then-COMMIT (no online
    adaptation; wastes the probe). Mirrors pilot_baselines.ESTR."""
    name = "estr"

    def __init__(self, cfg, m, n, d_guess, seed=0):
        super().__init__(cfg, m, n, d_guess, seed)
        self.explore_steps = int(cfg.estr_explore_frac * cfg.T)
        self.eps_floor = cfg.epsilon_min
        self.sum = np.zeros((m, n)); self.cnt = np.zeros((m, n))
        self.trans = False

    def act(self, obs):
        a = np.full(self.m, NO_OP, dtype=int)
        for i in range(self.m):
            off = self._offered(obs[i])
            if not off.size:
                continue
            if self.t < self.explore_steps or not self.trans or self.rng.random() < self.eps_floor:
                a[i] = int(self.rng.choice(off))
            else:
                scores = self.P[i][i] @ self.U[i][off].T
                a[i] = int(off[int(np.argmax(scores))])
        return a

    def observe(self, obs):
        for i in range(self.m):
            sel, rew = obs[i]["sel"], obs[i]["rew"]
            for k in range(self.m):
                j = sel[k]
                if j != NO_OP:
                    self.sum[i, j] += float(rew[k]); self.cnt[i, j] += 1
        self.t += 1
        if (not self.trans) and self.t >= self.explore_steps:
            for i in range(self.m):
                self._svd(i)
            self.trans = True

    def _svd(self, i):
        R_hat = np.where(self.cnt[i] > 0, self.sum[i] / np.maximum(self.cnt[i], 1), 0.0)
        d = self.d
        try:
            Us, S, Vt = np.linalg.svd(R_hat, full_matrices=False)
            r = min(d, len(S)); sq = np.sqrt(np.maximum(S[:r], 0))
            P = Us[:, :r] * sq[None, :]
            U = (Vt[:r, :].T * sq[None, :])
            if r < d:
                P = np.hstack([P, np.zeros((self.m, d - r))])
                U = np.hstack([U, np.zeros((self.n, d - r))])
            self.P[i], self.U[i] = P, U
        except np.linalg.LinAlgError:
            self.P[i] = np.zeros((self.m, d)); self.U[i] = np.zeros((self.n, d))


@algorithm("swarmcf_batch")
class SwarmCFBatch(_LowRankPolicy):
    """SwarmCF-batch (= PTF, Probe-Then-Fit): probe with own-row UCB1 to fill an empirical R_hat,
    SVD warm-start rank-d factors, then ONLINE SGD-MF finetune. A stronger low-rank baseline than
    plain MF (warm start) and than ESTR (targeted probe + online adaptation rather than
    uniform-explore-then-commit). Mirrors pilot_baselines.PTF; rewards live in [-1, 1] so R_hat and
    the SGD step are clipped. Returns a completed [m, n] (predict_rows)."""
    name = "swarmcf_batch"

    def __init__(self, cfg, m, n, d_guess, seed=0):
        super().__init__(cfg, m, n, d_guess, seed)
        self.probe_steps = int(cfg.ptf_probe_frac * cfg.T)
        self.counts = np.zeros((m, n)); self.sums = np.zeros((m, n)); self.tot = np.zeros(m)
        self.trans = False

    def act(self, obs):
        a = np.full(self.m, NO_OP, dtype=int)
        for i in range(self.m):
            off = self._offered(obs[i])
            if not off.size:
                continue
            if not self.trans:                       # PROBE: own-row UCB1
                cnt = self.counts[i, off]; tot = max(self.tot[i], 1)
                mean = np.where(cnt > 0, self.sums[i, off] / np.maximum(cnt, 1), 0.0)
                bonus = np.where(cnt > 0, self.cfg.ucb_c * np.sqrt(np.log(tot) / np.maximum(cnt, 1)), np.inf)
                sc = mean + bonus
                if np.isinf(sc.max()):
                    a[i] = int(self.rng.choice(off[np.isinf(sc)]))
                else:
                    a[i] = int(off[int(np.argmax(sc))])
            else:                                    # FIT: eps-greedy on the low-rank model
                if self.rng.random() < self.eps:
                    a[i] = int(self.rng.choice(off))
                else:
                    scores = self.P[i][i] @ self.U[i][off].T
                    a[i] = int(off[int(np.argmax(scores))])
        if self.trans:
            self.eps = max(self.cfg.epsilon_min, self.eps * self.cfg.epsilon_decay)
        return a

    def observe(self, obs):
        for i in range(self.m):
            sel, rew = obs[i]["sel"], obs[i]["rew"]
            for k in range(self.m):
                j = sel[k]
                if j != NO_OP:
                    self.counts[i, j] += 1; self.sums[i, j] += float(rew[k]); self.tot[i] += 1
        self.t += 1
        if (not self.trans) and self.t >= self.probe_steps:
            for i in range(self.m):
                self._warm(i)
            self.trans = True
        elif self.trans:
            lr, reg = self.cfg.mf_lr, self.cfg.mf_ridge
            for i in range(self.m):
                sel, rew = obs[i]["sel"], obs[i]["rew"]
                Pi, Ui = self.P[i], self.U[i]
                for k in range(self.m):
                    j = sel[k]
                    if j == NO_OP:
                        continue
                    rj = float(np.clip(rew[k], -1.0, 1.0))
                    err = float(np.clip(Pi[k] @ Ui[j] - rj, -5.0, 5.0))
                    pk = Pi[k].copy()
                    Pi[k] -= lr * (err * Ui[j] + reg * Pi[k])
                    Ui[j] -= lr * (err * pk + reg * Ui[j])

    def _warm(self, i):
        R_hat = np.where(self.counts[i] > 0, self.sums[i] / np.maximum(self.counts[i], 1), 0.0)
        R_hat = np.clip(R_hat, -1.0, 1.0)
        d = self.d
        try:
            Us, S, Vt = np.linalg.svd(R_hat, full_matrices=False)
            r = min(d, len(S)); sq = np.sqrt(np.maximum(S[:r], 0))
            P = Us[:, :r] * sq[None, :]; U = (Vt[:r, :].T * sq[None, :])
            if r < d:
                P = np.hstack([P, self.rng.normal(0, 0.01, (self.m, d - r))])
                U = np.hstack([U, self.rng.normal(0, 0.01, (self.n, d - r))])
            self.P[i], self.U[i] = P, U
        except np.linalg.LinAlgError:
            pass

    def predict_rows(self):
        # Before the warm-start the factors are noise; fall back to the own-row empirical means
        # (the floor on unseen), matching PTF.predict_scores's pre-transition branch.
        if not self.trans:
            out = np.where(self.counts > 0, self.sums / np.maximum(self.counts, 1), 0.0)
            return out
        return np.stack([self.P[i][i] @ self.U[i].T for i in range(self.m)])


@algorithm("bpmf")
class BPMF(Policy):
    """Bayesian Probabilistic Matrix Factorization (Salakhutdinov-Mnih 2008 style): each robot keeps
    a private conjugate posterior over rank-d factors P (m,d) and U (n,d), in natural parameters
    (precision Lambda + precision-weighted mean eta). Closed-form MAP plug-in updates consume the
    per-observation noise (1/sigma^2) as the likelihood precision; selection is Thompson-sampled
    (precision shrinks the posterior -> exploration without epsilon). Generalizes to unseen pairs via
    the posterior MEAN (predict_rows). Mirrors pilot_baselines.BPMF."""
    name = "bpmf"

    def __init__(self, cfg, m, n, d_guess, seed=0):
        super().__init__(cfg, m, n, d_guess, seed)
        d = self.d
        pv = cfg.bpmf_prior_var
        self.LP = np.tile(np.eye(d) / pv, (m, 1, 1)).astype(float)
        self.LU = np.tile(np.eye(d) / pv, (n, 1, 1)).astype(float)
        s0 = cfg.factor_init_scale
        muP0 = self.rng.normal(0, s0, (m, d)); muU0 = self.rng.normal(0, s0, (n, d))
        self.eP = np.einsum('idk,ik->id', self.LP, muP0)
        self.eU = np.einsum('jdk,jk->jd', self.LU, muU0)

    def act(self, obs):
        d = self.d
        a = np.full(self.m, NO_OP, dtype=int)
        for i in range(self.m):
            off = self._offered(obs[i])
            if not off.size:
                continue
            muPi = np.linalg.solve(self.LP[i], self.eP[i]); SPi = np.linalg.inv(self.LP[i])
            try:
                p_t = muPi + np.linalg.cholesky(SPi + 1e-9 * np.eye(d)) @ self.rng.standard_normal(d)
            except np.linalg.LinAlgError:
                p_t = muPi + np.sqrt(np.abs(np.diag(SPi))) * self.rng.standard_normal(d)
            Lc = self.LU[off]; ec = self.eU[off]
            muUc = np.linalg.solve(Lc, ec[..., None])[..., 0]; SUc = np.linalg.inv(Lc)
            z = self.rng.standard_normal((off.size, d))
            try:
                Lch = np.linalg.cholesky(SUc + 1e-9 * np.eye(d)[None, :, :])
                u_t = muUc + np.einsum('qde,qe->qd', Lch, z)
            except np.linalg.LinAlgError:
                u_t = muUc + np.sqrt(np.abs(np.diagonal(SUc, axis1=1, axis2=2))) * z
            a[i] = int(off[int(np.argmax(u_t @ p_t))])
        return a

    def observe(self, obs):
        for i in range(self.m):
            sel, rew = obs[i]["sel"], obs[i]["rew"]
            rvar = _rvar_row(self.cfg, i, self.m)
            muP = np.linalg.solve(self.LP, self.eP[..., None])[..., 0]   # pre-update snapshot
            muU = np.linalg.solve(self.LU, self.eU[..., None])[..., 0]
            for k in range(self.m):
                j = sel[k]
                if j == NO_OP:
                    continue
                r = float(rew[k]); s2 = max(rvar[k], 1e-6)
                mk = muP[k]; uj = muU[j]
                self.LP[k] += np.outer(uj, uj) / s2; self.eP[k] += (r * uj) / s2
                self.LU[j] += np.outer(mk, mk) / s2; self.eU[j] += (r * mk) / s2

    def predict_rows(self):
        muP = np.linalg.solve(self.LP, self.eP[..., None])[..., 0]   # (m,d)
        muU = np.linalg.solve(self.LU, self.eU[..., None])[..., 0]   # (n,d)
        return muP @ muU.T
