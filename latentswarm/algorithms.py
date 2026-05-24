"""Decentralized policy library for ZK-MRTA. Every policy is communication-free: robot i learns
only from the partial, private broadcast it observes (its own engagements plus visible teammates).
Each policy manages all m robots internally (one independent per-robot estimator each), exposing:

    act(obs)          -> int[m]   action per robot (task index or ZKMRTAEnv.NO_OP)
    observe(obs)      -> None      update from the post-step observation (the broadcast)
    predict_rows()    -> [m,n] | None   each robot's predicted reward row (for unseen-pair skill)

Register new policies with @algorithm(name).
"""
import numpy as np

from .registry import algorithm
from .env import ZKMRTAEnv

NO_OP = ZKMRTAEnv.NO_OP


class Policy:
    name = "base"

    def __init__(self, cfg, m, n, d_guess, seed=0):
        self.cfg, self.m, self.n, self.d = cfg, m, n, d_guess
        self.rng = np.random.RandomState(seed)
        self.t = 0

    def _offered(self, obs_i):
        return np.where(obs_i["offer"])[0]

    def act(self, obs):
        raise NotImplementedError

    def observe(self, obs):
        pass

    def predict_rows(self):
        return None


@algorithm("random")
class RandomPolicy(Policy):
    name = "random"

    def act(self, obs):
        a = np.full(self.m, NO_OP, dtype=int)
        for i in range(self.m):
            off = self._offered(obs[i])
            if off.size:
                a[i] = int(self.rng.choice(off))
        return a


@algorithm("ucb_indep")
class UCBIndep(Policy):
    """Structure-free per-(robot,task) UCB1: robot i estimates each task only from its OWN
    engagements, so it cannot generalize to tasks it never engaged (predict_rows is None)."""
    name = "ucb_indep"

    def __init__(self, cfg, m, n, d_guess, seed=0):
        super().__init__(cfg, m, n, d_guess, seed)
        self.count = np.zeros((m, n))
        self.sum = np.zeros((m, n))

    def act(self, obs):
        self.t += 1
        a = np.full(self.m, NO_OP, dtype=int)
        for i in range(self.m):
            off = self._offered(obs[i])
            if not off.size:
                continue
            cnt = self.count[i, off]
            mean = np.where(cnt > 0, self.sum[i, off] / np.maximum(cnt, 1), 0.0)
            bonus = np.where(cnt > 0, self.cfg.ucb_c * np.sqrt(np.log(self.t + 1) / np.maximum(cnt, 1)), np.inf)
            a[i] = int(off[int(np.argmax(mean + bonus))])
        return a

    def observe(self, obs):
        for i in range(self.m):
            j = obs[i]["sel"][i]
            if j != NO_OP:
                self.count[i, j] += 1
                self.sum[i, j] += obs[i]["rew"][i]


class _LowRankPolicy(Policy):
    """Shared scaffolding for per-robot low-rank estimators with eps-greedy action."""

    def __init__(self, cfg, m, n, d_guess, seed=0):
        super().__init__(cfg, m, n, d_guess, seed)
        self.eps = cfg.epsilon
        s0 = cfg.factor_init_scale
        self.P = [self.rng.normal(0, s0, (m, self.d)) for _ in range(m)]   # per-robot robot factors
        self.U = [self.rng.normal(0, s0, (n, self.d)) for _ in range(m)]   # per-robot task factors

    def act(self, obs):
        a = np.full(self.m, NO_OP, dtype=int)
        for i in range(self.m):
            off = self._offered(obs[i])
            if not off.size:
                continue
            if self.rng.random() < self.eps:
                a[i] = int(self.rng.choice(off))
            else:
                scores = self.P[i][i] @ self.U[i][off].T
                a[i] = int(off[int(np.argmax(scores))])
        self.eps = max(self.cfg.epsilon_min, self.eps * self.cfg.epsilon_decay)
        return a

    def predict_rows(self):
        return np.stack([self.P[i][i] @ self.U[i].T for i in range(self.m)])

    def per_robot_full(self):
        """Each robot's FULL predicted reward matrix R_hat^(i) = P_i @ U_i^T (frame-invariant), as a
        list of m arrays [m, n]. Used by the state_uniqueness metric (mirrors run_masked's uniq, which
        compares whole predicted matrices, not the trivially-different per-robot own rows)."""
        return [self.P[i] @ self.U[i].T for i in range(self.m)]


@algorithm("mf_sgd")
class MFSGD(_LowRankPolicy):
    """Per-robot SGD matrix factorization on the observed broadcast (the MF-SGD baseline)."""
    name = "mf_sgd"

    def observe(self, obs):
        lr, lam = self.cfg.mf_lr, self.cfg.mf_ridge
        for i in range(self.m):
            sel, rew = obs[i]["sel"], obs[i]["rew"]
            Pi, Ui = self.P[i], self.U[i]
            for k in range(self.m):
                j = sel[k]
                if j == NO_OP:
                    continue
                err = float(Pi[k] @ Ui[j] - rew[k])
                pk = Pi[k].copy()
                Pi[k] -= lr * (err * Ui[j] + lam * Pi[k])
                Ui[j] -= lr * (err * pk + lam * Ui[j])


@algorithm("swarm_cf")
class SwarmCF(_LowRankPolicy):
    """Per-robot online weighted ridge ALS over the observed broadcast (our method)."""
    name = "swarm_cf"

    def __init__(self, cfg, m, n, d_guess, seed=0):
        super().__init__(cfg, m, n, d_guess, seed)
        self.buf = [([], [], []) for _ in range(m)]   # per robot: (engager idx, task idx, reward)
        self.window = cfg.buffer_window

    def observe(self, obs):
        for i in range(self.m):
            sel, rew = obs[i]["sel"], obs[i]["rew"]
            K, J, V = self.buf[i]
            for k in range(self.m):
                j = sel[k]
                if j != NO_OP:
                    K.append(k); J.append(int(j)); V.append(float(rew[k]))
            if len(K) > self.window:
                self.buf[i] = (K[-self.window:], J[-self.window:], V[-self.window:])
        self.t += 1
        if self.t % self.cfg.refit_every == 0:
            for i in range(self.m):
                self._als(i)

    def _als(self, i):
        K, J, V = self.buf[i]
        if not K:
            return
        K = np.asarray(K); J = np.asarray(J); V = np.asarray(V, dtype=float)
        d = self.d
        Imat = self.cfg.ridge * np.eye(d)
        P, U = self.P[i], self.U[i]
        for _ in range(self.cfg.als_sweeps):
            A = np.tile(Imat, (self.m, 1, 1)); b = np.zeros((self.m, d))
            Uj = U[J]
            np.add.at(A, K, np.einsum('ni,nj->nij', Uj, Uj))
            np.add.at(b, K, V[:, None] * Uj)
            P = np.linalg.solve(A, b[..., None])[..., 0]
            A2 = np.tile(Imat, (self.n, 1, 1)); b2 = np.zeros((self.n, d))
            Pk = P[K]
            np.add.at(A2, J, np.einsum('ni,nj->nij', Pk, Pk))
            np.add.at(b2, J, V[:, None] * Pk)
            U = np.linalg.solve(A2, b2[..., None])[..., 0]
        self.P[i], self.U[i] = P, U


@algorithm("swarm_cf_r")
class SwarmCFRand(SwarmCF):
    """SwarmCF with RANDOMIZED near-best selection (communication-free de-confliction). Identical
    estimator, learning, and eps-greedy exploration as SwarmCF; the only change is the exploit step:
    instead of strict argmax it picks UNIFORMLY among the offered tasks whose predicted score is
    within `near_best_margin` of the best. This is NOT exploration (the candidate set is the
    near-OPTIMAL tasks only, never a uniformly random task): it is a stochastic tie-break that spreads
    robots with similar models across near-equivalent tasks, reducing capacity-1 collisions with no
    communication. With a small margin it reduces to argmax when one task is clearly best, so it costs
    almost nothing where there is no near-tie / no contention."""
    name = "swarm_cf_r"

    def act(self, obs):
        a = np.full(self.m, NO_OP, dtype=int)
        for i in range(self.m):
            off = self._offered(obs[i])
            if not off.size:
                continue
            if self.rng.random() < self.eps:
                a[i] = int(self.rng.choice(off))                      # exploration (same as SwarmCF)
            else:
                scores = self.P[i][i] @ self.U[i][off].T
                near = off[scores >= scores.max() - self.cfg.near_best_margin]   # near-best set
                a[i] = int(self.rng.choice(near))                     # randomized greedy (de-conflict)
        self.eps = max(self.cfg.epsilon_min, self.eps * self.cfg.epsilon_decay)
        return a
