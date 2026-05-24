"""The ZK-MRTA environment: one robot x task mission under partial, private observation.

A hidden low-rank reward R = P U^T governs which robot suits which task. Each round every robot
is offered a menu (all tasks by default, or a random size-c subset), selects one, and engages it;
under capacity-1 contention only the first robot to pick a task that round succeeds. Robots do not
communicate; each only passively senses a partial, per-observer-private slice of the public
outcome stream, set by the observation mask (persistent blind spots, or per-round dynamic
line-of-sight).
"""
import numpy as np

from .config import RunConfig


def reward_matrix(P, U, model="inner_product"):
    """Full robot x task reward. inner_product: R_ij=<p_i,u_j>; cosine: normalized."""
    R = P @ U.T
    if model == "cosine":
        pn = np.linalg.norm(P, axis=1, keepdims=True)
        un = np.linalg.norm(U, axis=1, keepdims=True)
        R = R / np.maximum(pn, 1e-9) / np.maximum(un.T, 1e-9)
    return R


class ZKMRTAEnv:
    """One T-round mission. reset() -> per-robot observations; step(actions) -> (obs, rewards, info).

    Per-robot observation i is a dict:
      'offer' : bool[n]  tasks offered to robot i this round (all True by default)
      'sel'   : int[m]   each teammate's last selected task, or -1 if not visible / no-op
      'rew'   : float[m] each visible teammate's last outcome, read with private per-observer noise
      'i'     : int      the robot's own index
    Masked teammates appear as sel=-1 (the policies skip them). Robot i's own entry (k=i) is its
    own clean (or sigma_own-noisy) reading.
    """

    NO_OP = -1

    def __init__(self, cfg: RunConfig, P, U, d_guess: int, seed: int = 0):
        self.cfg = cfg
        self.P, self.U = P, U
        self.m, self.n = P.shape[0], U.shape[0]
        self.d_guess = d_guess
        self.R = reward_matrix(P, U, cfg.reward_model)
        self.rng = np.random.RandomState(seed)
        self.mask = None
        self.last_sel = None
        self.last_rew = None
        self.engaged = None
        self.t = 0

    def _build_mask(self):
        if self.cfg.rho >= 1.0:
            self.mask = np.ones((self.m, self.m), dtype=bool)
        else:
            self.mask = self.rng.random((self.m, self.m)) < self.cfg.rho
            np.fill_diagonal(self.mask, True)   # always observe self

    def _offers(self):
        c = self.cfg.offer_size
        if not c or c <= 0 or c >= self.n:
            return np.ones((self.m, self.n), dtype=bool)        # all tasks offered (default)
        offers = np.zeros((self.m, self.n), dtype=bool)
        for i in range(self.m):
            offers[i, self.rng.choice(self.n, size=c, replace=False)] = True
        return offers

    def reset(self):
        self.t = 0
        self.last_sel = np.full(self.m, self.NO_OP, dtype=int)
        self.last_rew = np.zeros(self.m)
        self.engaged = [set() for _ in range(self.m)]
        self._build_mask()
        return self._observe()

    def _observe(self):
        offers = self._offers()
        obs = []
        for i in range(self.m):
            sel = np.full(self.m, self.NO_OP, dtype=int)
            rew = np.zeros(self.m)
            for k in range(self.m):
                if not self.mask[i, k] or self.last_sel[k] == self.NO_OP:
                    continue
                sel[k] = self.last_sel[k]
                if k == i:
                    sig = self.cfg.sigma_own
                else:
                    sig = self.cfg.sigma_obs
                noise = self.rng.normal(0.0, sig) if sig > 0 else 0.0
                rew[k] = self.last_rew[k] + noise
            obs.append({"offer": offers[i], "sel": sel, "rew": rew, "i": i})
        return obs

    def step(self, actions):
        """actions: int[m] task index or NO_OP. Returns (obs, rewards[m], info)."""
        if self.cfg.mask_mode == "per_round":
            self._build_mask()
        rewards = np.zeros(self.m)
        taken = set()
        collisions = 0
        for i in self.rng.permutation(self.m):
            a = int(actions[i])
            if a == self.NO_OP:
                self.last_sel[i] = self.NO_OP
                self.last_rew[i] = 0.0
                continue
            self.engaged[i].add(a)
            if self.cfg.capacity_one and a in taken:
                collisions += 1
                self.last_sel[i] = a
                self.last_rew[i] = 0.0
                rewards[i] = 0.0
                continue
            taken.add(a)
            r = float(self.R[i, a])
            rewards[i] = r
            self.last_sel[i] = a
            self.last_rew[i] = r
        self.t += 1
        return self._observe(), rewards, {"t": self.t, "collisions": collisions}
