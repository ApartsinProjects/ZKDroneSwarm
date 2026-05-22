"""
Weighted-ALS collaborative-filtering policy for the ZK-MRTA env (our RewardCF in
the env's per-drone policy interface). Each drone keeps its own P (num_agents x d)
and U (d x num_targets), accumulates the public (drone, target, reward) broadcast
events (a bounded sliding window for tractability), and refits by ridge-regularized
ALS every few steps. Action = epsilon-greedy on P[idx] @ U[:, active].

This validates that our online weighted-ALS method ports cleanly to the realistic
simulator (spatial targets, HP depletion, episodic resets) and beats the env's
SGD matrix-factorization policy. ZK-compliant: only the public broadcast.
"""
from typing import Any, Dict, Optional
import numpy as np


class WeightedALSPolicy:
    is_deterministic: bool = False

    def __init__(self, num_targets: int, agent_idx: int, num_agents: int,
                 latent_dim: int = 8, lam: float = 1.0, als_sweeps: int = 8,
                 refit_every: int = 8, window: int = 6000,
                 epsilon: float = 0.3, epsilon_decay: float = 0.9995,
                 epsilon_min: float = 0.02, seed: Optional[int] = None):
        self.num_targets = num_targets; self.agent_idx = agent_idx
        self.num_agents = num_agents; self.latent_dim = latent_dim
        self.lam = lam; self.sweeps = als_sweeps; self.refit_every = refit_every
        self.window = window
        self.epsilon = epsilon; self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay; self.epsilon_min = epsilon_min
        self.seed = seed; self.rng = np.random.RandomState(seed)
        self.P = self.rng.normal(0, 0.1, (num_agents, latent_dim))
        self.U = self.rng.normal(0, 0.1, (latent_dim, num_targets))
        self.dk = []; self.tj = []; self.rv = []; self.step_count = 0

    def predict_reward(self, t: int) -> float:
        return float(self.P[self.agent_idx] @ self.U[:, t])

    def select_action(self, observation: Dict[str, Any], allow_noop: bool = False) -> int:
        targets = observation["targets"]; nt = len(targets) // 3
        active = [t for t in range(nt) if targets[t * 3 + 2] > 0.5]
        if not active:
            return 0
        if self.rng.random() < self.epsilon:
            a = int(self.rng.choice([t + 1 for t in active]))
        else:
            scores = self.P[self.agent_idx] @ self.U[:, active]
            a = int(active[int(np.argmax(scores))]) + 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.step_count += 1
        return a

    def update_from_observation(self, observation: Dict[str, Any]) -> None:
        sel = observation["selected_targets"]; rew = observation["observed_rewards"]
        for k in range(self.num_agents):
            ta = int(sel[k])
            if ta <= 0 or ta - 1 >= self.num_targets:
                continue
            self.dk.append(k); self.tj.append(ta - 1); self.rv.append(float(rew[k]))
        if len(self.dk) > self.window:                       # bounded buffer (avoid O(T^2))
            self.dk = self.dk[-self.window:]; self.tj = self.tj[-self.window:]; self.rv = self.rv[-self.window:]
        if (self.step_count + 1) % self.refit_every == 0:
            self._als()

    def _als(self) -> None:
        if not self.dk:
            return
        K = np.asarray(self.dk); J = np.asarray(self.tj); V = np.asarray(self.rv)
        I = self.lam * np.eye(self.latent_dim)
        for _ in range(self.sweeps):
            A = np.tile(I, (self.num_agents, 1, 1)).astype(float); b = np.zeros((self.num_agents, self.latent_dim))
            Uj = self.U[:, J].T
            np.add.at(A, K, np.einsum('ni,nj->nij', Uj, Uj))
            np.add.at(b, K, V[:, None] * Uj)
            self.P = np.linalg.solve(A, b[..., None])[..., 0]
            A2 = np.tile(I, (self.num_targets, 1, 1)).astype(float); b2 = np.zeros((self.num_targets, self.latent_dim))
            Pk = self.P[K]
            np.add.at(A2, J, np.einsum('ni,nj->nij', Pk, Pk))
            np.add.at(b2, J, V[:, None] * Pk)
            self.U = np.linalg.solve(A2, b2[..., None])[..., 0].T

    def soft_reset(self, episode_count: Optional[int] = None) -> None:
        pass

    def reset(self) -> None:
        self.rng = np.random.RandomState(self.seed)
        self.P = self.rng.normal(0, 0.1, (self.num_agents, self.latent_dim))
        self.U = self.rng.normal(0, 0.1, (self.latent_dim, self.num_targets))
        self.dk = []; self.tj = []; self.rv = []; self.step_count = 0
        self.epsilon = self.initial_epsilon

    def get_learning_state(self) -> Optional[Dict[str, Any]]:
        return {"agent_idx": self.agent_idx, "P": self.P.tolist(), "U": self.U.tolist()}
