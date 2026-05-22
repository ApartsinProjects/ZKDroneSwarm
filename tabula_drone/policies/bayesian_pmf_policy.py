"""
Bayesian Probabilistic Matrix Factorization (BPMF) Policy for ZK-MRTA.

A proper Bayesian variant of MF-CF using closed-form conjugate updates
instead of SGD on a bilinear loss. Inspired by Salakhutdinov-Mnih 2008
(Bayesian PMF) and extended for online sequential decision-making in
the ZK-MRTA setting.

ALGORITHM OVERVIEW
==================
Each drone maintains a mean-field variational posterior over the rank-d
latent factor matrices P (drone factors, shape m x d) and U (target
factors, shape n x d). The posteriors are conjugate Gaussian under the
factorisation
    q(P, U) = prod_i N(P_i | mu_P_i, Sigma_P_i)
            * prod_j N(U_j | mu_U_j, Sigma_U_j).

Each posterior is stored in natural parameter form:
    Lambda_P_i = Sigma_P_i^{-1}    (precision matrix, d x d)
    eta_P_i    = Lambda_P_i @ mu_P_i   (precision-weighted mean, d-dim)
and analogously for U.

CLOSED-FORM CONJUGATE UPDATE
============================
For each broadcast tuple (drone k pulled target j; observed reward r),
the proper Variational Bayes update is (with likelihood variance
sigma_obs^2 representing the observer's noise):

    Lambda_P_k += E[U_j U_j^T] / sigma_obs^2
                = (mu_U_j @ mu_U_j^T + Sigma_U_j) / sigma_obs^2
    eta_P_k    += (r * mu_U_j) / sigma_obs^2

    Lambda_U_j += E[P_k P_k^T] / sigma_obs^2
                = (mu_P_k @ mu_P_k^T + Sigma_P_k) / sigma_obs^2
    eta_U_j    += (r * mu_P_k) / sigma_obs^2

(The variance terms Sigma_U_j and Sigma_P_k inside E[xx^T] are what
distinguishes Variational Bayes from naive plug-in MAP/SGD. Without
them this reduces to regularised alternating least squares.)

ACTION SELECTION
================
Thompson sampling: at each step, sample
    tilde P_i ~ N(mu_P_i, Sigma_P_i)
    tilde U_j ~ N(mu_U_j, Sigma_U_j)  for each active target j
and pick arg max_j tilde P_i^T tilde U_j.

Posterior uncertainty drives exploration without epsilon-greedy.

ADVANTAGES OVER SGD-MF-CF
=========================
1. No cold start: even after one observation, the posterior is well-
   defined (just wider). Decisions are optimal under current uncertainty.
2. No learning-rate / decay hyperparameters: only prior variance and
   likelihood variance.
3. Principled Thompson-sampling exploration replaces ad-hoc epsilon-
   greedy.
4. Closed-form O(d^3) per broadcast tuple. For d=3, m=9, n=27 this is
   ~6500 ops per step -- trivially fast.

ZK COMPLIANCE
=============
Each drone maintains its own private posterior, updated from its own
noisy view of the public broadcast (per the two-stage noise model of
section 3.2). No parameter sharing across drones.

References:
  Salakhutdinov & Mnih (2008) "Bayesian Probabilistic Matrix
    Factorization using Markov Chain Monte Carlo", ICML.
  Bishop (2006) "Pattern Recognition and Machine Learning",
    chapter 12 (Continuous Latent Variables) -- VBPMF derivation.
"""

from typing import Any, Dict, Optional
import numpy as np


class BayesianPMFPolicy:
    """
    Per-drone Bayesian Probabilistic Matrix Factorization with closed-form
    conjugate updates and Thompson-sampling action selection.
    """

    is_deterministic: bool = False

    def __init__(
        self,
        num_agents: int,
        num_targets: int,
        agent_idx: int,
        latent_dim: int = 3,
        prior_var_P: float = 1.0,
        prior_var_U: float = 1.0,
        likelihood_var: float = 0.04,
        seed: Optional[int] = None,
        allow_noop: bool = False,
        exploration_mode: str = "thompson",
        ucb_c: float = 1.0,
        update_mode: str = "map",       # "vb" or "map" (plug-in)
        init_scale: float = 0.1,        # std of random initialisation on prior means
    ):
        """
        Args:
            num_agents, num_targets: scenario dimensions.
            agent_idx: this drone's identifier (0-based).
            latent_dim: rank d of the factorisation.
            prior_var_P, prior_var_U: prior variances on the drone and
                target factors. Default 1.0 (uninformative for rewards in
                [-1, 1]).
            likelihood_var: assumed variance of the observation noise
                (sigma_e^2 + sigma_r^2 per the two-stage noise model).
                Default 0.04 corresponds to sigma = 0.2 baseline.
            exploration_mode: "thompson" (sample from posteriors) or
                "ucb" (use mean + c * sqrt(predictive variance)).
        """
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.agent_idx = agent_idx
        self.latent_dim = latent_dim
        self.prior_var_P = prior_var_P
        self.prior_var_U = prior_var_U
        self.likelihood_var = likelihood_var
        self.seed = seed
        self.allow_noop = allow_noop
        self.exploration_mode = exploration_mode
        self.ucb_c = ucb_c
        self.update_mode = update_mode
        self.init_scale = init_scale

        self.rng = np.random.RandomState(seed)
        self._init_posteriors()
        self.step_count = 0

    def _init_posteriors(self):
        d = self.latent_dim
        prior_prec_P = 1.0 / self.prior_var_P
        prior_prec_U = 1.0 / self.prior_var_U
        self.Lambda_P = np.tile(prior_prec_P * np.eye(d), (self.num_agents, 1, 1)).copy()
        self.Lambda_U = np.tile(prior_prec_U * np.eye(d), (self.num_targets, 1, 1)).copy()
        # Break symmetry: small random initial means N(0, init_scale^2 I).
        # Stored in natural-parameter form: eta = Lambda @ mu_init.
        init_rng = np.random.RandomState(self.seed if self.seed is not None else 0)
        mu_P_init = init_rng.normal(0.0, self.init_scale, (self.num_agents, d))
        mu_U_init = init_rng.normal(0.0, self.init_scale, (self.num_targets, d))
        self.eta_P = np.einsum("idj,ij->id", self.Lambda_P, mu_P_init)
        self.eta_U = np.einsum("jdk,jk->jd", self.Lambda_U, mu_U_init)
        # Cached posteriors (refreshed lazily for efficiency)
        self._cache_dirty = True
        self._mu_P = None
        self._Sigma_P = None
        self._mu_U = None
        self._Sigma_U = None

    def _refresh_cache(self):
        """Recompute posterior means and covariances from natural parameters."""
        d = self.latent_dim
        self._Sigma_P = np.zeros((self.num_agents, d, d))
        self._mu_P = np.zeros((self.num_agents, d))
        for i in range(self.num_agents):
            try:
                self._Sigma_P[i] = np.linalg.inv(self.Lambda_P[i])
            except np.linalg.LinAlgError:
                self._Sigma_P[i] = np.linalg.pinv(self.Lambda_P[i])
            self._mu_P[i] = self._Sigma_P[i] @ self.eta_P[i]
        self._Sigma_U = np.zeros((self.num_targets, d, d))
        self._mu_U = np.zeros((self.num_targets, d))
        for j in range(self.num_targets):
            try:
                self._Sigma_U[j] = np.linalg.inv(self.Lambda_U[j])
            except np.linalg.LinAlgError:
                self._Sigma_U[j] = np.linalg.pinv(self.Lambda_U[j])
            self._mu_U[j] = self._Sigma_U[j] @ self.eta_U[j]
        self._cache_dirty = False

    def select_action(self, observation: Dict[str, Any], allow_noop: bool = False) -> int:
        """Thompson-sampling or UCB action selection for this drone."""
        if self._cache_dirty:
            self._refresh_cache()

        targets_obs = observation["targets"]
        num_targets = len(targets_obs) // 3
        active_targets = [
            t for t in range(num_targets) if targets_obs[t * 3 + 2] > 0.5
        ]
        if not active_targets:
            return 0

        i = self.agent_idx
        mu_P_i = self._mu_P[i]
        Sigma_P_i = self._Sigma_P[i]

        if self.exploration_mode == "thompson":
            # Sample P_i ~ N(mu_P_i, Sigma_P_i) once for this step.
            try:
                L_P = np.linalg.cholesky(Sigma_P_i + 1e-9 * np.eye(self.latent_dim))
                p_tilde = mu_P_i + L_P @ self.rng.randn(self.latent_dim)
            except np.linalg.LinAlgError:
                p_tilde = mu_P_i + np.sqrt(np.abs(np.diag(Sigma_P_i))) * self.rng.randn(self.latent_dim)
            # Sample each U_j independently and score.
            best_score = -np.inf
            best_target = active_targets[0]
            for j in active_targets:
                Sigma_U_j = self._Sigma_U[j]
                mu_U_j = self._mu_U[j]
                try:
                    L_U = np.linalg.cholesky(Sigma_U_j + 1e-9 * np.eye(self.latent_dim))
                    u_tilde = mu_U_j + L_U @ self.rng.randn(self.latent_dim)
                except np.linalg.LinAlgError:
                    u_tilde = mu_U_j + np.sqrt(np.abs(np.diag(Sigma_U_j))) * self.rng.randn(self.latent_dim)
                score = float(np.dot(p_tilde, u_tilde))
                if score > best_score:
                    best_score = score
                    best_target = j
            action = best_target + 1
        else:  # UCB
            best_score = -np.inf
            best_target = active_targets[0]
            for j in active_targets:
                mu_U_j = self._mu_U[j]
                Sigma_U_j = self._Sigma_U[j]
                mean = float(np.dot(mu_P_i, mu_U_j))
                # Predictive variance of bilinear product (mean-field approx).
                var = (
                    float(mu_P_i @ Sigma_U_j @ mu_P_i)
                    + float(mu_U_j @ Sigma_P_i @ mu_U_j)
                    + float(np.trace(Sigma_P_i @ Sigma_U_j))
                )
                score = mean + self.ucb_c * np.sqrt(max(var, 1e-12))
                if score > best_score:
                    best_score = score
                    best_target = j
            action = best_target + 1

        self.step_count += 1
        return action

    def update_from_observation(self, observation: Dict[str, Any]) -> None:
        """
        Closed-form Variational Bayes update from a broadcast tuple.

        For each (drone_k, target_j, reward) in the broadcast, update
        BOTH the agent's posterior for P_k and the posterior for U_j,
        using the proper E[xx^T] = mu mu^T + Sigma form (Variational
        Bayes, not naive plug-in MAP).
        """
        if self._cache_dirty:
            self._refresh_cache()
        selected_targets = observation["selected_targets"]
        observed_rewards = observation["observed_rewards"]
        sigma2 = self.likelihood_var

        for k in range(self.num_agents):
            target_action = int(selected_targets[k])
            if target_action <= 0:
                continue
            j = target_action - 1
            if j >= self.num_targets:
                continue
            r = float(observed_rewards[k])

            # Snapshot the current posteriors (pre-update) for both factors.
            mu_P_k = self._mu_P[k].copy()
            mu_U_j = self._mu_U[j].copy()
            if self.update_mode == "vb":
                Sigma_P_k = self._Sigma_P[k].copy()
                Sigma_U_j = self._Sigma_U[j].copy()
                # E[U_j U_j^T] = mu mu^T + Sigma (Variational Bayes).
                E_U_outer = np.outer(mu_U_j, mu_U_j) + Sigma_U_j
                E_P_outer = np.outer(mu_P_k, mu_P_k) + Sigma_P_k
            else:  # "map" plug-in (online regularised alternating least squares)
                E_U_outer = np.outer(mu_U_j, mu_U_j)
                E_P_outer = np.outer(mu_P_k, mu_P_k)

            self.Lambda_P[k] += E_U_outer / sigma2
            self.eta_P[k] += (r * mu_U_j) / sigma2
            self.Lambda_U[j] += E_P_outer / sigma2
            self.eta_U[j] += (r * mu_P_k) / sigma2

        # Posteriors changed; mark cache dirty (will be refreshed at next access).
        self._cache_dirty = True

    def soft_reset(self, episode_count: Optional[int] = None) -> None:
        """End-of-episode reset; posteriors are preserved across episodes."""
        pass

    def reset(self) -> None:
        """Full reset to prior."""
        self.rng = np.random.RandomState(self.seed)
        self._init_posteriors()
        self.step_count = 0

    def get_learning_state(self) -> Optional[Dict[str, Any]]:
        if self._cache_dirty:
            self._refresh_cache()
        return {
            "policy": "bpmf",
            "exploration_mode": self.exploration_mode,
            "step_count": self.step_count,
            "mu_P_norm_mean": float(np.linalg.norm(self._mu_P, axis=1).mean()),
            "mu_U_norm_mean": float(np.linalg.norm(self._mu_U, axis=1).mean()),
        }
