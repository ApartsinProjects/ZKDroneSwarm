"""
tabula_drone/policies/probe_then_fit_policy.py

Probe-Then-Fit (PTF) hybrid policy for ZK-MRTA.

Addresses MF-CF's cold-start problem by borrowing the "offline pre-training
then online fine-tuning" pattern from hybrid recommender systems:

  Phase 1  (episodes 1 .. probe_episodes):
      Run UCB-Indep to rapidly fill an empirical reward matrix R̂ ∈ ℝ^{m×n}.
      UCB-Indep has no cold-start cost; it immediately identifies high-reward
      arms and accumulates per-(drone, target) statistics.

  Transition  (end of episode probe_episodes):
      Truncated SVD of R̂  →  rank-d_f approximation
      P_init = U_svd[:, :d_f] * sqrt(S[:d_f])    shape (m, d_f)
      Q_init = Vt_svd[:d_f, :].T * sqrt(S[:d_f])  shape (n, d_f)
      These are injected into each MF-CF drone's embedding matrices,
      replacing the random Gaussian initialisation with an informed
      starting point that already encodes the dominant latent structure.

  Phase 2  (episodes probe_episodes+1 .. T):
      MF-CF runs with warm-started P, U.  The cold-start phase is
      skipped; the policy enters exploitation immediately.

Why this helps:
  - UCB-Indep after K episodes has filled R̂ with ~K*steps/m observations
    per (drone, target) pair on average.  For K=8, each cell has 3-5 obs.
  - SVD of a partially-observed matrix recovers the dominant rank-d factor
    reliably once ~d*(m+n) entries are observed (Candès–Recht 2009).
  - MF-CF started from SVD factors needs only fine-tuning, not full
    structure recovery:  the first post-transition episode achieves LMQ
    close to UCB-Indep's converged LMQ rather than sub-random.

ZK compliance: Uses only the public observation stream (selected_targets,
               observed_rewards broadcast).  No latent vectors, no HP.

Parameters:
  probe_episodes:    Number of UCB-Indep episodes before switching to MF-CF.
                     Recommended: 5-10 for d=3, 10-20 for d=8.
  latent_dim:        MF-CF factorisation rank d_f (matched to env d).
  All other MF-CF kwargs are forwarded unchanged.
"""

from typing import Any, Dict, Optional

import numpy as np

from .base import IPolicy, EnvInfos
from .ucb_indep_policy import UCBIndepPolicy
from .matrix_factorization_policy import MatrixFactorizationPolicy
from .multi_agent_policy import MultiAgentPolicy


class ProbeThenFitPolicy(IPolicy):
    """
    Hybrid UCB-Indep (probe) then MF-CF (fit) policy.

    Wraps one UCBIndepPolicy for the probe phase and a MultiAgentPolicy
    wrapping per-drone MatrixFactorizationPolicy instances for the fit phase.
    The phase transition is triggered automatically after probe_episodes.
    """

    is_deterministic: bool = False

    def __init__(
        self,
        num_agents: int,
        num_targets: int,
        latent_dim: int = 3,
        probe_episodes: int = 8,
        # UCB probe parameters
        ucb_c: float = 2.0,
        # MF-CF parameters (all forwarded)
        learning_rate: float = 0.01,
        lambda_reg: float = 0.02,
        epsilon: float = 0.20,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.02,
        anti_signal_weight: float = 0.1,
        use_integration_matrix: bool = False,
        seed: Optional[int] = None,
        allow_noop: bool = False,
    ):
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.latent_dim = latent_dim
        self.probe_episodes = probe_episodes
        self.allow_noop = allow_noop
        self.seed = seed

        # Episode counter (incremented in soft_reset)
        self._episode = 0
        self._phase = "probe"  # "probe" | "fit"
        self._transitioned = False

        # Probe policy: UCBIndep
        self._probe = UCBIndepPolicy(
            num_agents=num_agents,
            num_targets=num_targets,
            c=ucb_c,
            seed=seed,
            allow_noop=allow_noop,
        )

        # Fit policy: MF-CF MultiAgentPolicy (one per drone)
        mf_policies = {}
        for i in range(num_agents):
            mf_policies[f"drone_{i}"] = MatrixFactorizationPolicy(
                num_targets=num_targets,
                agent_idx=i,
                num_agents=num_agents,
                latent_dim=latent_dim,
                learning_rate=learning_rate,
                lambda_reg=lambda_reg,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
                anti_signal_weight=anti_signal_weight,
                use_integration_matrix=use_integration_matrix,
                seed=(seed + i) if seed is not None else None,
            )
        self._fit = MultiAgentPolicy(mf_policies)

        # Track when warm-start was applied for logging
        self._warm_start_applied = False
        self._svd_error_before: Optional[float] = None  # Frobenius error before warm
        self._svd_error_after: Optional[float] = None   # Frobenius error after warm

    # ------------------------------------------------------------------
    # IPolicy interface
    # ------------------------------------------------------------------

    def select_actions(
        self,
        obs: Dict[str, Any],
        infos: EnvInfos,
        env: Any = None,
    ) -> Dict[str, int]:
        if self._phase == "probe":
            return self._probe.select_actions(obs, infos, env=env)
        else:
            return self._fit.select_actions(obs, infos)

    def update(self, obs: Dict[str, Any]) -> None:
        if self._phase == "probe":
            self._probe.update(obs)
        else:
            self._fit.update(obs)

    def soft_reset(self) -> None:
        """Called at end of each episode.  Handles phase transition."""
        self._episode += 1

        if self._phase == "probe":
            self._probe.soft_reset()
            if self._episode >= self.probe_episodes:
                self._apply_warm_start()
                self._phase = "fit"
        else:
            self._fit.soft_reset()

    # ------------------------------------------------------------------
    # Warm-start logic
    # ------------------------------------------------------------------

    def _build_R_hat(self) -> np.ndarray:
        """
        Construct the empirical reward matrix from UCBIndep's arm statistics.

        R̂[i, j] = total_reward(i, j) / max(count(i, j), 1)

        Unobserved cells (count=0) get R̂=0, which is fine for SVD since
        the matrix-completion guarantee only requires that *observed* entries
        are accurate.  We do NOT impute missing values.
        """
        counts = self._probe._counts.astype(np.float64)           # (m, n)
        reward_sums = self._probe._reward_sums.astype(np.float64)  # (m, n)
        R_hat = np.where(counts > 0, reward_sums / counts, 0.0)
        return R_hat

    def _apply_warm_start(self) -> None:
        """
        SVD-decompose R̂ and inject rank-d_f factors into each MF-CF drone.

        After: pol.P[i, :] ≈ left singular vector * sqrt(singular value)
               pol.U[:, j] ≈ right singular vector * sqrt(singular value)
        so that P[i] @ U[:, j] ≈ R̂[i, j] at the probe-phase mean.
        """
        R_hat = self._build_R_hat()                                # (m, n)
        d_f = self.latent_dim

        # Truncated SVD: keep top-d_f components
        try:
            U_svd, S_svd, Vt_svd = np.linalg.svd(R_hat, full_matrices=False)
            rank = min(d_f, len(S_svd))

            # Scale factors by sqrt(sigma) so P @ U ≈ R_hat
            sqrt_S = np.sqrt(np.maximum(S_svd[:rank], 0.0))       # (rank,)
            P_init = U_svd[:, :rank] * sqrt_S[np.newaxis, :]      # (m, rank)
            Q_init = Vt_svd[:rank, :].T * sqrt_S[np.newaxis, :]   # (n, rank)

            # Pad to d_f if rank < d_f (rare: R_hat nearly zero)
            if rank < d_f:
                rng = np.random.RandomState(self.seed)
                pad_P = rng.normal(0.0, 0.01, (self.num_agents, d_f - rank))
                pad_Q = rng.normal(0.0, 0.01, (self.num_targets, d_f - rank))
                P_init = np.hstack([P_init, pad_P])
                Q_init = np.hstack([Q_init, pad_Q])

            # Inject into each MF-CF drone
            for agent_id, pol in self._fit.policies.items():
                pol.P[:, :] = P_init                               # (m, d_f)
                pol.U[:, :] = Q_init.T                             # (d_f, n)

            self._warm_start_applied = True

        except np.linalg.LinAlgError:
            # SVD failed (degenerate R_hat) — keep random init, log and continue
            self._warm_start_applied = False

    # ------------------------------------------------------------------
    # Compatibility shims
    # ------------------------------------------------------------------

    def get_learning_state(self) -> Optional[Dict[str, Any]]:
        state = {
            "phase": self._phase,
            "episode": self._episode,
            "probe_episodes": self.probe_episodes,
            "warm_start_applied": self._warm_start_applied,
        }
        if self._phase == "fit":
            state["mf"] = self._fit.get_learning_state()
        return state
