"""RunConfig: the single source of every LatentSwarm knob (world, observation, dynamics,
scenario, policy, evaluation). Components read what they need from this object, so a run is
fully described by one config (serializable to JSON)."""
from dataclasses import dataclass, field, asdict
from typing import List, Union
import json


@dataclass
class RunConfig:
    # --- world ---
    m: int = 30                 # robots
    n: int = 240                # tasks (n >> T: task-scarce)
    d: int = 5                  # TRUE latent rank
    T: int = 50                 # mission horizon (rounds)

    # --- rank guess (the estimator's ASSUMED rank) ---
    # "random" draws d-hat ~ Uniform{rank_lo, ..., rank_hi} once per run (robustness to the guess).
    # d-hat must be >= d for exact recovery, so rank_lo defaults to d (see docs).
    rank_guess: Union[int, str] = "random"
    rank_lo: int = 5            # = d
    rank_hi: int = 10           # = 2d

    # --- offered menu ---
    # 0 = ALL tasks offered each round (default). Else: per-robot random size-c subset.
    offer_size: int = 0

    # --- observation channel ---
    mask_mode: str = "persistent"   # "persistent" (fixed) | "per_round" (dynamic) | "line_of_sight" (geometry-induced)
    rho: float = 0.5                # broadcast visibility rate per pair; target LOS density for "line_of_sight"
    sigma_obs: float = 0.3          # per-observer (private) noise on a broadcast reading
    sigma_own: float = 0.0          # noise on a robot's own reading

    # --- geometry (only for mask_mode="line_of_sight"): 2-D patrol positions induce a persistent,
    # range-limited disk-graph mask and distance-dependent per-observer noise (SNR ~ 1/r^2) ---
    field_size: float = 10.0        # side length of the 2-D field
    n_clusters: int = 5             # spatial clusters (sectorized patrol) -> persistent visibility structure
    cluster_std: float = 1.2        # within-cluster position spread
    sensing_radius: float = 0.0     # R_s; 0 -> set to the rho-quantile of pairwise distances (density parity with rho)
    noise_r0: float = 0.0           # distance-noise scale R0; 0 -> use R_s
    noise_alpha: float = 2.0        # per-observer noise variance grows as sigma_obs^2 * (1 + (r/R0)^alpha)

    # --- dynamics ---
    capacity_one: bool = True       # only the first robot to pick a task each round succeeds
    reward_model: str = "inner_product"   # "inner_product" (R_ij=<p_i,u_j>) | "cosine"

    # --- scenario (latent-trait generation) ---
    scenario: str = "gaussian_mixture"
    n_modes: int = 5
    jitter: float = 0.2
    # block_cosine (parity) scenario: number of latent types K (= core's K1=K2). Each type is forced
    # present (as experiments/core.make_world does); within-type spread is `jitter` there.
    n_types: int = 10
    # sensing_coalition scenario knobs (lifted from former literals): a robot/site profile is a small
    # baseline competence in every modality plus a specialty bump on its archetype's modality.
    sensing_base_competence: float = 0.15    # baseline competence in every modality (was 0.15)
    sensing_specialty: float = 1.0           # specialty-modality strength bump (was 1.0)

    # --- policy (shared exploration + estimator knobs) ---
    epsilon: float = 0.4
    epsilon_decay: float = 0.99
    epsilon_min: float = 0.05
    ridge: float = 1.0
    als_sweeps: int = 8
    refit_every: int = 3
    mf_lr: float = 0.05
    mf_ridge: float = 1e-2          # MF-SGD factor L2 regularization (was the hard-coded lam=1e-2)
    ucb_c: float = 2.0
    factor_init_scale: float = 0.1  # std of the low-rank factor random init (was rng.normal(0, 0.1, ...))
    buffer_window: int = 6000       # SwarmCF per-robot observation buffer length (was self.window=6000)

    # --- baselines.py hyperparameters (mirror the analytical-harness REGISTRY) ---
    estr_explore_frac: float = 0.4  # ESTR uniform-explore fraction of the horizon before the SVD commit
    ptf_probe_frac: float = 0.4     # SwarmCF-batch (PTF) own-row-UCB probe fraction before warm-start
    bpmf_prior_var: float = 1.0     # BPMF factor prior variance (precision = 1/prior_var)

    # --- refinements.py: the SwarmCF-* family (follow-up paper) ----------------------------------
    # All knobs default to the prototype settings (experiments/pilot_noise.py); ported faithfully.
    # SwarmCF-B (em_cf): variational Bayesian PMF + predictive-variance UCB exploration.
    em_lam: float = 1.0             # factor prior precision lambda (also the ARD column-prior init)
    em_sweeps: int = 6              # variational EM sweeps per refit
    em_refit_every: int = 4         # rounds between variational refits
    em_beta: float = 1.0            # predictive-sd UCB exploration weight (0 -> eps-greedy)
    em_collective: bool = True      # UCB bonus uses the COLLECTIVE (shared-u_j) variance only (anneals cleanly)
    em_shrink: float = 0.0          # >0 shrinks high-variance predictions toward the popularity prior
    # SwarmCF-B-ARD (ard_em_cf): same estimator with automatic relevance determination on factor columns.
    ard_eff_rank_thresh: float = 0.05   # a column counts toward the effective rank if 1/alpha_r > thresh * max
    # SwarmCF-X / SwarmCF-Xc (active_cf / coord_cf): count-bonus exploration in latent space.
    c_active: float = 0.5           # ActiveCF own-count UCB bonus weight: <p_i,u_j> + c_active/sqrt(gcount+1)
    c_explore: float = 0.5          # CoordCF negative-correlated (swarm-count) exploration bonus weight
    # SwarmCF-D / SwarmCF-D+ (contention_cf / contention_ada_cf): private de-confliction offset.
    eps_break: float = 0.1          # std of the FIXED private per-task offset h_i (symmetry breaking)
    deconflict_eps_lo: float = 0.02     # min offset magnitude (D+ self-tuning floor; near-greedy when winning)
    deconflict_eps_hi: float = 0.8      # max offset magnitude (D+/U saturation; spread hard when losing)
    deconflict_lr: float = 0.15         # EMA rate of the own loss-rate signal
    deconflict_loss0: float = 0.3       # D+ initial loss-EMA (U starts at 0; see refinements.py)
    deconflict_coll_pow: float = 2.0    # convexity of the loss->offset-scale law (1 = linear)
    deconflict_scarcity_k: float = 4.0  # D+ hard scarcity gate: offset engages only if |offer| <= k * m
    # SwarmCF-U (unified_cf): EMCF + loss-gated offset + loss-gated exploration anneal + abundance gate.
    unified_beta_anneal: float = 0.4    # loss-EMA at which the UCB exploration bonus is fully damped
    unified_abundance_k: float = 4.0    # damp UCB + fall back to eps-greedy when |offer| > k * m (no scarcity)
    unified_horizon: int = 0            # >0 enables a finite-horizon exploration anneal (VoI -> 0 near T); 0 = off
    # SwarmCF-Ch / SwarmCF-RC (choice_cf / both_cf): the action/choice channel.
    choice_s2c: float = 0.2         # choice-pseudo-observation variance (weight = gamma / s2c)
    choice_n_neg: int = 1           # negatives sampled per observed choice (the not-chosen pseudo-targets)
    choice_within: bool = True      # sample negatives from the teammate's OFFER (True) or globally (False)
    choice_competence: bool = True  # competence-weight choices (True = SwarmCF-Ch; False = naive/unweighted)
    choice_warm_frac: float = 0.3   # ignore the choice channel before this fraction of the horizon (ramp start)

    # --- evaluation ---
    seeds: List[int] = field(default_factory=lambda: list(range(16)))
    algorithms: List[str] = field(default_factory=lambda: ["random", "ucb_indep", "mf_sgd", "swarm_cf"])
    metrics: List[str] = field(default_factory=lambda: ["earned_skill", "unseen_pair_skill"])

    def rank_for_run(self, rng) -> int:
        """The guessed rank d-hat for one run (fixed int, or random in [rank_lo, rank_hi])."""
        if isinstance(self.rank_guess, int):
            return int(self.rank_guess)
        return int(rng.randint(self.rank_lo, self.rank_hi + 1))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        json.dump(self.to_dict(), open(path, "w"), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        return cls(**d)

    @classmethod
    def load(cls, path: str) -> "RunConfig":
        return cls(**json.load(open(path)))
