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

    # --- policy (shared exploration + estimator knobs) ---
    epsilon: float = 0.4
    epsilon_decay: float = 0.99
    epsilon_min: float = 0.05
    ridge: float = 1.0
    als_sweeps: int = 8
    refit_every: int = 3
    mf_lr: float = 0.05
    ucb_c: float = 2.0

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
