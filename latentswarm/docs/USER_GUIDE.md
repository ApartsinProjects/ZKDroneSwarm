# LatentSwarm user guide

How to configure and run studies. For extending the suite (new policies, scenarios, metrics) see the
[developer guide](DEVELOPER_GUIDE.md).

## 1. Running

**Python API.** Build a `RunConfig`, call `run`:
```python
from latentswarm import RunConfig
from latentswarm.run import run

cfg = RunConfig(seeds=list(range(16)),
                algorithms=["random", "ucb_indep", "mf_sgd", "swarm_cf"])
out = run(cfg)                         # prints a per-policy table; returns the results dict (below)
```

**Command line.**
```bash
python -m latentswarm.run --out results/pilots/latentswarm_main.json
python -m latentswarm.run --config my_study.json          # load a saved RunConfig (see to_json/load)
```

## 2. Configuration reference (`RunConfig`)

All fields have sensible defaults; override only what you need.

**World**
| Field | Default | Meaning |
|---|---|---|
| `m`, `n`, `d` | 30, 240, 5 | robots, tasks, true latent rank |
| `T` | 50 | mission horizon (rounds); task-scarce means `n >> T` |

**Guessed rank** (the estimator's assumed rank `d̂`)
| Field | Default | Meaning |
|---|---|---|
| `rank_guess` | `"random"` | `"random"` draws `d̂ ~ Uniform{rank_lo..rank_hi}` once per run; or set an int |
| `rank_lo`, `rank_hi` | 5, 10 | the `[d, 2d]` interval; over-ranking is safe, under-ranking is mis-specification |

**Offered menu**
| Field | Default | Meaning |
|---|---|---|
| `offer_size` | 0 | 0 = all `n` tasks offered each round; else a per-robot random size-`c` subset |

**Observation channel**
| Field | Default | Meaning |
|---|---|---|
| `mask_mode` | `"persistent"` | `"persistent"` (fixed blind spots), `"per_round"` (redrawn each round), `"line_of_sight"` (geometry-induced) |
| `rho` | 0.5 | per-pair visibility rate; for `line_of_sight` it is the target line-of-sight density |
| `sigma_obs` | 0.3 | per-observer noise std on a broadcast reading |
| `sigma_own` | 0.0 | noise on a robot's own reading |

**Geometry** (only for `mask_mode="line_of_sight"`)
| Field | Default | Meaning |
|---|---|---|
| `field_size` | 10.0 | side length of the 2-D field |
| `n_clusters` | 5 | spatial clusters (sectorized patrol) giving a persistent visibility structure |
| `cluster_std` | 1.2 | within-cluster position spread |
| `sensing_radius` | 0.0 | `R_s`; 0 sets it to the `rho`-quantile of pairwise distances (density parity with `rho`) |
| `noise_r0`, `noise_alpha` | 0.0, 2.0 | per-observer noise variance grows as `sigma_obs^2 (1 + (r/R0)^alpha)`; `R0=0` uses `R_s` |

**Dynamics**
| Field | Default | Meaning |
|---|---|---|
| `capacity_one` | True | only the first robot to pick a task each round succeeds (capacity-1 contention) |
| `reward_model` | `"inner_product"` | `"inner_product"` (`R_ij=<p_i,u_j>`) or `"cosine"` (normalized) |

**Scenario (latent-trait world)**
| Field | Default | Meaning |
|---|---|---|
| `scenario` | `"gaussian_mixture"` | `uniform_cosine` (i.i.d. unit-sphere, NO types; the paper's headline world), `block_cosine` (block, unit-cosine in [-1,1]; parity with the analytical harness), `gaussian_mixture` (block, unnormalized inner product), `iid_gaussian`, `sensing_coalition` (robotics-grounded sensing modalities), `approx_lowrank` (rank-d + full-rank perturbation, Appendix F) |
| `n_modes` | 5 | latent types for `gaussian_mixture` / `sensing_coalition` |
| `n_types` | 10 | latent types for `block_cosine` (= the analytical harness `K1=K2`) |
| `jitter` | 0.2 | within-type spread (use `0.15` for exact `block_cosine` parity with `core.make_world`) |
| `sensing_base_competence`, `sensing_specialty` | 0.15, 1.0 | `sensing_coalition` modality baseline and specialty bump |

**Policy hyperparameters**
| Field | Default | Meaning |
|---|---|---|
| `epsilon`, `epsilon_decay`, `epsilon_min` | 0.4, 0.99, 0.05 | exploration schedule shared by structured policies |
| `ridge`, `als_sweeps`, `refit_every` | 1.0, 8, 3 | SwarmCF weighted-ALS hyperparameters |
| `mf_lr`, `mf_ridge` | 0.05, 1e-2 | MF-SGD step size and factor L2 regularization |
| `factor_init_scale`, `buffer_window` | 0.1, 6000 | low-rank factor init std; SwarmCF observation buffer length |
| `ucb_c` | 2.0 | UCB exploration constant |
| `estr_explore_frac`, `ptf_probe_frac`, `bpmf_prior_var` | 0.4, 0.4, 1.0 | ESTR / SwarmCF-batch / BPMF baseline hyperparameters |

**Refinement hyperparameters (the SwarmCF-\* family)**
All default to the prototype settings; override only when a study needs to. Grouped by refinement.

| Field | Default | Used by | Meaning |
|---|---|---|---|
| `em_lam` | 1.0 | `em_cf`, `ard_em_cf`, `unified_cf` | factor prior precision lambda (and ARD column-prior init) |
| `em_sweeps` | 6 | same | variational EM sweeps per refit |
| `em_refit_every` | 4 | same | rounds between variational refits |
| `em_beta` | 1.0 | same | predictive-sd UCB exploration weight (0 -> eps-greedy) |
| `em_collective` | True | same | UCB bonus uses the collective (shared-`u_j`) variance only (anneals cleanly; the own-factor term over-explores early) |
| `em_shrink` | 0.0 | same | >0 shrinks high-variance predictions toward the popularity prior |
| `ard_eff_rank_thresh` | 0.05 | `ard_em_cf`, `effective_rank` | a latent column counts toward the effective rank if its energy `1/alpha_r` exceeds this fraction of the largest |
| `c_active` | 0.5 | `active_cf` | own/broadcast-count latent-UCB bonus weight |
| `c_explore` | 0.5 | `coord_cf` | negative-correlated (swarm-count) exploration bonus weight |
| `eps_break` | 0.1 | `contention_cf`, `contention_ada_cf` | std of the fixed private per-task offset (symmetry breaking) |
| `deconflict_eps_lo`, `deconflict_eps_hi` | 0.02, 0.8 | `contention_ada_cf` | min/max offset magnitude (near-greedy when winning; spread hard when losing) |
| `deconflict_lr` | 0.15 | `contention_ada_cf`, `unified_cf` | EMA rate of the own loss-rate signal |
| `deconflict_loss0` | 0.3 | `contention_ada_cf` | initial loss-EMA (`unified_cf` starts at 0, off until losses) |
| `deconflict_coll_pow` | 2.0 | `contention_ada_cf`, `unified_cf` | convexity of the loss -> offset-scale law (1 = linear) |
| `deconflict_scarcity_k` | 4.0 | `contention_ada_cf` | hard scarcity gate: the offset engages only if `|offer| <= k * m` |
| `unified_beta_anneal` | 0.4 | `unified_cf` | loss-EMA at which the UCB exploration bonus is fully damped |
| `unified_abundance_k` | 4.0 | `unified_cf` | damp UCB + fall back to eps-greedy when `|offer| > k * m` (no scarcity) |
| `unified_horizon` | 0 | `unified_cf` | >0 enables a finite-horizon exploration anneal (value of info -> 0 near `T`); 0 = off |
| `choice_s2c` | 0.2 | `choice_cf`, `both_cf` | choice-pseudo-observation variance (weight = gamma / s2c) |
| `choice_n_neg` | 1 | same | negatives sampled per observed choice (not-chosen pseudo-targets) |
| `choice_within` | True | same | sample negatives from the observer's own offer (True) or globally (False) |
| `choice_competence` | True | same | competence-weight choices (True = SwarmCF-Ch / SwarmCF-RC; False = naive/unweighted) |
| `choice_warm_frac` | 0.3 | same | ignore the choice channel before this fraction of the horizon (ramp start) |

**Evaluation**
| Field | Default | Meaning |
|---|---|---|
| `seeds` | `range(16)` | random seeds (each is one independent world + run) |
| `algorithms` | `[random, ucb_indep, mf_sgd, swarm_cf]` | policies (`random` always included); baselines: `tabular`, `ucb_homo`, `estr`, `swarmcf_batch` (PTF), `bpmf`; SwarmCF-\* refinements: `em_cf`, `ard_em_cf`, `active_cf`, `coord_cf`, `contention_cf`, `contention_ada_cf`, `choice_cf`, `both_cf`, `unified_cf` |
| `metrics` | `[earned_skill, unseen_pair_skill]` | also: `unseen_pair_skill_heldout`, `anytime_trajectory`, `cumulative_regret`, `time_to_competence`, `state_uniqueness`, `effective_rank` (ARD rank recovery) |

## 3. Output format

`run(cfg)` returns and (via the CLI) saves a dict:
```python
{
  "meta":    {"d_guesses": [...per seed...], "config": {...RunConfig.to_dict()...}},
  "earned":  {algo: [per-seed earned skill], ..., "oracle": [1.0, ...]},
  "unseen":  {algo: [per-seed unseen-pair skill or None], ...},
  "summary": {algo: {"earned": (mean, lo, hi), "unseen": (mean, lo, hi)}},   # bootstrap 95% CIs
}
```

## 4. Recipes

**Default masked-broadcast (no geometry).**
```python
run(RunConfig())
```

**Capacity-1 contention on vs off.**
```python
run(RunConfig(capacity_one=True))     # collisions: only the first robot to pick a task succeeds
run(RunConfig(capacity_one=False))    # no contention (note: the Hungarian oracle is then a loose ceiling)
```

**Robotics-grounded instance (geometry mask + sensing-modality traits).**
```python
run(RunConfig(scenario="sensing_coalition", n_modes=5,
              mask_mode="line_of_sight", rho=0.5, noise_alpha=2.0))
```
Here visibility comes from 2-D patrol positions (a range-limited line-of-sight disk graph) and the
per-observer noise grows with distance; traits are non-negative sensing modalities (EO / IR / acoustic /
LiDAR / range), so the reward is a modality match.

**Worlds: uniform (headline) vs block vs unnormalized.**
```python
run(RunConfig(scenario="uniform_cosine"))                         # HEADLINE: i.i.d. unit-sphere traits, NO types, cosine in [-1,1]
run(RunConfig(scenario="block_cosine", n_types=10, jitter=0.15))  # block, UNIT-COSINE (parity with core.make_world)
run(RunConfig(scenario="gaussian_mixture"))                       # block, UNNORMALIZED inner product (R is O(1))
```
`uniform_cosine` is the paper's headline world (i.i.d. Gaussian traits L2-normalized to the unit
sphere, no discrete types). `block_cosine` is the parity world: at `jitter=0.15` it reproduces
`experiments/core.make_world(model="block")` bit-for-bit. `gaussian_mixture` is the
unnormalized inner-product world. Alternatively set `reward_model="cosine"` to normalize at reward time
rather than at trait generation.

**Offer size (scarcity knob).**
```python
run(RunConfig(offer_size=0))    # all tasks offered each round
run(RunConfig(offer_size=20))   # per-robot random size-20 menu
```

**Masking model.**
```python
run(RunConfig(mask_mode="persistent"))   # fixed blind spots (the primary, harder case)
run(RunConfig(mask_mode="per_round"))    # i.i.d. per-round visibility (reduces to uniform sub-sampling)
```

**Rank guess.**
```python
run(RunConfig(rank_guess="random", rank_lo=5, rank_hi=10))   # robust to a random guess in [d, 2d]
run(RunConfig(rank_guess=8))                                  # a fixed over-guess
```

**Choosing methods / metrics.** Any registered name works:
```python
from latentswarm import ALGORITHMS, METRICS
print(sorted(ALGORITHMS), sorted(METRICS))
run(RunConfig(algorithms=["random", "ucb_indep", "swarm_cf"]))
```

**The SwarmCF-\* refinements (follow-up paper).** Each refinement is a registered algorithm; select
them like any other policy.
```python
# confidence-directed exploration (SwarmCF-B) vs the core estimator
run(RunConfig(algorithms=["swarm_cf", "em_cf"]))

# rank self-determination (SwarmCF-B-ARD): set an over-guess and read the recovered effective rank
cfg = RunConfig(algorithms=["ard_em_cf"], rank_guess=12, metrics=["effective_rank"])

# the action/choice channel (SwarmCF-Ch), strongest when the broadcast reward is noisy
run(RunConfig(algorithms=["swarm_cf", "choice_cf"], sigma_obs=2.0))

# the unified communication-free method (SwarmCF-U): one policy, refinements gate on their condition
run(RunConfig(algorithms=["unified_cf"]))
```
The de-confliction methods (`contention_cf` = SwarmCF-D, `contention_ada_cf` = SwarmCF-D+) target
capacity-1 contention; evaluate them with the dedicated contention sweep (below), which posts a shared
shrinking offer pool and reports matching-normalized earned reward. Their offset is tuned by
`eps_break`, `deconflict_*`; see the refinement hyperparameter table.

## 5. Reproducing paper figures

**Unified sweep drivers (single codebase).** The package ships config-driven sweeps that emit the same
JSON schema the figure pipeline reads (use `block_cosine` with `jitter=0.15` for parity with the
analytical harness):
```bash
python -m latentswarm.sweeps --which crossover          # broadcast-rate / unseen-pair skill (Fig 2)
python -m latentswarm.sweeps --which anytime            # anytime cumulative-reward trajectory (Fig 3)
python -m latentswarm.sweeps --which collab             # value of the broadcast incl rho=0 (Fig 4a)
python -m latentswarm.sweeps --which scale_m            # scaling with team size m (Fig 4b)
python -m latentswarm.sweeps --which ranksweep          # rank-guess robustness (Fig 7)
python -m latentswarm.sweeps --which offersize          # offer size c=20 vs c=n (Fig 8)
python -m latentswarm.sweeps --which iid_vs_persistent  # masking model (Fig 9)
python -m latentswarm.sweeps --which contention         # de-confliction under contention (follow-up Fig 2)
```
The `contention` sweep belongs to the **follow-up paper** (Section 4): it posts a SHARED size-`pool`
offer that shrinks from plentiful (240) to severe (15), resolves capacity-1 collisions randomly, and
reports matching-normalized earned reward, contention-free unseen skill, and the collision rate for the
private-offset methods (`contention_ada_cf`, `contention_cf`) against the no-offset and structure-free
policies. Hold `rho=1.0` (the default in `base_config`) to isolate contention from masking.
Single-config study drivers also live in `experiments/`:

| Script | Produces |
|---|---|
| `experiments/latentswarm_contention.py` | contention bake-off + collision / de-confliction (Figs 5, 6) |
| `experiments/latentswarm_grounded.py` | the robotics-grounded instance |
| `experiments/ranksweep.py` | the rank-guess robustness sweep (Fig 7) |

`experiments/make_figures.py` turns the JSONs in `results/pilots/` into the figures. A 3-seed parity check
(`results/smoke/parity_check.py`) confirms the package path reproduces the analytical-harness numbers
within overlapping ranges (statistical parity, not bit-identical except for the world itself; see the
developer guide's parity note).

## 6. Reproducibility notes
Every reported number is a mean over `seeds` with a bootstrap 95% confidence interval (`bootstrap_ci`).
A run is fully specified by its `RunConfig`; save it with `cfg.to_json(path)` and reload with
`RunConfig.load(path)` to reproduce a study exactly.
