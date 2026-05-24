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

**Scenario, policy, evaluation**
| Field | Default | Meaning |
|---|---|---|
| `scenario` | `"gaussian_mixture"` | trait generator: `gaussian_mixture`, `iid_gaussian`, `sensing_coalition` |
| `n_modes`, `jitter` | 5, 0.2 | latent types and within-type spread |
| `epsilon`, `epsilon_decay`, `epsilon_min` | 0.4, 0.99, 0.05 | exploration schedule shared by structured policies |
| `ridge`, `als_sweeps`, `refit_every`, `mf_lr`, `ucb_c` | 1.0, 8, 3, 0.05, 2.0 | estimator hyperparameters |
| `seeds` | `range(16)` | random seeds (each is one independent world + run) |
| `algorithms` | `[random, ucb_indep, mf_sgd, swarm_cf]` | policies to evaluate (`random` is always included) |
| `metrics` | `[earned_skill, unseen_pair_skill]` | metrics to report |

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

## 5. Reproducing paper figures

The study drivers (one script per experiment) live in `experiments/` and import this package:

| Script | Produces |
|---|---|
| `experiments/latentswarm_contention.py` | the contention bake-off and the collision / de-confliction figures |
| `experiments/latentswarm_grounded.py` | the robotics-grounded instance (geometry mask + sensing modalities) |
| `experiments/ranksweep.py` | the rank-guess robustness sweep |

Each writes a JSON to `results/pilots/`; `experiments/make_figures.py` turns those into the figures. The
remaining headline sweeps (the broadcast-rate, anytime, scaling, and offer-size figures and the
consolidated bake-off table) currently run from the analytical harness; folding them into this package is
the roadmap item in the developer guide.

## 6. Reproducibility notes
Every reported number is a mean over `seeds` with a bootstrap 95% confidence interval (`bootstrap_ci`).
A run is fully specified by its `RunConfig`; save it with `cfg.to_json(path)` and reload with
`RunConfig.load(path)` to reproduce a study exactly.
