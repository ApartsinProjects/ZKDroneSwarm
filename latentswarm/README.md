# LatentSwarm

A small, pluggable simulation suite for **Zero-Knowledge Multi-Robot Task Allocation (ZK-MRTA)**: a team
of robots must repeatedly choose which task to engage, with **no prior task model, no communication, and
only a partial, per-observer-private, noisy view** of teammates' outcomes, in a **task-scarce** regime
(`n >> T`). The hidden reward is **low-rank** (`R = P Uᵀ` over robot capability traits and task
requirement traits), so a robot can act well on tasks it never engaged by running online collaborative
filtering over the passive broadcast.

LatentSwarm is the simulator behind the paper *"Acting on the Unseen: Communication-Free Collaborative
Filtering for Decentralized Multi-Robot Task Allocation."*

## Design in one sentence
Every component (scenario, policy, metric, visualization) is a **named plugin** selected from a single
`RunConfig`, so an entire study is one config object and the suite is extended by writing a class and
decorating it, with no changes to the runner.

## Install
Pure Python; the only dependencies are `numpy` and `scipy`.

```bash
pip install numpy scipy
# Use from the repo root, where the `latentswarm/` folder is importable as the package `latentswarm`.
```

## Quick start

Python:
```python
from latentswarm import RunConfig
from latentswarm.run import run

out = run(RunConfig())          # default config; prints a per-policy skill table, returns a results dict
print(out["summary"]["swarm_cf"])   # {'earned': (mean, lo, hi), 'unseen': (mean, lo, hi)}
```

Command line:
```bash
python -m latentswarm.run --out results/pilots/latentswarm_main.json
```

## The model

- **Reward.** Robot `i` has a hidden capability vector `p_i ∈ R^d`, task `j` a requirement vector
  `u_j ∈ R^d`; the expected reward is the inner product `R_ij = <p_i, u_j>`, so `R = P Uᵀ` has rank `d`.
  The team does not know `P`, `U`, or `d`; it uses a **guessed rank** `d̂` drawn at random per run.
- **Observation (no communication).** Each robot passively senses a public stream of engagement outcomes,
  but only through a **persistent** per-pair mask `M_ik ~ Bernoulli(rho)` (fixed for the mission,
  `M_ii = 1`), and reads each visible outcome with **independent per-observer noise** `N(0, sigma_obs^2)`,
  so no two robots see the same stream. (A geometry-induced line-of-sight mask is also available; see below.)
- **Interaction.** Each round every robot is offered a menu (all `n` tasks by default, or a random
  size-`c` subset), selects one, and earns its reward. Under **capacity-1 contention** only the first
  robot to pick a task each round succeeds; colliding robots earn nothing. The regime is task-scarce
  (`n >> T`).
- **Metrics.** *Earned (anytime) skill* normalizes mean reward to `(policy - random)/(oracle - random)`
  with the per-round Hungarian (capacity-1) oracle; *unseen-pair skill* measures decision quality on
  tasks a robot never engaged, self-normalized so oracle = 1 and random ~ 0.

## Package layout

| File | Purpose |
|---|---|
| `config.py` | `RunConfig`: every knob (world, observation, geometry, dynamics, scenario, policy, baseline hyperparameters, evaluation). |
| `registry.py` | Name -> class registries plus the `@scenario` / `@algorithm` / `@metric` / `@visualization` decorators. |
| `scenarios.py` | Latent-trait generators: `uniform_cosine` (i.i.d. unit-sphere traits, NO types; the headline paper world), `gaussian_mixture` / `block_cosine` (type/block worlds, kept for parity), `iid_gaussian`, `sensing_coalition` (robotics-grounded sensing modalities), `approx_lowrank` (rank-d + full-rank perturbation, Appendix F). |
| `env.py` | `ZKMRTAEnv`: masking (`persistent` / `per_round` / `line_of_sight`), per-observer noise, offered menus, capacity-1 contention. |
| `algorithms.py` | Core decentralized policies: `random`, `ucb_indep` (structure-free), `mf_sgd`, `swarm_cf` (ours). |
| `baselines.py` | Competitor `@algorithm` drop-ins: `tabular`, `ucb_homo`, `estr`, `swarmcf_batch` (= PTF), `bpmf`, `club` (clustering-of-bandits; discrete-clustering control), `bias_model` (additive popularity, rank&le;2 control). All per-observer (one estimator per robot). |
| `refinements.py` | The **SwarmCF-\* family** (follow-up paper) as `@algorithm` drop-ins: `em_cf` (SwarmCF-B), `ard_em_cf` (SwarmCF-B-ARD), `active_cf` (SwarmCF-X), `coord_cf` (SwarmCF-Xc), `contention_cf` (SwarmCF-D), `contention_ada_cf` (SwarmCF-D+), `choice_cf` (SwarmCF-Ch), `both_cf` (SwarmCF-RC), `unified_cf` (SwarmCF-U). |
| `metrics.py` | `earned_skill` (Hungarian or best-in-subset oracle), `unseen_pair_skill`, `unseen_pair_skill_heldout`, `anytime_trajectory`, `cumulative_regret`, `time_to_competence`, `state_uniqueness`, `effective_rank` (ARD rank self-determination), the Hungarian oracle, `bootstrap_ci`. |
| `run.py` | Config-driven runner and CLI. |
| `sweeps.py` | Config-driven, process-pool-parallel sweep drivers + CLI (`bakeoff` [Table 3], `crossover`, `anytime`, `collab`, `scale_m`, `ranksweep`, `offersize`, `iid_vs_persistent`, `approxrank` [Appendix F], `contention`) emitting the figure-pipeline JSON schema. CLUB + BiasModel are in every comparison. |
| `tests/` | `pytest` smoke and invariant tests. |

## The SwarmCF refinement family (follow-up paper)

`refinements.py` adds the six refinements of the core `swarm_cf` estimator studied in the follow-up
paper, each a registered `@algorithm` drop-in and each a faithful port of an `experiments/pilot_*.py`
prototype (the update rules are taken from there, not reinvented). All are decentralized and
communication-free, run on the same harness, and complete unseen entries (`predict_rows -> [m, n]`):

| Name | Paper name | Refinement |
|---|---|---|
| `em_cf` | SwarmCF-B | confidence-directed exploration: variational Bayesian PMF with a predictive-variance (UCB) rule |
| `ard_em_cf` | SwarmCF-B-ARD | rank self-determination: SwarmCF-B + automatic relevance determination (learns the rank, removing `d̂`) |
| `active_cf` | SwarmCF-X | active exploration: own-count latent-UCB (probe where the swarm's coverage is low) |
| `coord_cf` | SwarmCF-Xc | coordinated exploration: negative-correlated, broadcast-count division of labor (no comms) |
| `contention_cf` | SwarmCF-D | de-confliction: a fixed, private per-task offset that breaks symmetry under capacity-1 contention |
| `contention_ada_cf` | SwarmCF-D+ | de-confliction: a scarcity-gated, loss-self-tuning private offset |
| `choice_cf` | SwarmCF-Ch | the action/choice channel: learn from *which* task a teammate engaged (noise-immune) |
| `both_cf` | SwarmCF-RC | fuse reward + competence-weighted choice in one weighted ALS |
| `unified_cf` | SwarmCF-U | a single method: SwarmCF-B + a loss-gated offset and a loss-gated exploration anneal that activate only on contention |

Every knob is configurable via `RunConfig` (no hard-coded constants); see the user guide for the
field reference. The follow-up's de-confliction sweep (Section 4 / Figure 2) is
`python -m latentswarm.sweeps --which contention`. Effective-rank recovery (Section 5) is the
`effective_rank` metric, read off an `ard_em_cf` policy.

## Documentation
- **[User guide](docs/USER_GUIDE.md)**: configuration reference, recipes (contention, geometry mask,
  grounded instance, offer sizes, masking models), output format, and how to reproduce paper figures.
- **[Developer guide](docs/DEVELOPER_GUIDE.md)**: architecture, the plugin registry, the contracts for
  scenarios / policies / metrics / visualizations with worked examples, the environment contract, testing,
  and the unification roadmap.

## Tests
```bash
pytest latentswarm/tests -q
```

## Status and roadmap
The package powers the paper's contention figures, the rank-robustness sweep, and the robotics-grounded
instance, and now also folds in the analytical sweeps: the competitor baselines (`baselines.py`), the
anytime / regret / time-to-competence / state-uniqueness / held-out-unseen / effective-rank metrics
(`metrics.py`), the `block_cosine` unit-cosine parity scenario, and config-driven sweep drivers
(`sweeps.py`, e.g. `python -m latentswarm.sweeps --which crossover`) that emit the same JSON schema
`experiments/make_figures.py` reads. A 3-seed parity check confirms the package path reproduces the
analytical numbers within overlapping ranges; see the roadmap and parity note in the developer guide.

It also now supports the **follow-up paper** end to end: the `refinements.py` SwarmCF-\* family (the six
refinements above), the `effective_rank` metric, and the `contention` de-confliction sweep, all ported
faithfully from the `experiments/` prototypes and configurable through `RunConfig`. See the developer
guide for the per-refinement prototype provenance and the one interface adaptation (the de-confliction
loss signal) flagged for review.

## Citation
Apartsin, Meshulam, Aperstein. *Acting on the Unseen: Communication-Free Collaborative Filtering for
Decentralized Multi-Robot Task Allocation.* Code and data: https://github.com/ApartsinProjects/ZKDroneSwarm
