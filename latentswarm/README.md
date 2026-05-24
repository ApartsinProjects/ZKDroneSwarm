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
| `config.py` | `RunConfig`: every knob (world, observation, geometry, dynamics, scenario, policy, evaluation). |
| `registry.py` | Name -> class registries plus the `@scenario` / `@algorithm` / `@metric` / `@visualization` decorators. |
| `scenarios.py` | Latent-trait generators: `gaussian_mixture` (block model), `iid_gaussian`, `sensing_coalition` (robotics-grounded sensing modalities). |
| `env.py` | `ZKMRTAEnv`: masking (`persistent` / `per_round` / `line_of_sight`), per-observer noise, offered menus, capacity-1 contention. |
| `algorithms.py` | Decentralized policies: `random`, `ucb_indep` (structure-free), `mf_sgd`, `swarm_cf` (ours). |
| `metrics.py` | `earned_skill`, `unseen_pair_skill`, the Hungarian oracle, `bootstrap_ci`. |
| `run.py` | Config-driven runner and CLI. |
| `tests/` | `pytest` smoke and invariant tests. |

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
The package currently powers the paper's contention figures, the rank-robustness sweep, and the
robotics-grounded instance. Porting the remaining analytical sweeps (the method bake-off and the
broadcast-rate / scaling / offer-size / masking sweeps) into the package, so a single command reproduces
every paper figure, is in progress; see the roadmap in the developer guide.

## Citation
Apartsin, Meshulam, Aperstein. *Acting on the Unseen: Communication-Free Collaborative Filtering for
Decentralized Multi-Robot Task Allocation.* Code and data: https://github.com/ApartsinProjects/ZKDroneSwarm
