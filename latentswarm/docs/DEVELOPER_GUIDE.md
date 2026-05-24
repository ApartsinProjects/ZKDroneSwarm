# LatentSwarm developer guide

How the package is put together and how to extend it. For running studies see the
[user guide](USER_GUIDE.md).

## 1. Architecture

One `RunConfig` selects, by name, a **scenario** (generates the latent traits), a set of **algorithms**
(decentralized policies), and a set of **metrics**, which the **runner** drives through the
**environment**:

```
RunConfig ──► build_scenario ──► P, U  ─┐
          ──► get(ALGORITHMS, name) ────┤
          ──► ZKMRTAEnv(cfg, P, U) ─────┴─► run_mission ──► metrics ──► summary (+ bootstrap CIs)
```

Everything pluggable is registered in `registry.py`:
```python
SCENARIOS, ALGORITHMS, METRICS, VISUALIZATIONS   # name -> class/callable dicts
@scenario(name) @algorithm(name) @metric(name) @visualization(name)   # decorators that register
get(reg, name)                                   # lookup with a helpful error
```
Registration happens on import: `latentswarm/__init__.py` imports `scenarios`, `algorithms`, `metrics`
for their decorator side effects, so importing the package populates the registries.

## 2. The environment contract (`env.py`)

`ZKMRTAEnv(cfg, P, U, d_guess, seed)`:
- `reset() -> obs` : initializes the persistent mask (and, for `line_of_sight`, the positions and the
  distance-dependent per-observer noise), returns the first observation.
- `step(actions) -> (obs, rewards, info)` : `actions` is `int[m]` (a task index or `ZKMRTAEnv.NO_OP`).
  Resolves capacity-1 contention in a random order, returns per-robot `rewards[m]` and
  `info = {"t": round, "collisions": count}`. For `mask_mode="per_round"` the mask is redrawn here.
- `engaged` : `list[set]`, the tasks each robot has engaged (used by the unseen-pair metric).
- `NO_OP = -1`.

**Observation schema.** `obs` is a list of `m` dicts, one per robot `i`:
```python
{
  "offer": bool[n],   # tasks offered to robot i this round
  "sel":   int[m],    # each teammate's last selected task, or NO_OP if not visible / no-op
  "rew":   float[m],  # each visible teammate's last outcome, with per-observer noise (NO_OP -> 0)
  "i":     int,       # the robot's own index
}
```
A masked teammate appears as `sel[k] = NO_OP`; the robot's own entry `k == i` is its own (lower-noise)
reading. This dict is the **only** information a policy may use: there is no communication.

## 3. Component contracts and how to add one

### Add a policy (algorithm)
Subclass `Policy` (`algorithms.py`) and register it. A policy manages all `m` robots internally (one
independent per-robot estimator each) and exposes:
```python
act(obs)        -> int[m]          # action per robot (a task index in its offer, or NO_OP)
observe(obs)    -> None            # update from the post-step observation (the broadcast)
predict_rows()  -> [m, n] | None   # each robot's predicted reward row (None => structure-free, no unseen skill)
```
Constructor signature is `(cfg, m, n, d_guess, seed=0)`. `self._offered(obs_i)` returns the offered task
indices. Example, a popularity / bias baseline:
```python
from latentswarm.registry import algorithm
from latentswarm.algorithms import Policy, NO_OP
import numpy as np

@algorithm("popularity")
class Popularity(Policy):
    """Global per-task mean over everything the robot sees (rank-1 popularity; no personalization)."""
    name = "popularity"
    def __init__(self, cfg, m, n, d_guess, seed=0):
        super().__init__(cfg, m, n, d_guess, seed)
        self.sum = np.zeros(n); self.cnt = np.zeros(n)
    def act(self, obs):
        a = np.full(self.m, NO_OP, dtype=int)
        mean = np.where(self.cnt > 0, self.sum / np.maximum(self.cnt, 1), 0.0)
        for i in range(self.m):
            off = self._offered(obs[i])
            if off.size:
                a[i] = int(off[int(np.argmax(mean[off]))])
        return a
    def observe(self, obs):
        for i in range(self.m):
            for k in range(self.m):
                j = obs[i]["sel"][k]
                if j != NO_OP:
                    self.sum[j] += obs[i]["rew"][k]; self.cnt[j] += 1
    def predict_rows(self):
        mean = np.where(self.cnt > 0, self.sum / np.maximum(self.cnt, 1), 0.0)
        return np.tile(mean, (self.m, 1))   # same row for every robot (no personalization)
```
Then `run(RunConfig(algorithms=["random", "swarm_cf", "popularity"]))`. The shared low-rank scaffolding
`_LowRankPolicy` (eps-greedy action + per-robot factors `self.P[i]`, `self.U[i]` + `predict_rows`) is the
base for `mf_sgd` and `swarm_cf`; reuse it for any factorization-based policy and just implement
`observe`. `SwarmCF._als` is a vectorized weighted-ridge ALS sweep worth reusing.

### Add a scenario
Subclass `Scenario` (`scenarios.py`), return `(P[m,d], U[n,d])` so `R = P Uᵀ` is rank `d`:
```python
@scenario("my_world")
class MyWorld(Scenario):
    name = "my_world"
    def generate(self):
        c, rng = self.cfg, self.rng
        return rng.normal(0, 1, (c.m, c.d)), rng.normal(0, 1, (c.n, c.d))
```

### Add a metric
Subclass `Metric` (`metrics.py`), implement `compute(**kw)`; the runner passes what it has
(`mean_reward, random_mean, oracle_mean` for earned skill; `P, U, pred_rows, engaged, rng` for the
unseen-pair metric). `hungarian_oracle_per_step(P, U)` is the capacity-1 ceiling; `bootstrap_ci(xs)`
returns `(mean, lo, hi)`.

### Add a visualization
`@visualization("name")` registers a plotting callable; keep plotting out of the core loop.

## 4. The run loop (`run.py`)
`run_mission(env, policy, T)` runs one mission and returns `(per_round_rewards, env.engaged)`. `run(cfg)`
loops over seeds: builds one world per seed (shared across policies for a fair comparison), draws `d̂` via
`cfg.rank_for_run`, runs each policy, computes earned + unseen skill against the same per-seed oracle, and
returns per-seed lists plus bootstrap-CI summaries.

## 5. Conventions and invariants
- **No communication.** A policy may read only its own `obs` dicts. Do not share state between the
  per-robot estimators except through what the env exposes in the broadcast.
- **RNG.** Worlds use `RandomState(seed)`; policies are seeded `1000 + seed`; keep new randomness seeded
  and documented so runs are reproducible.
- **`predict_rows`.** Return `None` for a genuinely structure-free policy (it then scores 0 on unseen-pair
  skill by construction, the categorical floor); return an `[m, n]` matrix for any method that completes
  unseen entries.

## 6. Testing
```bash
pytest latentswarm/tests -q
```
`tests/test_latentswarm.py` covers smoke runs and invariants (registries populate, the env respects the
mask and capacity-1, structure-free policies stay at the unseen floor, SwarmCF beats the floor). Add a
test when you add a component.

## 7. Roadmap: a single codebase for the whole paper
The package currently produces the contention figures, the rank sweep, and the grounded instance. Folding
the remaining analytical sweeps in (so one command reproduces every paper figure) needs, in priority order:

1. **`baselines.py`** (new): the missing policies as `@algorithm` drop-ins (`Tabular`, `UCBHomo`, `ESTR`,
   `PTF` = SwarmCF-batch, `BPMF`, and optionally `SoftImpute`, `BiasModel`, `KNNCF`). This plus
   anytime/regret metrics reproduces the bake-off table and the broadcast-rate / anytime / scaling figures.
2. **`metrics.py`** additions: anytime cumulative-reward trajectory + AUC, cumulative regret,
   time-to-competence, and state-uniqueness; plus a held-out unseen-pair eval matching the analytical
   protocol (fresh size-`min(c,20)` offers over repeats).
3. **`sweeps.py`** (new): config-driven drivers (broadcast-rate crossover, anytime, collaboration including
   `rho=0`, scale-with-`m`, offer-size, persistent-vs-i.i.d., recovery validation) emitting the same JSON
   schema `make_figures.py` already reads.
4. **A parity scenario.** Important caveat for a "matches the analytical harness" claim: the current
   `gaussian_mixture` scenario is an **unnormalized inner-product** world, whereas the analytical harness
   (`experiments/core.make_world`) builds a **unit-cosine** world (traits L2-normalized, reward in
   `[-1,1]`, every latent type forced present). A faithful `block_cosine` scenario replicating that
   construction is the linchpin for statistical (within-CI) parity; bit-identical parity is not expected
   because the two implementations use different RNG stream structures and ALS vectorization. A small
   2-3-seed parity check (compare SwarmCF overall + unseen skill and the structure-free floor against the
   analytical numbers) is the acceptance test, not a full rerun.
