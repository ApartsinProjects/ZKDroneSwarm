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
Registration happens on import: `latentswarm/__init__.py` imports `scenarios`, `algorithms`,
`baselines`, `refinements`, `metrics` for their decorator side effects, so importing the package
populates the registries. (`refinements` holds the SwarmCF-\* follow-up family; see section 8.)

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
The package powers the contention figures, the rank sweep, and the grounded instance, and now also
folds in the analytical sweeps so the package path reproduces the headline figures. Status of the
unification (all items below are DONE; see the parity note at the end):

1. **`baselines.py`** (DONE): the missing policies as `@algorithm` drop-ins, ported faithfully from
   `experiments/pilot_baselines.py` + `pilot_noise.py` and adapted to the Policy interface:
   `tabular` (own-row eps-greedy), `ucb_homo` (shared arm table), `estr` (explore-then-spectral-commit),
   `swarmcf_batch` (= PTF: probe-then SVD warm-start + SGD-MF finetune), `bpmf` (Bayesian PMF). The
   low-rank ones return a completed `[m,n]` from `predict_rows`; `tabular`/`ucb_homo` return `None`
   (the unseen floor). Imported in `__init__.py` so they register on import. (`SoftImpute`/`BiasModel`/
   `KNNCF` remain available in the analytical harness; port them here the same way if a figure needs them.)
2. **`metrics.py`** additions (DONE): `anytime_trajectory` (per-round cumulative `(real-random)/
   (oracle-random)` + final AUC), `cumulative_regret` (`sum_t (1-skill_t)`), `time_to_competence`
   (rounds to a fraction of oracle), `state_uniqueness` (`1 - mean pairwise cosine` of the per-robot
   predicted reward matrices, via `Policy.per_robot_full()`), and `unseen_pair_skill_heldout` (the
   analytical held-out protocol: fresh size-`min(c,20)` menus over ~120 reps from each robot's
   never-engaged tasks). `earned_skill` also accepts `oracle_mode="best_in_subset"` (the no-contention
   ceiling) in addition to the default Hungarian ceiling.
3. **`sweeps.py`** (DONE) + CLI: config-driven drivers reproducing `crossover` (unseen vs rho),
   `anytime` (trajectory), `collab` (rho incl 0), `scale_m` (team size), `ranksweep` (d_hat),
   `offersize` (c=20 vs c=n), and `iid_vs_persistent`. Each emits the SAME JSON schema (and analytical
   method KEYS: `RewardCF`, `PTF`, `ESTR`, `BPMF`, `UCBIndep`, `Tabular`) that `experiments/make_figures.py`
   reads (`c15_crossover`, `c16_anytime`, `collab`, `scale_m`, `e246_scaling`, `e12_iid_masking`). Run with
   `python -m latentswarm.sweeps --which crossover`. **By default sweeps write to `results/smoke/`**; point
   `--out results/pilots` only for a real (non-smoke) sweep that should feed the figure pipeline.
4. **A parity scenario** (DONE): `block_cosine` replicates `experiments/core.make_world` for `signed=True`
   (the **unit-cosine** world: type centers and per-entity traits L2-normalized so `R = P Uᵀ` is a signed
   cosine in `[-1,1]`, rank `<= min(d, n_types)`, every latent type forced present, within-type jitter =
   `cfg.jitter`, with `cfg.n_types` = core's `K1=K2`, default 10). It reproduces `make_world`'s exact
   `RandomState(seed)` draw order, so its `P,U` are **bit-identical** to `core.make_world(..., signed=True)`.
   `gaussian_mixture` is left as the unnormalized-inner-product world; use `block_cosine` to match the
   analytical numbers.

**Parity status.** A 3-seed parity check (`results/smoke/parity_check.py`) compares the package path
(`block_cosine` + persistent mask + the new metrics) against `experiments/pilot_c11_masking.run_masked`
called in-process, at `m=30, n=240, d=5, n_types=10, T=50, c=20, rho=0.5, sigma_own=0.10, sigma_obs=0.30,
d_hat = RandomState(9000+seed).randint(5,11)`. **Statistical parity holds**: SwarmCF overall 0.629 vs 0.638,
SwarmCF unseen 0.356 vs 0.398, Tabular unseen ~0 vs ~0 (both at the structure-free floor), state-uniqueness
0.724 vs 0.707 (analytical vs package, 3-seed means; all within overlapping ranges). **Bit-identical parity
is not expected** because the two implementations use different RNG stream structures: `run_masked` draws
the per-round candsets, the persistent mask, and the per-(i,k) observation noise in one explicit loop and
evaluates overall/unseen from a single interleaved eval RNG, whereas the package draws candsets and
per-observer noise inside `ZKMRTAEnv._offers`/`_observe` (a different call order) and the held-out / overall
metrics each own an independent eval stream. The per-seed differences this induces (largest on the noisier
unseen metric) average out across seeds; the categorical claims (CF lifts unseen well above the floor;
tabular stays at it; state-uniqueness substantial under masking) reproduce cleanly.

## 8. The follow-up paper: the SwarmCF-\* refinement family (`refinements.py`)

The follow-up paper (`experiments/make_ras_paper2.py` -> `docs/ras_paper2.html`) studies six refinements
of the core `swarm_cf` estimator. All six are implemented as registered `@algorithm` drop-ins in
`refinements.py`, plus the `effective_rank` metric (`metrics.py`) and the `contention` sweep
(`sweeps.py`). Each is a **faithful port** of an `experiments/` prototype: the update rules are mined
from there, not reinvented. Everything is configurable via `RunConfig` (no hard-coded constants; the
refinement knobs live in `config.py`, grouped under "refinements.py").

### Registry names, paper names, and prototype provenance

| Name | Paper name | Refinement | Prototype (read-only source) |
|---|---|---|---|
| `em_cf` | SwarmCF-B | variational Bayesian PMF + predictive-variance UCB exploration | `pilot_noise.EMCF` (`ard=False`) |
| `ard_em_cf` | SwarmCF-B-ARD | `em_cf` + automatic relevance determination (rank self-determination) | `pilot_noise.EMCF` (`ard=True`); `pilot_ardrank.py`, `pilot_ard.py` |
| `active_cf` | SwarmCF-X | own/broadcast-count latent-UCB exploration | `pilot_noise.ActiveCF` |
| `coord_cf` | SwarmCF-Xc | negative-correlated (swarm-count) coordinated exploration | `pilot_noise.CoordCF` |
| `contention_cf` | SwarmCF-D | fixed private per-task offset (capacity-1 de-confliction) | `pilot_noise.ContentionCF`; `pilot_contention.py` |
| `contention_ada_cf` | SwarmCF-D+ | scarcity-gated, loss-self-tuning private offset | `pilot_noise.ContentionAdaptiveCF`; `pilot_contention.py` |
| `choice_cf` | SwarmCF-Ch | the action/choice channel only (noise-immune) | `pilot_noise.ChoiceCF`; `pilot_choiceem.py` |
| `both_cf` | SwarmCF-RC | fuse reward + competence-weighted choice | `pilot_noise.BothCF` |
| `unified_cf` | SwarmCF-U | `em_cf` + loss-gated offset + loss-gated exploration + abundance gate | `pilot_noise.UnifiedCF`; `pilot_unified.py` |

The `effective_rank` metric reads `policy.eff_rank()` (the mean number of ARD-retained columns across
robots). For `ard_em_cf` this is the identifiable rank (<= d), invariant to the guessed `d̂`; for a
non-ARD policy every column survives the flat prior, so it returns the full `d̂`.

### Interface adaptation (one item flagged for review)

The prototypes use a per-drone interface, `select(t, cand)` and `observe(t, choices, revealed,
cand_sets, rvar)`, where the harness reconstructs each learner's view. The package's Policy contract is
`act(obs)` / `observe(obs)` with the class managing all `m` robots. The reconstruction follows the
established `baselines.py` pattern exactly: teammate `k` is observed by `i` iff `obs[i]["sel"][k] !=
NO_OP`, the reading is `obs[i]["rew"][k]`, and the per-observer variance is `_rvar_row(cfg, i, m)`
(`sigma_own^2` on the diagonal, else `sigma_obs^2`; the `line_of_sight` distance-dependent noise is
approximated by `sigma_obs^2` for the precision-weighted methods, the same approximation `baselines.py`
documents).

Two ports needed a decision the prototype interface does not directly translate; both are implemented as
clearly as the prototype allows and flagged here:

1. **De-confliction loss signal (`contention_cf` is unaffected; `contention_ada_cf` and `unified_cf`
   gate on it).** The prototypes detect a lost contest with `choices[idx] == -1`, a flag the separate
   contention harness sets for losers. The package env keeps a colliding robot's own `sel` set and
   zeroes its reward, so that exact flag is not present in `obs`. Two paths are provided:
   - the **dedicated contention sweep** (`sweeps.run_contention_cell`) knows won/lost exactly and calls
     `policy.set_lost(lost_array)` before `observe`, so the gated methods see the SAME signal as the
     prototype (this is the path the follow-up's Figure 2 numbers should use);
   - in the **generic env** (other sweeps, `run`), `_update_loss` falls back to a communication-free
     VISIBLE-CONTENTION proxy: robot `i` counts a loss when it engaged a task and a teammate it can see
     engaged the same task. This is exact for detecting contention on the robot's own pick and monotone
     in the true loss rate, but it is a proxy, not the prototype's exact win/loss. **For human review:**
     if a future study runs `contention_ada_cf` / `unified_cf` through the *generic* env under
     capacity-1 and needs the exact loss rate, expose a per-robot won/lost flag from `ZKMRTAEnv.step`
     (e.g. in `info` or a new `obs` field) and wire `set_lost` to it; the hook is already there.

2. **Choice-channel negative sampling (`choice_cf`, `both_cf`).** The prototype samples "not-chosen"
   negative pseudo-targets from the teammate's own offer (`cand_sets[k]`). In the communication-free
   env a robot does not see teammates' offers, so with `choice_within=True` (default) negatives are
   drawn from the OBSERVER's own offer as a proxy; `choice_within=False` draws them globally (the
   prototype's strict-ZK option). The chosen target itself (the positive) is always exact; only the
   negatives use the proxy offer. This does not affect the noise-immunity claim (the positive choice is
   what carries the signal) but is a faithfulness note worth knowing.

Beyond these two items the ports are line-for-line: the variational EM sweep (`_EMCF._refit`), the ARD
`alpha` M-step, the predictive/collective variance, the precision-weighted ALS (`_ALSReward._als`), the
choice pseudo-observation + competence ramp (`_ChoiceBase`), the fixed/adaptive offset and its scarcity
gate, and the unified loss-gated exploration anneal + abundance gate all reproduce the prototype math.

### Adding to or tuning a refinement
The refinements reuse two internal scaffolds: `_EMCF` (Bayesian PMF; subclass and set `_ard`) and
`_ALSReward` (precision-weighted online ALS with a pluggable `_select_row`; the exploration variants and
the contention/choice families build on it). To add a new exploration rule, subclass `_ALSReward` and
implement `_select_row(i, off)`. To add a confidence/rank variant, subclass `_EMCF`. Keep the
constructor signature `(cfg, m, n, d_guess, seed=0)` and read every constant from `cfg`. Add a test to
`tests/test_latentswarm.py` (see `test_refinements_run_and_predict_rows`,
`test_ard_self_determines_rank_below_guess`, `test_contention_sweep_cell_deconflicts`).
