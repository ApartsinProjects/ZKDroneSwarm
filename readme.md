# Acting on the Unseen: Zero-Knowledge Swarm Coordination

**Decentralized multi-robot task allocation under minimal assumptions: no prior
knowledge, zero communication, only partial and noisy observation, fully
distributed decisions.**

![Hero: ZK drone swarm engaging distributed targets](docs/hero.png)

> **Interactive tutorial (self-contained, graduate-level, with proofs):**
> https://apartsinprojects.github.io/ZKDroneSwarm/tutorial.html
> **Streamlined paper (v2):**
> https://apartsinprojects.github.io/ZKDroneSwarm/paper_v2.html

---

## The setting in one breath

A swarm of agents must repeatedly choose which task to take, under the LEAST
possible information:

- **No prior knowledge** — no labels, no task models, no known latent structure,
  not even the true rank of the problem.
- **Zero communication** — no messages, no parameter sharing, no coordinator, no
  consensus. Agents only PASSIVELY SENSE a public stream of outcomes.
- **Only partial, noisy observables** — each agent senses a masked, noisy slice of
  the public outcomes (limited detection, not radio transmission).
- **Fully distributed decisions** — every agent decides alone, from its own
  private state.

**The question:** can such a swarm still act intelligently, in particular on tasks
it has never tried? **Yes** — by running online collaborative filtering over the
public outcome stream, with an advantage that is *categorical* (proven), not a few
percent.

## Why it works (one paragraph)

Compatibility between agents and tasks is hidden but low-rank (a few latent factors
explain it), and there are far more tasks than rounds (`n >> T`). A structure-free
learner that estimates a task only from its own attempts is, on any task it never
tried, at the prior-mean error floor, which dominates under starvation.
Collaborative filtering recovers the shared task factors from the public broadcast,
so each agent can complete its OWN value for tasks it never touched. The result is
a per-agent sample-complexity separation of `Theta(d)` (CF) vs `Theta(n)`
(tabular), categorical on unseen pairs (error to 0 vs a constant floor).

## Headline results

- **Acting on the unseen (categorical).** CF acts well on never-observed
  agent-task pairs at every observation density; structure-free learners sit at the
  floor by construction.
- **Onboarding and cold-start.** A new task is onboarded for the whole swarm from
  about `d` shared probes (vs `n`); a new agent with zero history acts from the
  broadcast alone.
- **Anytime / operational.** On reward actually earned over time ("targets
  destroyed by round K"), our online methods dominate at every horizon; per-arm
  bandits stay near random because, with `n >> T`, they never stop exploring.
- **Masking-robust and dominant.** Our online weighted-ALS is robust to
  observation masking (batch-SVD hybrids decay); with a converged estimator our
  methods dominate the strongest prior-art competitor (PTF) on every metric and
  density.
- **Theory (with proofs).** Five results, each matched to an experiment: tabular
  floor; O(d) row completion; anytime separation under starvation; persistent-vs-
  i.i.d. masking dichotomy; additive rank-ceiling.
- **Validated in a real simulator.** Ported into the tabula_drone PettingZoo env
  (spatial, depleting HP, episodic), our method reaches skill ~0.81, beating the
  env's SGD matrix-factorization (~0.25) and UCBIndep (~0.72) and approaching the
  oracle; the advantage also survives approximate-low-rank and nonlinear-reward
  stress tests.

## Methods (all strictly zero-knowledge, fully distributed)

| method | cross-agent signal | note |
|---|---|---|
| RewardCF | teammates' rewards | simple workhorse; anytime-optimal |
| ChoiceZK | teammates' actions only | noise-immune fallback |
| HybridCFconv | probe + SVD warm-start + converged online ALS | best final-policy unseen |
| ActiveCFconv | latent-space UCB with broadcast-count uncertainty bonus | best balanced; collective active exploration |

Each agent runs its own online weighted ALS (precision-weighted; a masked event is
zero precision, not zero value), estimation separated from the decision policy. No
parameters are shared; collaboration emerges only through the public observation
stream.

## Documents

- **Tutorial** (everything, step by step, with proofs and figures):
  `docs/tutorial.html` ([live](https://apartsinprojects.github.io/ZKDroneSwarm/tutorial.html))
- **Paper v2** (streamlined): `docs/paper_v2.html`
  ([live](https://apartsinprojects.github.io/ZKDroneSwarm/paper_v2.html))
- **Paper v1** (full draft): `docs/PAPER_DRAFT.md`
- **Theory with proofs:** `docs/THEORY_FORMAL.md`
- **Zero-knowledge audit:** `docs/ZK_COMPLIANCE.md`
- **Experiment log / data catalogue / backlog / plan:** `docs/PROJECT_LOG.md`,
  `docs/DATA_CATALOGUE.md`, `docs/BACKLOG.md`, `docs/EXPERIMENT_PLAN.md`

## Reproducibility

All numbers come from complete per-seed JSON in `results/pilots/` (registry in
`docs/DATA_CATALOGUE.md`). Regenerate the figures and pages:

```bash
python experiments/make_figures.py     # F2..F11 -> docs/figures/
python experiments/make_table1.py      # comparison table
python experiments/make_tutorial.py    # docs/tutorial.html
python experiments/make_paper_v2.py    # docs/paper_v2.html
```

Key code: world / reward / oracle in `experiments/core.py`; our methods in
`experiments/pilot_noise.py`; baselines in `experiments/pilot_baselines.py`;
experiment harnesses `experiments/pilot_*.py` (compare, crossover, anytime,
channels, iid, scaling, robust, newcomer, active, conv, e8). Every method uses a
guessed rank and broadcast-only inputs; the Oracle is used only to normalize
scores. A separate `tabula_drone/` package holds an earlier PettingZoo/Gymnasium
ZK-MRTA environment and policies.

## Team

**Yigal Meshulam** — researcher, Department of Computer Science

**Alexander Apartsin** — MSc Adviser, Department of Computer Science

## License

MIT
