---
title: System Integration Guide - Latent ZK-MRTA Benchmark
description: How to assemble and orchestrate benchmark components to run experiments
last_updated: 2026-04-12
---

# System Integration Guide: Latent ZK-MRTA Benchmark

This guide documents the **patterns and architecture** for assembling the benchmark components into an experiment harness. It bridges the gap between understanding individual components (technical specification) and using the system (usage guide) by explaining **how the pieces fit together**.

**Audience:** Researchers building custom experiment harnesses, contributors modifying orchestration logic, or anyone needing to understand system integration without reading `main_zk_mrta.py` line-by-line.

**Related documentation:**
- **Technical specification**: [`technical-overview.md`](policies/classic-collaborative-filtering/technical-overview.md) — Environment mechanics, policy algorithms, implementation details
- **Usage guide**: [`usage-guide.md`](usage-guide.md) — How to configure and run experiments

---

## 1. Component Overview

### Architecture Diagram

```
Configuration (JSON)
        ↓
LatentScenarioBuilder
        ↓
(drones_config, targets_config)
        ↓
DroneEngageLatentMRTA (environment)
        ↓
Policy (random / oracle / matrix_factorization)
        ↓
Episode Loop (reset → step → update → terminate)
        ↓
EnvironmentLogger + MetricsManager
        ↓
Artifacts (episodes, learning_state, metrics, analysis)
```

### Component Responsibilities

| Component | Responsibility | Reference |
|-----------|---------------|-----------|
| **`LatentScenarioBuilder`** | Generate scenario configs with hidden latent vectors | Technical overview §1 |
| **`DroneEngageLatentMRTA`** | Execute episodes, manage environment state | Technical overview §7 |
| **`MatrixFactorizationPolicy`** | Learn embeddings, select actions | Technical overview §5 |
| **`RandomPolicy`** | Baseline random action selection | — |
| **`OptimalAssignmentOracle`** | Optimal assignment using ground truth | — |
| **`EnvironmentLogger`** | Orchestrate logging (when to save what) | — |
| **`MetricsManager`** | Aggregate episode metrics, select best episode | — |
| **`MultiAgentPolicy`** | Wrapper for decentralized per-agent policies | — |

### Data Flow

1. **Configuration** → Parsed into dataclass structures
2. **Builder** → Generates `drones_config` and `targets_config` with latent vectors
3. **Environment** → Initialized with configs, executes episodes
4. **Policy** → Observes, acts, learns from public interaction stream
5. **Logger** → Captures episode traces, learning states, metrics
6. **Metrics** → Aggregated across episodes, best episode selected

---

## 2. Initialization Sequence

### Pattern: Config → Builder → Environment → Policies

The initialization follows a strict dependency order to ensure reproducibility and proper component wiring.

```python
# Step 1: Load configuration
config = load_config("config/scenario.json")

# Step 2: Build scenario (deterministic given seed)
builder = LatentScenarioBuilder(
    world_size=config.world.size,
    config=config.latent_world,
    target_hp=config.targets.target_hp,
    seed=config.seed
)
builder.with_drones(
    count=config.drones.count,
    region=config.drones.region,
    min_distance_between_drones=config.drones.min_distance_between_drones
)
builder.with_targets(
    count=config.targets.count,
    region=config.targets.region,
    min_distance_from_drones=config.targets.min_distance_from_drones,
    min_distance_between_targets=config.targets.min_distance_between_targets
)
drones_config, targets_config = builder.build()

# Step 3: Create environment (single instance, reused across policies)
env = DroneEngageLatentMRTA(
    world_size=config.world.size,
    max_steps=config.environment.max_steps,
    drones_config=drones_config,
    targets_config=targets_config,
    scenario_id=config.environment.scenario_id,
    reward_noise=config.collaborative_filtering.reward_noise,
    observation_noise=config.collaborative_filtering.observation_noise,
    target_hp=config.targets.target_hp,
    builder=builder,  # Reference for re-generation if needed
    latent_world=latent_world_dict  # Metadata for logging
)

# Step 4: Create all policies upfront (not per-episode)
policies = create_all_policies(config, drones_config, num_targets)
# Returns: {"random": RandomPolicy(...), "matrix_factorization_cf": MultiAgentPolicy(...), ...}
```

### Key Decisions

**Why single environment instance?**
- Scenario is fixed (same drones, targets, latent vectors)
- Only policies vary across runs
- Ensures fair comparison (all policies face identical scenario)

**Why create policies upfront?**
- Learning policies accumulate knowledge across episodes
- Creating fresh policy per episode would reset learned state
- Allows policy-level initialization (e.g., seed distribution for decentralized agents)

**Why separate builder from environment?**
- **Separation of concerns**: Builder generates configs, environment executes episodes
- **Reusability**: Builder can generate multiple scenarios without environment coupling
- **Testability**: Can test scenario generation independently of environment mechanics

---

## 3. Policy Lifecycle Management

### Pattern: Instantiate → Run Episodes → Soft Reset → Repeat

Policies have two reset modes that serve different purposes:

```python
for policy_type, policy in policies.items():
    # Determine episode count based on policy type
    num_episodes = 1 if policy.is_deterministic else config.environment.num_episodes
    
    for episode_num in range(1, num_episodes + 1):
        # Soft reset: preserve learned state, reset episode-specific counters
        if episode_num > 1:
            policy.soft_reset()
        
        # Run episode
        episode_seed = config.seed + episode_num  # Reproducible env randomness
        metrics = run_episode(env, policy, episode_num, seed=episode_seed)
        
        # Log learning state (if applicable)
        if hasattr(policy, 'get_learning_state'):
            learning_state = policy.get_learning_state()
            logger.save_learning_state(learning_state, episode_num)
```

### Policy Interface Contract

```python
class IPolicy:
    is_deterministic: bool  # True = run 1 episode, False = run N episodes
    
    def select_actions(self, obs: Dict, infos: Dict) -> Dict[str, int]:
        """Return action dict: {agent_id: action}"""
        pass
    
    def update(self, obs: Dict) -> None:
        """Update policy state after environment step (learning happens here)"""
        pass
    
    def soft_reset(self) -> None:
        """Reset for new episode (preserve learned knowledge)"""
        pass
    
    def reset(self) -> None:
        """Full reset (reinitialize all state)"""
        pass
    
    def get_learning_state(self) -> Optional[Dict]:
        """Return learning state for logging (optional)"""
        return None
```

### Reset Semantics

| Method | When Called | What It Resets | What It Preserves |
|--------|-------------|----------------|-------------------|
| `soft_reset()` | Between episodes | Step counters, episode-specific state | P/U matrices, learned knowledge, epsilon continues decaying |
| `reset()` | New experiment run | Everything (P/U matrices, epsilon, counters) | Nothing (full reinitialization) |

**Design rationale:**
- `soft_reset()` enables **cumulative learning** across episodes
- Epsilon decay continues across episodes (exploration → exploitation over time)
- Full `reset()` used only when starting completely fresh experiment

### Seed Management

```python
episode_seed = config.seed + episode_num
```

**Why this pattern?**
- **Reproducibility**: Same config.seed produces same episode sequence
- **Variation**: Each episode has different env randomness (action processing order, noise)
- **Learning**: Policy state evolves across episodes despite env randomness

---

## 4. Episode Execution Loop

### Pattern: Reset → Step → Update → Terminate

The canonical episode loop follows PettingZoo's parallel execution model:

```python
def run_episode(env, policy, episode_num, seed):
    # Phase 1: Reset environment
    obs, infos = env.reset(seed=seed)
    
    # Phase 2: Bind diagnostics provider (for oracle policies)
    bind_diagnostics_provider(policy, lambda: env.diagnostics.to_dict())
    
    # Phase 3: Initialize tracking
    total_rewards = {agent_id: 0.0 for agent_id in env.agents}
    step_count = 0
    done = False
    
    # Phase 4: Episode loop
    while not done:
        step_count += 1
        
        # Policy selects actions for all agents
        actions = policy.select_actions(obs, infos)
        # Returns: {"drone_0": 5, "drone_1": 12, ...}
        
        # Environment step (parallel execution)
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Policy update (learns from new observations)
        policy.update(obs)
        
        # Track cumulative rewards
        for agent_id in env.agents:
            total_rewards[agent_id] += rewards[agent_id]
        
        # Check termination (all agents have same termination state)
        reference_agent = env.agents[0]
        done = terminations[reference_agent] or truncations[reference_agent]
    
    # Phase 5: Extract metrics from final diagnostics
    diagnostics = env.diagnostics.to_dict()
    metrics = EpisodeMetrics(
        episode=episode_num,
        steps=step_count,
        targets_neutralized=diagnostics["cumulative_neutralizations"],
        total_ammo_used=sum(diagnostics["ammo_used"].values()),
        total_net_damage=diagnostics["net_damage"],
        total_gross_damage=diagnostics["total_gross_damage"],
        total_overkill=sum(sum(event.values()) for event in overkill_events),
        total_collisions=total_collisions,
        # ... other metrics
    )
    
    return metrics
```

### Key Patterns

**Diagnostics binding:**
```python
bind_diagnostics_provider(policy, lambda: env.diagnostics.to_dict())
```
- Oracle policies need access to ground truth (latent vectors, true compatibility)
- Binding provides access without polluting agent observations
- Lambda ensures fresh diagnostics on each access

**Uniform action selection:**
```python
actions = policy.select_actions(obs, infos)
```
- Single method returns actions for **all agents**
- Handles both single-policy (random, oracle) and multi-agent (MF) cases
- `MultiAgentPolicy` wrapper delegates to per-agent policies internally

**Post-step update:**
```python
policy.update(obs)
```
- Called **after** environment step, **before** next action selection
- MF policy processes public interaction stream here (updates P and U matrices)
- Observation contains: `selected_targets`, `observed_rewards`, `target_was_active_at_engagement`

**Termination check:**
```python
done = terminations[reference_agent] or truncations[reference_agent]
```
- All agents have identical termination state (parallel env property)
- Can use any agent as reference
- Termination: all targets neutralized; Truncation: max steps reached

---

## 5. Multi-Policy Orchestration

### Pattern: Sequential Policy Runs with Isolated State

Policies are executed sequentially to ensure fair comparison on identical scenarios:

```python
for policy_type, policy in policies.items():
    # Start policy run (logging context)
    logger.start_policy(policy_type, is_deterministic=policy.is_deterministic)
    
    # Run episodes
    policy_metrics = []
    num_episodes = 1 if policy.is_deterministic else config.environment.num_episodes
    
    for episode_num in range(1, num_episodes + 1):
        # Soft reset for episodes 2+
        if episode_num > 1:
            policy.soft_reset()
        
        # Run episode
        episode_seed = config.seed + episode_num
        metrics = run_episode(env, policy, episode_num, seed=episode_seed)
        policy_metrics.append(metrics)
        
        # Log episode artifacts
        logger.log_step(...)  # Per-step trace
        logger.log_metrics(metrics)  # Episode summary
        
        # Log learning state (if applicable)
        if not policy.is_deterministic:
            learning_state = policy.get_learning_state()
            logger.save_learning_state(learning_state, episode_num)
    
    # Save policy artifacts (writes all episodes to disk)
    result = logger.save_policy_episodes()
    
    # Compute policy summary
    summary = metrics_manager.calc_total_episodes_metrics(policy_metrics)
```

### Key Patterns

**Sequential execution:**
- Policies run one after another (not in parallel)
- Each policy gets fresh environment state
- Ensures identical scenario for all policies

**Isolated state:**
- Each policy maintains independent learned state
- No cross-policy contamination
- Fair comparison (policies don't see each other's learning)

**Shared environment:**
- Same `env` instance used for all policies
- Same scenario (drones, targets, latent vectors)
- Only randomness: action processing order, noise (controlled by episode seed)

**Logging orchestration:**
- Logger manages when to persist artifacts
- Episode traces saved per-episode
- Learning states saved per-episode (MF only)
- Metrics aggregated and saved per-policy

---

## 6. Logging Architecture

### Three-Tier Logging Structure

The logging system separates concerns across three artifact types:

```
logs/{scenario_id}/{policy_type}/
├── episodes/
│   ├── episode_01.json  ← Tier 1: Per-step trace
│   ├── episode_02.json
│   └── ...
├── learning_state/
│   ├── state_ep01.json  ← Tier 2: Learning snapshots
│   ├── state_ep02.json
│   └── ...
├── analysis/
│   └── analysis_ep{best}.json  ← Tier 3: Engagement analysis
└── metrics.json  ← Tier 3: Aggregated metrics
```

### Tier 1: Per-Step Episode Trace

**Purpose:** Full episode replay capability

```python
logger.start_episode(env, reset_info, seed, episode_num, total_episodes)

for step in episode:
    logger.log_step(
        step_num=step_count,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        info=diagnostics
    )

logger.end_episode(total_rewards, done_reason)
logger.persist_episode_outputs(episode_num, steps)
```

**Content:**
- Initial environment state (drones, targets, latent vectors)
- Per-step: actions, rewards, observations, diagnostics
- Final: total rewards, done reason

**Use case:** Offline analysis, visualization, debugging

### Tier 2: Learning State Snapshots

**Purpose:** Track policy learning progression

```python
learning_state = policy.get_learning_state()
# Returns: {
#   "agent_idx": 0,
#   "P": [[...], ...],  # (num_agents, latent_dim)
#   "U": [[...], ...],  # (latent_dim, num_targets)
#   "epsilon": 0.025,
#   "match": {"predicted_rewards": [...], "ranked_targets": [...]}
# }

logger.save_learning_state(
    episode_state=learning_state,
    episode_num=episode_num,
    num_agents=policy.num_agents,
    num_targets=policy.num_targets,
    latent_dim=policy.latent_dim
)
```

**Content:**
- P and U matrices (learned embeddings)
- Epsilon (current exploration rate)
- Predicted rewards and target rankings
- Integration matrix (if enabled)

**Use case:** Visualize learned structure, track convergence, compare learned vs true latent vectors

**Post-processing:**
- t-SNE enrichment (if enabled): Projects embeddings to 2D
- Saves `state_epXX_enriched.json` with t-SNE coordinates

### Tier 3: Aggregated Metrics

**Purpose:** Policy-level performance summary

```python
logger.log_metrics(episode_metrics)  # Called per-episode
result = logger.save_policy_episodes()  # Called after all episodes

# metrics.json contains:
# {
#   "episodes": [
#     {"episode": 1, "steps": 23, "targets_neutralized": 27, ...},
#     {"episode": 2, "steps": 21, "targets_neutralized": 27, ...},
#     ...
#   ]
# }
```

**Content:**
- Per-episode metrics (steps, ammo, overkill, efficiency)
- Best episode number (by damage efficiency)

**Use case:** Compare policies, track learning progress, identify best episode

### When Artifacts Are Persisted

| Artifact | Persistence Timing | Trigger |
|----------|-------------------|---------|
| Episode trace | After episode completes | `persist_episode_outputs()` |
| Learning state | After episode completes | `save_learning_state()` |
| Metrics | After all episodes complete | `save_policy_episodes()` |
| Analysis | After all episodes complete | Generated for best episode only |

### Best-Episode Selection

```python
# Logger tracks best episode by damage efficiency
result = logger.save_policy_episodes()
best_episode_num = result["best_episode_num"]

# Analysis file generated only for best episode
analysis_path = f"analysis/analysis_ep{best_episode_num:02d}.json"
```

**Selection criterion:** Highest damage efficiency (net damage / gross damage)

**Rationale:** Damage efficiency indicates coordination quality (low overkill, efficient target selection)

---

## 7. Policy Factory Pattern

### Why a Factory?

The factory pattern centralizes policy creation logic and handles:
- Hyperparameter injection from configuration
- Seed distribution for decentralized policies
- Uniform interface across policy types

### Pattern

```python
def create_policy(policy_type, config, drones_config, num_targets):
    if policy_type == "random":
        return RandomPolicy(
            seed=config.seed,
            allow_noop=config.policy.allow_noop
        )
    
    elif policy_type == "max_damage_oracle":
        return OptimalAssignmentOracle(
            seed=config.seed,
            allow_noop=config.policy.allow_noop
        )
    
    elif policy_type == "matrix_factorization_cf":
        # Extract hyperparameters from config
        mf_cfg = config.collaborative_filtering.matrix_factorization_cf
        
        # Create per-agent policies (true decentralization)
        num_agents = len(drones_config)
        policies = {}
        
        for agent_idx in range(num_agents):
            agent_id = f"drone_{agent_idx}"
            policies[agent_id] = MatrixFactorizationPolicy(
                num_targets=num_targets,
                agent_idx=agent_idx,
                num_agents=num_agents,
                latent_dim=mf_cfg.latent_dim,
                learning_rate=mf_cfg.learning_rate,
                lambda_reg=mf_cfg.lambda_reg,
                epsilon=mf_cfg.epsilon,
                epsilon_decay=mf_cfg.epsilon_decay,
                epsilon_min=mf_cfg.epsilon_min,
                anti_signal_weight=mf_cfg.anti_signal_weight,
                use_integration_matrix=mf_cfg.use_integration_matrix,
                seed=config.seed + agent_idx  # Unique seed per agent
            )
        
        # Wrap in MultiAgentPolicy for uniform interface
        return MultiAgentPolicy(policies)
    
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
```

### Key Patterns

**Config-driven hyperparameters:**
- All hyperparameters come from `config`, not hardcoded
- Enables easy experimentation (change config, not code)
- Single source of truth for experiment configuration

**Seed derivation:**
```python
seed=config.seed + agent_idx
```
- Ensures reproducibility (same config.seed → same agent seeds)
- Provides per-agent randomness (different agents have different RNG states)
- Maintains decentralization (agents don't share RNG)

**Wrapper pattern:**
```python
return MultiAgentPolicy(policies)
```
- Provides uniform `select_actions()` / `update()` interface
- Hides per-agent delegation from caller
- Allows single-policy and multi-agent policies to be used interchangeably

### MultiAgentPolicy Wrapper

```python
class MultiAgentPolicy:
    def __init__(self, policies: Dict[str, MatrixFactorizationPolicy]):
        self.policies = policies
        self.is_deterministic = False
    
    def select_actions(self, obs: Dict, infos: Dict) -> Dict[str, int]:
        actions = {}
        for agent_id, agent_policy in self.policies.items():
            actions[agent_id] = agent_policy.select_action(obs[agent_id])
        return actions
    
    def update(self, obs: Dict) -> None:
        for agent_id, agent_policy in self.policies.items():
            agent_policy.update_from_observation(obs[agent_id])
    
    def soft_reset(self) -> None:
        for agent_policy in self.policies.values():
            agent_policy.soft_reset()
    
    def get_learning_state(self) -> Dict:
        # Return first agent's state (all agents have same structure)
        first_agent = next(iter(self.policies.values()))
        return first_agent.get_learning_state()
```

**Design rationale:**
- Decouples multi-agent coordination from policy interface
- Allows per-agent policies to focus on single-agent logic
- Simplifies episode loop (treats all policies uniformly)

---

## 8. Metrics Aggregation

### Pattern: Collect Per-Episode → Aggregate → Select Representative

```python
class MetricsManager:
    def calc_total_episodes_metrics(
        self, 
        episode_metrics: List[EpisodeMetrics]
    ) -> PolicyRunSummary:
        # Compute averages
        avg_steps = mean([m.steps for m in episode_metrics])
        avg_targets = mean([m.targets_neutralized for m in episode_metrics])
        avg_ammo = mean([m.total_ammo_used for m in episode_metrics])
        avg_overkill = mean([m.total_overkill for m in episode_metrics])
        avg_reward = mean([m.total_reward for m in episode_metrics])
        avg_dmg_eff = mean([m.dmg_eff for m in episode_metrics])
        
        # Compute success rate
        successful_episodes = [m for m in episode_metrics if m.done_reason == "all_targets_neutralized"]
        success_rate = len(successful_episodes) / len(episode_metrics) * 100
        
        # Select representative episode (best damage efficiency)
        representative_episode = max(episode_metrics, key=lambda m: m.dmg_eff)
        
        return PolicyRunSummary(
            avg_steps=avg_steps,
            avg_targets=avg_targets,
            avg_ammo=avg_ammo,
            avg_overkill=avg_overkill,
            avg_reward=avg_reward,
            avg_dmg_eff=avg_dmg_eff,
            success_rate=success_rate,
            representative_episode=representative_episode,
            ammo_eff=representative_episode.ammo_eff,
            dmg_eff=representative_episode.dmg_eff
        )
```

### Representative Episode Selection

**Criterion:** Highest damage efficiency

```python
representative_episode = max(episode_metrics, key=lambda m: m.dmg_eff)
```

**Rationale:**
- Damage efficiency = `net_damage / gross_damage`
- High efficiency indicates good coordination (low overkill, efficient target selection)
- Best episode for detailed analysis (engagement patterns, learned structure)

**Usage:**
- Analysis file generated only for representative episode
- Engagement matrix shows which drones engaged which targets
- Reveals coordination patterns and policy behavior

### Efficiency Calculations

Computed in `EpisodeMetrics.__post_init__()`:

```python
@dataclass
class EpisodeMetrics:
    # ... fields
    
    def __post_init__(self):
        # Ammo efficiency: targets per shot
        self.ammo_eff = (
            self.targets_neutralized / self.total_ammo_used
            if self.total_ammo_used > 0 else 0.0
        )
        
        # Damage efficiency: useful damage / total damage
        self.dmg_eff = (
            self.total_net_damage / self.total_gross_damage
            if self.total_gross_damage > 0 else 0.0
        )
        
        # Shots per target: inverse of ammo efficiency
        self.shots_per_target = (
            self.total_ammo_used / self.targets_neutralized
            if self.targets_neutralized > 0 else float('inf')
        )
        
        # Total reward across all agents
        self.total_reward = sum(self.agent_rewards.values())
```

---

## 9. Extension Points

### Adding a New Policy

**Step 1:** Implement `IPolicy` interface

```python
class MyCustomPolicy:
    is_deterministic: bool = False  # or True
    
    def select_actions(self, obs: Dict, infos: Dict) -> Dict[str, int]:
        # Your action selection logic
        actions = {}
        for agent_id in obs.keys():
            # ... compute action for this agent
            actions[agent_id] = selected_action
        return actions
    
    def update(self, obs: Dict) -> None:
        # Your learning logic (if applicable)
        pass
    
    def soft_reset(self) -> None:
        # Reset episode-specific state, preserve learned knowledge
        pass
    
    def reset(self) -> None:
        # Full reinitialization
        pass
    
    def get_learning_state(self) -> Optional[Dict]:
        # Return learning state for logging (optional)
        return None
```

**Step 2:** Add factory case

```python
def create_policy(policy_type, config, drones_config, num_targets):
    # ... existing cases
    
    elif policy_type == "my_custom_policy":
        return MyCustomPolicy(
            num_targets=num_targets,
            seed=config.seed,
            # ... your hyperparameters from config
        )
```

**Step 3:** Add to configuration

```json
"policy": {
  "type": ["random", "my_custom_policy"]
}
```

### Adding Custom Metrics

**Step 1:** Extend `EpisodeMetrics` dataclass

```python
@dataclass
class EpisodeMetrics:
    # ... existing fields
    my_custom_metric: float = 0.0
```

**Step 2:** Compute in `run_episode()`

```python
def run_episode(env, policy, episode_num, seed):
    # ... episode loop
    
    # Extract from diagnostics or compute from episode data
    my_custom_value = compute_my_metric(diagnostics, total_rewards)
    
    metrics = EpisodeMetrics(
        # ... existing fields
        my_custom_metric=my_custom_value
    )
    
    return metrics
```

**Step 3:** Update aggregation (optional)

```python
class MetricsManager:
    def calc_total_episodes_metrics(self, episode_metrics):
        # ... existing aggregations
        avg_custom = mean([m.my_custom_metric for m in episode_metrics])
        # ... include in PolicyRunSummary
```

### Customizing Logging

**Subclass `EnvironmentLogger`:**

```python
class CustomLogger(EnvironmentLogger):
    def log_step(self, step_num, actions, rewards, terminated, truncated, info):
        # Custom per-step logging
        super().log_step(step_num, actions, rewards, terminated, truncated, info)
        # ... additional logging
    
    def save_learning_state(self, episode_state, episode_num, **kwargs):
        # Custom learning state processing
        # ... custom enrichment
        super().save_learning_state(episode_state, episode_num, **kwargs)
```

**Use custom logger:**

```python
logger = CustomLogger(output_dir=config.logging.output_dir, scenario_id=config.environment.scenario_id)
```

### Modifying Scenario Generation

**Subclass `LatentScenarioBuilder`:**

```python
class CustomScenarioBuilder(LatentScenarioBuilder):
    def _sample_latent_vector(self, variance, entity_type='drone'):
        # Custom latent vector sampling
        # ... your sampling logic
        return mode_id, latent_vector
    
    def _generate_positions(self, count, region, existing_positions, ...):
        # Custom placement logic
        # ... your placement algorithm
        return positions
```

**Use custom builder:**

```python
builder = CustomScenarioBuilder(
    world_size=config.world.size,
    config=config.latent_world,
    target_hp=config.targets.target_hp,
    seed=config.seed
)
```

---

## 10. Design Decisions & Rationale

### Why Policies Created Upfront, Not Per-Episode?

**Decision:** All policies instantiated before episode loop begins.

**Rationale:**
- Learning policies accumulate knowledge across episodes
- Creating fresh policy each episode would reset learned state (P/U matrices)
- Upfront creation allows policy-level initialization (seed distribution, hyperparameter setup)

**Alternative considered:** Create policy per-episode
- **Rejected:** Would prevent cumulative learning, defeating purpose of multi-episode runs

---

### Why `soft_reset()` Instead of `reset()`?

**Decision:** Use `soft_reset()` between episodes, `reset()` only for new experiment runs.

**Rationale:**
- `soft_reset()` preserves P/U matrices (learned knowledge)
- Epsilon continues decaying across episodes (exploration → exploitation)
- Enables cumulative learning while resetting episode-specific counters

**Alternative considered:** Full `reset()` between episodes
- **Rejected:** Would treat each episode as independent, losing learning progression

---

### Why Diagnostics Are Environment-Owned, Not in PettingZoo `info`?

**Decision:** Diagnostics stored on environment (`env.diagnostics`), not in per-agent `info` dict.

**Rationale:**
- PettingZoo `info` is per-agent, diagnostics are shared/global
- Keeps environment telemetry separate from agent observations
- Allows oracle policies to access ground truth without polluting agent observations
- Maintains zero-knowledge constraint (agents never see diagnostics in observations)

**Alternative considered:** Include diagnostics in `info` dict
- **Rejected:** Would violate zero-knowledge constraint, complicate agent observation parsing

---

### Why Learning State Logged Separately from Episode Trace?

**Decision:** Learning state saved in separate `learning_state/` directory.

**Rationale:**
- Episode trace is large (per-step observations, actions, rewards)
- Learning state is small (P, U matrices, epsilon)
- Separation allows fast access to learned embeddings without parsing full episode
- Enables post-processing (t-SNE enrichment) without modifying episode logs

**Alternative considered:** Embed learning state in episode trace
- **Rejected:** Would bloat episode files, slow down learning state access

---

### Why t-SNE Enrichment Is Post-Processing, Not Inline?

**Decision:** t-SNE runs after all episodes complete, not during training loop.

**Rationale:**
- t-SNE is computationally expensive (~1-2 seconds per episode)
- Running inline would slow training loop significantly
- Post-processing allows training to complete quickly, visualization later
- Enriched files saved separately (`_enriched.json` suffix)

**Alternative considered:** Inline t-SNE during episode loop
- **Rejected:** Would add 35-70 seconds to training time for 35 episodes

---

### Why Episode Seed Is `config.seed + episode_num`?

**Decision:** Each episode gets unique seed derived from base seed.

**Rationale:**
- **Reproducibility**: Same `config.seed` produces same episode sequence
- **Variation**: Each episode has different env randomness (action processing order, noise)
- **Learning**: Policy state evolves across episodes despite env randomness
- **Debugging**: Can replay specific episode by using `config.seed + episode_num`

**Alternative considered:** Same seed for all episodes
- **Rejected:** Would produce identical env behavior across episodes (no variation)

---

### Why MultiAgentPolicy Wrapper Exists?

**Decision:** Wrap per-agent policies in `MultiAgentPolicy` container.

**Rationale:**
- Provides uniform interface (`select_actions`, `update`) across policy types
- Hides per-agent delegation from episode loop
- Allows single-policy (random, oracle) and multi-agent (MF) policies to be used interchangeably
- Simplifies policy factory (returns uniform interface)

**Alternative considered:** Episode loop handles per-agent policies directly
- **Rejected:** Would complicate episode loop, duplicate delegation logic

---

### Why Best Episode Selected by Damage Efficiency?

**Decision:** Representative episode = highest `dmg_eff` (net damage / gross damage).

**Rationale:**
- Damage efficiency indicates coordination quality (low overkill, efficient target selection)
- More meaningful than steps (can be fast but wasteful) or ammo (can be efficient but slow)
- Best episode for detailed analysis (reveals coordination patterns)

**Alternative considered:** Select by steps (fastest) or ammo (most efficient)
- **Rejected:** Steps can be fast but wasteful; ammo doesn't capture overkill

---

## Summary

This guide documents the **integration patterns** for assembling the latent ZK-MRTA benchmark components. It complements the technical specification (what components do) and usage guide (how to run experiments) by explaining **how components fit together**.

**Key takeaways:**
- **Initialization**: Config → Builder → Environment → Policies (strict dependency order)
- **Policy lifecycle**: Soft reset preserves learned state across episodes
- **Episode loop**: Reset → Step → Update → Terminate (PettingZoo parallel execution)
- **Multi-policy orchestration**: Sequential runs with isolated state
- **Logging**: Three-tier structure (traces, learning states, metrics)
- **Factory pattern**: Centralized policy creation with config-driven hyperparameters
- **Extension points**: New policies, metrics, logging, scenario generation

**Related documentation:**
- **Technical specification**: [`technical-overview.md`](policies/classic-collaborative-filtering/technical-overview.md)
- **Usage guide**: [`usage-guide.md`](usage-guide.md)

---

**Last Updated:** 2026-04-12
