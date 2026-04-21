---
title: Usage Guide - Latent ZK-MRTA Benchmark
description: Operational guide for running experiments with the latent zero-knowledge multi-robot task allocation benchmark
last_updated: 2026-04-12
---

# Usage Guide: Latent ZK-MRTA Benchmark

This guide provides practical instructions for running experiments with the latent zero-knowledge multi-robot task allocation (ZK-MRTA) benchmark environment. For technical specifications of the environment and learning algorithms, see [`technical-overview.md`](policies/classic-collaborative-filtering/technical-overview.md).

---

## 1. Prerequisites & Installation

### Requirements

- Python 3.11+
- Virtual environment recommended

### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip3 install -r requirements.txt
```

### Key Dependencies

- `pettingzoo` — Multi-agent RL environment framework
- `gymnasium` — Action/observation space definitions
- `numpy` — Numerical computations
- `scikit-learn` — Cosine similarity, t-SNE visualization

---

## 2. Configuration Reference

Experiments are configured via `config/scenario.json`. The configuration file controls scenario generation, environment parameters, policy selection, and hyperparameters.

### Configuration Structure

```json
{
  "latent_world": { ... },        // Latent benchmark construction
  "seed": 42,                     // Random seed for reproducibility
  "world": { ... },               // World dimensions
  "environment": { ... },         // Episode settings
  "drones": { ... },              // Drone placement
  "targets": { ... },             // Target placement and HP
  "policy": { ... },              // Policy selection
  "collaborative_filtering": { ... },  // MF hyperparameters
  "logging": { ... }              // Output directory
}
```

### Latent World Configuration

Controls the hidden latent structure that governs drone-target compatibility.

```json
"latent_world": {
  "mode": "independent",          // "common" or "independent"
  "center_mode": "one_hot",       // "one_hot", "orthogonal", or "random"
  "epsilon": 0.1,                 // Smoothing for one_hot centers
  "latent_dim": 3,                // Latent vector dimension
  "independent": {
    "drones": {
      "num_modes": 3,             // Number of drone mode centers
      "drone_variance": 0.2       // Sampling variance around centers
    },
    "targets": {
      "num_modes": 3,             // Number of target mode centers
      "target_variance": 0.2      // Sampling variance around centers
    }
  }
}
```

**Mode options:**
- `"common"`: Drones and targets share the same mode centers
- `"independent"`: Drones and targets have separate mode centers

**Center mode options:**
- `"one_hot"`: Smoothed axis-aligned centers (controlled by `epsilon`)
- `"orthogonal"`: Orthogonal centers via QR decomposition
- `"random"`: Random positive centers in log-space

### Environment Configuration

```json
"environment": {
  "max_steps": 250,               // Maximum steps per episode
  "num_episodes": 35,             // Episodes to run (for learning policies)
  "verbose": false,               // Print step-by-step details
  "scenario_id": "latent_mrta_benchmark"
}
```

### Drone and Target Placement

```json
"drones": {
  "count": 9,
  "region": {
    "x_fraction": [0.25, 0.65],   // Placement region (fraction of world size)
    "y_fraction": [0.2, 0.4]
  },
  "min_distance_between_drones": 50.0
},
"targets": {
  "count": 27,
  "target_hp": 10.0,              // Initial HP per target
  "region": {
    "x_fraction": [0.1, 0.9],
    "y_fraction": [0.5, 0.85]
  },
  "min_distance_from_drones": 50.0,
  "min_distance_between_targets": 25.0
}
```

### Policy Selection

```json
"policy": {
  "type": [
    "random",
    "max_damage_oracle",
    "matrix_factorization_cf"
  ],
  "allow_noop": false             // Whether NoOp (action 0) is valid
}
```

### Matrix Factorization Hyperparameters

```json
"collaborative_filtering": {
  "reward_noise": 0.2,            // Gaussian noise std dev on rewards
  "observation_noise": 0.2,       // Probability of corrupting observed actions
  "enable_tsne_enrichment": true, // Post-process with t-SNE visualization
  "matrix_factorization_cf": {
    "use_integration_matrix": true,  // Learn from running mean vs raw rewards
    "latent_dim": 3,                 // Embedding dimension
    "learning_rate": 0.01,           // SGD step size
    "lambda_reg": 0.02,              // L2 regularization coefficient
    "epsilon": 0.3,                  // Initial exploration rate
    "epsilon_decay": 0.9995,         // Multiplicative decay per step
    "epsilon_min": 0.02,             // Minimum exploration rate
    "anti_signal_weight": 1.0        // Weight for negative reward updates
  }
}
```

### Logging Configuration

```json
"logging": {
  "output_dir": "logs/"           // Base directory for all outputs
}
```

---

## 3. Running Experiments

### Basic Execution

```bash
python main_zk_mrta.py
```

### Execution Flow

When you run `main_zk_mrta.py`, the following occurs:

1. **Load configuration** from `config/scenario.json`
2. **Build latent scenario** using `LatentScenarioBuilder`:
   - Sample mode centers based on `center_mode`
   - Generate drone and target latent vectors via log-normal sampling
   - Place entities according to region and spacing constraints
3. **Create environment** with configured noise levels
4. **For each policy** in `config.policy.type`:
   - Instantiate policy with hyperparameters
   - Run N episodes (1 for deterministic policies, `num_episodes` for learning policies)
   - Log episode data, metrics, and learning state (if applicable)
   - Compute performance summaries
5. **Output comparison tables** and save artifacts

### Console Output

During execution, you'll see:

- **Episode progress**: Step count, targets neutralized, total reward per episode
- **Per-policy summaries**: Average steps, ammo efficiency, damage efficiency
- **Best-episode comparisons**: Performance vs random baseline
- **File save confirmations**: Paths to saved episode logs and metrics

Example output:
```
=== Episode 1/35 ===
Step 15: 12 targets neutralized | Net damage: 120.0 | Reward: 8.5
...
Episode complete: 23 steps | 27/27 targets | Ammo efficiency: 1.17

Saved: logs/latent_mrta_benchmark/matrix_factorization_cf/episodes/episode_01.json
```

---

## 4. Output Artifacts

All outputs are saved under `{output_dir}/{scenario_id}/{policy_type}/`.

### Directory Structure

```
logs/
└── latent_mrta_benchmark/
    ├── random/
    │   ├── episodes/
    │   │   └── episode_01.json
    │   ├── analysis/
    │   │   └── analysis_ep01.json
    │   └── metrics.json
    ├── max_damage_oracle/
    │   └── ...
    └── matrix_factorization_cf/
        ├── episodes/
        │   ├── episode_01.json
        │   ├── episode_02.json
        │   └── ...
        ├── learning_state/
        │   ├── state_ep01.json
        │   ├── state_ep01_enriched.json  (if t-SNE enabled)
        │   └── ...
        ├── analysis/
        │   └── analysis_ep{best}.json
        └── metrics.json
```

### File Types

| File | Description | Content |
|------|-------------|---------|
| `episode_XX.json` | Full episode trace | Step-by-step observations, actions, rewards, diagnostics |
| `state_epXX.json` | Learning state snapshot | P and U matrices, epsilon, predictions (MF policy only) |
| `state_epXX_enriched.json` | Learning state + visualization | Adds t-SNE 2D projections of embeddings |
| `metrics.json` | Aggregated episode metrics | Steps, ammo, overkill, efficiency per episode |
| `analysis_epXX.json` | Engagement analysis | Which drones engaged which targets |

### Episode Log Schema (Excerpt)

```json
{
  "episode_num": 1,
  "seed": 43,
  "steps": [
    {
      "step_num": 1,
      "actions": {"drone_0": 5, "drone_1": 12, ...},
      "rewards": {"drone_0": 0.87, "drone_1": -1.0, ...},
      "diagnostics": {
        "target_hps": [10.0, 8.3, ...],
        "target_active": [true, true, false, ...],
        "collisions": 2,
        "net_damage": 15.2,
        ...
      }
    },
    ...
  ]
}
```

### Learning State Schema (Excerpt)

```json
{
  "episode_num": 35,
  "agents": [
    {
      "agent_idx": 0,
      "P": [[0.123, -0.045, 0.678], ...],  // (num_agents, latent_dim)
      "U": [[0.234, 0.567, ...], ...],     // (latent_dim, num_targets)
      "epsilon": 0.025,
      "match": {
        "predicted_rewards": [0.45, 0.78, -0.12, ...],
        "ranked_targets": [12, 5, 18, ...],
        "best_target": 12
      }
    }
  ]
}
```

---

## 5. Metrics Explained

Metrics are computed per episode and aggregated across runs.

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Steps** | Episode length | Lower = faster completion |
| **Targets Neutralized** | Targets destroyed | Should equal total target count for success |
| **Total Ammo Used** | Shots fired across all drones | Lower = more efficient |
| **Shots per Target** | `Total Ammo / Targets Neutralized` | Ideal ≈ `HP / avg_damage`; lower = better coordination |
| **Ammo Efficiency** | `Targets Neutralized / Total Ammo` | Higher = better; inverse of shots per target |
| **Damage Efficiency** | `Net Damage / Gross Damage` | 1.0 = no waste; <1.0 indicates overkill |
| **Total Net Damage** | Effective damage applied to target HP | Should equal `target_hp × num_targets` for full success |
| **Total Gross Damage** | Total damage produced (including wasted) | Includes overkill and shots on inactive targets |
| **Total Overkill** | Excess damage beyond target HP | Lower = better coordination |
| **Total Collisions** | Simultaneous target selections | Lower = better decentralized coordination |

### Efficiency Calculations

**Shots per Target:**
```
shots_per_target = total_ammo_used / targets_neutralized
```

---

## 6. Policy Types

### Random (`random`)

**Description:** Selects a random active target each step.

**Characteristics:**
- Deterministic: Runs 1 episode
- No learning
- Baseline for comparison

**Use case:** Establish lower-bound performance.

### Max Damage Oracle (`max_damage_oracle`)

**Description:** Has access to true latent vectors (cheating) and computes optimal assignment via Hungarian algorithm.

**Characteristics:**
- Deterministic: Runs 1 episode
- No learning (uses ground truth)
- Upper bound on performance

**Use case:** Establish theoretical performance ceiling.

### Matrix Factorization (`matrix_factorization_cf`)

**Description:** Learns P (drone) and U (target) embedding matrices from public interaction history using decentralized SGD.

**Characteristics:**
- Non-deterministic: Runs N episodes with cumulative learning
- Each drone maintains independent P and U matrices
- ε-greedy exploration with multiplicative decay

**Use case:** Evaluate decentralized collaborative filtering in ZK-MRTA.

**Hyperparameters:** See §7 below.

---

## 7. Hyperparameter Reference

### Matrix Factorization Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `latent_dim` | 8 | 1–20 | Embedding dimension; match to true `latent_world.latent_dim` for best recovery |
| `learning_rate` | 0.01 | 0.001–0.1 | SGD step size; too high = instability, too low = slow learning |
| `lambda_reg` | 0.02 | 0.0–0.1 | L2 regularization; prevents overfitting to noise |
| `epsilon` | 0.20 | 0.0–1.0 | Initial exploration rate; higher = more random exploration |
| `epsilon_decay` | 1.0 | 0.9–1.0 | Multiplicative decay per step; <1.0 = gradual shift to exploitation |
| `epsilon_min` | 0.02 | 0.0–0.2 | Floor for exploration; prevents pure exploitation |
| `use_integration_matrix` | false | true/false | If true, learn from running mean instead of raw rewards |
| `anti_signal_weight` | 0.1 | 0.0–1.0 | Weight for negative-cosine reward updates (direct mode only) |

### Environment Noise Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `reward_noise` | 0.0 | 0.0–0.5 | Std dev of Gaussian noise added to observed rewards |
| `observation_noise` | 0.0 | 0.0–0.5 | Probability of corrupting observed action IDs |

**Noise impact:**
- `reward_noise`: Distorts the *value* of interactions, making utility estimates noisier
- `observation_noise`: Distorts the *identity* of interaction partners, creating spurious entries in learned models

---

## 8. Common Workflows

### Compare Policies on Same Scenario

```json
"policy": {
  "type": ["random", "max_damage_oracle", "matrix_factorization_cf"]
}
```

Run once to compare all three policies on identical scenarios.

### Ablation Study: Integration Matrix

Test whether learning from running means improves performance vs. raw rewards.

**Config A (baseline):**
```json
"collaborative_filtering": {
  "matrix_factorization_cf": {
    "use_integration_matrix": false
  }
}
```

**Config B (integration matrix):**
```json
"collaborative_filtering": {
  "matrix_factorization_cf": {
    "use_integration_matrix": true
  }
}
```

### Noise Robustness Test

Evaluate policy performance under observation and reward corruption.

```json
"collaborative_filtering": {
  "reward_noise": 0.2,
  "observation_noise": 0.2
}
```

### Dimension Mismatch Experiment

Test whether the policy can learn with fewer dimensions than the true latent structure.

```json
"latent_world": {
  "latent_dim": 5
},
"collaborative_filtering": {
  "matrix_factorization_cf": {
    "latent_dim": 3  // Underparameterized
  }
}
```

### Exploration Schedule Tuning

Test different exploration decay rates.

**Fast decay (aggressive exploitation):**
```json
"epsilon": 0.3,
"epsilon_decay": 0.995,
"epsilon_min": 0.02
```

**Slow decay (prolonged exploration):**
```json
"epsilon": 0.3,
"epsilon_decay": 0.9995,
"epsilon_min": 0.05
```

---

## 9. Extension Guide

### Adding a New Policy

**Step 1:** Implement policy class in `tabula_drone/policies/`.

Your policy must implement the `IPolicy` interface:

```python
class MyPolicy:
    is_deterministic: bool = False  # True if policy runs 1 episode
    
    def select_actions(self, obs: Dict, infos: Dict) -> Dict[str, int]:
        """Return action dict: {agent_id: action}"""
        pass
    
    def update(self, obs: Dict) -> None:
        """Update policy state after environment step"""
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

**Step 2:** Add factory logic in `main_zk_mrta.py`.

Edit `create_policy()`:

```python
def create_policy(policy_type: str, config, drones_config, num_targets):
    if policy_type == "my_policy":
        return MyPolicy(
            num_targets=num_targets,
            seed=config.seed,
            # ... your hyperparameters
        )
    # ... existing policies
```

**Step 3:** Add policy type to config.

```json
"policy": {
  "type": ["random", "my_policy"]
}
```

### Modifying Scenario Generation

Edit `tabula_drone/scenarios/latent_scenario_builder.py`:

- Adjust mode center generation in `_generate_mode_centers()`
- Modify sampling distributions in `_sample_latent_vector()`
- Change placement logic in `with_drones()` / `with_targets()`

### Custom Metrics

**Step 1:** Extend `EpisodeMetrics` dataclass in `tabula_drone/utils/metrics_manager.py`.

```python
@dataclass
class EpisodeMetrics:
    # ... existing fields
    my_custom_metric: float = 0.0
```

**Step 2:** Compute metric in `run_episode()` in `main_zk_mrta.py`.

```python
# Extract from diagnostics or compute from episode data
my_custom_value = compute_my_metric(shared_info)

metrics = EpisodeMetrics(
    # ... existing fields
    my_custom_metric=my_custom_value,
)
```

---

## Cross-References

For detailed technical specifications, see:

- **§1 Latent Benchmark Construction:** [`technical-overview.md §1`](policies/classic-collaborative-filtering/technical-overview.md#1-latent-benchmark-construction)
- **§2 Zero-Knowledge Observation Model:** [`technical-overview.md §2`](policies/classic-collaborative-filtering/technical-overview.md#2-zero-knowledge-observation-model)
- **§5 Matrix Factorization Learning:** [`technical-overview.md §5`](policies/classic-collaborative-filtering/technical-overview.md#5-decentralized-matrix-factorization-learning-and-prediction)
- **§6 Evaluation Metrics:** [`technical-overview.md §6`](policies/classic-collaborative-filtering/technical-overview.md#6-evaluation-and-metrics)
- **§7 Environment Implementation:** [`technical-overview.md §7`](policies/classic-collaborative-filtering/technical-overview.md#7-environment-implementation)

---

## Troubleshooting

**Issue:** `ValueError: latent world_model requires parsed latent_world config`  
**Solution:** Ensure `config/scenario.json` includes a `latent_world` section with required fields.

**Issue:** Policy runs only 1 episode instead of N  
**Solution:** Check `is_deterministic` attribute. Deterministic policies (random, oracle) run 1 episode by design.

**Issue:** t-SNE enrichment not generating `_enriched.json` files  
**Solution:** Verify `"enable_tsne_enrichment": true` in `collaborative_filtering` config section.

**Issue:** Metrics show 0.0 damage efficiency  
**Solution:** Check that targets are being neutralized. If all targets remain active, net damage = 0.

---

**Last Updated:** 2026-04-12
