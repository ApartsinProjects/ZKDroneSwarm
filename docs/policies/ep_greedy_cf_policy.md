# EpGreedyCFPolicy (ε-Greedy Collaborative Filtering)

## Overview

The `EpGreedyCFPolicy` is a **learning-based ZK-compliant policy** for the ZK-MRTA (Zero-Knowledge Multi-Robot Task Allocation) environment. It uses **SGD-based matrix factorization** to learn agent-target compatibility from observed rewards, combined with **ε-greedy exploration** for action selection.

**Key Characteristic**: Learns latent representations of agents and targets from reward feedback, enabling intelligent target selection without privileged state information.

---

## Design Philosophy

This policy is **fully ZK-compliant** because it only uses information available through the collaborative observation mode:
- Binary active/inactive status of each target
- Observed rewards from previous actions (all agents)
- Selected targets from previous step (all agents)

It does **not** use:
- True HP values or remaining attributes
- Target class types
- Weapon damage profiles directly

The policy exists to:

1. Demonstrate **learning-based target selection** in ZK settings
2. Leverage **collaborative filtering** to infer agent-target compatibility
3. Balance **exploration vs exploitation** via ε-greedy
4. Provide a middle-ground between random (no learning) and oracle (privileged info)

---

## Core Algorithm


### Matrix Factorization Model

The policy maintains **latent vectors** for each agent and target:

```
Agent Latent Vectors: A ∈ ℝ^(num_agents × latent_dim)
Target Latent Vectors: T ∈ ℝ^(num_targets × latent_dim)
```

The predicted reward for agent `i` attacking target `j` is:

```
predicted_reward(i, j) = (1 + dot(A[i], T[j])) / 2
```

This scales the dot product from [-1, 1] to [0, 1].

#### Why Matrix Factorization?

Matrix factorization is a classic collaborative filtering technique that:
- **Discovers latent structure**: Agents with similar weapons cluster together in latent space
- **Generalizes from sparse data**: Learning about one agent-target pair informs predictions for similar pairs
- **Scales efficiently**: O(latent_dim) per prediction, not O(num_agents × num_targets)

#### Worked Example: Latent Space Interpretation

```
Scenario: 2 agents, 3 targets, latent_dim=2

After training, the latent vectors might look like:

Agent Vectors (weapon characteristics):
  drone_0: [0.9, 0.1]  → "armor-piercing" weapon
  drone_1: [0.2, 0.95] → "shield-breaker" weapon

Target Vectors (defensive characteristics):
  target_0: [0.85, 0.2]  → "armored" target
  target_1: [0.3, 0.9]   → "shielded" target
  target_2: [0.6, 0.6]   → "balanced" target

Predicted Rewards:
  drone_0 → target_0: (1 + 0.9*0.85 + 0.1*0.2) / 2 = 0.89 (high!)
  drone_0 → target_1: (1 + 0.9*0.3 + 0.1*0.9) / 2 = 0.68
  drone_1 → target_0: (1 + 0.2*0.85 + 0.95*0.2) / 2 = 0.68
  drone_1 → target_1: (1 + 0.2*0.3 + 0.95*0.9) / 2 = 0.96 (high!)

The policy learns that drone_0 is effective against target_0,
and drone_1 is effective against target_1, without ever seeing
the actual weapon/target attributes!
```

### SGD Update Rule

When an agent attacks a target and receives a reward, the policy updates both latent vectors using stochastic gradient descent:

```
# Skip updates for NoOp or negative rewards (wasted shots)
if target_idx < 0 or observed_reward < 0:
    return

error = observed_reward - predicted_reward

A[agent] += learning_rate * error * T[target]
T[target] += learning_rate * error * A[agent]

# Re-normalize to unit sphere
A[agent] = normalize(A[agent])
T[target] = normalize(T[target])
```

#### Why Skip Negative Rewards?

The environment returns -1.0 for "wasted shots" (firing at already-neutralized targets). Due to sequential processing, this can happen when:
1. Drone A fires at Target X and kills it
2. Drone B (who also targeted X) fires but target is already dead → gets -1.0

If we learned from this -1.0, the policy would incorrectly conclude that Drone B is "incompatible" with Target X's class — when in reality it was just a timing issue. Skipping negative rewards prevents this false incompatibility learning.

#### Why Normalize?

Keeping vectors on the unit sphere:
- **Bounds predictions**: Dot products stay in [-1, 1], predictions in [0, 1]
- **Prevents explosion**: No unbounded growth during training
- **Improves stability**: Consistent scale across all vectors

#### Worked Example: SGD Update

```
Before Update:
  A[0] = [0.6, 0.8]
  T[0] = [0.8, 0.6]
  
  predicted = (1 + 0.6*0.8 + 0.8*0.6) / 2 = 0.98
  observed = 0.5  (actual reward from environment)
  error = 0.5 - 0.98 = -0.48

Update (learning_rate = 0.1):
  A[0] += 0.1 * (-0.48) * [0.8, 0.6] = [0.6 - 0.038, 0.8 - 0.029]
  A[0] = [0.562, 0.771] → normalize → [0.589, 0.808]
  
  T[0] += 0.1 * (-0.48) * [0.6, 0.8] = [0.8 - 0.029, 0.6 - 0.038]
  T[0] = [0.771, 0.562] → normalize → [0.808, 0.589]

After Update:
  new_predicted = (1 + 0.589*0.808 + 0.808*0.589) / 2 = 0.976

The prediction moved closer to the observed reward (0.5).
```

### ε-Greedy Action Selection

The policy balances exploration and exploitation:

```
1. Parse observation to identify active targets
2. Build list of valid actions:
   - If allow_noop=True: [0] + [active target indices]
   - If allow_noop=False: [active target indices only]
3. With probability ε: select random action (explore)
4. With probability 1-ε: select action with highest predicted reward (exploit)
5. Decay ε: ε = max(ε_min, ε * ε_decay)
6. Return selected action
```

#### Why ε-Greedy?

- **Exploration**: Random actions discover new agent-target compatibilities
- **Exploitation**: Greedy selection uses learned knowledge
- **Decay**: Reduces exploration as policy becomes more confident
- **Simplicity**: Easy to tune and understand

#### Worked Example: ε-Greedy Selection

```
Scenario: 3 active targets, ε=0.2

Predicted Rewards:
  target_0: 0.7
  target_1: 0.9  ← highest
  target_2: 0.4

Random draw: 0.15 (< ε=0.2)
→ EXPLORE: Select random action from [1, 2, 3]
→ Action = 3 (fire at target_2)

Random draw: 0.85 (> ε=0.2)
→ EXPLOIT: Select action with highest predicted reward
→ Action = 2 (fire at target_1)
```

### Collaborative Learning

In collaborative observation mode, each agent observes:
- All agents' selected targets from the previous step
- All agents' observed rewards from the previous step

The policy learns from **all** observations, not just its own:

```python
for each agent_idx in range(num_agents):
    target_action = selected_targets[agent_idx]
    reward = observed_rewards[agent_idx]
    if target_action > 0:  # Not NoOp
        target_idx = target_action - 1
        update(agent_idx, target_idx, reward)
```

#### Why Collaborative Learning?

- **Faster convergence**: Learn from N agents' experiences per step
- **Better generalization**: See how different weapons affect different targets
- **True collaborative filtering**: The "collaborative" in CF refers to learning from multiple users/agents

---

## ZK-Compliance Deep Dive

### What Makes a Policy ZK-Compliant?

In the ZK-MRTA environment, agents receive **limited observations** that hide certain information:

| Information | Available in ZK Observation? |
|-------------|------------------------------|
| Target position (x, y) | ✅ Yes |
| Target active status | ✅ Yes |
| Selected targets (previous step) | ✅ Yes (collaborative mode) |
| Observed rewards (previous step) | ✅ Yes (collaborative mode) |
| Target HP / remaining attributes | ❌ No (hidden) |
| Target class / type | ❌ No (hidden) |
| Weapon damage profiles | ❌ No (hidden) |

A ZK-compliant policy can only use the "Yes" information.

### Why EpGreedyCFPolicy is ZK-Compliant

The policy **never accesses**:
- `targets_state` (privileged HP values)
- `class_attribute_mapping` (target class definitions)
- `weapon_damage_profile_mapping` (weapon specifications)

Instead, it **learns** agent-target compatibility from observed rewards, which implicitly encode the hidden weapon-target interactions.

### Comparison with Other Policies

| Aspect | RandomPolicy | EpGreedyCFPolicy | Min TTK Oracle |
|--------|--------------|------------------|----------------|
| **ZK-Compliant** | ✅ Yes | ✅ Yes | ❌ No |
| **Uses HP values** | No | No | Yes |
| **Uses weapon profiles** | No | No | Yes |
| **Learns from rewards** | No | Yes | No |
| **Decision basis** | Random | Learned predictions | Hits-to-kill |
| **Expected performance** | Lowest | Medium | Highest |

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 2 | Dimension of latent vectors |
| `learning_rate` | 0.01 | SGD step size |
| `epsilon` | 0.3 | Initial exploration rate |
| `epsilon_decay` | 0.99 | Decay factor per action |
| `epsilon_min` | 0.05 | Minimum exploration rate |

### Tuning Guidelines

**`latent_dim`**:
- Higher = more expressive, but needs more data
- Start with 2-4 for simple scenarios
- Increase if policy struggles to differentiate targets

**`learning_rate`**:
- Higher = faster learning, but less stable
- Lower = slower but more stable convergence
- 0.01-0.1 is typical range

**`epsilon` / `epsilon_decay` / `epsilon_min`**:
- High initial ε (0.3-0.5) for exploration
- Decay rate depends on episode length
- Keep ε_min > 0 to maintain some exploration

---

## Edge Case Handling

| Scenario | Behavior |
|----------|----------|
| All targets active | Select based on ε-greedy over predictions |
| No active targets | Returns NoOp (action 0) |
| Single active target | ε-greedy between NoOp (if allowed) and that target |
| NoOp action in update | Skipped (no learning from NoOp) |
| Negative reward (wasted shot) | Skipped (no learning from timing-based penalties) |

---

## Action Space

| Action | Meaning |
|--------|---------|
| `0` | NoOp (do nothing) - only when `allow_noop=True` |
| `1` to `N` | Fire at target index `i-1` (1-indexed) |

---

## API Reference

### Constructor

```python
EpGreedyCFPolicy(
    num_agents: int,
    num_targets: int,
    latent_dim: int = 2,
    learning_rate: float = 0.01,
    epsilon: float = 0.3,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.05,
    seed: Optional[int] = None,
)
```

**Parameters**:
- **`num_agents`**: Number of agents in the environment
- **`num_targets`**: Number of targets in the environment
- **`latent_dim`**: Dimension of latent vectors (default 2)
- **`learning_rate`**: SGD learning rate (default 0.01)
- **`epsilon`**: Initial exploration rate for ε-greedy (default 0.3)
- **`epsilon_decay`**: Decay factor for epsilon per step (default 0.99)
- **`epsilon_min`**: Minimum epsilon value (default 0.05)
- **`seed`**: Random seed for reproducibility

### Methods

#### `predict_reward`

```python
def predict_reward(
    agent_idx: int,
    target_idx: int,
) -> float
```

Predict reward for agent-target pair using dot product.

**Parameters**:
- **`agent_idx`**: Agent index (0-based)
- **`target_idx`**: Target index (0-based)

**Returns**: Predicted reward in [0, 1] range.

---

#### `update`

```python
def update(
    agent_idx: int,
    target_idx: int,
    observed_reward: float,
) -> None
```

Update latent vectors based on observed reward using SGD.

**Skipped when**:
- `target_idx < 0` (NoOp action)
- `observed_reward < 0` (wasted shot penalty — avoids learning false incompatibilities from timing issues)

**Parameters**:
- **`agent_idx`**: Agent index that took action
- **`target_idx`**: Target index that was selected (-1 for NoOp, skipped)
- **`observed_reward`**: Observed reward value

---

#### `update_from_observation`

```python
def update_from_observation(
    observation: Dict[str, Any],
    agent_id: str,
) -> None
```

Update latent vectors from collaborative mode observation. Learns from all agents' observed rewards.

**Parameters**:
- **`observation`**: Dict observation from collaborative mode containing `selected_targets` and `observed_rewards`
- **`agent_id`**: ID of the agent receiving this observation

---

#### `select_action`

```python
def select_action(
    agent_idx: int,
    observation: Dict[str, Any],
    allow_noop: bool = False,
) -> int
```

Select action using ε-greedy over predicted rewards.

**Parameters**:
- **`agent_idx`**: Index of the agent selecting action
- **`observation`**: Dict observation from collaborative mode
- **`allow_noop`**: If `True`, include NoOp (0) as valid action

**Returns**: Integer action in `[0, num_targets]` or `[1, num_targets]` depending on `allow_noop`.

---

#### `select_actions`

```python
def select_actions(
    observations: Dict[str, Dict[str, Any]],
    allow_noop: bool = False,
) -> Dict[str, int]
```

Select actions for **all agents**.

**Parameters**:
- **`observations`**: Dict of `{agent_id: observation}`
- **`allow_noop`**: If `True`, include NoOp as valid action

**Returns**: Dict of `{agent_id: action}`

---

#### `reset`

```python
def reset() -> None
```

Full reset: reinitialize all latent vectors and reset epsilon. Use between independent experiments.

---

#### `soft_reset`

```python
def soft_reset() -> None
```

Soft reset for new episode. Currently a no-op (preserves learned knowledge across episodes).

---

## Usage Example

```python
from tabula_drone.policies.ep_greedy_cf_policy import EpGreedyCFPolicy
from tabula_drone.envs.drone_engage_zk_mrta_v0 import DroneEngageZKMRTA

# Create environment in collaborative mode
env = DroneEngageZKMRTA(
    drones_config=[
        {'position': (100, 100), 'weapon_type': 'heavy'},
        {'position': (200, 200), 'weapon_type': 'light'},
    ],
    targets_config=[
        {'position': (500, 500), 'class_type': 'A'},
        {'position': (600, 600), 'class_type': 'B'},
    ],
    class_attribute_mapping={
        'A': {'armor': 100, 'shields': 50},
        'B': {'armor': 50, 'shields': 100},
    },
    weapon_damage_profile_mapping={
        'heavy': {'armor': 50, 'shields': 10},
        'light': {'armor': 10, 'shields': 50},
    },
    observation_mode='collaborative',
)

# Initialize policy
policy = EpGreedyCFPolicy(
    num_agents=env.num_drones,
    num_targets=env.num_targets,
    latent_dim=2,
    learning_rate=0.1,
    epsilon=0.3,
    seed=42,
)

# Run episode
obs, _ = env.reset(seed=42)

for step in range(100):
    # Select actions
    actions = policy.select_actions(obs)
    
    # Step environment
    obs, rewards, terminations, truncations, _ = env.step(actions)
    
    # Update policy from observations (learn from all agents)
    for agent_id, agent_obs in obs.items():
        policy.update_from_observation(agent_obs, agent_id)
    
    if terminations['drone_0'] or truncations['drone_0']:
        break

# Check learned predictions
print(f"drone_0 → target_0: {policy.predict_reward(0, 0):.2f}")
print(f"drone_0 → target_1: {policy.predict_reward(0, 1):.2f}")
print(f"drone_1 → target_0: {policy.predict_reward(1, 0):.2f}")
print(f"drone_1 → target_1: {policy.predict_reward(1, 1):.2f}")
```

---

## Implementation Details

### Observation Parsing

The policy extracts target active status from the collaborative observation:
- `observation['targets']`: Array with format `[t0_x, t0_y, t0_active, t1_x, ...]`
- Active status is at index `target_idx * 3 + 2`
- A target is considered active if `active > 0.5`

### Latent Vector Initialization

- Vectors are initialized uniformly in [-1, 1] then normalized
- Same seed produces identical initial vectors
- Normalization ensures all vectors start on unit sphere

### Random Number Generator

- Uses `numpy.random.RandomState` for reproducibility
- Same seed produces identical exploration/exploitation decisions
- RNG is shared across all operations (initialization, exploration, tie-breaking)

---

## Limitations

1. **Requires Collaborative Mode**: Only works with `observation_mode='collaborative'`
2. **No Coordination**: Agents select independently; may all fire at same target
3. **Cold Start**: Initial predictions are random until sufficient learning
4. **Single-Step Greedy**: Does not plan ahead or consider future states
5. **Fixed Latent Dimension**: Cannot adapt complexity during training

### The Cold Start Problem

When the policy starts, latent vectors are random, so predictions are meaningless. The policy relies on exploration (high ε) to gather initial data.

**Mitigation strategies**:
- Start with high ε (0.3-0.5)
- Use slower ε decay in early episodes
- Consider warm-starting from previous runs

### No Coordination

Like the oracle policies, agents select independently:

```
Scenario: 2 agents, 2 targets

Both agents learn that target_0 is "easier" (higher predicted reward).
Without coordination, both fire at target_0, wasting one attack.

With coordination, one would attack each target.
```

This is a fundamental limitation of independent learners, not specific to CF.

---

## File Location

`tabula_drone/policies/ep_greedy_cf_policy.py`

## Related Tests

`tests/test_ep_greedy_cf_policy.py`
