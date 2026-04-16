# RandomPolicy

## Overview

The `RandomPolicy` is a **ZK-compliant baseline policy** for the ZK-MRTA (Zero-Knowledge Multi-Robot Task Allocation) environment. It implements **uniform random selection** over active targets, serving as a lower-bound benchmark for policy performance.

**Key Characteristic**: Selects uniformly at random from all valid actions (active targets + optional NoOp) without using any privileged information.

---

## Design Philosophy

This policy is **fully ZK-compliant** because it only uses information available in the zero-knowledge observation:
- Binary active/inactive status of each target
- No HP values, class types, or damage profiles
- No memory or learning across steps
- Treats all active targets as identical

It exists to:

1. Establish a **lower performance bound** for comparison with intelligent policies
2. Validate that the environment rewards intelligent behavior over random
3. Provide a simple, correct reference implementation
4. Serve as a sanity check during development and debugging

---

## Core Algorithm

### Action Selection

The algorithm is intentionally simple:

```
1. Parse observation to identify active targets
2. Build list of valid actions:
   - If allow_noop=True: [0] + [active target indices]
   - If allow_noop=False: [active target indices only]
3. Select uniformly at random from valid actions
4. Return selected action
```

#### Why Uniform Random?

Uniform random selection ensures:
- **No bias**: Each valid action has equal probability
- **Reproducibility**: With a seed, the sequence is deterministic
- **Simplicity**: No hyperparameters or tuning required
- **Baseline validity**: Any intelligent policy should outperform random

#### Worked Example: Action Selection

```
Scenario: 4 targets, allow_noop=True

Observation (parsed):
  Target 0: active = True
  Target 1: active = False  (eliminated)
  Target 2: active = True
  Target 3: active = True

Valid Actions:
  [0, 1, 3, 4]
  │   │  │  └── Fire at Target 3 (1-indexed)
  │   │  └── Fire at Target 2 (1-indexed)
  │   └── Fire at Target 0 (1-indexed)
  └── NoOp (do nothing)

Probability Distribution:
  P(action=0) = 1/4 = 25%
  P(action=1) = 1/4 = 25%
  P(action=3) = 1/4 = 25%
  P(action=4) = 1/4 = 25%

Note: Action 2 is NOT valid because Target 1 is inactive.
```

### Understanding allow_noop

The `allow_noop` parameter controls whether "do nothing" is a valid action:

| `allow_noop` | Valid Actions | Use Case |
|--------------|---------------|----------|
| `True` | NoOp + active targets | When inaction is sometimes optimal |
| `False` | Active targets only | When agents must always engage |

#### Worked Example: allow_noop Impact

```
Same scenario: 3 active targets (indices 0, 2, 3)

With allow_noop=True:
  Valid actions: [0, 1, 3, 4]
  P(NoOp) = 25%
  P(any fire action) = 75%

With allow_noop=False:
  Valid actions: [1, 3, 4]
  P(NoOp) = 0%
  P(any fire action) = 100%
```

When `allow_noop=False`, the agent is **forced to fire** at some active target every step.

---

## ZK-Compliance Deep Dive

### What Makes a Policy ZK-Compliant?

In the ZK-MRTA environment, agents receive **limited observations** that hide certain information:

| Information | Available in ZK Observation? |
|-------------|------------------------------|
| Target position (x, y) | ✅ Yes |
| Target active status | ✅ Yes |
| Target HP / remaining attributes | ❌ No (hidden) |
| Target class / type | ❌ No (hidden) |
| Other agents' actions | ❌ No (hidden) |

A ZK-compliant policy can only use the "Yes" information.

### Why RandomPolicy is ZK-Compliant

```python
# The policy ONLY reads the active status from observations
for target_idx in range(num_targets):
    obs_idx = target_idx * 3 + 2  # Index of active field
    is_active = observation[obs_idx] > 0.5  # Binary check only
```

It does **not**:
- Access `targets_state` (privileged HP values)
- Use weapon damage profiles for decision-making
- Remember previous observations
- Coordinate with other agents

### Comparison with Oracle Policies

| Aspect | RandomPolicy | Min TTK Oracle | Max Damage Oracle |
|--------|--------------|----------------|-------------------|
| **ZK-Compliant** | ✅ Yes | ❌ No | ❌ No |
| **Uses HP values** | No | Yes | Yes |
| **Uses weapon profiles** | No | Yes | Yes |
| **Decision basis** | Random | Hits-to-kill | Dot-product score |
| **Expected performance** | Lowest | High | Highest |

---

## Edge Case Handling

| Scenario | Behavior |
|----------|----------|
| All targets active | Uniform over [NoOp + all targets] or [all targets] |
| No active targets + `allow_noop=True` | Returns NoOp (action 0) |
| No active targets + `allow_noop=False` | **Error**: Empty valid_actions list |
| Single active target | 50% NoOp, 50% fire (if allow_noop=True) |

#### Warning: No Active Targets with allow_noop=False

```python
# This will raise an error:
valid_actions = []  # No NoOp, no active targets
action = self.rng.choice(valid_actions)  # ValueError!
```

If your environment can have zero active targets, ensure `allow_noop=True` or handle this case externally.

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
RandomPolicy(
    seed: Optional[int] = None,
    allow_noop: bool = True,
)
```

**Parameters**:
- **`seed`**: Random seed for reproducibility. Same seed = same action sequence.
- **`allow_noop`**: If `True`, action 0 (NoOp) is included in valid actions.

**Note**: Unlike oracle policies, RandomPolicy does **not** require `agent_weapon_profiles` because it doesn't use weapon information.

### Methods

#### `select_action`

```python
def select_action(
    observation: np.ndarray,
    num_targets: int,
) -> int
```

Select a random action for a **single agent**.

**Parameters**:
- **`observation`**: ZK observation array with shape `(3 * num_targets,)`
  - Format: `[target_0_x, target_0_y, target_0_active, target_1_x, ...]`
- **`num_targets`**: Total number of targets in environment

**Returns**: Integer action in `[0, num_targets]` or `[1, num_targets]` depending on `allow_noop`.

---

#### `select_actions`

```python
def select_actions(
    observations: Dict[str, np.ndarray],
    num_targets: int,
) -> Dict[str, int]
```

Select random actions for **all agents** independently.

**Parameters**:
- **`observations`**: Dict of `{agent_id: observation_array}`
- **`num_targets`**: Total number of targets

**Returns**: Dict of `{agent_id: action}`

**Note**: Each agent samples independently. There is **no coordination**—agents may randomly select the same target.

---

## Usage Example

```python
from tabula_drone.policies.random_policy import RandomPolicy
import numpy as np

# Initialize policy with seed for reproducibility
policy = RandomPolicy(seed=42, allow_noop=True)

# Observation: 3 targets, targets 0 and 2 are active
# Format: [t0_x, t0_y, t0_active, t1_x, t1_y, t1_active, t2_x, t2_y, t2_active]
obs = np.array([
    100.0, 200.0, 1.0,   # Target 0: active
    150.0, 250.0, 0.0,   # Target 1: inactive
    200.0, 300.0, 1.0,   # Target 2: active
], dtype=np.float32)

# Select action for single agent
action = policy.select_action(obs, num_targets=3)
# Returns one of: 0 (NoOp), 1 (Target 0), or 3 (Target 2)
# Note: 2 is not valid because Target 1 is inactive

# Select actions for multiple agents
observations = {
    "drone_0": obs,
    "drone_1": obs,
    "drone_2": obs,
}
actions = policy.select_actions(observations, num_targets=3)
# Returns e.g.: {"drone_0": 1, "drone_1": 0, "drone_2": 3}
# Each agent samples independently
```

---

## Implementation Details

### Observation Parsing

The policy extracts target active status from the ZK observation array:
- Each target occupies 3 consecutive values: `[x, y, active]`
- Active status is at index `target_idx * 3 + 2`
- A target is considered active if `active > 0.5`

### Random Number Generator

- Uses `numpy.random.RandomState` for reproducibility
- Same seed produces identical action sequences
- Each call to `select_action` advances the RNG state

#### Reproducibility Example

```python
# Same seed = same sequence
policy1 = RandomPolicy(seed=42)
policy2 = RandomPolicy(seed=42)

actions1 = [policy1.select_action(obs, 3) for _ in range(10)]
actions2 = [policy2.select_action(obs, 3) for _ in range(10)]

assert actions1 == actions2  # Identical sequences
```

### No Coordination

When using `select_actions` for multiple agents:
- Each agent calls `select_action` independently
- The same RNG is used sequentially (not per-agent)
- Agents may randomly select the same target (collisions possible)

---

## Statistical Properties

### Expected Behavior

With `allow_noop=True` and `k` active targets:

```
P(NoOp) = 1 / (k + 1)
P(any specific target) = 1 / (k + 1)
P(fire at some target) = k / (k + 1)
```

### Expected Collisions (Multi-Agent)

With `n` agents and `k` active targets (allow_noop=True):

```
P(at least one collision) ≈ 1 - (k+1)! / ((k+1)^n * (k+1-n)!)

Example: 3 agents, 3 active targets
  Valid actions per agent: 4 (NoOp + 3 targets)
  P(all different) = (4 * 3 * 2) / (4^3) = 24/64 = 37.5%
  P(at least one collision) ≈ 62.5%
```

Random policies have **high collision rates** in multi-agent scenarios.

---

## Limitations

1. **No Intelligence**: Treats all active targets identically
2. **No Coordination**: Agents act independently, causing collisions
3. **No Learning**: Same behavior regardless of history
4. **Potential Error**: Crashes if no valid actions (allow_noop=False with no active targets)

### When Random is Actually Optimal

In rare cases, random selection is optimal:
- All targets have identical value/threat
- No coordination benefit (single agent)
- Unpredictability is strategically valuable (adversarial settings)

In most ZK-MRTA scenarios, intelligent policies significantly outperform random.

---

## File Location

`tabula_drone/policies/random_policy.py`

## Related Tests

No dedicated test file found. Consider adding `tests/test_random_policy.py`.
