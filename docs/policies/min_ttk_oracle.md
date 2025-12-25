# OracleTimeToKillPolicy (Min TTK Oracle)

## Overview

The `OracleTimeToKillPolicy` is a **privileged baseline policy** for the ZK-MRTA (Zero-Knowledge Multi-Robot Task Allocation) environment. It serves as an **upper-bound benchmark** by using privileged (non-ZK-compliant) information to make optimal target selection decisions.

**Key Characteristic**: Selects the active target that can be eliminated in the **fewest estimated hits** given each agent's weapon damage profile.

---

## Design Philosophy

This policy is intentionally **not ZK-compliant** because it uses true remaining attribute values of targets (privileged state information). It exists to:

1. Establish an upper performance bound for comparison with ZK-compliant policies
2. Validate environment mechanics and reward structures
3. Provide a reference implementation for target prioritization logic

---

## Core Algorithm

### Hits-to-Kill Estimation

The policy estimates the number of hits required to neutralize a target using the formula:

```
hits = max_a ceil(rem_a / dmg_a)
```

Where:
- `rem_a` = remaining value of attribute `a` on the target
- `dmg_a` = damage per hit dealt to attribute `a` by the weapon
- The maximum is taken across all attributes

#### Why Maximum Across Attributes?

A target is only neutralized when **all** of its defensive attributes are depleted to zero. Each attribute depletes independently based on the weapon's damage profile. The "bottleneck" attribute—the one requiring the most hits to deplete—determines the total hits needed.

Think of it like breaking through multiple layers of defense: you must break through ALL layers, so the thickest layer determines your total effort.

#### Worked Example: Multi-Attribute Target

Consider a target with two defensive attributes and a weapon that damages both:

```
Target State:
  - armor:   25.0
  - shields: 12.0

Weapon Damage Profile:
  - armor:   10.0 per hit
  - shields:  5.0 per hit
```

**Step-by-step calculation:**

1. **Armor**: `ceil(25.0 / 10.0) = ceil(2.5) = 3 hits`
2. **Shields**: `ceil(12.0 / 5.0) = ceil(2.4) = 3 hits`
3. **Result**: `max(3, 3) = 3 hits` to neutralize target

Now consider a different target:

```
Target State:
  - armor:   10.0
  - shields: 30.0

Weapon Damage Profile (same):
  - armor:   10.0 per hit
  - shields:  5.0 per hit
```

1. **Armor**: `ceil(10.0 / 10.0) = 1 hit`
2. **Shields**: `ceil(30.0 / 5.0) = 6 hits`
3. **Result**: `max(1, 6) = 6 hits` — shields are the bottleneck

The policy would prefer the first target (3 hits) over the second (6 hits).

**Special Cases**:
| Condition | Result |
|-----------|--------|
| `rem_a <= 0` for all attributes | 0 hits (already dead) |
| `dmg_a == 0` and `rem_a > 0` | Infinity (unkillable) |
| Attribute not in weapon profile | Infinity (unkillable) |

### Target Selection Logic

The core decision loop follows a greedy strategy: always attack the target that can be eliminated fastest.

```
1. Parse observation to identify active targets
2. For each active target:
   - Compute estimated hits-to-kill using privileged state
3. Select target(s) with minimum hits-to-kill
4. If tie: random selection among tied targets (seeded RNG)
5. Return 1-indexed action (target_index + 1)
```

#### Why Greedy Minimum TTK?

This strategy maximizes the **rate of target elimination**. By always focusing on the "easiest" target:

- Targets are removed from the battlefield faster
- Each eliminated target stops contributing to enemy actions
- Rewards (if tied to eliminations) are earned sooner

This is optimal when:
- All targets have equal threat/value
- There's no penalty for overkill (multiple agents hitting same target)
- Future state doesn't change the relative difficulty of targets

#### Worked Example: Target Selection

```
Scenario: 3 active targets, 1 agent

Agent Weapon Profile:
  - hp: 10.0 per hit

Target States:
  Target 0: {"hp": 45.0}  → ceil(45/10) = 5 hits
  Target 1: {"hp": 18.0}  → ceil(18/10) = 2 hits  ← MINIMUM
  Target 2: {"hp": 30.0}  → ceil(30/10) = 3 hits

Decision: Select Target 1 (action = 2, since actions are 1-indexed)
```

If Target 1 were inactive (already eliminated), the policy would select Target 2 (3 hits) as the next best option.

### Edge Case Handling

| Scenario | Behavior |
|----------|----------|
| No active targets | Return NoOp (action 0) |
| All targets unkillable + `allow_noop=True` | Return NoOp (action 0) |
| All targets unkillable + `allow_noop=False` | Random active target |
| Multiple targets with same min hits | Random tie-break (deterministic with seed) |

#### Understanding "Unkillable" Targets

A target is considered **unkillable** by a specific agent when the agent's weapon cannot deplete at least one of the target's attributes. This happens when:

1. **Zero damage**: The weapon deals 0 damage to an attribute that has remaining value
2. **Missing attribute**: The target has an attribute not defined in the weapon's damage profile

**Example: Unkillable Scenario**

```
Agent Weapon Profile:
  - armor: 10.0 per hit
  (no shields damage defined)

Target State:
  - armor:   0.0   (already depleted)
  - shields: 50.0  (still active)

Calculation:
  - armor: 0.0 remaining → skip (already zero)
  - shields: 50.0 remaining, but weapon has no shields damage → INFINITY

Result: Target is unkillable by this agent
```

This is critical in heterogeneous multi-agent scenarios where different agents have different weapon capabilities. An agent with only armor-piercing weapons cannot eliminate shield-only targets.

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
OracleTimeToKillPolicy(
    agent_weapon_profiles: Dict[str, Dict[str, float]],
    seed: Optional[int] = None,
    allow_noop: bool = True,
)
```

**Parameters**:
- **`agent_weapon_profiles`**: Dict mapping agent IDs to their weapon damage profiles.
  - Example: `{"drone_0": {"armor": 10.0, "shields": 5.0}, "drone_1": {"armor": 15.0}}`
- **`seed`**: Random seed for reproducibility (affects tie-breaking only)
- **`allow_noop`**: If `True`, action 0 (NoOp) is valid. If `False`, agents must always fire.

### Methods

#### `select_action`

```python
def select_action(
    agent_id: str,
    observation: np.ndarray,
    num_targets: int,
    targets_state: List[Dict[str, float]],
) -> int
```

Select action for a **single agent**.

**Parameters**:
- **`agent_id`**: ID of the agent (e.g., `"drone_0"`)
- **`observation`**: ZK observation array with shape `(3 * num_targets,)`
  - Format: `[target_0_x, target_0_y, target_0_active, target_1_x, ...]`
- **`num_targets`**: Total number of targets in environment
- **`targets_state`**: **Privileged** list of target attribute dicts
  - Example: `[{"armor": 25.0, "shields": 10.0}, {"armor": 0.0, "shields": 5.0}]`

**Returns**: Integer action in `[0, num_targets]` or `[1, num_targets]` depending on `allow_noop`.

---

#### `select_actions`

```python
def select_actions(
    observations: Dict[str, np.ndarray],
    num_targets: int,
    targets_state: List[Dict[str, float]],
) -> Dict[str, int]
```

Select actions for **all agents** independently.

**Parameters**:
- **`observations`**: Dict of `{agent_id: observation_array}`
- **`num_targets`**: Total number of targets
- **`targets_state`**: Privileged target state list

**Returns**: Dict of `{agent_id: action}`

**Note**: Each agent selects independently based on their own weapon profile. There is **no coordination** between agents.

---

## Usage Example

```python
from tabula_drone.policies.min_ttk_oracle import OracleTimeToKillPolicy
import numpy as np

# Define weapon profiles for each agent
agent_profiles = {
    "drone_0": {"armor": 10.0, "shields": 5.0},
    "drone_1": {"armor": 15.0, "shields": 3.0},
}

# Initialize policy
policy = OracleTimeToKillPolicy(
    agent_weapon_profiles=agent_profiles,
    seed=42,
    allow_noop=True,
)

# Observation: 2 targets, both active
# Format: [t0_x, t0_y, t0_active, t1_x, t1_y, t1_active]
obs = np.array([100.0, 200.0, 1.0, 150.0, 250.0, 1.0], dtype=np.float32)

# Privileged target state
targets_state = [
    {"armor": 30.0, "shields": 10.0},  # Target 0: needs 3 hits (armor: 30/10)
    {"armor": 15.0, "shields": 20.0},  # Target 1: needs 4 hits (shields: 20/5)
]

# Select action for drone_0
action = policy.select_action("drone_0", obs, num_targets=2, targets_state=targets_state)
# Returns 1 (fire at target 0, which has min hits-to-kill)

# Select actions for all agents
observations = {"drone_0": obs, "drone_1": obs}
actions = policy.select_actions(observations, num_targets=2, targets_state=targets_state)
# Returns {"drone_0": 1, "drone_1": 1} (both select target 0)
```

---

## Implementation Details

### Observation Parsing

The policy extracts target active status from the ZK observation array:
- Each target occupies 3 consecutive values: `[x, y, active]`
- Active status is at index `target_idx * 3 + 2`
- A target is considered active if `active > 0.5`

### Weapon Damage Profiles

- Each agent can have a **different weapon profile**
- Damage is applied per-hit to each attribute independently
- Attributes not in the weapon profile are treated as having 0 damage (unkillable)

### Tie-Breaking

- When multiple targets have the same minimum hits-to-kill, one is selected randomly
- The random selection is **deterministic** when a seed is provided
- Uses `numpy.random.RandomState` for reproducibility

#### Why Deterministic Tie-Breaking Matters

In reinforcement learning and simulation environments, **reproducibility** is essential for:

1. **Debugging**: Reproduce exact sequences of actions to diagnose issues
2. **Benchmarking**: Compare policies under identical conditions
3. **Testing**: Write deterministic unit tests that don't flake

**Example: Tie-Breaking in Action**

```python
# Two targets with identical hits-to-kill
targets_state = [
    {"hp": 20.0},  # 2 hits
    {"hp": 20.0},  # 2 hits (tie!)
]

# With seed=42, the policy will ALWAYS choose the same target
policy = OracleTimeToKillPolicy(
    agent_weapon_profiles={"drone_0": {"hp": 10.0}},
    seed=42
)

# Run 100 times - all results identical
actions = [policy.select_action("drone_0", obs, 2, targets_state) for _ in range(100)]
assert len(set(actions)) == 1  # All same action
```

Without a seed, the same scenario would produce different actions across runs, making debugging and testing difficult.

---

## Limitations

1. **Not ZK-Compliant**: Uses privileged state information (true remaining attributes)
2. **No Coordination**: Agents select independently; may all fire at the same target
3. **Static Weapon Profiles**: Assumes constant damage per hit (no weapon degradation)
4. **Single-Step Greedy**: Does not plan ahead or consider future states

### Deep Dive: The Coordination Problem

The most significant practical limitation is the **lack of coordination**. When multiple agents use this policy, they all independently compute the same "best" target and may all fire at it simultaneously.

**Example: Wasteful Overkill**

```
Scenario: 3 agents, 3 targets

All agents have identical weapon profiles: {"hp": 10.0}

Target States:
  Target 0: {"hp": 10.0}  → 1 hit to kill
  Target 1: {"hp": 20.0}  → 2 hits to kill
  Target 2: {"hp": 30.0}  → 3 hits to kill

Without Coordination:
  - All 3 agents select Target 0 (minimum hits)
  - Target 0 receives 3 hits but only needed 1
  - 2 hits are "wasted" (overkill)
  - Targets 1 and 2 remain untouched

With Optimal Coordination:
  - Agent 0 → Target 0 (1 hit, eliminated)
  - Agent 1 → Target 1 (1 of 2 hits needed)
  - Agent 2 → Target 1 (2 of 2 hits, eliminated)
  - Or distribute across all targets for faster overall elimination
```

This limitation is **by design**—the oracle policy establishes an upper bound for what's achievable with perfect target state knowledge but **without** inter-agent communication. A coordinated policy would be a separate, more complex baseline.

---

## File Location

`tabula_drone/policies/min_ttk_oracle.py`

## Related Tests

`tests/test_min_ttk_oracle.py`
