# OptimalAssignmentOracle (Max Damage Oracle)

## Overview

The `OptimalAssignmentOracle` is a **privileged baseline policy** for the ZK-MRTA (Zero-Knowledge Multi-Robot Task Allocation) environment. It computes **globally optimal drone-to-target assignments** that maximize total damage output while enforcing **collision-free constraints** (at most one drone per target).

**Key Characteristic**: Uses linear sum assignment (Hungarian algorithm) to find the optimal one-to-one matching between agents and targets based on dot-product scores.

---

## Design Philosophy

This policy is intentionally **not ZK-compliant** because it uses true remaining attribute values of targets (privileged state information). It exists to:

1. Establish an upper performance bound for coordinated multi-agent assignment
2. Demonstrate optimal resource allocation under collision-free constraints
3. Provide a reference for comparing learned coordination policies

**Key Difference from Min TTK Oracle**: While the Min TTK Oracle has each agent independently select the "easiest" target (potentially causing collisions), this oracle **coordinates all agents** to maximize global utility with no target collisions.

---

## Core Algorithm

### The Assignment Problem

The oracle solves a classic **linear sum assignment problem**:

- **Input**: N agents, M targets, and a score matrix S where S[i,j] represents the "value" of assigning agent i to target j
- **Output**: One-to-one assignment that maximizes total score
- **Constraint**: Each target can be assigned to at most one agent

#### Why Linear Sum Assignment?

This is the mathematically optimal solution for one-to-one matching problems. The Hungarian algorithm (used by SciPy's `linear_sum_assignment`) finds the global optimum in O(n³) time, guaranteeing no better assignment exists.

### Score Matrix Construction

The score for assigning agent i to target j is computed as a **dot product**:

```
S[i,j] = dot(agent_vector[i], target_vector[j])
       = Σ_a (damage_a[i] × remaining_a[j])
```

Where:
- `damage_a[i]` = damage per hit that agent i deals to attribute a
- `remaining_a[j]` = remaining value of attribute a on target j

#### Why Dot Product?

The dot product captures the **total potential damage** an agent can deal to a target in a single hit. Higher scores indicate better "fit" between an agent's weapon profile and a target's remaining attributes.

#### Worked Example: Score Matrix

```
Scenario: 2 agents, 2 targets

Agent Weapon Profiles:
  drone_0: {armor: 10.0, shields: 5.0}
  drone_1: {armor: 8.0,  shields: 12.0}

Target States:
  Target 0: {armor: 50.0, shields: 30.0}
  Target 1: {armor: 40.0, shields: 20.0}

Score Matrix Calculation:

S[drone_0, target_0] = (10 × 50) + (5 × 30) = 500 + 150 = 650
S[drone_0, target_1] = (10 × 40) + (5 × 20) = 400 + 100 = 500
S[drone_1, target_0] = (8 × 50) + (12 × 30) = 400 + 360 = 760
S[drone_1, target_1] = (8 × 40) + (12 × 20) = 320 + 240 = 560

Score Matrix:
              Target 0    Target 1
  drone_0       650         500
  drone_1       760         560
```

#### Worked Example: Optimal Assignment

From the score matrix above, we need to find the assignment that maximizes total score:

```
Option A: drone_0 → Target 0, drone_1 → Target 1
  Total = 650 + 560 = 1210

Option B: drone_0 → Target 1, drone_1 → Target 0
  Total = 500 + 760 = 1260  ← OPTIMAL

The oracle selects Option B because it yields higher total damage.
```

**Key Insight**: Even though drone_0 would prefer Target 0 (650 > 500), the global optimum assigns drone_0 to Target 1 because drone_1 is even better suited for Target 0. This is the power of coordinated assignment.

### Collision-Free Guarantee

The linear sum assignment algorithm **inherently guarantees** that:
- Each agent is assigned to at most one target
- Each target is assigned to at most one agent

This eliminates the "overkill" problem where multiple agents waste resources on the same target.

#### Why Collision-Free Matters

```
Scenario: 3 agents, 3 targets (all with {hp: 10.0})

Without Coordination (Min TTK style):
  - All agents independently select the same "easiest" target
  - 3 hits on 1 target, 0 hits on other 2
  - 2 targets survive

With Collision-Free Assignment:
  - Each agent assigned to different target
  - 1 hit on each target
  - All 3 targets eliminated in same time step
```

---

## Edge Case Handling

| Scenario | Behavior |
|----------|----------|
| No active targets | All agents get NoOp (action 0) |
| More agents than active targets | Excess agents get NoOp (action 0) |
| More targets than agents | Best N targets are assigned (others unattacked) |
| Empty targets_state | All agents get NoOp (action 0) |

#### Understanding "More Agents Than Targets"

When there are more drones than active targets, not all drones can be assigned. The oracle assigns the **best-suited** drones to targets and leaves the rest unassigned.

**Example: 3 Agents, 1 Target**

```
Agent Weapon Profiles:
  drone_0: {armor: 10.0}  → Score = 10 × 50 = 500
  drone_1: {armor: 8.0}   → Score = 8 × 50 = 400
  drone_2: {armor: 15.0}  → Score = 15 × 50 = 750  ← HIGHEST

Target State:
  Target 0: {armor: 50.0}

Assignment:
  drone_2 → Target 0 (action = 1)
  drone_0 → NoOp (action = 0)
  drone_1 → NoOp (action = 0)
```

The oracle assigns drone_2 because it has the highest score (best weapon match).

---

## Action Space

| Action | Meaning |
|--------|---------|
| `0` | NoOp (unassigned or no active targets) |
| `1` to `N` | Fire at target index `i-1` (1-indexed) |
| `-1` | Unassigned (only when `allow_noop=False`) |

---

## API Reference

### Constructor

```python
OptimalAssignmentOracle(
    agent_weapon_profiles: Dict[str, Dict[str, float]],
    seed: Optional[int] = None,
    allow_noop: bool = True,
)
```

**Parameters**:
- **`agent_weapon_profiles`**: Dict mapping agent IDs to their weapon damage profiles.
  - Example: `{"drone_0": {"armor": 10.0, "shields": 5.0}, "drone_1": {"armor": 15.0}}`
- **`seed`**: Random seed (unused in this policy, kept for interface consistency)
- **`allow_noop`**: If `True`, unassigned agents get action 0. If `False`, they get -1.

### Methods

#### `select_actions`

```python
def select_actions(
    observations: Dict[str, np.ndarray],
    num_targets: int,
    targets_state: List[Dict[str, float]],
) -> Dict[str, int]
```

Select **globally optimal** actions for all agents simultaneously.

**Parameters**:
- **`observations`**: Dict of `{agent_id: observation_array}`
  - Format: `[target_0_x, target_0_y, target_0_active, target_1_x, ...]`
- **`num_targets`**: Total number of targets in environment
- **`targets_state`**: **Privileged** list of target attribute dicts
  - Example: `[{"armor": 25.0, "shields": 10.0}, {"armor": 0.0, "shields": 5.0}]`

**Returns**: Dict of `{agent_id: action}`

**Note**: Unlike Min TTK Oracle, this policy does **not** have a `select_action` method for single agents because the assignment is inherently a multi-agent coordination problem.

---

## Usage Example

```python
from tabula_drone.policies.max_damage_oracle import OptimalAssignmentOracle
import numpy as np

# Define weapon profiles for each agent
agent_profiles = {
    "drone_0": {"armor": 10.0, "shields": 5.0},
    "drone_1": {"armor": 8.0, "shields": 12.0},
}

# Initialize oracle
oracle = OptimalAssignmentOracle(
    agent_weapon_profiles=agent_profiles,
    allow_noop=True,
)

# Observation: 2 targets, both active
# Format: [t0_x, t0_y, t0_active, t1_x, t1_y, t1_active]
obs = np.array([100.0, 200.0, 1.0, 150.0, 250.0, 1.0], dtype=np.float32)

# Privileged target state
targets_state = [
    {"armor": 50.0, "shields": 30.0},  # Target 0
    {"armor": 40.0, "shields": 20.0},  # Target 1
]

# Select actions for all agents (coordinated)
observations = {"drone_0": obs, "drone_1": obs}
actions = oracle.select_actions(observations, num_targets=2, targets_state=targets_state)

# Result: {"drone_0": 2, "drone_1": 1}
# drone_0 assigned to Target 1, drone_1 assigned to Target 0
# This is the globally optimal assignment (see worked example above)
```

---

## Implementation Details

### Vector Construction

The policy converts weapon profiles and target states into vectors for efficient matrix operations:

1. **Attribute Names**: Extracted from first target's keys, sorted alphabetically
2. **Agent Vectors**: `[damage_attr1, damage_attr2, ...]` for each agent
3. **Target Vectors**: `[remaining_attr1, remaining_attr2, ...]` for each target

Missing attributes default to 0.0.

### Active Target Filtering

Before solving the assignment:
1. Parse observation to identify active targets (active flag > 0.5)
2. Build score matrix only for active targets
3. Map assignment results back to original target indices

### SciPy's Linear Sum Assignment

The oracle uses `scipy.optimize.linear_sum_assignment`:
- Input: Cost matrix (negated score matrix, since scipy minimizes)
- Output: Row and column indices of optimal assignment
- Complexity: O(n³) where n = max(agents, targets)

---

## Comparison: Min TTK vs Max Damage Oracle

| Aspect | Min TTK Oracle | Max Damage Oracle |
|--------|----------------|-------------------|
| **Strategy** | Greedy per-agent | Global optimization |
| **Coordination** | None (independent) | Full (collision-free) |
| **Objective** | Minimize hits to kill | Maximize total damage |
| **Collisions** | Possible (overkill) | Impossible (one-to-one) |
| **Complexity** | O(n) per agent | O(n³) for all agents |
| **API** | `select_action` + `select_actions` | `select_actions` only |

### When to Use Which?

- **Min TTK Oracle**: When you want to measure performance of independent greedy decisions
- **Max Damage Oracle**: When you want to measure performance of optimal coordinated assignment

---

## Limitations

1. **Not ZK-Compliant**: Uses privileged state information (true remaining attributes)
2. **Static Assignment**: Computes assignment once per step; doesn't adapt mid-step
3. **Homogeneous Objective**: All agents optimize the same dot-product metric
4. **No Temporal Planning**: Greedy per-step; doesn't consider future states

### Deep Dive: The Dot Product Assumption

The dot-product score assumes that **higher remaining attributes = more valuable target**. This may not always align with mission objectives:

**Example: When Dot Product Misleads**

```
Scenario: Eliminate high-threat targets first

Target 0: {armor: 100.0, shields: 100.0}  → High dot-product score
Target 1: {armor: 10.0, shields: 10.0}    → Low dot-product score

If Target 1 is actually more dangerous (e.g., about to fire),
the dot-product metric would incorrectly prioritize Target 0.
```

The oracle maximizes **damage dealt**, not necessarily **threat reduction**. For threat-aware assignment, a different scoring function would be needed.

### Deep Dive: Computational Cost

The Hungarian algorithm has O(n³) complexity. For large numbers of agents/targets:

```
Agents × Targets    Approximate Operations
10 × 10             1,000
50 × 50             125,000
100 × 100           1,000,000
```

For real-time applications with many agents, approximate algorithms may be needed.

---

## File Location

`tabula_drone/policies/max_damage_oracle.py`

## Related Tests

`tests/test_max_damage_oracle.py`
