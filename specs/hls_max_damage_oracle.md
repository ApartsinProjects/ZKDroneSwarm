## Max Damage Oracle Policy - High-Level Specification

> **Last Updated:** December 2024

### 1. Scope

This specification defines the **OptimalAssignmentOracle** (Max Damage Oracle), a **privileged baseline policy** for the ZK-MRTA simulation. This policy computes **globally optimal drone-to-target assignments** that maximize total damage output while enforcing **collision-free constraints** (at most one drone per target).

**Key Characteristic**: Uses linear sum assignment (Hungarian algorithm) to find the optimal one-to-one matching between agents and targets based on dot-product scores.

---

### 2. Design Philosophy

The Max Damage Oracle exists to:

1. **Establish an upper performance bound** for coordinated multi-agent assignment
2. **Demonstrate optimal resource allocation** under collision-free constraints
3. **Provide a reference** for comparing learned coordination policies

**Key Difference from Min TTK Oracle**: While the Min TTK Oracle has each agent independently select the "easiest" target (potentially causing collisions), this oracle **coordinates all agents** to maximize global utility with no target collisions.

**Privileged Information Used:**
- True remaining attribute values of targets
- All agents' weapon damage profiles (centralized view)

---

### 3. Core Algorithm

#### 3.1 The Assignment Problem

The oracle solves a classic **linear sum assignment problem**:

- **Input**: N agents, M targets, and a score matrix S where S[i,j] represents the "value" of assigning agent i to target j
- **Output**: One-to-one assignment that maximizes total score
- **Constraint**: Each target can be assigned to at most one agent

#### 3.2 Score Matrix Construction

The score for assigning agent i to target j is computed as a **dot product**:

```
S[i,j] = dot(agent_vector[i], target_vector[j])
       = Σ_a (damage_a[i] × remaining_a[j])
```

Where:
- `damage_a[i]` = damage per hit that agent i deals to attribute a
- `remaining_a[j]` = remaining value of attribute a on target j

#### 3.3 Why Dot Product?

The dot product captures the **total potential damage** an agent can deal to a target in a single hit. Higher scores indicate better "fit" between an agent's weapon profile and a target's remaining attributes.

#### 3.4 Worked Example

```
Scenario: 2 agents, 2 targets

Agent Weapon Profiles:
  drone_0: {armor: 10.0, shields: 5.0}
  drone_1: {armor: 8.0,  shields: 12.0}

Target States:
  Target 0: {armor: 50.0, shields: 30.0}
  Target 1: {armor: 40.0, shields: 20.0}

Score Matrix Calculation:
  S[drone_0, target_0] = (10 × 50) + (5 × 30) = 650
  S[drone_0, target_1] = (10 × 40) + (5 × 20) = 500
  S[drone_1, target_0] = (8 × 50) + (12 × 30) = 760
  S[drone_1, target_1] = (8 × 40) + (12 × 20) = 560

Possible Assignments:
  Option A: drone_0 → Target 0, drone_1 → Target 1 → Total = 650 + 560 = 1210
  Option B: drone_0 → Target 1, drone_1 → Target 0 → Total = 500 + 760 = 1260 ← OPTIMAL

Result: drone_0 → Target 1, drone_1 → Target 0
```

**Key Insight**: Even though drone_0 would prefer Target 0 (650 > 500), the global optimum assigns drone_0 to Target 1 because drone_1 is even better suited for Target 0.

---

### 4. Collision-Free Guarantee

The linear sum assignment algorithm **inherently guarantees** that:
- Each agent is assigned to at most one target
- Each target is assigned to at most one agent

This eliminates the "overkill" problem where multiple agents waste resources on the same target.

#### 4.1 Why Collision-Free Matters

```
Without Coordination (Min TTK style):
  - All agents independently select the same "easiest" target
  - 3 hits on 1 target, 0 hits on other 2
  - 2 targets survive

With Collision-Free Assignment:
  - Each agent assigned to different target
  - 1 hit on each target
  - All 3 targets can be eliminated faster
```

---

### 5. Edge Case Handling

| Scenario | Behavior |
|----------|----------|
| No active targets | All agents get NoOp (action 0) |
| More agents than active targets | Excess agents get NoOp (action 0) |
| More targets than agents | Best N targets are assigned (others unattacked) |
| Empty targets_state | All agents get NoOp (action 0) |

#### 5.1 More Agents Than Targets

When there are more drones than active targets, the oracle assigns the **best-suited** drones to targets and leaves the rest unassigned:

```
Example: 3 Agents, 1 Target

Agent Weapon Profiles:
  drone_0: {armor: 10.0}  → Score = 500
  drone_1: {armor: 8.0}   → Score = 400
  drone_2: {armor: 15.0}  → Score = 750 ← HIGHEST

Assignment:
  drone_2 → Target 0 (action = 1)
  drone_0 → NoOp (action = 0)
  drone_1 → NoOp (action = 0)
```

---

### 6. Action Space

| Action | Meaning |
|--------|---------|
| `0` | NoOp (unassigned or no active targets) |
| `1` to `N` | Fire at target index `i-1` (1-indexed) |
| `-1` | Unassigned (only when `allow_noop=False`) |

---

### 7. Policy Requirements

#### 7.1 Privileged State Access

The policy **requires** access to privileged target state at action-selection time:

```python
targets_state: List[Dict[str, float]]
# Example: [{"armor": 50.0, "shields": 30.0}, {"armor": 40.0, "shields": 20.0}]
```

#### 7.2 Centralized Coordination

Unlike Min TTK Oracle, this policy requires a **centralized view** of all agents:
- All agent weapon profiles must be provided at initialization
- Assignment is computed for all agents simultaneously
- No `select_action` method for single agents (only `select_actions`)

#### 7.3 Per-Agent Weapon Profiles

Each agent can have a **different weapon profile**:

```python
agent_weapon_profiles: Dict[str, Dict[str, float]]
# Example: {
#     "drone_0": {"armor": 10.0, "shields": 5.0},
#     "drone_1": {"armor": 8.0, "shields": 12.0},
# }
```

---

### 8. Comparison with Other Policies

| Property | RandomPolicy | Min TTK Oracle | Max Damage Oracle |
|----------|--------------|----------------|-------------------|
| **ZK-Compliant** | Yes | No | No |
| **Coordination** | None | None | Full (collision-free) |
| **Strategy** | Uniform random | Greedy per-agent | Global optimization |
| **Objective** | N/A | Min hits-to-kill | Max total damage |
| **Collisions** | Random | Possible (overkill) | Impossible |
| **Complexity** | O(1) | O(n) per agent | O(n³) for all agents |

---

### 9. Limitations

1. **Not ZK-Compliant**: Uses privileged state information (true remaining attributes)
2. **Static Assignment**: Computes assignment once per step; doesn't adapt mid-step
3. **Homogeneous Objective**: All agents optimize the same dot-product metric
4. **No Temporal Planning**: Greedy per-step; doesn't consider future states
5. **Computational Cost**: O(n³) complexity for Hungarian algorithm

#### 9.1 The Dot Product Assumption

The dot-product score assumes that **higher remaining attributes = more valuable target**. This may not always align with mission objectives:

```
Example: When Dot Product Misleads

Target 0: {armor: 100.0, shields: 100.0}  → High dot-product score
Target 1: {armor: 10.0, shields: 10.0}    → Low dot-product score

If Target 1 is actually more dangerous (e.g., higher threat),
the dot-product metric would incorrectly prioritize Target 0.
```

The oracle maximizes **damage dealt**, not necessarily **threat reduction**.

---

### 10. API Summary

#### 10.1 Constructor

```python
OptimalAssignmentOracle(
    agent_weapon_profiles: Dict[str, Dict[str, float]],
    seed: Optional[int] = None,  # Unused, kept for interface consistency
    allow_noop: bool = True,
)
```

#### 10.2 Methods

```python
def select_actions(
    observations: Dict[str, np.ndarray],
    num_targets: int,
    targets_state: List[Dict[str, float]],  # PRIVILEGED
) -> Dict[str, int]
```

**Note**: No `select_action` method for single agents because the assignment is inherently a multi-agent coordination problem.

---

### 11. Usage Example

```python
from tabula_drone.policies.max_damage_oracle import OptimalAssignmentOracle

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

# Privileged target state (from environment info dict)
targets_state = [
    {"armor": 50.0, "shields": 30.0},  # Target 0
    {"armor": 40.0, "shields": 20.0},  # Target 1
]

# Select actions for all agents (coordinated)
actions = oracle.select_actions(observations, num_targets=2, targets_state=targets_state)
# Result: {"drone_0": 2, "drone_1": 1}
# drone_0 → Target 1, drone_1 → Target 0 (globally optimal)
```

---

### 12. Implementation Details

#### 12.1 Vector Construction

1. **Attribute Names**: Extracted from first target's keys, sorted alphabetically
2. **Agent Vectors**: `[damage_attr1, damage_attr2, ...]` for each agent
3. **Target Vectors**: `[remaining_attr1, remaining_attr2, ...]` for each target
4. Missing attributes default to 0.0

#### 12.2 Active Target Filtering

Before solving the assignment:
1. Parse observation to identify active targets (active flag > 0.5)
2. Build score matrix only for active targets
3. Map assignment results back to original target indices

#### 12.3 SciPy's Linear Sum Assignment

- Uses `scipy.optimize.linear_sum_assignment`
- Input: Cost matrix (negated score matrix, since scipy minimizes)
- Output: Row and column indices of optimal assignment
- Complexity: O(n³) where n = max(agents, targets)

---

### 13. File References

| File | Purpose |
|------|---------|
| `tabula_drone/policies/max_damage_oracle.py` | Implementation |
| `tests/test_max_damage_oracle.py` | Unit tests |
| `docs/policies/max_damage_oracle.md` | Detailed documentation |

---

*Specification for OptimalAssignmentOracle. Last updated: December 2024.*
