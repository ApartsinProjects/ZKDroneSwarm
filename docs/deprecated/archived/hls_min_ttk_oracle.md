## Min TTK Oracle Policy - High-Level Specification

> **Last Updated:** December 2024

### 1. Scope

This specification defines the **OracleTimeToKillPolicy** (Min TTK Oracle), a **privileged baseline policy** for the ZK-MRTA simulation. This policy is intentionally **not ZK-compliant** and serves as an upper-bound benchmark for comparison with ZK-compliant policies.

**Key Characteristic**: Selects the active target that can be eliminated in the **fewest estimated hits** given each agent's weapon damage profile.

---

### 2. Design Philosophy

The Min TTK Oracle exists to:

1. **Establish an upper performance bound** for comparison with ZK-compliant policies
2. **Validate environment mechanics** and reward structures
3. **Provide a reference implementation** for target prioritization logic

**Privileged Information Used:**
- True remaining attribute values of targets
- Agent's own weapon damage profile

**Not Used (ZK-Compliant Aspects):**
- No inter-agent communication
- No coordination between agents
- No planning ahead (single-step greedy)

---

### 3. Core Algorithm

#### 3.1 Hits-to-Kill Estimation

The policy estimates the number of hits required to neutralize a target using:

```
hits = max_a ceil(rem_a / dmg_a)
```

Where:
- `rem_a` = remaining value of attribute `a` on the target
- `dmg_a` = damage per hit dealt to attribute `a` by the weapon
- The **maximum** is taken across all attributes

#### 3.2 Why Maximum Across Attributes?

A target is only neutralized when **all** of its defensive attributes are depleted to zero. Each attribute depletes independently based on the weapon's damage profile. The "bottleneck" attribute—the one requiring the most hits to deplete—determines the total hits needed.

**Example:**

```
Target State:
  - armor:   25.0
  - shields: 12.0

Weapon Damage Profile:
  - armor:   10.0 per hit
  - shields:  5.0 per hit

Calculation:
  - Armor:   ceil(25.0 / 10.0) = 3 hits
  - Shields: ceil(12.0 / 5.0)  = 3 hits
  - Result:  max(3, 3) = 3 hits to neutralize
```

#### 3.3 Special Cases

| Condition | Result |
|-----------|--------|
| `rem_a <= 0` for all attributes | 0 hits (already neutralized) |
| `dmg_a == 0` and `rem_a > 0` | Infinity (unkillable by this agent) |
| Attribute not in weapon profile | Infinity (unkillable by this agent) |

---

### 4. Target Selection Logic

```
1. Parse observation to identify active targets
2. For each active target:
   - Compute estimated hits-to-kill using privileged state
3. Select target(s) with minimum hits-to-kill
4. If tie: random selection among tied targets (seeded RNG)
5. Return 1-indexed action (target_index + 1)
```

#### 4.1 Edge Case Handling

| Scenario | Behavior |
|----------|----------|
| No active targets | Return NoOp (action 0) |
| All targets unkillable + `allow_noop=True` | Return NoOp (action 0) |
| All targets unkillable + `allow_noop=False` | Random active target |
| Multiple targets with same min hits | Random tie-break (deterministic with seed) |

---

### 5. Action Space

| Action | Meaning |
|--------|---------|
| `0` | NoOp (do nothing) - only when `allow_noop=True` |
| `1` to `N` | Fire at target index `i-1` (1-indexed) |

---

### 6. Policy Requirements

#### 6.1 Privileged State Access

The policy **requires** access to privileged target state at action-selection time:

```python
targets_state: List[Dict[str, float]]
# Example: [{"armor": 25.0, "shields": 10.0}, {"armor": 0.0, "shields": 5.0}]
```

This is the key difference from ZK-compliant policies, which only receive the standard observation array.

#### 6.2 Per-Agent Weapon Profiles

Each agent can have a **different weapon profile**:

```python
agent_weapon_profiles: Dict[str, Dict[str, float]]
# Example: {
#     "drone_0": {"armor": 10.0, "shields": 5.0},
#     "drone_1": {"armor": 15.0, "shields": 3.0},
# }
```

This enables heterogeneous agent scenarios where different drones have different capabilities.

#### 6.3 No Coordination

Agents select independently based on their own weapon profile. There is **no coordination** between agents:

- Each agent computes its own best target
- Multiple agents may select the same target
- This can lead to overkill (wasteful damage)

#### 6.4 Reproducibility

The policy uses a seeded RNG for tie-breaking:

- Same `seed` + same scenario → same action sequences
- Deterministic behavior for testing and benchmarking

---

### 7. Comparison with Random Policy

| Property | RandomPolicy | OracleTimeToKillPolicy |
|----------|--------------|------------------------|
| **ZK-Compliant** | Yes | No |
| **Target Selection** | Uniform random | Min hits-to-kill |
| **Uses Privileged State** | No | Yes (target attributes) |
| **Coordination** | None | None |
| **Memory** | None | None |
| **Tie-Breaking** | N/A (all equal) | Random (seeded) |

---

### 8. Limitations

1. **Not ZK-Compliant**: Uses privileged state information (true remaining attributes)
2. **No Coordination**: Agents select independently; may all fire at the same target
3. **Static Weapon Profiles**: Assumes constant damage per hit (no weapon degradation)
4. **Single-Step Greedy**: Does not plan ahead or consider future states

#### 8.1 The Coordination Problem

When multiple agents use this policy, they all independently compute the same "best" target and may all fire at it simultaneously, leading to overkill:

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
```

This limitation is **by design**—the oracle establishes an upper bound for what's achievable with perfect target state knowledge but **without** inter-agent communication.

---

### 9. API Summary

#### 9.1 Constructor

```python
OracleTimeToKillPolicy(
    agent_weapon_profiles: Dict[str, Dict[str, float]],
    seed: Optional[int] = None,
    allow_noop: bool = True,
)
```

#### 9.2 Methods

```python
def select_action(
    agent_id: str,
    observation: np.ndarray,
    num_targets: int,
    targets_state: List[Dict[str, float]],  # PRIVILEGED
) -> int

def select_actions(
    observations: Dict[str, np.ndarray],
    num_targets: int,
    targets_state: List[Dict[str, float]],  # PRIVILEGED
) -> Dict[str, int]
```

---

### 10. Usage Example

```python
from tabula_drone.policies.min_ttk_oracle import OracleTimeToKillPolicy

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

# Privileged target state (from environment info dict)
targets_state = [
    {"armor": 30.0, "shields": 10.0},  # Target 0: 3 hits (armor bottleneck)
    {"armor": 15.0, "shields": 20.0},  # Target 1: 4 hits (shields bottleneck)
]

# Select actions for all agents
actions = policy.select_actions(observations, num_targets=2, targets_state=targets_state)
# Both agents select Target 0 (fewer hits required)
```

---

### 11. File References

| File | Purpose |
|------|---------|
| `tabula_drone/policies/min_ttk_oracle.py` | Implementation |
| `tests/test_min_ttk_oracle.py` | Unit tests |
| `docs/policies/min_ttk_oracle.md` | Detailed documentation |

---

*Specification for OracleTimeToKillPolicy. Last updated: December 2024.*
