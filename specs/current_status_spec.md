# Current Implementation Status

## Overview

This document describes the **current state** of the TabulaDrone ZK-MRTA simulation as implemented. It serves as a factual reference for what exists in the codebase today.

> **Last Updated:** December 2024

---

## 1. Entity Model (Implemented)

### 1.1 TargetState

Defined in `tabula_drone/core/states.py`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str` | Unique identifier (e.g., `"target_0"`) |
| `position` | `Tuple[float, float]` | Static 2D coordinates `(x, y)` |
| `class_type` | `str` | Class label determining initial attributes |
| `attributes` | `AttributeProfile` | Multi-attribute health profile (see 1.4) |
| `is_active` | `bool` | `True` if any attribute > 0, else `False` |

**Backward Compatibility Properties:**
- `hp_current` → Sum of all current attribute values
- `hp_initial` → Sum of all initial attribute values

**Class-to-Attribute Mapping:** Configurable via `class_attribute_mapping` parameter (no hardcoded defaults).

---

### 1.2 DroneState

Defined in `tabula_drone/core/states.py`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str` | Unique identifier (e.g., `"drone_0"`) |
| `position` | `Tuple[float, float]` | Static 2D coordinates `(x, y)` |
| `ammo_used` | `int` | Running count of shots fired (starts at 0) |
| `weapon_type` | `str` | Weapon category (e.g., `"light"`, `"medium"`, `"heavy"`) |
| `damage_profile` | `Dict[str, float]` | Damage per attribute (e.g., `{"armor": 10.0, "shields": 5.0}`) |

**Backward Compatibility Property:**
- `damage_per_shot` → Sum of all damage profile values

**Weapon-to-Damage Mapping:** Configurable via `weapon_damage_profile_mapping` parameter (no hardcoded defaults).

> **Note:** Drones have **unlimited ammo**. The `ammo_used` field is a counter for metrics, not a constraint.

---

### 1.3 WorldState

Defined in `tabula_drone/core/states.py`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `world_size` | `Tuple[float, float]` | `(width, height)` bounds of 2D environment |
| `time_step` | `int` | Current discrete step index (0-indexed) |
| `max_steps` | `int` | Maximum allowed steps per episode |
| `scenario_id` | `str` | Identifier for scenario configuration |
| `seed` | `Optional[int]` | Random seed for reproducibility |

---

### 1.4 AttributeProfile

Defined in `tabula_drone/core/states.py`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `attributes` | `Dict[str, float]` | Current values for each attribute (mutable) |
| `initial_values` | `Dict[str, float]` | Original values at creation (immutable reference) |

**Methods:**
- `apply_damage(damage_profile)` → Reduces each attribute by corresponding damage value
- `is_depleted()` → Returns `True` if ALL attributes are ≤ 0
- `get_total()` → Returns sum of all current attribute values

> **Note:** A target is considered neutralized when ALL attributes reach zero or below.

---

## 2. High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        EPISODE LIFECYCLE                         │
└─────────────────────────────────────────────────────────────────┘

1. SCENARIO GENERATION
   ┌──────────────────┐
   │ ScenarioBuilder  │
   │                  │
   │ • Drone positions│──────┐
   │ • Weapon distrib.│      │
   │ • Target count   │      ▼
   │ • Class distrib. │   drones_config, targets_config
   │ • Spatial constr.│
   └──────────────────┘

2. ENVIRONMENT INITIALIZATION
   ┌──────────────────┐
   │ DroneEngageZKMRTA│
   │                  │
   │ • Creates drones │◄──── drones_config
   │ • Creates targets│◄──── targets_config
   │ • Initializes    │
   │   WorldState     │
   └──────────────────┘

3. EPISODE LOOP (Sequential Processing)
   ┌──────────────────────────────────────────────────────────────┐
   │                                                              │
   │  ┌─────────────┐    observations    ┌─────────────────────┐ │
   │  │ Environment │ ─────────────────► │       Policy        │ │
   │  │             │                    │                     │ │
   │  │ • Targets   │    actions         │ • RandomPolicy      │ │
   │  │ • Drones    │ ◄───────────────── │ • OracleTimeToKill  │ │
   │  │ • World     │                    │ • OptimalAssignment │ │
   │  └─────────────┘                    └─────────────────────┘ │
   │        │                                                    │
   │        ▼                                                    │
   │  ┌─────────────┐                                            │
   │  │ Step Logic  │  (Drones processed sequentially in        │
   │  │             │   random order each step)                  │
   │  │ • Validate  │                                            │
   │  │ • Apply dmg │                                            │
   │  │ • Rewards   │  (Killing blow only)                       │
   │  │ • Check end │                                            │
   │  └─────────────┘                                            │
   │                                                              │
   └──────────────────────────────────────────────────────────────┘

4. TERMINATION
   Episode ends when:
   • All targets neutralized (is_active == False for all), OR
   • max_steps reached (truncation)
```

---

## 3. Entity Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                           WORLD                                  │
│  WorldState: bounds, time_step, max_steps, scenario_id, seed    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                     CONTAINS                                 ││
│  │                                                              ││
│  │   ┌──────────────┐              ┌──────────────┐            ││
│  │   │   DRONES     │              │   TARGETS    │            ││
│  │   │              │   engages    │              │            ││
│  │   │ • drone_0    │ ──────────►  │ • target_0   │            ││
│  │   │ • drone_1    │              │ • target_1   │            ││
│  │   │ • ...        │              │ • ...        │            ││
│  │   │              │              │              │            ││
│  │   │ weapon_type  │              │ class_type   │            ││
│  │   │ damage_per   │              │ hp_current   │            ││
│  │   │ _shot        │              │ is_active    │            ││
│  │   └──────────────┘              └──────────────┘            ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘

RELATIONSHIPS:
• World CONTAINS multiple Drones and multiple Targets
• Drones ENGAGE Targets via actions (1-indexed target selection)
• Engagement REDUCES target hp_current by drone's damage_per_shot
• Multiple drones CAN engage same target in same step (cumulative damage)
• Target becomes INACTIVE when hp_current <= 0
```

---

## 4. Action Model

### Action Space (per drone)
- `Discrete(num_targets + 1)`
- **Action 0**: NoOp (do nothing)
- **Action 1 to N**: Fire at target index (1-indexed)

### Action Processing (per step)
1. All drones submit actions simultaneously
2. For each `Engage(target_id)` action:
   - Drone's `ammo_used` increments by 1
   - Target's `hp_current` decreases by drone's `damage_per_shot`
3. If `hp_current <= 0`: target's `is_active` set to `False`
4. Overkill tracked when damage exceeds remaining HP

---

## 5. Observation Model (ZK-Compliant)

Each drone observes a flat array of shape `(3 * num_targets,)`:

```
[target_0_x, target_0_y, target_0_active,
 target_1_x, target_1_y, target_1_active,
 ...]
```

**Exposed:**
- Target positions (x, y)
- Target active status (1.0 or 0.0)

**Hidden (ZK compliance):**
- Target HP values (current or initial)
- Target class types
- Drone damage values
- Other drones' states or actions

---

## 6. Policies

### 6.1 RandomPolicy (ZK-Compliant)

Implemented in `tabula_drone/policies/random_policy.py`:

| Property | Implementation |
|----------|----------------|
| **Selection** | Uniform random over valid actions |
| **Valid Actions** | `[NoOp]` (optional) + `[active target indices]` |
| **Memory** | None — stateless per step |
| **Coordination** | None — independent per drone |
| **Seed** | Accepts seed for reproducibility |
| **ZK-Compliant** | Yes — uses only binary active/inactive status |

---

### 6.2 OracleTimeToKillPolicy (Privileged Baseline)

Implemented in `tabula_drone/policies/min_ttk_oracle.py`:

| Property | Implementation |
|----------|----------------|
| **Selection** | Target with minimum estimated hits-to-kill |
| **Formula** | `hits = max_a ceil(remaining_a / damage_a)` |
| **Tie-Breaking** | Random among equal candidates |
| **Memory** | None — stateless per step |
| **Coordination** | None — independent per drone |
| **ZK-Compliant** | **No** — uses true remaining attribute values |

---

### 6.3 OptimalAssignmentOracle (Privileged Baseline)

Implemented in `tabula_drone/policies/max_damage_oracle.py`:

| Property | Implementation |
|----------|----------------|
| **Selection** | Globally optimal drone-to-target assignment |
| **Algorithm** | Linear sum assignment (SciPy) |
| **Objective** | Maximize total dot-product score |
| **Constraint** | At most one drone per target (collision-free) |
| **Coordination** | **Yes** — centralized global optimization |
| **ZK-Compliant** | **No** — uses true remaining attribute values |

---

## 7. Reward Model

- **+1.0** to the drone that delivers the **killing blow** (neutralizes the target)
- Sequential processing: only one drone can neutralize a target per step
- Shots at already-neutralized targets are wasted (ammo counted, no reward)
- No penalty for NoOp or missed shots

---

## 8. Metrics Collected

| Metric | Scope | Description |
|--------|-------|-------------|
| `steps` | Episode | Number of steps until termination |
| `targets_neutralized` | Episode | Count of targets with `is_active == False` |
| `total_ammo_used` | Episode | Sum of all drones' `ammo_used` |
| `total_overkill` | Episode | Sum of excess damage beyond neutralization |
| `agent_rewards` | Per-drone | Cumulative reward per drone |
| `done_reason` | Episode | `"all_targets_neutralized"` or `"max_steps"` |

---

## 9. Module Structure

```
tabula_drone/
├── config/
│   ├── __init__.py
│   └── config_loader.py       # Configuration loading utilities
├── core/
│   ├── __init__.py
│   └── states.py              # DroneState, TargetState, WorldState, AttributeProfile
├── envs/
│   ├── __init__.py
│   └── drone_engage_zk_mrta_v0.py  # PettingZoo ParallelEnv implementation
├── logging/
│   ├── __init__.py
│   └── episode_logger.py      # JSON episode capture for replay/analysis
├── policies/
│   ├── __init__.py
│   ├── random_policy.py       # ZK-compliant random action selection
│   ├── min_ttk_oracle.py      # Oracle: minimum time-to-kill selection
│   └── max_damage_oracle.py   # Oracle: optimal assignment via linear sum
├── scenarios/
│   ├── __init__.py
│   ├── scenario_builder.py    # Fluent API for scenario generation
│   └── weapon_assignment.py   # Weapon distribution utilities
└── utils/
    └── __init__.py

viewer/                        # Visualization module
├── components/
│   ├── base/
│   ├── containers/
│   └── panels/
├── __main__.py
├── cli.py
└── draw.py

docs/
└── policies/
    ├── random_policy.md       # Policy documentation
    ├── min_ttk_oracle.md
    └── max_damage_oracle.md

main_zk_mrta.py                # Demo script / entry point
specs/
├── current_status_spec.md     # This document
└── high_level_specification_random.md
tests/
├── test_config_loader.py
├── test_drone_engage_zk_mrta_v0.py
├── test_episode_logger.py
├── test_max_damage_oracle.py
├── test_min_ttk_oracle.py
├── test_scenario_builder.py
└── test_state_adapter.py
```

---

## 10. Known Deviations from Original Specification

| Spec Requirement | Current Implementation | Notes |
|------------------|------------------------|-------|
| Finite ammo (`ammo`, `ammo_max`) | Unlimited ammo (`ammo_used` counter) | Intentional simplification for MVP |
| Single HP value per target | Multi-attribute `AttributeProfile` | More flexible damage model |
| Single damage value per drone | `damage_profile` dict per attribute | Matches multi-attribute targets |
| Simultaneous action processing | Sequential (random order) | Enables killing-blow reward |
| Shared cooperative reward | Killing-blow-only reward | Single drone rewarded per kill |
| Per-class/zone metrics | Aggregate metrics only | Not yet implemented |
| Hardcoded class/weapon mappings | Configurable via parameters | Required at env/builder init |

---

## 11. Logging & Replay

Implemented in `tabula_drone/logging/episode_logger.py`:

**EpisodeLogger** captures:
- Initial scenario setup (positions, weapons, classes)
- Per-step actions, rewards, and state
- Episode summary (total steps, rewards, termination reason)

**JSON Schema (v1.1):**
```json
{
  "version": "1.1",
  "episode_id": "<uuid>",
  "timestamp": "<ISO8601>",
  "rng_seed": <int|null>,
  "config": {...},
  "scenario": {...},
  "steps": [...],
  "summary": {...}
}
```

---

*Document generated from codebase analysis. Last updated: December 2024.*
