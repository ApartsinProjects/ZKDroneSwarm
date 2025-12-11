# Current Implementation Status

## Overview

This document describes the **current state** of the TabulaDrone ZK-MRTA simulation as implemented. It serves as a factual reference for what exists in the codebase today.

---

## 1. Entity Model (Implemented)

### 1.1 TargetState

Defined in `tabula_drone/core/states.py`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str` | Unique identifier (e.g., `"target_0"`) |
| `position` | `Tuple[float, float]` | Static 2D coordinates `(x, y)` |
| `class_type` | `str` | Class label (`"A"`, `"B"`, `"C"`) determining initial HP |
| `hp_initial` | `float` | Initial hit points based on class_type |
| `hp_current` | `float` | Current remaining hit points |
| `is_active` | `bool` | `True` if `hp_current > 0`, else `False` |

**Default Class HP Mapping:**
- `"A"` → 100.0 HP
- `"B"` → 150.0 HP
- `"C"` → 200.0 HP

---

### 1.2 DroneStateZK

Defined in `tabula_drone/envs/drone_engage_zk_mrta_v0.py`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str` | Unique identifier (e.g., `"drone_0"`) |
| `position` | `Tuple[float, float]` | Static 2D coordinates `(x, y)` |
| `ammo_used` | `int` | Running count of shots fired (starts at 0) |
| `weapon_type` | `str` | Weapon category (`"light"`, `"medium"`, `"heavy"`) |
| `damage_per_shot` | `float` | Damage value based on weapon_type |

**Default Weapon Damage Mapping:**
- `"light"` → 10.0 damage
- `"medium"` → 25.0 damage
- `"heavy"` → 50.0 damage

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

3. EPISODE LOOP
   ┌──────────────────────────────────────────────────────────────┐
   │                                                              │
   │  ┌─────────────┐    observations    ┌─────────────────────┐ │
   │  │ Environment │ ─────────────────► │    RandomPolicy     │ │
   │  │             │                    │                     │ │
   │  │ • Targets   │    actions         │ • Uniform selection │ │
   │  │ • Drones    │ ◄───────────────── │ • ZK-compliant      │ │
   │  │ • World     │                    │ • No memory         │ │
   │  └─────────────┘                    └─────────────────────┘ │
   │        │                                                    │
   │        ▼                                                    │
   │  ┌─────────────┐                                            │
   │  │ Step Logic  │                                            │
   │  │             │                                            │
   │  │ • Validate  │                                            │
   │  │ • Apply dmg │                                            │
   │  │ • Rewards   │                                            │
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

## 6. Random Policy

Implemented in `tabula_drone/policies/random_policy.py`:

| Property | Implementation |
|----------|----------------|
| **Selection** | Uniform random over valid actions |
| **Valid Actions** | `[NoOp]` (optional) + `[active target indices]` |
| **Memory** | None — stateless per step |
| **Coordination** | None — independent per drone |
| **Seed** | Accepts seed for reproducibility |

---

## 7. Reward Model

- **+1.0** to each drone that fired at a target that was neutralized this step
- Multiple drones can receive reward for same target (shared cooperative reward)
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
├── core/
│   └── states.py          # DroneState, TargetState, WorldState dataclasses
├── envs/
│   └── drone_engage_zk_mrta_v0.py  # PettingZoo ParallelEnv implementation
├── policies/
│   └── random_policy.py   # ZK-compliant random action selection
├── scenarios/
│   ├── scenario_builder.py    # Fluent API for scenario generation
│   └── weapon_assignment.py   # Weapon distribution utilities
└── utils/
    └── (empty)

main_zk_mrta.py            # Demo script / entry point
specs/
└── high_level_specification_random.md  # Target specification
tests/
├── test_drone_engage_zk_mrta_v0.py
└── test_scenario_builder.py
```

---

## 10. Known Deviations from Specification

| Spec Requirement | Current Implementation | Notes |
|------------------|------------------------|-------|
| Finite ammo (`ammo`, `ammo_max`) | Unlimited ammo (`ammo_used` counter) | Intentional simplification for MVP |
| `DroneState` with ammo fields | `DroneStateZK` with `ammo_used` | Separate dataclass in env module |
| Per-class/zone metrics | Aggregate metrics only | Not yet implemented |
| Single seed for full reproducibility | Separate env/policy seeds | Set to same value in demo |

---

*Document generated from codebase analysis. Last updated: December 2024.*
