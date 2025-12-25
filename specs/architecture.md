# TabulaDrone Architecture

## Overview

TabulaDrone is a **Zero-Knowledge Multi-Robot Task Allocation (ZK-MRTA)** simulation framework built on PettingZoo. It models scenarios where multiple static drones engage multiple static targets under information constraints.

> **Last Updated:** December 2024

---

## 1. System Context

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TABULADRONE SYSTEM                                │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Scenario   │    │ Environment  │    │   Policies   │                   │
│  │   Builder    │───►│  (PettingZoo)│◄───│              │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         │                   ▼                   │                            │
│         │            ┌──────────────┐           │                            │
│         │            │   Episode    │           │                            │
│         └───────────►│   Logger     │◄──────────┘                            │
│                      └──────────────┘                                        │
│                             │                                                │
│                             ▼                                                │
│                      ┌──────────────┐                                        │
│                      │    Viewer    │                                        │
│                      │  (Replay)    │                                        │
│                      └──────────────┘                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Actors:**
- **ScenarioBuilder** — Generates randomized drone/target configurations
- **Environment** — PettingZoo ParallelEnv managing simulation state
- **Policies** — Action selection strategies (ZK-compliant and oracle baselines)
- **EpisodeLogger** — Captures episode data for replay and analysis
- **Viewer** — Visualizes logged episodes

---

## 2. Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION LAYER                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ main_zk_mrta.py │  │     viewer/     │  │     tests/      │              │
│  │   (Demo/CLI)    │  │  (Visualization)│  │  (Test Suite)   │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               DOMAIN LAYER                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │    policies/    │  │    scenarios/   │  │    logging/     │              │
│  │                 │  │                 │  │                 │              │
│  │ • RandomPolicy  │  │ • ScenarioBuilder│ │ • EpisodeLogger │              │
│  │ • MinTTKOracle  │  │ • WeaponAssign  │  │                 │              │
│  │ • MaxDmgOracle  │  │                 │  │                 │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ENVIRONMENT LAYER                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        envs/                                         │    │
│  │                                                                      │    │
│  │  DroneEngageZKMRTA (PettingZoo ParallelEnv)                         │    │
│  │  • Action/Observation spaces                                         │    │
│  │  • Step logic (sequential processing)                                │    │
│  │  • Reward computation (killing blow)                                 │    │
│  │  • Termination conditions                                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               CORE LAYER                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        core/states.py                                │    │
│  │                                                                      │    │
│  │  • DroneState        — Agent representation                          │    │
│  │  • TargetState       — Task representation                           │    │
│  │  • WorldState        — Environment metadata                          │    │
│  │  • AttributeProfile  — Multi-attribute health system                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        config/                                       │    │
│  │                                                                      │    │
│  │  • config_loader.py  — Configuration utilities                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Diagram

### 3.1 Core Data Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA MODEL                                      │
│                                                                              │
│  ┌─────────────────┐         ┌─────────────────┐                            │
│  │   DroneState    │         │   TargetState   │                            │
│  ├─────────────────┤         ├─────────────────┤                            │
│  │ id: str         │         │ id: str         │                            │
│  │ position: (x,y) │         │ position: (x,y) │                            │
│  │ ammo_used: int  │         │ class_type: str │                            │
│  │ weapon_type: str│         │ is_active: bool │                            │
│  │ damage_profile: │ engages │ attributes:     │                            │
│  │   Dict[str,float│────────►│  AttributeProfile                            │
│  └─────────────────┘         └────────┬────────┘                            │
│                                       │                                      │
│                                       │ contains                             │
│                                       ▼                                      │
│                              ┌─────────────────┐                            │
│                              │AttributeProfile │                            │
│                              ├─────────────────┤                            │
│                              │ attributes:     │                            │
│                              │   Dict[str,float│                            │
│                              │ initial_values: │                            │
│                              │   Dict[str,float│                            │
│                              ├─────────────────┤                            │
│                              │ apply_damage()  │                            │
│                              │ is_depleted()   │                            │
│                              │ get_total()     │                            │
│                              └─────────────────┘                            │
│                                                                              │
│  ┌─────────────────┐                                                        │
│  │   WorldState    │                                                        │
│  ├─────────────────┤                                                        │
│  │ world_size:(w,h)│                                                        │
│  │ time_step: int  │                                                        │
│  │ max_steps: int  │                                                        │
│  │ scenario_id: str│                                                        │
│  │ seed: int|None  │                                                        │
│  └─────────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Policy Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            POLICY ARCHITECTURE                               │
│                                                                              │
│                         ┌─────────────────────┐                             │
│                         │   <<interface>>     │                             │
│                         │      Policy         │                             │
│                         ├─────────────────────┤                             │
│                         │ select_action()     │                             │
│                         │ select_actions()    │                             │
│                         └──────────┬──────────┘                             │
│                                    │                                         │
│              ┌─────────────────────┼─────────────────────┐                  │
│              │                     │                     │                  │
│              ▼                     ▼                     ▼                  │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐           │
│  │  RandomPolicy   │   │OracleTimeToKill │   │OptimalAssignment│           │
│  │  (ZK-Compliant) │   │   Policy        │   │    Oracle       │           │
│  ├─────────────────┤   ├─────────────────┤   ├─────────────────┤           │
│  │ • Uniform random│   │ • Min hits-to-  │   │ • Linear sum    │           │
│  │ • Active targets│   │   kill selection│   │   assignment    │           │
│  │ • No memory     │   │ • Per-agent     │   │ • Global optimal│           │
│  │ • Independent   │   │ • Privileged    │   │ • Coordinated   │           │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘           │
│         │                      │                      │                     │
│         │                      │                      │                     │
│    ZK-Compliant           Privileged             Privileged                 │
│    (observations)         (true state)           (true state)               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Sequence Diagram: Episode Lifecycle

```
┌─────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Main   │     │ScenarioBuilder    │ Environment │     │   Policy    │
└────┬────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
     │                 │                   │                   │
     │  with_drones()  │                   │                   │
     │────────────────►│                   │                   │
     │  with_targets() │                   │                   │
     │────────────────►│                   │                   │
     │     build()     │                   │                   │
     │────────────────►│                   │                   │
     │◄────────────────│                   │                   │
     │  (drones_config,│                   │                   │
     │   targets_config)                   │                   │
     │                 │                   │                   │
     │                 │    __init__()     │                   │
     │─────────────────────────────────────►                   │
     │                 │                   │                   │
     │                 │     reset()       │                   │
     │─────────────────────────────────────►                   │
     │◄────────────────────────────────────│                   │
     │                 │  (observations,   │                   │
     │                 │   infos)          │                   │
     │                 │                   │                   │
     │─────────────────────────────────────────────────────────►
     │                 │                   │  select_actions() │
     │◄────────────────────────────────────────────────────────│
     │                 │                   │     (actions)     │
     │                 │                   │                   │
     │                 │     step()        │                   │
     │─────────────────────────────────────►                   │
     │◄────────────────────────────────────│                   │
     │                 │  (obs, rewards,   │                   │
     │                 │   term, trunc,    │                   │
     │                 │   infos)          │                   │
     │                 │                   │                   │
     │         [loop until terminated/truncated]               │
     │                 │                   │                   │
```

---

## 5. Data Flow: Step Processing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STEP PROCESSING FLOW                                 │
│                                                                              │
│  INPUT: actions = {drone_0: 2, drone_1: 1, drone_2: 0}                      │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 1. VALIDATE ACTIONS                                                   │   │
│  │    • Check all agents provided actions                                │   │
│  │    • Check action range [0, num_targets]                              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 2. SHUFFLE PROCESSING ORDER                                           │   │
│  │    • Randomize drone order for fairness                               │   │
│  │    • Example: [drone_1, drone_0, drone_2]                             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 3. SEQUENTIAL PROCESSING (for each drone in shuffled order)           │   │
│  │                                                                       │   │
│  │    ┌─────────────────────────────────────────────────────────────┐   │   │
│  │    │ IF action == 0 (NoOp):                                       │   │   │
│  │    │    → Skip                                                    │   │   │
│  │    │                                                              │   │   │
│  │    │ ELSE (Fire at target):                                       │   │   │
│  │    │    → Increment drone.ammo_used                               │   │   │
│  │    │    → IF target.is_active:                                    │   │   │
│  │    │         → Apply damage_profile to target.attributes          │   │   │
│  │    │         → IF target.attributes.is_depleted():                │   │   │
│  │    │              → Set target.is_active = False                  │   │   │
│  │    │              → Award +1.0 reward to this drone               │   │   │
│  │    │              → Track overkill                                │   │   │
│  │    │    → ELSE:                                                   │   │   │
│  │    │         → Wasted shot (no damage, no reward)                 │   │   │
│  │    └─────────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 4. TIME PROGRESSION                                                   │   │
│  │    • world.time_step += 1                                             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 5. CHECK TERMINATION                                                  │   │
│  │    • all_targets_neutralized → terminated = True                      │   │
│  │    • max_steps_reached → truncated = True                             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 6. COMPUTE OBSERVATIONS                                               │   │
│  │    • For each target: [x, y, is_active]                               │   │
│  │    • All agents receive identical observations                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  OUTPUT: (observations, rewards, terminations, truncations, infos)          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Module Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MODULE DEPENDENCIES                                 │
│                                                                              │
│                           ┌─────────────┐                                   │
│                           │main_zk_mrta │                                   │
│                           └──────┬──────┘                                   │
│                                  │                                           │
│          ┌───────────────────────┼───────────────────────┐                  │
│          │                       │                       │                  │
│          ▼                       ▼                       ▼                  │
│  ┌───────────────┐       ┌───────────────┐       ┌───────────────┐         │
│  │   scenarios/  │       │    envs/      │       │   policies/   │         │
│  │               │       │               │       │               │         │
│  │ScenarioBuilder│──────►│DroneEngageZK  │◄──────│ RandomPolicy  │         │
│  │               │       │    MRTA       │       │ MinTTKOracle  │         │
│  └───────┬───────┘       └───────┬───────┘       │ MaxDmgOracle  │         │
│          │                       │               └───────┬───────┘         │
│          │                       │                       │                  │
│          │                       ▼                       │                  │
│          │               ┌───────────────┐               │                  │
│          │               │   logging/    │               │                  │
│          │               │               │               │                  │
│          │               │EpisodeLogger  │               │                  │
│          │               └───────────────┘               │                  │
│          │                       │                       │                  │
│          └───────────────────────┼───────────────────────┘                  │
│                                  │                                           │
│                                  ▼                                           │
│                          ┌───────────────┐                                  │
│                          │    core/      │                                  │
│                          │               │                                  │
│                          │ DroneState    │                                  │
│                          │ TargetState   │                                  │
│                          │ WorldState    │                                  │
│                          │AttributeProfile                                  │
│                          └───────────────┘                                  │
│                                  │                                           │
│                                  ▼                                           │
│                          ┌───────────────┐                                  │
│                          │   config/     │                                  │
│                          │               │                                  │
│                          │config_loader  │                                  │
│                          └───────────────┘                                  │
│                                                                              │
│  External Dependencies:                                                      │
│  • numpy — Array operations                                                  │
│  • gymnasium — Space definitions                                             │
│  • pettingzoo — ParallelEnv base class                                       │
│  • scipy — Linear sum assignment (OptimalAssignmentOracle)                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Key Design Decisions

### 7.1 Multi-Attribute Health System

**Decision:** Targets have multiple named attributes (e.g., armor, shields) instead of single HP.

**Rationale:**
- Enables heterogeneous weapon effectiveness
- Supports more complex damage models
- Backward compatible via `hp_current` property

### 7.2 Sequential Drone Processing

**Decision:** Drones are processed sequentially in random order each step.

**Rationale:**
- Enables clear "killing blow" attribution
- Avoids ambiguity in simultaneous neutralization
- Random order ensures fairness

### 7.3 Killing-Blow Reward Model

**Decision:** Only the drone that delivers the killing blow receives +1.0 reward.

**Rationale:**
- Clear credit assignment
- Encourages efficient target selection
- Simpler than shared credit models

### 7.4 Configurable Mappings (No Hardcoded Defaults)

**Decision:** `class_attribute_mapping` and `weapon_damage_profile_mapping` are required parameters.

**Rationale:**
- Explicit configuration over implicit defaults
- Prevents hidden assumptions
- Enables flexible scenario design

### 7.5 ZK-Compliant Observations

**Decision:** Observations expose only target positions and binary active status.

**Rationale:**
- Core ZK-MRTA constraint
- Agents cannot observe HP, class, or damage values
- Enables fair comparison of information-limited policies

---

## 8. Extension Points

| Extension Point | Location | Purpose |
|-----------------|----------|---------|
| New Policies | `tabula_drone/policies/` | Add new action selection strategies |
| New Metrics | `EpisodeLogger._build_summary()` | Capture additional episode metrics |
| New Observations | `DroneEngageZKMRTA._compute_observations()` | Expose additional state (breaks ZK) |
| New Reward Models | `DroneEngageZKMRTA.step()` | Alternative credit assignment |
| New Scenarios | `ScenarioBuilder` | Additional generation constraints |

---

## 9. File Reference

| File | Purpose |
|------|---------|
| `tabula_drone/core/states.py` | Core dataclasses |
| `tabula_drone/envs/drone_engage_zk_mrta_v0.py` | PettingZoo environment |
| `tabula_drone/policies/random_policy.py` | ZK-compliant baseline |
| `tabula_drone/policies/min_ttk_oracle.py` | Privileged min-TTK oracle |
| `tabula_drone/policies/max_damage_oracle.py` | Privileged optimal assignment |
| `tabula_drone/scenarios/scenario_builder.py` | Scenario generation |
| `tabula_drone/logging/episode_logger.py` | Episode capture |
| `tabula_drone/config/config_loader.py` | Configuration utilities |
| `viewer/` | Visualization module |
| `main_zk_mrta.py` | Demo entry point |

---

*Architecture document generated from codebase analysis. Last updated: December 2024.*
