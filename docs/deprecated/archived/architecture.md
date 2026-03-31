# TabulaDrone Architecture

## Overview

TabulaDrone is a **Zero-Knowledge Multi-Robot Task Allocation (ZK-MRTA)** simulation framework built on PettingZoo. It models scenarios where multiple static drones engage multiple static targets under information constraints.

> **Last Updated:** January 2025

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
│  │ • MinTTKOracle  │  │ • WeaponAssign  │  │ • RunManager    │              │
│  │ • MaxDmgOracle  │  │                 │  │                 │              │
│  │ • CF Policies   │  │                 │  │                 │              │
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
│  │  • Reward computation (dominant attribute / HP reduction)            │    │
│  │  • Observation modes (minimal / collaborative)                        │    │
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
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         PRIVILEGED ORACLES                             │  │
│  │  ┌─────────────────────┐         ┌─────────────────────┐              │  │
│  │  │ OracleTimeToKill    │         │ OptimalAssignment   │              │  │
│  │  │     Policy          │         │      Oracle         │              │  │
│  │  ├─────────────────────┤         ├─────────────────────┤              │  │
│  │  │ • Min hits-to-kill  │         │ • Linear sum        │              │  │
│  │  │ • Per-agent greedy  │         │   assignment        │              │  │
│  │  │ • True state access │         │ • Global optimal    │              │  │
│  │  └─────────────────────┘         └─────────────────────┘              │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      ZK-COMPLIANT POLICIES                             │  │
│  │                                                                        │  │
│  │  ┌─────────────────┐                                                  │  │
│  │  │  RandomPolicy   │  Baseline: uniform random over active targets    │  │
│  │  └─────────────────┘                                                  │  │
│  │                                                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │              COLLABORATIVE FILTERING (CF) POLICIES               │  │  │
│  │  │                                                                  │  │  │
│  │  │  ┌─────────────────────┐                                        │  │  │
│  │  │  │    BaseCFPolicy     │  Abstract base class for SGD-based     │  │  │
│  │  │  │    (abstract)       │  matrix factorization policies         │  │  │
│  │  │  └──────────┬──────────┘                                        │  │  │
│  │  │             │                                                    │  │  │
│  │  │      ┌──────┴──────┐                                            │  │  │
│  │  │      ▼             ▼                                            │  │  │
│  │  │  ┌─────────────┐ ┌─────────────────┐                            │  │  │
│  │  │  │SelfishEp    │ │CoordinatedEp    │                            │  │  │
│  │  │  │GreedyCF     │ │GreedyCFPolicy   │                            │  │  │
│  │  │  │Policy       │ │                 │                            │  │  │
│  │  │  ├─────────────┤ ├─────────────────┤                            │  │  │
│  │  │  │• ε-greedy   │ │• Hungarian alg  │                            │  │  │
│  │  │  │• Per-agent  │ │• Implicit coord │                            │  │  │
│  │  │  │• Private LV │ │• Private LV     │                            │  │  │
│  │  │  └─────────────┘ └─────────────────┘                            │  │  │
│  │  │                                                                  │  │  │
│  │  │  ┌─────────────────────┐                                        │  │  │
│  │  │  │    UCBCFPolicy      │  UCB1 exploration with shared latent   │  │  │
│  │  │  │                     │  vectors (centralized learning)        │  │  │
│  │  │  └─────────────────────┘                                        │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Legend:                                                                     │
│  • LV = Latent Vectors (agent_lv, target_lv, other_agents_lv)               │
│  • CF policies learn agent-target compatibility from observed rewards        │
│  • Decentralized: one policy instance per agent, no shared state            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 CF Policy Learning Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     COLLABORATIVE FILTERING MODEL                            │
│                                                                              │
│  Each agent maintains PRIVATE latent vectors:                                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Agent i's Private State:                                           │    │
│  │                                                                      │    │
│  │  agent_lv[latent_dim]        — This agent's latent vector           │    │
│  │  target_lv[num_targets, latent_dim] — Estimates of all targets      │    │
│  │  other_agents_lv[num_agents, latent_dim] — Estimates of other agents│    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Reward Prediction:                                                          │
│    predicted_reward = (1 + dot(agent_lv, target_lv[t])) / 2                 │
│                                                                              │
│  SGD Update (on observed reward):                                            │
│    error = observed_reward - predicted_reward                                │
│    agent_lv += lr * error * target_lv[t]                                    │
│    target_lv[t] += lr * error * agent_lv                                    │
│    (both vectors normalized after update)                                    │
│                                                                              │
│  Action Selection:                                                           │
│    • SelfishEpGreedyCF: ε-greedy over predicted rewards                     │
│    • CoordinatedEpGreedyCF: Hungarian algorithm on belief matrix            │
│    • UCBCFPolicy: UCB1 score = predicted + c*sqrt(log(t)/visits)            │
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
│  │ 2. FIXED PROCESSING ORDER                                             │   │
│  │    • Drones processed in fixed order (deterministic)                  │   │
│  │    • Order: [drone_0, drone_1, drone_2, ...]                          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 3. SEQUENTIAL PROCESSING (for each drone in fixed order)              │   │
│  │                                                                       │   │
│  │    ┌─────────────────────────────────────────────────────────────┐   │   │
│  │    │ IF action == 0 (NoOp):                                       │   │   │
│  │    │    → Skip                                                    │   │   │
│  │    │                                                              │   │   │
│  │    │ ELSE (Fire at target):                                       │   │   │
│  │    │    → Increment drone.ammo_used                               │   │   │
│  │    │    → IF target.is_active:                                    │   │   │
│  │    │         → Apply damage_profile to target.attributes          │   │   │
│  │    │         → Compute reward (dominant attr or HP reduction)     │   │   │
│  │    │         → IF target.attributes.is_depleted():                │   │   │
│  │    │              → Set target.is_active = False                  │   │   │
│  │    │              → Track overkill                                │   │   │
│  │    │    → ELSE:                                                   │   │   │
│  │    │         → Wasted shot (reward = -1.0)                        │   │   │
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
│  │ 6. COMPUTE OBSERVATIONS (based on observation_mode)                   │   │
│  │                                                                       │   │
│  │    MINIMAL MODE:                                                      │   │
│  │    • For each target: [x, y, is_active]                               │   │
│  │    • All agents receive identical observations                        │   │
│  │                                                                       │   │
│  │    COLLABORATIVE MODE:                                                │   │
│  │    • targets: [x, y, is_active] for each target                       │   │
│  │    • selected_targets: actions from last step (all agents)            │   │
│  │    • observed_rewards: rewards from last step (with noise)            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  OUTPUT: (observations, rewards, terminations, truncations, infos)          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Observation Modes

The environment supports two observation modes, configured via the `observation_mode` parameter:

#### Minimal Mode (`observation_mode="minimal"`)

Default mode for ZK-compliant policies. Each agent receives a flat array:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  MINIMAL OBSERVATION SPACE                                                   │
│                                                                              │
│  Shape: Box(shape=(3 * num_targets,), dtype=float32)                        │
│                                                                              │
│  Layout: [t0_x, t0_y, t0_active, t1_x, t1_y, t1_active, ...]               │
│                                                                              │
│  Example (3 targets):                                                        │
│  [150.0, 200.0, 1.0, 300.0, 450.0, 1.0, 500.0, 100.0, 0.0]                  │
│       ↑     ↑    ↑                              ↑                           │
│    target_0      active=True              target_2 inactive                  │
│                                                                              │
│  Properties:                                                                 │
│  • All agents receive IDENTICAL observations                                 │
│  • No information about HP, class, or damage profiles                        │
│  • Fully ZK-compliant                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Collaborative Mode (`observation_mode="collaborative"`)

Extended mode for CF policies that learn from swarm behavior. Each agent receives a Dict:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  COLLABORATIVE OBSERVATION SPACE                                             │
│                                                                              │
│  Type: Dict with three keys                                                  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  "targets": Box(shape=(3 * num_targets,), dtype=float32)            │    │
│  │                                                                      │    │
│  │  Same as minimal mode: [t0_x, t0_y, t0_active, ...]                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  "selected_targets": Box(shape=(num_agents,), dtype=int32)          │    │
│  │                                                                      │    │
│  │  Actions from PREVIOUS step for all agents                          │    │
│  │  Values: 0 = NoOp, 1-N = target index (1-indexed)                   │    │
│  │                                                                      │    │
│  │  Example (4 agents): [2, 1, 0, 3]                                    │    │
│  │  → drone_0 fired at target 1, drone_1 at target 0,                  │    │
│  │    drone_2 did NoOp, drone_3 fired at target 2                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  "observed_rewards": Box(shape=(num_agents,), dtype=float32)        │    │
│  │                                                                      │    │
│  │  Rewards from PREVIOUS step for all agents (with noise)             │    │
│  │                                                                      │    │
│  │  Example (4 agents): [0.85, 0.12, 0.0, -1.0]                         │    │
│  │  → drone_0 got high reward, drone_3 wasted shot                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Properties:                                                                 │
│  • Enables learning from other agents' experiences                          │
│  • Still ZK-compliant (no HP/class/damage exposed)                          │
│  • Noise applied to rewards for realistic information sharing               │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Noise Model

Collaborative mode supports configurable noise to simulate imperfect information sharing:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  NOISE APPLICATION                                                           │
│                                                                              │
│  For agent i observing agent j's reward:                                     │
│                                                                              │
│  IF i == j (own reward):                                                     │
│      observed_reward = actual_reward + N(0, reward_noise)                   │
│                                                                              │
│  ELSE (other agent's reward):                                                │
│      total_σ = sqrt(reward_noise² + observation_noise²)                     │
│      observed_reward = actual_reward + N(0, total_σ)                        │
│                                                                              │
│  Parameters:                                                                 │
│  • reward_noise: Base noise on all rewards (environment stochasticity)      │
│  • observation_noise: Additional noise when observing others (comm noise)   │
│                                                                              │
│  Example configuration:                                                      │
│    reward_noise = 0.1      → ±10% noise on own rewards                      │
│    observation_noise = 0.05 → Additional ±5% when observing others          │
│    → Others' rewards have σ = sqrt(0.1² + 0.05²) ≈ 0.112                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Mode | Description | Use Case |
|------|-------------|----------|
| `minimal` | Target positions + active status only | RandomPolicy, Oracle policies |
| `collaborative` | Adds `selected_targets` and `observed_rewards` | CF policies (Selfish, Coordinated, UCB) |

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
│          │                       │               │ CF Policies   │         │
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
│  • scipy — Linear sum assignment (OptimalAssignmentOracle, CoordinatedCF)    │
│  • sklearn — Cosine similarity (alignment analysis in main)                  │
│  • tabulate — Output formatting                                              │
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

**Decision:** Drones are processed sequentially in fixed order each step.

**Rationale:**
- Enables clear "killing blow" attribution
- Avoids ambiguity in simultaneous neutralization
- Fixed order ensures deterministic behavior for reproducibility

### 7.3 Reward Model (Configurable)

**Decision:** Reward is computed based on damage effectiveness, with two modes:

1. **Dominant Attribute Mode** (default): Reward = damage to target's dominant attribute / max weapon damage
2. **HP Reduction Mode**: Reward = actual HP reduction / drone's weapon damage

**Additional rules:**
- Wasted shots (firing at inactive targets) receive **-1.0** reward
- Rewards are normalized to enable fair comparison across weapon types

**Rationale:**
- Dominant attribute mode encourages weapon-target matching
- Normalized rewards enable CF policies to learn meaningful compatibility
- Negative reward for wasted shots discourages inefficient behavior

### 7.4 Configurable Mappings (No Hardcoded Defaults)

**Decision:** `class_attribute_mapping` and `weapon_damage_profile_mapping` are required parameters.

**Rationale:**
- Explicit configuration over implicit defaults
- Prevents hidden assumptions
- Enables flexible scenario design

### 7.5 ZK-Compliant Observations

**Decision:** Observations expose only target positions and binary active status (minimal mode).

**Rationale:**
- Core ZK-MRTA constraint
- Agents cannot observe HP, class, or damage values
- Enables fair comparison of information-limited policies

### 7.6 Collaborative Observation Mode

**Decision:** Optional mode that exposes other agents' actions and rewards for CF learning.

**Rationale:**
- Enables collaborative filtering policies to learn from swarm behavior
- Maintains ZK compliance (no HP/class/damage exposed)
- Configurable noise parameters for realistic information sharing

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
| `tabula_drone/core/states.py` | Core dataclasses (DroneState, TargetState, WorldState, AttributeProfile) |
| `tabula_drone/envs/drone_engage_zk_mrta_v0.py` | PettingZoo environment |
| `tabula_drone/policies/random_policy.py` | ZK-compliant baseline |
| `tabula_drone/policies/min_ttk_oracle.py` | Privileged min-TTK oracle |
| `tabula_drone/policies/max_damage_oracle.py` | Privileged optimal assignment oracle |
| `tabula_drone/policies/base_cf_policy.py` | Abstract base class for CF policies |
| `tabula_drone/policies/selfish_ep_greedy_cf_policy.py` | Decentralized ε-greedy CF policy |
| `tabula_drone/policies/coordinated_ep_greedy_cf_policy.py` | Decentralized Hungarian-based CF policy |
| `tabula_drone/policies/ucb_cf_policy.py` | UCB1 exploration CF policy |
| `tabula_drone/scenarios/scenario_builder.py` | Scenario generation |
| `tabula_drone/scenarios/weapon_assignment.py` | Weapon assignment utilities |
| `tabula_drone/logging/episode_logger.py` | Episode capture |
| `tabula_drone/logging/run_manager.py` | Multi-episode run management |
| `tabula_drone/config/config_loader.py` | Configuration utilities |
| `viewer/` | Visualization module |
| `main_zk_mrta.py` | Demo entry point |

---

*Architecture document generated from codebase analysis. Last updated: January 2025.*
