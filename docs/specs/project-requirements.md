# TabulaDrone — Project Requirements

## 1. Introduction

TabulaDrone is a simulation framework for studying **Zero-Knowledge Multi-Robot Task Allocation (ZK-MRTA)** — a class of problems where autonomous drones must allocate themselves to targets without prior knowledge of task properties, their own capabilities, or the capabilities of other drones, and without any form of inter-drone communication.

The simulation models a discrete-time, 2D world in which a heterogeneous swarm of static drones engages a set of static targets. Each drone selects a target to fire upon at every time step based solely on limited local observations. The framework serves as a controlled testbed for evaluating and comparing different task-allocation strategies — from naive baselines (random selection) through privileged oracles (with full knowledge) to learning-based approaches (such as collaborative filtering) — under identical simulated conditions.

This document defines **what** the system must model and enforce. It does not prescribe architecture, design, or policy algorithms.

---

## 2. Glossary

| Domain Term | Definition | Implementation Reference |
|-------------|-----------|--------------------------|
| **Environment** | The simulated 2D world that governs all entities, enforces rules, progresses time, computes rewards, and produces observations. It is the authoritative source of truth for all state. | `DroneEngageZKMRTA` (PettingZoo ParallelEnv) |
| **Drone** | An autonomous agent positioned in the environment. Each drone carries exactly one missile type and acts independently each time step by selecting a target to engage or choosing to do nothing. Drones are static (they do not move) and heterogeneous (different drones may carry different missile types). | `DroneState` |
| **Target** | A task entity positioned in the environment. Each target belongs to a class that determines its resilience profile — a set of named attributes with numeric health values. A target starts active and becomes inactive (neutralized) when all of its attributes are depleted. Targets are static. | `TargetState` |
| **Missile** | The weapon a drone launches when it engages a target. Each missile type has a damage profile that specifies how much damage it deals to each attribute of a target's resilience. A drone carries exactly one missile type for the entire mission. Missiles are unlimited — a drone can fire every time step without running out. | `weapon_type` / `damage_profile` |
| **Attribute** | A named component of a target's resilience (e.g., structural integrity, envelope integrity, utilities & life-safety). Each attribute has an initial value and a current value that decreases when damaged. | `AttributeProfile` |
| **Target Class** | A category label assigned to a target that determines its initial attribute values. Different classes produce different resilience profiles, making some targets harder to neutralize with certain missile types. | `class_type` / `class_attribute_mapping` |
| **Missile Type** | A category label assigned to a drone's weapon that determines its damage profile — how much damage each shot deals to each target attribute. Different missile types are more or less effective against different target classes. | `weapon_type` / `weapon_damage_profile_mapping` |
| **Engagement** | The act of a drone firing its missile at a target during a single time step. The target's attributes are reduced according to the drone's damage profile. | Action processing in environment step |
| **Neutralization** | The event when all of a target's attributes reach zero or below, causing the target to become inactive. Once neutralized, a target cannot be damaged further. | `is_active` flag / `is_depleted()` |
| **Time Step** | A single discrete tick of the simulation. During each time step, every drone selects an action, the environment processes all actions, updates state, and produces new observations. | `world.time_step` |
| **Episode** | A complete simulation run from initialization to termination. An episode ends when all targets are neutralized or the maximum number of time steps is reached. | Episode lifecycle |
| **Observation** | The information the environment provides to each drone at each time step. Observations are constrained by the Zero-Knowledge rules and the selected observation mode. | Observation space |

---

## 3. Requirements

### 3.1 Environment Requirements

- **ENV-1:** The environment shall model a bounded 2D world with configurable dimensions.
- **ENV-2:** The environment shall progress in discrete time steps, advancing by one step each tick.
- **ENV-3:** The environment shall enforce a configurable maximum number of time steps per episode (mission duration).
- **ENV-4:** The environment shall terminate an episode when either all targets are neutralized or the maximum time steps are reached.
- **ENV-5:** The environment shall support deterministic reproducibility via a configurable random seed.
- **ENV-6:** The environment shall be the sole authority on all state — no entity may modify state outside the environment's step processing.
- **ENV-7:** The environment shall process drone actions sequentially in a fixed order each time step, ensuring deterministic outcomes.

### 3.2 Drone Requirements

- **DRN-1:** Each drone shall have a unique identifier, a fixed 2D position, and exactly one missile type.
- **DRN-2:** Drones shall not move during an episode.
- **DRN-3:** Drones shall have unlimited ammunition — the number of shots is not constrained.
- **DRN-4:** Each drone shall independently select one action per time step: engage a specific target, or do nothing (no-operation).
- **DRN-5:** A drone's damage profile (how effective its missile is against each attribute) shall be determined by its missile type and shall remain constant throughout an episode.
- **DRN-6:** The environment shall track the total number of shots fired by each drone for metrics purposes.

### 3.3 Target Requirements

- **TGT-1:** Each target shall have a unique identifier, a fixed 2D position, a class label, and a multi-attribute resilience profile.
- **TGT-2:** Targets shall not move during an episode.
- **TGT-3:** A target's initial attribute values shall be determined by its class via a configurable class-to-attribute mapping.
- **TGT-4:** Each attribute of a target shall decrease independently when damaged by a missile.
- **TGT-5:** A target shall be considered neutralized (inactive) when **all** of its attributes reach zero or below.
- **TGT-6:** Once neutralized, a target shall not be further damaged. Shots fired at a neutralized target are wasted.

### 3.4 Missile and Engagement Requirements

- **MSL-1:** Each missile type shall have a damage profile that specifies damage values per attribute (e.g., structural: 5, envelope: 1, utilities: 1).
- **MSL-2:** Missile types and their damage profiles shall be configurable via a weapon-to-damage-profile mapping.
- **MSL-3:** When a drone engages an active target, the target's attributes shall be reduced by the corresponding values in the drone's damage profile.
- **MSL-4:** Multiple drones may engage the same target in the same time step; damage is applied sequentially per the processing order.
- **MSL-5:** Engaging a neutralized target shall result in a wasted shot — no damage is applied, and a negative reward is returned.

### 3.5 Zero-Knowledge Constraint Requirements

- **ZK-1:** Drones shall have no prior knowledge of any target's attributes, class, or resilience values.
- **ZK-2:** Drones shall have no knowledge of their own damage profile or missile effectiveness.
- **ZK-3:** Drones shall have no knowledge of other drones' capabilities, missile types, or damage profiles.
- **ZK-4:** Drones shall not communicate with one another in any form.
- **ZK-5:** All drone decisions shall be decentralized — each drone acts independently based solely on its own observations and internal memory.

### 3.6 Observation Requirements

The environment shall support two observation modes:

#### 3.6.1 Minimal Mode (Strict Zero-Knowledge)

- **OBS-M1:** Each drone shall receive only: the position (x, y) and active/inactive status of every target.
- **OBS-M2:** All drones shall receive identical observations.
- **OBS-M3:** No information about target health values, target classes, damage profiles, or other drones' states shall be exposed.

#### 3.6.2 Collaborative Mode (Extended Zero-Knowledge)

- **OBS-C1:** Each drone shall receive the same target information as minimal mode (positions and active status).
- **OBS-C2:** Each drone shall additionally receive: the action selected by every drone in the previous time step (which target each drone engaged, or no-operation).
- **OBS-C3:** Each drone shall additionally receive: the reward obtained by every drone in the previous time step, subject to configurable noise.
- **OBS-C4:** Noise on observed rewards shall be configurable via two parameters: base reward noise (applied to all rewards including the drone's own) and observation noise (additional noise when observing other drones' rewards).
- **OBS-C5:** No information about target health values, target classes, or damage profiles shall be exposed. The collaborative mode remains compliant with the core Zero-Knowledge constraints (ZK-1 through ZK-3).

### 3.7 Reward Requirements

- **RWD-1:** The environment shall compute a reward for each drone at each time step based on the outcome of its action.
- **RWD-2:** A drone that delivers the neutralizing blow to a target (the shot that depletes the final attribute) shall receive a positive reward.
- **RWD-3:** A drone that fires at an already-neutralized target shall receive a negative reward (−1.0).
- **RWD-4:** Rewards shall be normalized to enable fair comparison across different missile types and target classes.
- **RWD-5:** The reward model shall be configurable, supporting at minimum: dominant-attribute mode (reward based on damage to the target's strongest attribute relative to maximum weapon damage) and HP-reduction mode (reward based on actual health reduction relative to weapon damage).

### 3.8 Scenario Configuration Requirements

- **SCN-1:** The environment shall be instantiated from a configuration that specifies: number of drones, number of targets, world dimensions, mission duration, random seed, observation mode, and reward mode.
- **SCN-2:** The number and types (missile types) of drones shall be configurable.
- **SCN-3:** The number and classes of targets shall be configurable.
- **SCN-4:** Class-to-attribute mappings and weapon-to-damage-profile mappings shall be provided as explicit configuration — no hardcoded defaults.
- **SCN-5:** The scenario builder shall support randomized placement of drones and targets within the world bounds.

### 3.9 Performance Metrics Requirements

- **MET-1:** The environment shall record the number of time steps until episode termination.
- **MET-2:** The environment shall record the count of neutralized targets at episode end.
- **MET-3:** The environment shall record the total ammunition used across all drones.
- **MET-4:** The environment shall record total overkill damage (excess damage applied beyond what was needed to neutralize targets).
- **MET-5:** The environment shall record cumulative reward per drone.
- **MET-6:** The environment shall record the termination reason (all targets neutralized vs. mission time exceeded).
- **MET-7:** All metrics shall be computed by the environment and remain independent of the chosen strategy.

### 3.10 Logging and Replay Requirements

- **LOG-1:** The environment shall support episode logging that captures: initial scenario configuration, per-step actions, rewards, state changes, and episode summary.
- **LOG-2:** Logs shall be stored in a structured format suitable for post-hoc analysis and replay.
- **LOG-3:** Logging shall not alter what drones observe or how the environment behaves.

---

*Requirements derived from the ZK-MRTA research proposal and TabulaDrone architecture specification. Scope limited to currently implemented behavior.*
