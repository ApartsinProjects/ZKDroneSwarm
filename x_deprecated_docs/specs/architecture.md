# TabulaDrone — High-Level Architecture

## 1. Introduction

TabulaDrone is a simulation framework designed for studying **Zero-Knowledge Multi-Robot Task Allocation (ZK-MRTA)**. In this domain, autonomous drones must allocate themselves to targets without prior knowledge of task properties, their own capabilities, or the capabilities of other drones, and without inter-drone communication.

To support this research reliably, TabulaDrone's architecture enforces a **strict bipartite model**. It completely separates the authoritative simulation rules from the decision-making logic.

## 2. High-Level System Design (The Bipartite Architecture)

The system is split into two non-overlapping halves, bridged only by standard data contracts (Gymnasium spaces) and a structural protocol (`IPolicy`).

1. **The Environment (Simulation Engine):** The absolute authority over the world state, physics, rules, and outcomes.
2. **The Policy Plugin (Decision Logic):** An external, interchangeable "black box" that receives observations and returns actions.

The core architectural rule is that the environment **never** references, imports, or knows about specific policy algorithms (e.g., Random, Collaborative Filtering). Conversely, policies **never** have direct access to the environment's internal state—they only "see" what the environment explicitly passes to them via the observation space.

## 3. The Simulation Environment (Authoritative State)

The core engine is implemented as a PettingZoo `ParallelEnv` (`DroneEngageZKMRTA`). It models a discrete-time, 2D world where all agents act simultaneously.

### 3.1 Key Entities
- **Drones (`DroneState`):** Static entities with fixed 2D positions and unlimited ammunition. Each drone carries exactly one missile type, dictating its damage profile.
- **Targets (`TargetState`):** Static task entities with a multi-attribute resilience profile (e.g., structural, envelope, utilities). Their initial health values are determined by configurable target classes.

### 3.2 Time and Progression
- The simulation advances in discrete time steps.
- At each step, actions from all drones are collected and processed sequentially to ensure deterministic, reproducible outcomes (via a configurable random seed).
- Targets are neutralized when all attributes drop to zero or below. 

## 4. Data Contracts & Constraints (Gymnasium Spaces)

The boundary between the environment and the policy plugin is strictly formalized using Gymnasium spaces. This enforces the Zero-Knowledge (ZK) constraints natively at the API level.

- **Action Space (`spaces.Discrete`):** A drone can choose to do nothing (NoOp, `0`) or fire at a specific target index.
- **Observation Space (`spaces.Dict`):** Depending on the strictness of the scenario (Minimal vs. Collaborative mode), the environment exposes a highly filtered view of the world:
  - Positions and active/inactive status of targets.
  - *(In Collaborative mode)* Noisy rewards and previous actions of other drones.
  - **Crucially:** Target health values, target classes, and weapon damage profiles are never exposed in the observation space.

## 5. The Policy Plugin Boundary

Because the goal of TabulaDrone is to evaluate different task-allocation strategies, the policy is intentionally kept out of the core architecture.

All decision-making logic must satisfy the `IPolicy` structural protocol:
- **`select_actions(obs, info)`:** Receives the constrained Gymnasium observations and returns discrete actions.
- **`update(obs)`:** An optional hook called after each step to allow learning algorithms to update their internal state.
- **`soft_reset()`:** A hook called between episodes to clear episode-level counters while preserving learned weights.

By keeping the architecture strictly bipartite, researchers can easily swap out baselines, oracles, or complex learning models by simply pointing the configuration at a new `IPolicy` implementation, without altering the simulation environment.
