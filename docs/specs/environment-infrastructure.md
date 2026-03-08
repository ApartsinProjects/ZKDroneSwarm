# TabulaDrone — Environment Infrastructure

## 1. Introduction

TabulaDrone is a simulation framework for studying Zero-Knowledge Multi-Robot Task Allocation (ZK-MRTA). At its core, it provides a multi-agent environment where drones engage targets under strict information constraints, and a plugin system where **policies are interchangeable components**.

The infrastructure is designed around a clean separation: the **environment** owns all simulation state and enforces rules, while **policies** are plugged in from outside and interact only through a well-defined interface. The environment never imports or references any specific policy. This means a developer can write a new policy — from a trivial random baseline to a sophisticated learning algorithm — without modifying any infrastructure code.

This document explains the technology stack, the interaction model between environment and policy, the plugin contract, and how to integrate a new policy.

---

## 2. Technology Stack

### 2.1 Gymnasium Spaces

[Gymnasium](https://gymnasium.faraday.ai/) (the maintained successor to OpenAI Gym) provides standardized definitions for action and observation spaces. TabulaDrone uses `gymnasium.spaces` to formally declare what actions a drone can take and what observations it receives.

The project uses three space types:
- **`spaces.Discrete(n)`** — defines the action space. Each drone's action is an integer in `[0, n)`, where `0` is NoOp and `1` through `num_targets` selects a target to engage.
- **`spaces.Box(low, high, shape, dtype)`** — defines continuous array observations. Used for target position/status arrays and reward arrays.
- **`spaces.Dict({...})`** — composes multiple spaces into a single observation structure. Each drone's observation is a Dict containing `"targets"` (Box), `"selected_targets"` (Box), and `"observed_rewards"` (Box).

These space definitions serve as a formal contract between the environment and any policy: the environment guarantees observations will conform to the declared space, and policies must return actions within the declared action space.

### 2.2 PettingZoo ParallelEnv

[PettingZoo](https://pettingzoo.faraday.ai/) extends the single-agent Gymnasium interface to multi-agent settings. TabulaDrone's environment (`DroneEngageZKMRTA`) extends `pettingzoo.utils.env.ParallelEnv`, which models scenarios where all agents act simultaneously each step.

The project implements the following ParallelEnv APIs:

| API | Purpose |
|-----|---------|
| `reset(seed, options)` | Initialize or reinitialize the environment. Returns `(observations, infos)` where both are dicts keyed by agent ID. |
| `step(actions)` | Accept a dict of `{agent_id: action}` from the policy, advance simulation by one time step, and return `(observations, rewards, terminations, truncations, infos)`. |
| `agents` | Property returning the list of currently active agent IDs (e.g., `["drone_0", "drone_1", ...]`). |
| `possible_agents` | The full list of agent IDs that can ever appear. |
| `observation_spaces` | Dict of `{agent_id: Space}` defining each agent's observation structure. |
| `action_spaces` | Dict of `{agent_id: Space}` defining each agent's valid actions. |

The ParallelEnv model means all drones submit actions at the same time, the environment processes them in a deterministic order, and all drones receive their next observation simultaneously.

---

## 3. High-Level Interaction Flow

The interaction between the environment and a policy follows a simple loop. The environment is the passive authority — it produces observations and processes actions. The policy is the active decision-maker — it receives observations and returns actions.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        EPISODE LIFECYCLE                              │
│                                                                       │
│   ┌─────────────┐          observations, info          ┌──────────┐ │
│   │             │ ──────────────────────────────────►  │          │ │
│   │ Environment │                                      │  Policy  │ │
│   │  (ParallelEnv)        actions (per drone)          │ (IPolicy)│ │
│   │             │ ◄──────────────────────────────────  │          │ │
│   └──────┬──────┘                                      └─────┬────┘ │
│          │                                                   │      │
│          │  1. env.reset()  ───────────► observations, info  │      │
│          │                                                   │      │
│          │  2. policy.select_actions(obs, info) ◄──── actions│      │
│          │                                                   │      │
│          │  3. env.step(actions) ──► obs, rewards, done, info│      │
│          │                                                   │      │
│          │  4. policy.update(obs)  (learn, if applicable)    │      │
│          │                                                   │      │
│          │  5. Repeat 2–4 until episode ends                 │      │
│          │                                                   │      │
│          │  6. policy.soft_reset()  (between episodes)       │      │
│          │                                                   │      │
└──────────────────────────────────────────────────────────────────────┘
```

**Key points:**
- The environment never calls policy-specific methods — it only produces data.
- The orchestration code (e.g., `main_zk_mrta.py`) bridges environment and policy by calling both in sequence.
- Policies receive observations and return actions. They may optionally learn from observations via `update()`.
- Between episodes, `soft_reset()` allows policies to clear episode-level state while preserving learned knowledge.

---

## 4. The Plugin Contract: IPolicy Protocol

All policies must satisfy the `IPolicy` protocol, defined in `tabula_drone/policies/base.py`. This is a Python `Protocol` (structural typing) — any class that implements the required methods is automatically compatible, without needing to inherit from a base class.

### 4.1 Required Interface

| Member | Signature | Purpose |
|--------|-----------|---------|
| `is_deterministic` | `bool` (class attribute) | Declares whether the policy produces deterministic actions. Used by infrastructure for logging/reproducibility. |
| `select_actions` | `(obs: Dict[str, Any], info: Dict[str, Any]) → Dict[str, int]` | Given observations and info dicts (keyed by agent ID), return an action dict `{agent_id: action}`. This is the core decision method. |
| `update` | `(obs: Dict[str, Any]) → None` | Called after each environment step with the new observations. Learning policies update their internal state here. Non-learning policies implement this as a no-op. |
| `soft_reset` | `() → None` | Called between episodes. Resets episode-level state (e.g., step counters) while preserving learned parameters (e.g., latent vectors). Stateless policies implement this as a no-op. |
| `get_learning_state` | `() → Optional[Dict[str, Any]]` | Returns internal learning state for logging and visualization. Non-learning policies return `None`. |

### 4.2 Observation Format

The `obs` dict passed to `select_actions` and `update` is keyed by agent ID. Each agent's observation is a Dict with:

| Key | Shape | Content |
|-----|-------|---------|
| `"targets"` | `(3 × num_targets,)` | Flat array: `[t0_x, t0_y, t0_active, t1_x, t1_y, t1_active, ...]` |
| `"selected_targets"` | `(num_agents,)` | Previous step's actions for all agents (0 = NoOp, 1–N = target index) |
| `"observed_rewards"` | `(num_agents,)` | Previous step's rewards for all agents (with configurable noise) |

### 4.3 Action Format

Each action is an integer:
- `0` — NoOp (do nothing)
- `1` to `num_targets` — Fire at target (1-indexed)

---

## 5. Plugin Patterns

Two integration patterns exist, depending on whether the policy operates at the swarm level or per-agent level.

### 5.1 Pattern 1: Direct IPolicy Implementation (Swarm-Level)

For policies that make decisions for all agents in a single call, implement `IPolicy` directly.

**Structure:**
```
YourPolicy (implements IPolicy)
  ├── select_actions(obs, info) → {agent_id: action}    # decides for all agents
  ├── update(obs) → None
  ├── soft_reset() → None
  └── get_learning_state() → Optional[Dict]
```

**When to use:** The policy doesn't need per-agent internal state, or it manages all agents' state centrally.

**Example:** `RandomPolicy` — selects independently for each agent within `select_actions`, but has no per-agent state. Oracle policies (`OracleTimeToKillPolicy`, `OptimalAssignmentOracle`) also use this pattern.

### 5.2 Pattern 2: Per-Agent Policy + MultiAgentPolicy Wrapper

For policies where each agent maintains its own private state (e.g., learned latent vectors), implement a per-agent policy and wrap it with `MultiAgentPolicy`.

**Structure:**
```
YourAgentPolicy (extends BaseCFAgentPolicy)
  ├── select_action(observation, ...) → int              # decides for ONE agent
  ├── update_from_observation(observation) → None         # learns from ONE agent's obs
  ├── soft_reset() → None
  └── get_learning_state() → Optional[Dict]

MultiAgentPolicy (implements IPolicy)
  ├── wraps Dict[agent_id → YourAgentPolicy]
  ├── select_actions() → delegates to each agent's select_action()
  ├── update() → delegates to each agent's update_from_observation()
  └── satisfies IPolicy interface for the orchestration code
```

**When to use:** Each agent needs its own private state (e.g., latent vectors for collaborative filtering). True decentralization — no shared state between agent instances.

**Example:** `SelfishEpGreedyCFPolicy` — each drone has its own latent vectors. One instance per agent, all wrapped by `MultiAgentPolicy`.

### 5.3 Wiring a Policy into the System

Policies are instantiated in the orchestration code (`main_zk_mrta.py`) via a factory function `create_policy()`. The factory receives a policy type string from the configuration and returns an `IPolicy`-compliant object. The episode loop then interacts with it uniformly:

```
policy = create_policy(policy_type, config, drones_config, num_targets)

obs, info = env.reset()
while not done:
    actions = policy.select_actions(obs, info)
    obs, rewards, terminations, truncations, info = env.step(actions)
    policy.update(obs)
```

The environment and the episode loop never know which policy is running. This is the plugin boundary.

---

## 6. Reference Examples

### 6.1 RandomPolicy (Pattern 1 — Direct IPolicy)

**File:** `tabula_drone/policies/random_policy.py`

| Aspect | Implementation |
|--------|----------------|
| **Implements** | `IPolicy` directly |
| **`is_deterministic`** | `True` (deterministic given seed) |
| **`select_actions`** | Iterates over all agents, calls internal `select_action` per agent |
| **`select_action`** | Parses observation to find active targets, selects uniformly at random |
| **`update`** | No-op — stateless policy |
| **`soft_reset`** | No-op — no episode state |
| **`get_learning_state`** | Returns `None` |

**Key integration detail:** Accepts both flat array and Dict observations (handles both observation modes gracefully).

### 6.2 SelfishEpGreedyCFPolicy (Pattern 2 — Per-Agent + Wrapper)

**File:** `tabula_drone/policies/selfish_ep_greedy_cf_policy.py`
**Base:** `tabula_drone/policies/base_cf_agent_policy.py`
**Wrapper:** `tabula_drone/policies/multi_agent_policy.py`

| Aspect | Implementation |
|--------|----------------|
| **Extends** | `BaseCFAgentPolicy` (abstract, provides SGD learning infrastructure) |
| **One instance per agent** | Each drone gets its own policy object with private state |
| **`select_action`** | Receives single agent's observation, returns single action |
| **`update_from_observation`** | Receives single agent's observation, updates private latent vectors |
| **Wrapped by** | `MultiAgentPolicy` — aggregates all per-agent instances into one `IPolicy` |
| **Wiring** | `create_policy()` creates a dict of per-agent instances, passes to `MultiAgentPolicy(policies)` |

**Key integration detail:** The per-agent policy itself raises `NotImplementedError` on `select_actions()` — it is not usable directly as an `IPolicy`. The `MultiAgentPolicy` wrapper bridges this gap by delegating to each agent's `select_action()` individually.

---

*Developer guide for TabulaDrone environment infrastructure and policy plugin system.*
