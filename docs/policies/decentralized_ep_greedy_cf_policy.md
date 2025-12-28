# DecentralizedEpGreedyCFPolicy (Decentralized ε-Greedy Collaborative Filtering)

## Table of Contents

1. [Overview](#overview)
2. [Design Philosophy](#design-philosophy)
3. [High-Level Flow (5 Phases)](#high-level-flow-5-phases)
4. [Core Algorithm](#core-algorithm)
5. [ZK-Compliance Deep Dive](#zk-compliance-deep-dive)
6. [Hyperparameters](#hyperparameters)
7. [Edge Case Handling](#edge-case-handling)
8. [Action Space](#action-space)
9. [API Reference](#api-reference)
10. [Usage Example](#usage-example)
11. [Implementation Details](#implementation-details)
12. [Limitations](#limitations)
13. [File Location](#file-location)

---

## Overview

The `DecentralizedEpGreedyCFPolicy` is a **true ZK-MRTA compliant learning-based policy** for the ZK-MRTA (Zero-Knowledge Multi-Robot Task Allocation) environment. It uses **SGD-based matrix factorization** to learn agent-target compatibility from observed rewards, combined with **ε-greedy exploration** for action selection.

**Key Characteristic**: Each agent maintains its own **private latent vectors** and learns independently — one policy instance per agent with **no shared state** between agents. This fully adheres to the ZK-MRTA research proposal requirement that "each agent independently maintains and updates its own model."

---

## Design Philosophy

This policy is **fully ZK-MRTA compliant** because:

1. **No shared state**: Each agent has its own policy instance with private latent vectors
2. **No communication**: Agents don't share learned representations directly
3. **Indirect observation only**: Agents learn from observed rewards (collaborative mode), not from each other's internal state

It uses information available through the collaborative observation mode:
- Binary active/inactive status of each target
- Observed rewards from previous actions (all agents)
- Selected targets from previous step (all agents)

It does **not** use:
- True HP values or remaining attributes
- Target class types
- Weapon damage profiles directly
- Other agents' internal latent vectors

The policy exists to:

1. Demonstrate **true decentralized learning** in ZK-MRTA settings
2. Align with research proposal: "each agent independently maintains and updates its local matrix"
3. Enable **deployment-ready** single-agent extraction
4. Provide a ZK-compliant baseline for comparison with oracle approaches

---

## High-Level Flow (5 Phases)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EPISODE LOOP (PER AGENT)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. OBSERVE        →  Get observation (active targets, past rewards)│
│         ↓                                                           │
│  2. LEARN          →  Update PRIVATE vectors from past rewards (SGD)│
│         ↓                                                           │
│  3. SELECT ACTION  →  ε-greedy: explore randomly OR exploit best    │
│         ↓                                                           │
│  4. EXECUTE        →  Environment processes action (fires at target)│
│         ↓                                                           │
│  5. REWARD         →  Environment returns reward based on damage    │
│         ↓                                                           │
│       (loop back to 1)                                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    DECENTRALIZED ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │ Policy (drone_0)│  │ Policy (drone_1)│  │ Policy (drone_N)│     │
│  │ ─────────────── │  │ ─────────────── │  │ ─────────────── │     │
│  │ agent_lv (1D)   │  │ agent_lv (1D)   │  │ agent_lv (1D)   │     │
│  │ target_lv (2D)  │  │ target_lv (2D)  │  │ target_lv (2D)  │     │
│  │ other_agents_lv │  │ other_agents_lv │  │ other_agents_lv │     │
│  │ epsilon, rng    │  │ epsilon, rng    │  │ epsilon, rng    │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│           │                    │                    │               │
│           └────────── NO SHARED STATE ─────────────┘               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Responsibility Breakdown

| Phase | Responsible Component | Method/Location |
|-------|----------------------|-----------------|
| Phase 1: OBSERVE | Environment | `DroneEngageZKMRTA._compute_observations()` |
| Phase 2: LEARN | Policy (per agent) | `DecentralizedEpGreedyCFPolicy.update_from_observation()` |
| Phase 3: SELECT ACTION | Policy (per agent) | `DecentralizedEpGreedyCFPolicy.select_action()` |
| Phase 4: EXECUTE | Environment | `DroneEngageZKMRTA.step()` |
| Phase 5: REWARD | Environment | `DroneEngageZKMRTA.step()` |

**In short**: Each agent's policy controls *what action to take* (Phase 3) and *how to learn from feedback* (Phase 2) using its own private state. Everything else — observation, execution, reward calculation — is handled by the environment.

### Phase 1: OBSERVE — *Receive what happened last step*

Receive observation dictionary with 3 components:

**`targets`** — Shape: `(3 * num_targets,)`
- Each target contributes 3 values: x-coordinate, y-coordinate, active status
- Format: `[t0_x, t0_y, t0_active, t1_x, t1_y, t1_active, ...]`
- Active status: `1.0` if target is alive, `0.0` if neutralized
- Example: 2 targets → array of length 6: `[100, 200, 1.0, 300, 400, 0.0]` (target 0 at (100,200) alive, target 1 at (300,400) neutralized)

**`selected_targets`** — Shape: `(num_drones,)`
- History log of what each drone did **last step**
- Format: `[action_of_drone_0, action_of_drone_1, action_of_drone_2, ...]`
- Action `0` means NoOp (did nothing), action `N` means fired at target `N-1`
- Example with 3 drones and 4 targets:
  - Last step: drone_0 fired at target 2, drone_1 did nothing, drone_2 fired at target 0
  - This step's observation: `[3, 0, 1]`
- Why needed: The policy pairs this with `observed_rewards` to learn which drone-target combinations are effective

**`observed_rewards`** — Shape: `(num_drones,)` *(Dominant Attribute Reward)*
- The feedback signal telling how effective each drone's last action was
- Format: `[reward_of_drone_0, reward_of_drone_1, reward_of_drone_2, ...]`
- Uses the **Dominant Attribute** approach: `reward = damage_to_dominant_attribute / max_weapon_damage`
- Values:
  - Positive (0 to 1): Damage dealt, normalized by max possible damage
  - `0.0`: Drone did nothing (NoOp)
  - `-1.0`: Wasted shot (fired at already-dead target)
- Why "dominant attribute": Each target has multiple defensive attributes (e.g., armor, shields). The reward reflects damage to the attribute with the highest initial value. This means weapon-target compatibility affects the reward.
- Example: drone_0 with "heavy" weapon (50 armor damage) fires at "armored" target → reward = 50/50 = 1.0

The policy pairs `selected_targets` with `observed_rewards` to learn: *"drone X fired at target Y and got reward Z"* — this is how it discovers which drone-target combinations are effective without ever seeing weapon or target attributes.

### Phase 2: LEARN — *Update PRIVATE predictions based on observed rewards*

This is where the **decentralized** nature becomes apparent. Each agent updates its **own private vectors**.

**What each agent maintains** — Three sets of private latent vectors:
- `agent_lv`: Shape `(latent_dim,)` — **this agent's** "weapon characteristics" in latent space
- `target_lv`: Shape `(num_targets, latent_dim)` — **this agent's estimates** of all targets
- `other_agents_lv`: Shape `(num_agents, latent_dim)` — **this agent's estimates** of other agents

**Key decentralization principle**: Each agent has its own private `agent_lv` vector and must also maintain estimates of other agents (`other_agents_lv`) to learn from their observed rewards. There is no shared state between agents.

**SGD (Stochastic Gradient Descent) Update Process** — For each observed action/reward:
1. **If it's this agent's own action**:
   - Get the data: Which target did I fire at? What reward did I get?
   - Predict: `predicted = (1 + dot(agent_lv, target_lv[target])) / 2`
   - Compute error: `error = observed_reward - predicted_reward`
   - Update `agent_lv` (own weapon characteristics)
   - Update `target_lv[target]` (target estimate)
   - Normalize both vectors to unit length
2. **If it's another agent's action**:
   - Get the data: Which target did agent X fire at? What reward did they get?
   - Predict: `predicted = (1 + dot(other_agents_lv[X], target_lv[target])) / 2`
   - Compute error: `error = observed_reward - predicted_reward`
   - Update `other_agents_lv[X]` (estimate of that agent)
   - Update `target_lv[target]` (target estimate)
   - Normalize both vectors to unit length

**What gets skipped** — The policy does NOT learn from:
- NoOp actions (`target_idx < 0`) — no information gained
- Negative rewards (`reward < 0`) — wasted shots don't reflect true compatibility, just bad timing

**Why maintain `other_agents_lv`?** To learn from others' experiences, this agent needs to model what other agents' "weapon characteristics" might be. This allows the agent to update its target estimates based on all observed interactions, not just its own.

**Example**:
- Last step: drone_0 fired at target_1 (reward 0.8), drone_1 fired at target_0 (reward 0.3)
- drone_0's learning:
  - For own action: Update `agent_lv` and `target_lv[1]` so prediction → 0.8
  - For drone_1's action: Update `other_agents_lv[1]` and `target_lv[0]` so prediction → 0.3
- Over time, the policy learns which drone-target combinations are effective

### Phase 3: SELECT ACTION — *Choose which target to fire at*

This is where the policy decides which target to fire at based on its learned knowledge.

**The ε-greedy (epsilon-greedy) strategy** — Balances two competing goals:
- **Exploration**: Try new targets to discover better options
- **Exploitation**: Use learned knowledge to maximize reward
- Parameter **ε** controls the balance: higher ε = more exploration

**Step-by-step process**:
1. **Identify active targets**: Parse `targets` array, build list of targets where `active == 1.0`. Inactive targets are never valid choices.
2. **Build valid actions**:
   - Always includes: all active target indices
   - If `allow_noop=True`: also includes NoOp (action 0) as an option
   - If no active targets exist → return NoOp (nothing to fire at)
   - **What is NoOp?** NoOp = "No Operation" — the drone does nothing this step (doesn't fire). Action `0` = NoOp, Action `1` = fire at target 0, Action `2` = fire at target 1, etc.
   - **Why would a policy choose NoOp?** In scenarios with limited ammo, cooldowns, or energy costs, choosing not to act can be strategic (e.g., saving ammo for higher-value targets).
3. **ε-greedy decision**:
   - Generate random number [0, 1]
   - If random < ε → **Explore**: pick random target from active targets
   - If random ≥ ε → **Exploit**: pick target with highest `predict_reward(target)`
4. **Decay epsilon**: `ε = max(ε_min, ε * ε_decay)` — gradually shifts from exploration to exploitation

**Note**: Each agent decays epsilon independently, so exploration rates may differ across agents.

**Why ε-greedy?**
- Early training: High ε → lots of exploration → discover which drone-target pairs work
- Later training: Low ε → mostly exploitation → use learned knowledge
- ε_min > 0 ensures some exploration always continues

**Example**: 3 active targets, ε = 0.2, predicted rewards: target_0 = 0.4, target_1 = 0.9, target_2 = 0.6
- Random draw 0.15 (< ε) → Explore: pick random target
- Random draw 0.85 (≥ ε) → Exploit: pick target_1 (highest predicted reward)

### Phase 4: EXECUTE — *Environment applies damage to target*

This phase happens in the **environment**, not the policy. The policy has selected an action, now the environment processes it.

**What happens**:
1. **Environment receives actions**: Each drone submits its selected action (target index or NoOp)
2. **Process each drone**: Drones are processed sequentially in fixed order
   - If action = 0 (NoOp) → skip, do nothing
   - If action > 0 → fire at target (action - 1)
3. **Apply damage**: Drone's weapon damage profile is applied to target's attributes. Each attribute (e.g., armor, shields) is reduced by the corresponding damage value
4. **Check target status**: If all target attributes reach 0 → target becomes inactive (neutralized)

**Edge cases**:
- Drone fires at active target → damage applied, reward calculated
- Drone fires at already-dead target → wasted shot (ammo counted, reward = -1.0)
- Multiple drones fire at same target → each processed sequentially; later drones may hit dead target
- NoOp → nothing happens, reward = 0.0

**Key point**: The policy doesn't control this phase — it just submits an action and waits. The environment handles all mechanics.

### Phase 5: REWARD — *Environment calculates and returns reward*

This phase happens in the **environment** after damage is applied. The environment calculates and returns rewards to each drone.

**How reward is calculated** — Uses the **Dominant Attribute** approach:
- Formula: `reward = damage_to_dominant_attribute / max_weapon_damage`
- Dominant attribute = the target's attribute with the highest initial value (e.g., if target has armor: 100, shields: 50 → armor is dominant)

**Reward values**:
- Fire at active target → `0.0` to `1.0` (normalized damage to dominant attribute)
- NoOp → `0.0`
- Fire at already-dead target → `-1.0` (wasted shot penalty)

**What happens after**:
1. **Rewards stored**: Environment saves each drone's reward in `last_rewards`
2. **Actions stored**: Environment saves each drone's action in `last_actions`
3. **New observation built**: Contains updated `targets`, `selected_targets`, `observed_rewards`
4. **Loop back to Phase 1**: Next step begins with new observation

**Why this reward design?**
- Normalized (0-1): Makes learning stable across different weapon/target configurations
- Dominant attribute focus: Rewards matching weapon strengths to target weaknesses
- Wasted shot penalty (-1.0): Discourages firing at dead targets

**Connection to learning**: The reward from this phase becomes the `observed_rewards` in the next step's observation, which Phase 2 (LEARN) uses to update the latent vectors. This closes the loop.

---

## Core Algorithm

*Technical reference for implementation and modification. Some concepts overlap with the High-Level Flow section above — this is intentional for critical points.*

### 1. Private Matrix Factorization Model

Each agent maintains **private latent vectors**:

```
This Agent's Latent Vector: a ∈ ℝ^latent_dim
Target Latent Vectors: T ∈ ℝ^(num_targets × latent_dim)
Other Agents' Latent Vectors: O ∈ ℝ^(num_agents × latent_dim)
```

The predicted reward for **this agent** attacking target `j` is:

```
predicted_reward(j) = (1 + dot(a, T[j])) / 2
```

This scales the dot product from [-1, 1] to [0, 1].

The predicted reward for **another agent `k`** attacking target `j` (used for learning):

```
predicted_reward_other(k, j) = (1 + dot(O[k], T[j])) / 2
```

#### Why Matrix Factorization?

Matrix factorization is a classic collaborative filtering technique that:
- **Discovers latent structure**: Agents with similar weapons cluster together in latent space
- **Generalizes from sparse data**: Learning about one agent-target pair informs predictions for similar pairs
- **Scales efficiently**: O(latent_dim) per prediction, not O(num_agents × num_targets)

#### Why Three Sets of Vectors? (Decentralization Requirement)

- **`agent_lv`**: This agent's own characteristics — learned from its own rewards
- **`target_lv`**: Target characteristics — learned from all observed interactions
- **`other_agents_lv`**: Estimates of other agents — needed to learn from their rewards

**Key insight**: To learn from another agent's reward, this agent needs to model that agent's characteristics. Without `other_agents_lv`, the agent could only learn from its own actions, missing valuable information from collaborative observations.

#### Worked Example: Decentralized Learning

```
Scenario: 2 agents (drone_0, drone_1), 2 targets, latent_dim=2

drone_0's private state:
  agent_lv = [0.9, 0.1]           # drone_0's own characteristics
  target_lv = [[0.8, 0.2],        # drone_0's estimate of target_0
               [0.3, 0.9]]        # drone_0's estimate of target_1
  other_agents_lv = [[0.9, 0.1],  # drone_0's estimate of itself (unused)
                     [0.2, 0.8]]  # drone_0's estimate of drone_1

drone_1's private state:
  agent_lv = [0.2, 0.95]          # drone_1's own characteristics
  target_lv = [[0.7, 0.3],        # drone_1's estimate of target_0
               [0.4, 0.85]]       # drone_1's estimate of target_1
  other_agents_lv = [[0.85, 0.15],# drone_1's estimate of drone_0
                     [0.2, 0.95]] # drone_1's estimate of itself (unused)

Note: drone_0 and drone_1 have DIFFERENT estimates of the same targets!
This is the key property of decentralized learning — each agent builds
its own model of the world from its own perspective.
```

#### Worked Example: Latent Space Interpretation

```
After training, the latent vectors might look like:

drone_0's view:
  agent_lv: [0.9, 0.1]  → "armor-piercing" weapon (learned from own rewards)
  target_lv[0]: [0.85, 0.2]  → "armored" target
  target_lv[1]: [0.3, 0.9]   → "shielded" target

Predicted Rewards (drone_0's perspective):
  → target_0: (1 + 0.9*0.85 + 0.1*0.2) / 2 = 0.89 (high!)
  → target_1: (1 + 0.9*0.3 + 0.1*0.9) / 2 = 0.68

drone_0 learns it is effective against target_0 without ever seeing
the actual weapon/target attributes!
```

### 2. SGD Update Rule

When an agent observes an action and reward, it updates latent vectors using stochastic gradient descent.

**For this agent's own action**:

```python
# Skip updates for NoOp or negative rewards (wasted shots)
if target_idx < 0 or observed_reward < 0:
    return

error = observed_reward - predicted_reward(target_idx)

agent_lv += learning_rate * error * target_lv[target_idx]
target_lv[target_idx] += learning_rate * error * agent_lv

# Re-normalize to unit sphere
agent_lv = normalize(agent_lv)
target_lv[target_idx] = normalize(target_lv[target_idx])
```

**For another agent's action**:

```python
# Skip updates for NoOp or negative rewards (wasted shots)
if target_idx < 0 or observed_reward < 0:
    return

error = observed_reward - predicted_reward_other(other_agent_idx, target_idx)

other_agents_lv[other_agent_idx] += learning_rate * error * target_lv[target_idx]
target_lv[target_idx] += learning_rate * error * other_agents_lv[other_agent_idx]

# Re-normalize to unit sphere
other_agents_lv[other_agent_idx] = normalize(other_agents_lv[other_agent_idx])
target_lv[target_idx] = normalize(target_lv[target_idx])
```

#### Why Skip Negative Rewards?

The environment returns -1.0 for "wasted shots" (firing at already-neutralized targets). Due to sequential processing, this can happen when:
1. Drone A fires at Target X and kills it
2. Drone B (who also targeted X) fires but target is already dead → gets -1.0

If we learned from this -1.0, the policy would incorrectly conclude that Drone B is "incompatible" with Target X's class — when in reality it was just a timing issue. Skipping negative rewards prevents this false incompatibility learning.

#### Why Normalize?

Keeping vectors on the unit sphere:
- **Bounds predictions**: Dot products stay in [-1, 1], predictions in [0, 1]
- **Prevents explosion**: No unbounded growth during training
- **Improves stability**: Consistent scale across all vectors

#### Why Update `other_agents_lv`?

When drone_0 observes drone_1 firing at target_0 and getting reward 0.8, drone_0 can learn:
1. Something about drone_1's characteristics (update `other_agents_lv[1]`)
2. Something about target_0's characteristics (update `target_lv[0]`)

This allows drone_0 to improve its target estimates even from actions it didn't take.

#### Worked Example: SGD Update

```
Before Update (drone_0's state):
  agent_lv = [0.6, 0.8]
  target_lv[0] = [0.8, 0.6]
  
  predicted = (1 + 0.6*0.8 + 0.8*0.6) / 2 = 0.98
  observed = 0.5  (actual reward from environment)
  error = 0.5 - 0.98 = -0.48

Update (learning_rate = 0.1):
  agent_lv += 0.1 * (-0.48) * [0.8, 0.6] = [0.6 - 0.038, 0.8 - 0.029]
  agent_lv = [0.562, 0.771] → normalize → [0.589, 0.808]
  
  target_lv[0] += 0.1 * (-0.48) * [0.6, 0.8] = [0.8 - 0.029, 0.6 - 0.038]
  target_lv[0] = [0.771, 0.562] → normalize → [0.808, 0.589]

After Update:
  new_predicted = (1 + 0.589*0.808 + 0.808*0.589) / 2 = 0.976

The prediction moved closer to the observed reward (0.5).
```

### 3. ε-Greedy Action Selection

The policy balances exploration and exploitation using this agent's private vectors:

```
1. Parse observation to identify active targets
2. Build list of valid actions:
   - If allow_noop=True: [0] + [active target indices]
   - If allow_noop=False: [active target indices only]
3. With probability ε: select random action (explore)
4. With probability 1-ε: select action with highest predicted reward (exploit)
5. Decay ε: ε = max(ε_min, ε * ε_decay)
6. Return selected action
```

#### Why ε-Greedy?

- **Exploration**: Random actions discover new agent-target compatibilities
- **Exploitation**: Greedy selection uses learned knowledge
- **Decay**: Reduces exploration as policy becomes more confident
- **Simplicity**: Easy to tune and understand

#### Worked Example: ε-Greedy Selection

```
Scenario: 3 active targets, ε=0.2

Predicted Rewards (from this agent's private vectors):
  target_0: 0.7
  target_1: 0.9  ← highest
  target_2: 0.4

Random draw: 0.15 (< ε=0.2)
→ EXPLORE: Select random action from [1, 2, 3]
→ Action = 3 (fire at target_2)

Random draw: 0.85 (> ε=0.2)
→ EXPLOIT: Select action with highest predicted reward
→ Action = 2 (fire at target_1)
```

### 4. Collaborative Learning from Observations

In collaborative observation mode, each agent observes:
- All agents' selected targets from the previous step
- All agents' observed rewards from the previous step

The policy learns from **all** observations, not just its own:

```python
for each other_agent_idx in range(num_agents):
    target_action = selected_targets[other_agent_idx]
    reward = observed_rewards[other_agent_idx]
    if target_action > 0:  # Not NoOp
        target_idx = target_action - 1
        if other_agent_idx == self.agent_idx:
            # Own action: update agent_lv and target_lv
            update(target_idx, reward)
        else:
            # Other's action: update other_agents_lv and target_lv
            update_from_other(other_agent_idx, target_idx, reward)
```

#### Why Collaborative Learning?

- **Faster convergence**: Learn from N agents' experiences per step
- **Better generalization**: See how different weapons affect different targets
- **True collaborative filtering**: The "collaborative" in CF refers to learning from multiple users/agents

#### Key Decentralization Property

Even though agents learn from each other's rewards, they maintain **separate models**:
- drone_0's `target_lv[0]` may differ from drone_1's `target_lv[0]`
- drone_0's estimate of drone_1 (`other_agents_lv[1]`) may differ from drone_1's actual `agent_lv`
- No information flows between agents except through observed rewards

---

## ZK-Compliance Deep Dive

### What Makes a Policy ZK-Compliant?

In the ZK-MRTA environment, agents receive **limited observations** that hide certain information:

| Information | Available in ZK Observation? |
|-------------|------------------------------|
| Target position (x, y) | ✅ Yes |
| Target active status | ✅ Yes |
| Selected targets (previous step) | ✅ Yes (collaborative mode) |
| Observed rewards (previous step) | ✅ Yes (collaborative mode) |
| Target HP / remaining attributes | ❌ No (hidden) |
| Target class / type | ❌ No (hidden) |
| Weapon damage profiles | ❌ No (hidden) |

A ZK-compliant policy can only use the "Yes" information.

### What Makes This Policy Truly ZK-MRTA Compliant?

The ZK-MRTA research proposal states:

> "Each agent independently maintains and updates its local matrix, enabling autonomous task selection without requiring communication or centralized coordination."

This policy satisfies this requirement because:

| ZK-MRTA Requirement | How This Policy Complies |
|---------------------|--------------------------|
| No prior knowledge of task attributes | Learns from observed rewards only |
| No knowledge of own capabilities | Learns own `agent_lv` from experience |
| No knowledge of others' capabilities | Maintains private estimates `other_agents_lv` |
| No communication between agents | Each instance is independent — no shared state |
| Only observe outcomes of own actions | Updates own `agent_lv` from own rewards |
| Indirectly observe others' outcomes | Updates `other_agents_lv` and `target_lv` from observed rewards |
| **Each agent maintains its own model** | ✅ Private `agent_lv`, `target_lv`, `other_agents_lv` |

### The Decentralized Architecture

The policy **never accesses**:
- `targets_state` (privileged HP values)
- `class_attribute_mapping` (target class definitions)
- `weapon_damage_profile_mapping` (weapon specifications)
- Other agents' internal latent vectors

Instead, it **learns** agent-target compatibility from observed rewards, which implicitly encode the hidden weapon-target interactions.

```python
# Each agent has its own private state — NO SHARED OBJECTS
# drone_0's policy:
self.agent_lv = np.ndarray((latent_dim,))  # PRIVATE to drone_0
self.target_lv = np.ndarray((num_targets, latent_dim))  # drone_0's ESTIMATES
self.other_agents_lv = np.ndarray((num_agents, latent_dim))  # drone_0's ESTIMATES of others

# drone_1's policy (completely separate object):
self.agent_lv = np.ndarray((latent_dim,))  # PRIVATE to drone_1
self.target_lv = np.ndarray((num_targets, latent_dim))  # drone_1's ESTIMATES
self.other_agents_lv = np.ndarray((num_agents, latent_dim))  # drone_1's ESTIMATES of others
```

drone_0 and drone_1 have completely separate state. They can only learn from observed rewards, not from each other's internal representations. This is the key property that makes this policy truly decentralized.

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 2 | Dimension of latent vectors |
| `learning_rate` | 0.01 | SGD step size |
| `epsilon` | 0.3 | Initial exploration rate |
| `epsilon_decay` | 0.99 | Decay factor per action |
| `epsilon_min` | 0.05 | Minimum exploration rate |

### Tuning Guidelines

**`latent_dim`**:
- Higher = more expressive, but needs more data
- Start with 2-4 for simple scenarios
- Increase if policy struggles to differentiate targets
- Note: Each agent maintains its own vectors, so memory scales with `num_agents × latent_dim`

**`learning_rate`**:
- Higher = faster learning, but less stable
- Lower = slower but more stable convergence
- 0.01-0.1 is typical range
- Consider higher rates (0.05-0.1) for faster initial learning

**`epsilon` / `epsilon_decay` / `epsilon_min`**:
- High initial ε (0.3-0.5) for exploration
- Decay rate depends on episode length
- Keep ε_min > 0 to maintain some exploration
- Note: Each agent decays epsilon independently

### Seed Strategy

Each agent gets a derived seed: `seed = master_seed + agent_idx`

This ensures:
- Reproducibility across runs
- Different initial vectors per agent
- Independent exploration patterns

---

## Edge Case Handling

| Scenario | Behavior |
|----------|----------|
| All targets active | Select based on ε-greedy over predictions |
| No active targets | Returns NoOp (action 0) |
| Single active target | ε-greedy between NoOp (if allowed) and that target |
| NoOp action in update | Skipped (no learning from NoOp) |
| Negative reward (wasted shot) | Skipped (no learning from timing-based penalties) |
| Own action vs other's action | Different update paths (see SGD Update Rule) |

---

## Action Space

| Action | Meaning |
|--------|---------|
| `0` | NoOp (do nothing) - only when `allow_noop=True` |
| `1` to `N` | Fire at target `N-1` (e.g., action 1 → target 0, action 2 → target 1) |

---

## API Reference

### Constructor

```python
DecentralizedEpGreedyCFPolicy(
    num_targets: int,
    agent_idx: int,
    num_agents: int,
    latent_dim: int = 2,
    learning_rate: float = 0.01,
    epsilon: float = 0.3,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.05,
    seed: Optional[int] = None,
)
```

**Parameters**:
- **`num_targets`**: Number of targets in the environment
- **`agent_idx`**: This agent's index (0-based) — identifies self in observations
- **`num_agents`**: Total number of agents (for parsing observations)
- **`latent_dim`**: Dimension of latent vectors (default 2)
- **`learning_rate`**: SGD learning rate (default 0.01)
- **`epsilon`**: Initial exploration rate for ε-greedy (default 0.3)
- **`epsilon_decay`**: Decay factor for epsilon per step (default 0.99)
- **`epsilon_min`**: Minimum epsilon value (default 0.05)
- **`seed`**: Random seed for reproducibility

### Methods

#### `predict_reward`

```python
def predict_reward(
    target_idx: int,
) -> float
```

Predict reward for **this agent** attacking a target using private latent vectors.

**Parameters**:
- **`target_idx`**: Target index (0-based)

**Returns**: Predicted reward in [0, 1] range.

---

#### `update`

```python
def update(
    target_idx: int,
    observed_reward: float,
) -> None
```

Update **this agent's** latent vectors based on its own observed reward.

**Skipped when**:
- `target_idx < 0` (NoOp action)
- `observed_reward < 0` (wasted shot penalty)

**Parameters**:
- **`target_idx`**: Target index that was selected
- **`observed_reward`**: Observed reward value

---

#### `update_from_observation`

```python
def update_from_observation(
    observation: Dict[str, Any],
) -> None
```

Update latent vectors from collaborative mode observation. Learns from all agents' observed rewards:
- For this agent's own action: updates `agent_lv` and `target_lv`
- For other agents' actions: updates `other_agents_lv` and `target_lv`

**Parameters**:
- **`observation`**: Dict observation from collaborative mode containing `selected_targets` and `observed_rewards`

**Note**: This method does not take `agent_id` parameter — the agent knows its own index from construction.

---

#### `select_action`

```python
def select_action(
    observation: Dict[str, Any],
    allow_noop: bool = False,
) -> int
```

Select action using ε-greedy over predicted rewards.

**Parameters**:
- **`observation`**: Dict observation from collaborative mode
- **`allow_noop`**: If `True`, include NoOp (0) as valid action

**Returns**: Integer action in `[0, num_targets]` or `[1, num_targets]` depending on `allow_noop`.

**Note**: This method does not take `agent_idx` parameter — the agent uses its own private vectors.

---

#### `reset`

```python
def reset() -> None
```

Full reset: reinitialize all latent vectors (`agent_lv`, `target_lv`, `other_agents_lv`) and reset epsilon. Use between independent experiments.

---

#### `soft_reset`

```python
def soft_reset() -> None
```

Soft reset for new episode. Currently a no-op (preserves learned knowledge across episodes).

---

## Usage Example

```python
from tabula_drone.policies import DecentralizedEpGreedyCFPolicy
from tabula_drone.envs import DroneEngageZKMRTA

# Create environment in collaborative mode
env = DroneEngageZKMRTA(
    drones_config=[
        {'position': (100, 100), 'weapon_type': 'heavy'},
        {'position': (200, 200), 'weapon_type': 'light'},
    ],
    targets_config=[
        {'position': (500, 500), 'class_type': 'A'},
        {'position': (600, 600), 'class_type': 'B'},
    ],
    class_attribute_mapping={
        'A': {'armor': 100, 'shields': 50},
        'B': {'armor': 50, 'shields': 100},
    },
    weapon_damage_profile_mapping={
        'heavy': {'armor': 50, 'shields': 10},
        'light': {'armor': 10, 'shields': 50},
    },
    observation_mode='collaborative',
)

# Create one policy instance per agent (TRUE DECENTRALIZATION)
num_agents = env.num_drones
num_targets = env.num_targets
master_seed = 42

policies = {
    f"drone_{i}": DecentralizedEpGreedyCFPolicy(
        num_targets=num_targets,
        agent_idx=i,
        num_agents=num_agents,
        latent_dim=2,
        learning_rate=0.1,
        epsilon=0.3,
        seed=master_seed + i,  # Different seed per agent
    )
    for i in range(num_agents)
}

# Run episode
obs, _ = env.reset(seed=42)

for step in range(100):
    # Each agent selects its own action using its PRIVATE policy
    actions = {
        agent_id: policies[agent_id].select_action(agent_obs)
        for agent_id, agent_obs in obs.items()
    }
    
    # Step environment
    obs, rewards, terminations, truncations, _ = env.step(actions)
    
    # Each agent learns from its observation using its PRIVATE state
    for agent_id, agent_obs in obs.items():
        policies[agent_id].update_from_observation(agent_obs)
    
    if terminations['drone_0'] or truncations['drone_0']:
        break

# Check learned predictions (each agent has DIFFERENT predictions!)
print("drone_0's predictions:")
print(f"  → target_0: {policies['drone_0'].predict_reward(0):.2f}")
print(f"  → target_1: {policies['drone_0'].predict_reward(1):.2f}")

print("drone_1's predictions:")
print(f"  → target_0: {policies['drone_1'].predict_reward(0):.2f}")
print(f"  → target_1: {policies['drone_1'].predict_reward(1):.2f}")
```

---

## Implementation Details

### Private State Structure

Each agent maintains three sets of latent vectors:

```python
# This agent's own characteristics (1D vector)
self.agent_lv = np.ndarray((latent_dim,))

# This agent's estimates of all targets (2D matrix)
self.target_lv = np.ndarray((num_targets, latent_dim))

# This agent's estimates of all agents including self (2D matrix)
# Note: other_agents_lv[self.agent_idx] is unused
self.other_agents_lv = np.ndarray((num_agents, latent_dim))
```

### Observation Parsing

The policy extracts target active status from the collaborative observation:
- `observation['targets']`: Array with format `[t0_x, t0_y, t0_active, t1_x, ...]`
- Active status is at index `target_idx * 3 + 2`
- A target is considered active if `active > 0.5`

### Latent Vector Initialization

- Vectors are initialized uniformly in [-1, 1] then normalized
- Each agent gets a different seed: `master_seed + agent_idx`
- Normalization ensures all vectors start on unit sphere
- This ensures different initial vectors per agent

### Random Number Generator

- Each agent has its own `numpy.random.RandomState`
- Same seed produces identical exploration/exploitation decisions
- Different seeds produce different exploration patterns
- RNG is used for initialization, exploration, and tie-breaking

---

## Limitations

1. **Requires Collaborative Mode**: Only works with `observation_mode='collaborative'`
2. **No Coordination**: Agents select independently; may all fire at same target
3. **Cold Start**: Initial predictions are random until sufficient learning
4. **Independent Learning**: Each agent learns separately (no shared knowledge)
5. **Higher Memory**: Each agent maintains its own copy of all vectors
6. **Single-Step Greedy**: Does not plan ahead or consider future states

### Independent Learning (Expected Trade-off for True Decentralization)

Because each agent maintains its own private state:
- Each agent learns independently (no shared knowledge)
- Target estimates may differ across agents
- More exploration needed to discover patterns
- Convergence requires more episodes than shared-state approaches

This is the expected trade-off for true ZK-MRTA compliance. Performance improves with more training episodes.

### The Cold Start Problem

Each agent starts with:
- Random `agent_lv` (own characteristics)
- Random `target_lv` (target estimates)
- Random `other_agents_lv` (estimates of other agents)

This means:
- Initial predictions are essentially random
- More exploration needed initially
- Consider higher initial ε (0.4-0.5)
- Consider more episodes for training

**Mitigation strategies**:
- Start with high ε (0.3-0.5)
- Use slower ε decay in early episodes
- Consider warm-starting from previous runs

### No Coordination

Like all independent learners, agents may converge to similar preferences:

```
Scenario: 2 agents, 2 targets

Both agents independently learn that target_0 is "easier".
Without coordination, both fire at target_0, wasting one attack.

With coordination, one would attack each target.
```

This is a fundamental limitation of independent learners, not specific to this policy.

---

## File Location

`tabula_drone/policies/decentralized_ep_greedy_cf_policy.py`

## Policy Type

Use `"decentralized_ep_greedy_cf"` in config to select this policy.

## Related

- Research Proposal: `docs/research_proposal/Research Proposal-ZK_MRTA.md`
