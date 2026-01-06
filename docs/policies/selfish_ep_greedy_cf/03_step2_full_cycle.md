# Part 4: Step 2 — The Full Cycle

This is the core section. We'll trace drone_0 through the complete **Observe → Learn → Select → Execute** cycle in Step 2.

---

## Overview: The Step 2 Timeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STEP 2 TIMELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. OBSERVE ─────────────────────────────────────────────────────────────►  │
│     │  Environment sends observation to drone_0                              │
│     │  Contains: targets, Step 1 actions, Step 1 rewards                    │
│     │                                                                        │
│  2. LEARN ───────────────────────────────────────────────────────────────►  │
│     │  drone_0 calls update_from_observation()                              │
│     │  Updates latent vectors from ALL agents' Step 1 results               │
│     │                                                                        │
│  3. SELECT ACTION ───────────────────────────────────────────────────────►  │
│     │  drone_0 calls select_action()                                        │
│     │  ε-greedy: explore (30%) or exploit (70%)                             │
│     │                                                                        │
│  4. EXECUTE ─────────────────────────────────────────────────────────────►  │
│     │  Environment processes action, calculates reward                       │
│     │  Stores results for Step 3's observation                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Observation Received

### Code Path

```
main_zk_mrta.run_episode()
    └── obs, rewards, ... = env.step(actions)  # End of Step 1
        └── DroneEngageZKMRTA._compute_observations()
            └── Returns dict for each agent
```

### drone_0's Step 2 Observation

```python
observation = {
    "targets": np.array([
        # target_0: x, y, is_active
        276.35, 647.32, 1.0,
        # target_1: x, y, is_active
        747.54, 501.62, 1.0,
        # ... (25 targets × 3 values = 75 floats)
    ]),
    
    "selected_targets": np.array([1, 22, 23, 14, 22, 20]),
    #                             ▲   ▲   ▲   ▲   ▲   ▲
    #                             │   │   │   │   │   └─ drone_5 → target_19
    #                             │   │   │   │   └───── drone_4 → target_21
    #                             │   │   │   └───────── drone_3 → target_13
    #                             │   │   └───────────── drone_2 → target_22
    #                             │   └───────────────── drone_1 → target_21
    #                             └───────────────────── drone_0 → target_0
    
    "observed_rewards": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    #                             ▲    ▲    ▲    ▲    ▲    ▲
    #                             │    │    │    │    │    └─ drone_5
    #                             │    │    │    │    └────── drone_4
    #                             │    │    │    └─────────── drone_3
    #                             │    │    └──────────────── drone_2
    #                             │    └───────────────────── drone_1
    #                             └────────────────────────── drone_0
}
```

### Step 1 Results Summary

| Drone | Weapon | Action | Target | Target Class | Reward |
|-------|--------|--------|--------|--------------|--------|
| drone_0 | systems | 1 | target_0 | B | **1.0** |
| drone_1 | structural | 22 | target_21 | A | **1.0** |
| drone_2 | structural | 23 | target_22 | B | **1.0** |
| drone_3 | breach | 14 | target_13 | B | **1.0** |
| drone_4 | systems | 22 | target_21 | A | **1.0** |
| drone_5 | breach | 20 | target_19 | C | **1.0** |

**Note:** All rewards are 1.0 because `REWARD_DOMINANT_ATTRIBUTE=False` — rewards are based on total HP reduction (12/12 = 1.0).

---

## Phase 2: Learning

### Code Path

```
main_zk_mrta.run_episode()
    └── for agent_id, agent_obs in obs.items():
            policy[agent_id].update_from_observation(agent_obs)
                └── BaseCFPolicy.update_from_observation()
                    ├── self.update()           # For own action
                    └── self._update_from_other() # For others' actions
```

### The Learning Loop

`update_from_observation()` iterates through all 6 agents' actions:

```python
def update_from_observation(self, observation):
    selected_targets = observation['selected_targets']  # [1, 22, 23, 14, 22, 20]
    observed_rewards = observation['observed_rewards']  # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    for other_agent_idx in range(self.num_agents):  # 0, 1, 2, 3, 4, 5
        target_action = selected_targets[other_agent_idx]
        reward = observed_rewards[other_agent_idx]
        
        if target_action > 0:  # Skip NoOp (action=0)
            target_idx = target_action - 1  # Convert 1-indexed to 0-indexed
            
            if other_agent_idx == self.agent_idx:  # drone_0's own action
                self.update(target_idx, reward)
            else:  # Another drone's action
                self._update_from_other(other_agent_idx, target_idx, reward)
```

---

### Update 1: drone_0's Own Action

**Input:** target_idx=0, observed_reward=1.0

#### Step 2.1.1: Compute Predicted Reward

```python
predicted = self.predict_reward(0)
# dot = np.dot(agent_lv, target_lv[0])
```

**Calculation:**

```
agent_lv     = [-0.240, 0.863, 0.444]
target_lv[0] = [0.199, -0.693, -0.693]

dot = (-0.240)×(0.199) + (0.863)×(-0.693) + (0.444)×(-0.693)
dot = -0.048 - 0.598 - 0.308
dot = -0.954

predicted = (1 + (-0.954)) / 2 = 0.023
```

#### Step 2.1.2: Compute Error

```python
error = observed_reward - predicted
error = 1.0 - 0.023 = 0.977
```

**Interpretation:** drone_0 **massively underestimated** the reward. The actual reward (1.0) was much higher than predicted (0.023).

#### Step 2.1.3: SGD Update

```python
# Store copies for simultaneous update
agent_vec = self.agent_lv.copy()
target_vec = self.target_lv[0].copy()

# Update agent_lv
self.agent_lv += learning_rate * error * target_vec
self.agent_lv += 0.05 * 0.977 * [0.199, -0.693, -0.693]
self.agent_lv += [0.00972, -0.03385, -0.03385]
```

**Before normalization:**
```
agent_lv_new = [-0.240, 0.863, 0.444]
             + [0.00972, -0.03385, -0.03385]
             = [-0.230, 0.829, 0.410]
```

**After normalization:**
```
||agent_lv_new|| = sqrt(0.0529 + 0.6872 + 0.1681) = sqrt(0.9082) ≈ 0.953

agent_lv_normalized = [-0.241, 0.870, 0.430]
```

**Similarly for target_lv[0]:**
```python
self.target_lv[0] += learning_rate * error * agent_vec
self.target_lv[0] += 0.05 * 0.977 * [-0.240, 0.863, 0.444]
self.target_lv[0] += [-0.01172, 0.04216, 0.02169]
```

**Before normalization:**
```
target_lv[0]_new = [0.199, -0.693, -0.693]
                 + [-0.01172, 0.04216, 0.02169]
                 = [0.187, -0.651, -0.671]
```

**After normalization:**
```
target_lv[0]_normalized = [0.190, -0.662, -0.682]
```

#### Summary: Update 1

| Vector | Before | After | Change Direction |
|--------|--------|-------|------------------|
| `agent_lv` | [-0.240, 0.863, 0.444] | [-0.241, 0.870, 0.430] | Moved toward target_0 |
| `target_lv[0]` | [0.199, -0.693, -0.693] | [0.190, -0.662, -0.682] | Moved toward agent_lv |

**Effect:** Since error was positive (underestimated), both vectors moved **toward** each other, increasing their dot product. Future predictions for this pair will be higher.

---

### Update 2: drone_1's Action (Observed by drone_0)

**Input:** other_agent_idx=1, target_idx=21, observed_reward=1.0

#### Step 2.2.1: Compute Predicted Reward (for drone_1)

```python
predicted = self._predict_reward_for_other(1, 21)
# dot = np.dot(other_agents_lv[1], target_lv[21])
```

**Using drone_0's estimate of drone_1:**

```
other_agents_lv[1] = [0.255, -0.349, -0.902]
target_lv[21]      = [-0.567, 0.477, -0.672]

dot = (0.255)×(-0.567) + (-0.349)×(0.477) + (-0.902)×(-0.672)
dot = -0.145 - 0.166 + 0.606
dot = 0.295

predicted = (1 + 0.295) / 2 = 0.648
```

#### Step 2.2.2: Compute Error

```python
error = 1.0 - 0.648 = 0.352
```

**Interpretation:** drone_0 **underestimated** drone_1's reward. It predicted 0.648, but drone_1 got 1.0.

#### Step 2.2.3: SGD Update

```python
# Update drone_0's estimate of drone_1
self.other_agents_lv[1] += 0.05 * 0.352 * target_lv[21]
self.other_agents_lv[1] += 0.05 * 0.352 * [-0.567, 0.477, -0.672]
self.other_agents_lv[1] += [-0.00998, 0.00840, -0.01183]

# Update target_lv[21]
self.target_lv[21] += 0.05 * 0.352 * other_agents_lv[1]
self.target_lv[21] += 0.05 * 0.352 * [0.255, -0.349, -0.902]
self.target_lv[21] += [0.00449, -0.00614, -0.01588]
```

#### Summary: Update 2

| Vector | Change Direction |
|--------|------------------|
| `other_agents_lv[1]` | Moved **toward** target_21 |
| `target_lv[21]` | Moved **toward** drone_1's estimate |

**Effect:** Since error was positive (underestimated), vectors moved **together**. drone_0 now "believes" that drone_1 is more compatible with target_21.

---

### Updates 3-6: Other Drones' Actions

The same process repeats for drone_2 through drone_5:

| Update | Drone | Target | Reward | Error | Effect |
|--------|-------|--------|--------|-------|--------|
| 3 | drone_2 | target_22 | 1.0 | positive | Updates `other_agents_lv[2]`, `target_lv[22]` |
| 4 | drone_3 | target_13 | 1.0 | positive | Updates `other_agents_lv[3]`, `target_lv[13]` |
| 5 | drone_4 | target_21 | 1.0 | positive | Updates `other_agents_lv[4]`, `target_lv[21]` (again!) |
| 6 | drone_5 | target_19 | 1.0 | positive | Updates `other_agents_lv[5]`, `target_lv[19]` |

**Key insight:** In a single step, drone_0 updates:
- Its own `agent_lv` (once, from its own action)
- `target_lv[21]` (twice — from drone_1 AND drone_4's actions on the same target)
- `target_lv[0]`, `target_lv[22]`, `target_lv[13]`, `target_lv[19]` (once each)
- `other_agents_lv[1..5]` (once each)

**With HP reduction rewards:** In Step 1, all rewards are 1.0 because each target has full HP. However, differentiation emerges over time — when a drone repeatedly hits a mismatched target (e.g., structural → Class C), the low-HP attributes deplete quickly, causing subsequent hits to deal less damage and receive lower rewards (e.g., 7/12 = 0.58 instead of 1.0).

---

## Phase 3: Action Selection

### Code Path

```
main_zk_mrta.run_episode()
    └── actions[agent_id] = policy[agent_id].select_action(agent_obs)
        └── SelfishEpGreedyCFPolicy.select_action()
```

### Step 3.1: Parse Active Targets

```python
targets_obs = observation['targets']  # 75 floats
num_targets = len(targets_obs) // 3   # 25

active_targets = []
for t_idx in range(num_targets):
    is_active = targets_obs[t_idx * 3 + 2] > 0.5
    if is_active:
        active_targets.append(t_idx)

# Result: active_targets = [0, 1, 2, ..., 24]  (all 25 still active)
```

### Step 3.2: Build Valid Actions

```python
valid_actions = []  # allow_noop=False by default
valid_actions.extend([t + 1 for t in active_targets])
# valid_actions = [1, 2, 3, ..., 25]
```

### Step 3.3: ε-Greedy Decision

```python
if self.rng.random() < self.epsilon:  # epsilon = 0.297
    # EXPLORE: Random action
    action = self.rng.choice(valid_actions)
else:
    # EXPLOIT: Best predicted reward
    best_action = valid_actions[0]
    best_reward = -np.inf
    
    for action in valid_actions:
        if action == 0:
            continue
        target_idx = action - 1
        predicted = self.predict_reward(target_idx)
        if predicted > best_reward:
            best_reward = predicted
            best_action = action
    
    action = best_action
```

### Step 3.4: Exploitation Example

If drone_0 exploits (70% chance), it computes predicted rewards for all 25 targets using the **updated** vectors:

| Target | Class | Predicted Reward (after learning) |
|--------|-------|----------------------------------|
| target_0 | B | ~0.45 |
| target_1 | C | ~0.52 |
| ... | ... | ... |
| target_18 | A | ~0.62 (increased from 0.605!) |
| target_19 | C | ~0.43 |
| ... | ... | ... |

**Note:** target_18's predicted reward increased because Update 1 moved `agent_lv` and `target_lv[18]` closer together.

### Step 3.5: Decay Epsilon

```python
self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
self.epsilon = max(0.05, 0.297 * 0.99)
self.epsilon = max(0.05, 0.294)
self.epsilon = 0.294
```

### Step 2 Action Result

From the episode log, drone_0's Step 2 action was:
- **Action:** 1 (fire at target_0)
- **Reward:** 1.0

---

## Phase 4: Execution

### Code Path

```
main_zk_mrta.run_episode()
    └── obs, rewards, terminated, truncated, info = env.step(actions)
        └── DroneEngageZKMRTA.step()
            ├── Apply damage to targets
            ├── Calculate rewards
            └── Store last_actions, last_rewards for next observation
```

### What Happens in the Environment

For drone_0's action (fire at target_0):

```python
# Get target
target_idx = action - 1  # 0
target = self.targets[0]  # Class B

# Apply damage
damage_profile = drone.damage_profile  # systems: {structural: 1, envelope: 1, utilities: 10}
target.attributes.apply_damage(damage_profile)
# target_0 attributes: {structural: 15→14, envelope: 150→149, utilities: 15→5}

# Calculate reward (HP reduction mode, REWARD_DOMINANT_ATTRIBUTE=False)
hp_before = 180  # 15 + 150 + 15
hp_after = 168   # 14 + 149 + 5
total_damage = 12  # 1 + 1 + 10
reward = total_damage / self.max_weapon_damage  # 12 / 12 = 1.0
```

**With HP reduction mode:** Every hit deals 12 total damage (1+1+10 for systems weapon), and max_weapon_damage is also 12, so every successful hit gives reward 1.0 regardless of weapon-target compatibility.

### Stored for Next Step

```python
self.last_actions = {
    "drone_0": 5,   # target_4
    "drone_1": 4,   # target_3
    "drone_2": 4,   # target_3
    "drone_3": 1,   # target_0
    "drone_4": 8,   # target_7
    "drone_5": 1,   # target_0
}

self.last_rewards = {
    "drone_0": 1.0,
    "drone_1": 1.0,
    "drone_2": 1.0,
    "drone_3": 1.0,
    "drone_4": 1.0,
    "drone_5": 1.0,
}
```

These will appear in Step 3's observation.

---

## Summary: What drone_0 Learned in Step 2

### Vectors Updated

| Vector | Updates | Net Effect |
|--------|---------|------------|
| `agent_lv` | 1 (from own action) | Moved toward target_0 |
| `target_lv[0]` | 1 (from own action) | Moved toward agent_lv |
| `target_lv[21]` | 2 (drone_1 + drone_4) | Both pushed toward their estimates |
| `target_lv[22]` | 1 (from drone_2) | Moved toward drone_2's estimate |
| `target_lv[13]` | 1 (from drone_3) | Moved toward drone_3's estimate |
| `target_lv[19]` | 1 (from drone_5) | Moved toward drone_5's estimate |
| `other_agents_lv[1..5]` | 1 each | All moved toward their targets |

### Key Observations

1. **Collaborative learning works:** drone_0 learned about 5 targets it didn't engage, just by observing others' rewards.

2. **HP reduction mode:** In early steps, all rewards are 1.0 (full HP targets). Differentiation emerges as targets take damage — mismatched weapons deplete low-HP attributes quickly, causing diminishing rewards on subsequent hits.

3. **Epsilon decayed:** 0.297 → 0.294. Over many steps, drone_0 will explore less and exploit more.

4. **No explicit coordination:** drone_0 has no idea what actions other drones will take in Step 3. It just picks its best option independently.

5. **Reward mode matters:** To learn specialization, use `REWARD_DOMINANT_ATTRIBUTE=True` which gives different rewards based on weapon-class compatibility.

---

*Previous: [02_initial_state.md](02_initial_state.md) — drone_0's Initial State*

*Next: [04_insights.md](04_insights.md) — Key Insights*
