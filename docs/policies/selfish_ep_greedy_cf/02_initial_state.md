# Part 3: drone_0's Initial State

Before we trace Step 2, let's examine what drone_0 "knows" at the start of Episode 1. All data comes from `logs/run_20260106_122459/selfish_ep_greedy_cf/learning_state/learning_state_ep01.json`.

---

## 1. Policy Initialization

When `main_zk_mrta.py` creates the policy instances:

```python
policies[agent_id] = SelfishEpGreedyCFPolicy(
    num_targets=25,
    agent_idx=0,
    num_agents=6,
    latent_dim=3,
    learning_rate=0.05,
    epsilon=0.3,
    epsilon_decay=0.99,
    epsilon_min=0.05,
    seed=42,  # config.seed + agent_idx
)
```

The `BaseCFPolicy.__init__` method initializes three sets of latent vectors:

| Vector | Shape | Purpose |
|--------|-------|---------|
| `agent_lv` | (3,) | drone_0's own latent representation |
| `target_lv` | (25, 3) | drone_0's estimates of all 25 targets |
| `other_agents_lv` | (6, 3) | drone_0's estimates of all 6 drones (including itself) |

---

## 2. drone_0's Latent Vectors (Pre-Episode)

These are the **randomly initialized** vectors before any learning occurs.

### 2.1 Agent Latent Vector (`agent_lv`)

This 3D vector represents drone_0's "weapon signature" — what the policy will learn to associate with high/low rewards.

```python
agent_lv = [-0.240, 0.863, 0.444]
```

**Visualization:**

```
Dimension:    0      1      2
            ─────────────────────
agent_lv:  -0.24   0.86   0.44
            ███    █████  ████
              ▼      ▲      ▲
           (neg)  (pos)  (pos)
```

**Note:** This is random initialization. The policy doesn't know that drone_0 has a `systems` weapon. It will learn this implicitly through rewards.

### 2.2 Target Latent Vectors (`target_lv`)

drone_0's estimates of all 25 targets. Here are a few examples:

| Target | Class | Latent Vector (3D) |
|--------|-------|-------------------|
| target_0 | B | `[0.199, -0.693, -0.693]` |
| target_1 | C | `[-0.758, 0.628, 0.174]` |
| target_7 | C | `[-0.105, 0.685, -0.721]` |
| target_13 | B | `[-0.700, 0.615, -0.363]` |
| target_19 | C | `[-0.268, -0.550, 0.791]` |

**Key insight:** At initialization, these vectors are random. The policy has no idea which targets are Class A, B, or C. It will learn target characteristics through observed rewards.

### 2.3 Other Agents Latent Vectors (`other_agents_lv`)

drone_0's estimates of what other drones' "weapon signatures" might be:

| Drone | Weapon | drone_0's Estimate (3D) |
|-------|--------|------------------------|
| drone_0 | systems | `[-0.259, -0.702, 0.664]` |
| drone_1 | structural | `[0.255, -0.349, -0.902]` |
| drone_2 | structural | `[-0.548, -0.507, 0.666]` |
| drone_3 | breach | `[0.334, 0.940, -0.067]` |
| drone_4 | systems | `[-0.749, 0.420, 0.513]` |
| drone_5 | breach | `[0.221, 0.975, -0.022]` |

**Note:** drone_0's estimate of itself (`other_agents_lv[0]`) is different from its actual `agent_lv`. This is intentional — `other_agents_lv` is used when learning from others' rewards, while `agent_lv` is used for its own predictions.

---

## 3. Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `latent_dim` | 3 | Dimensionality of latent space |
| `learning_rate` | 0.05 | Step size for SGD updates |
| `epsilon` | 0.3 | Exploration probability (30% random actions) |
| `epsilon_decay` | 0.99 | Multiply epsilon by this after each action |
| `epsilon_min` | 0.05 | Floor for epsilon (always 5% exploration) |

### Epsilon Decay Schedule

| After N actions | Epsilon |
|-----------------|---------|
| 0 | 0.300 |
| 10 | 0.270 |
| 50 | 0.182 |
| 100 | 0.110 |
| 200 | 0.040 → 0.050 (floored) |

---

## 4. What Happened in Step 1

Before Step 2, drone_0 already took one action in Step 1. Here's what happened:

### Step 1 Observation (received by drone_0)

```python
{
    "targets": [...],  # 75 floats (25 targets × 3 values each)
    "selected_targets": [0, 0, 0, 0, 0, 0],  # No prior actions
    "observed_rewards": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # No prior rewards
}
```

**Key point:** In Step 1, `selected_targets` and `observed_rewards` are all zeros because there was no previous step. This means `update_from_observation()` does nothing — there's nothing to learn from.

### Step 1 Action Selection

drone_0 used ε-greedy to select an action:
- Random roll determined explore vs exploit
- Selected action: **1** (fire at target_0)

### Step 1 Execution

| Drone | Action | Target | Target Class | Reward |
|-------|--------|--------|--------------|--------|
| drone_0 | 1 | target_0 | B | **1.0** |
| drone_1 | 22 | target_21 | A | **1.0** |
| drone_2 | 23 | target_22 | B | **1.0** |
| drone_3 | 14 | target_13 | B | **1.0** |
| drone_4 | 22 | target_21 | A | **1.0** |
| drone_5 | 20 | target_19 | C | **1.0** |

**Note:** All drones got reward 1.0 because `REWARD_DOMINANT_ATTRIBUTE=False` — rewards are based on total HP reduction (12/12 = 1.0 for any hit), not dominant attribute damage.

### Step 1 Learning

**No learning occurred in Step 1** because the observation contained no prior action/reward data.

### Epsilon After Step 1

```python
epsilon = max(0.05, 0.3 * 0.99) = 0.297
```

---

## 5. State Entering Step 2

As drone_0 enters Step 2:

| State | Value |
|-------|-------|
| `agent_lv` | Unchanged from initialization |
| `target_lv` | Unchanged from initialization |
| `other_agents_lv` | Unchanged from initialization |
| `epsilon` | 0.297 (decayed once) |

**This is the first step where learning will actually occur**, because the Step 2 observation will contain Step 1's actions and rewards.

---

## 6. Predicted Rewards (Before Any Learning)

Using the initial vectors, let's compute what drone_0 would predict for a few targets:

### Prediction Formula

$$\hat{r} = \frac{1 + \mathbf{u} \cdot \mathbf{v}}{2}$$

### Example Calculations

**target_0 (Class B):**
```
agent_lv     = [-0.240, 0.863, 0.444]
target_lv[0] = [0.199, -0.693, -0.693]

dot product = (-0.240)(0.199) + (0.863)(-0.693) + (0.444)(-0.693)
            = -0.048 - 0.598 - 0.308
            = -0.954

predicted_reward = (1 + (-0.954)) / 2 = 0.023
```

**target_19 (Class C):**
```
agent_lv      = [-0.240, 0.863, 0.444]
target_lv[19] = [-0.268, -0.550, 0.791]

dot product = (-0.240)(-0.268) + (0.863)(-0.550) + (0.444)(0.791)
            = 0.064 - 0.475 + 0.351
            = -0.060

predicted_reward = (1 + (-0.060)) / 2 = 0.470
```

### Initial Predictions Summary

| Target | Class | Dot Product | Predicted Reward |
|--------|-------|-------------|------------------|
| target_0 | B | -0.954 | 0.023 |
| target_19 | C | -0.060 | 0.470 |

**Note:** These predictions are essentially random because the vectors haven't been trained yet. With HP reduction rewards (all 1.0), the policy will learn that all targets are equally good — there's no weapon-class specialization signal in this reward mode.

---

*Previous: [01_math_foundation.md](01_math_foundation.md) — The Math Foundation*

*Next: [03_step2_full_cycle.md](03_step2_full_cycle.md) — Step 2: The Full Cycle*
