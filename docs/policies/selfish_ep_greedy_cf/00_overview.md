# Part 1: The Big Picture

## What is the Selfish ε-Greedy CF Policy?

The `SelfishEpGreedyCFPolicy` is a **decentralized learning policy** for multi-agent task allocation. Each drone (agent) learns to predict which targets it should engage, without knowing:

- Its own weapon's damage profile
- The targets' vulnerability profiles (class attributes)
- What other drones are doing or learning

The only feedback each drone receives is a **scalar reward** after each action.

---

## The Problem: Zero-Knowledge Multi-Robot Task Allocation (ZK-MRTA)

In the TabulaDrone environment:

| Entity | Hidden Information |
|--------|-------------------|
| **Drones** | Don't know their weapon type's damage profile |
| **Targets** | Don't know their class's attribute distribution |
| **System** | No central coordinator, no shared state |

**The challenge**: How can drones learn to specialize and coordinate without communication?

---

## The Solution: Collaborative Filtering via Matrix Factorization

The policy borrows from **recommender systems**. Just as Netflix predicts "user × movie" ratings without knowing why users like certain movies, our drones predict "drone × target" rewards without knowing the underlying weapon-class compatibility.

### Core Idea

Each drone maintains **private latent vectors**:

```
agent_lv      : This drone's learned "weapon signature" (k-dimensional)
target_lv[j]  : This drone's estimate of target j's "vulnerability signature"
other_agents_lv[i] : This drone's estimate of drone i's "weapon signature"
```

**Predicted reward** = How well the signatures align (dot product, scaled to [0,1])

### Learning Signal

After each step, drones observe:
1. **Their own reward** → Update `agent_lv` and `target_lv`
2. **Other drones' rewards** → Update `other_agents_lv` and `target_lv`

This is **collaborative learning without communication** — each drone builds its own model of the world, refined by observing others' outcomes.

---

## How Learning Leads to Effectiveness

Here's the intuitive flow of how a drone learns to specialize, without any math:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    THE LEARNING JOURNEY (drone_0 with systems weapon)           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  PHASE 1: EXPLORATION (Early Episodes)                                          │
│  ─────────────────────────────────────                                          │
│  • drone_0 tries random targets (30% exploration rate)                          │
│  • Hits Class A target → gets reward 1.0 (first hit, full HP)                   │
│  • Hits Class A again → gets reward 0.58 (low-HP attributes depleted!)          │
│  • Hits Class C target → gets reward 1.0                                        │
│  • Hits Class C again → gets reward 1.0 (high-HP attribute still has HP)        │
│                                                                                  │
│  PHASE 2: PATTERN RECOGNITION                                                   │
│  ────────────────────────────────                                               │
│  • drone_0 notices: "Class C targets keep giving me high rewards"               │
│  • drone_0 notices: "Class A/B targets give diminishing rewards"                │
│  • Latent vectors adjust: drone_0's signature aligns with Class C signatures    │
│                                                                                  │
│  PHASE 3: EXPLOITATION (Later Episodes)                                         │
│  ──────────────────────────────────────                                         │
│  • Exploration rate decays to 5%                                                │
│  • drone_0 now predicts: "Class C targets will give me high reward"             │
│  • drone_0 preferentially selects Class C targets                               │
│  • Result: SPECIALIZATION without ever knowing it has a "systems" weapon        │
│                                                                                  │
│  PHASE 4: COLLABORATIVE BOOST                                                   │
│  ────────────────────────────────                                               │
│  • drone_0 also observes drone_4's rewards (also systems weapon)                │
│  • drone_4 gets high rewards on Class C → drone_0 learns faster                 │
│  • drone_0 observes drone_1 (structural) getting low rewards on Class C         │
│  • drone_0 learns: "drone_1 is different from me"                               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### The Key Insight: Reward Patterns Reveal Hidden Structure

| What drone_0 experiences | What it means (hidden from drone) |
|--------------------------|-----------------------------------|
| "Target X keeps giving me 1.0" | X is Class C (matches systems weapon) |
| "Target Y gave 1.0, then 0.5, then 0.2" | Y is Class A/B (mismatched, HP depletes fast) |
| "drone_4 gets same rewards as me" | drone_4 has same weapon type |
| "drone_1 gets opposite rewards" | drone_1 has different weapon type |

**The magic:** Without knowing weapon types or target classes, drones learn to specialize simply by chasing high rewards. The latent vectors become implicit encodings of the hidden weapon-class compatibility matrix.

---

## The Math (Gentle Introduction)

Now let's see how the learning actually works, with a simple example.

s### Step 0: How the Environment Calculates Rewards

Before we can learn, we need to understand what we're learning *from*. The environment calculates rewards based on **actual damage dealt**:

$$\text{reward} = \frac{\text{total damage dealt}}{\text{max weapon damage}}$$

**Example: systems weapon hits Class B target (first time)**

```
Weapon damage:  {structural: 1, envelope: 1, utilities: 10}  → max = 12
Target HP:      {structural: 15, envelope: 150, utilities: 15}

Damage dealt = min(1,15) + min(1,150) + min(10,15) = 1 + 1 + 10 = 12
Reward = 12/12 = 1.0 ✓
```

**Same weapon hits same target again:**

```
Target HP now:  {structural: 14, envelope: 149, utilities: 5}  (depleted!)

Damage dealt = min(1,14) + min(1,149) + min(10,5) = 1 + 1 + 5 = 7
Reward = 7/12 = 0.58 ✗ (diminishing!)
```

**The key insight:** Damage is capped by remaining HP. Mismatched weapons deplete low-HP attributes quickly → diminishing rewards. Matched weapons (systems → Class C with 150 utilities HP) sustain high rewards longer.

### Step 1: Prediction

Each drone predicts how good a target will be using a **dot product** of two vectors:

$$\text{predicted reward} = \frac{1 + (\text{drone vector} \cdot \text{target vector})}{2}$$

**Simple example with 2D vectors:**

```
drone_0 vector  = [0.8, 0.2]    (points mostly "right")
target_5 vector = [0.6, 0.8]    (points "up-right")

dot product = (0.8 × 0.6) + (0.2 × 0.8) = 0.48 + 0.16 = 0.64

predicted reward = (1 + 0.64) / 2 = 0.82
```

**Intuition:** Vectors pointing in similar directions → high dot product → high predicted reward.

### Step 2: Learning from Reality

After firing, drone_0 gets the **actual reward** (say, 1.0). Now it can learn:

```
predicted = 0.82
actual    = 1.00
error     = actual - predicted = +0.18  (underestimated!)
```

### Step 3: Update Both Vectors

The update rule nudges **both vectors toward each other** when we underestimated:

$$\text{new drone vector} = \text{normalize}(\text{drone vector} + \text{learning rate} \times \text{error} \times \text{target vector})$$

$$\text{new target vector} = \text{normalize}(\text{target vector} + \text{learning rate} \times \text{error} \times \text{drone vector})$$

**Continuing our example (learning_rate = 0.05):**

```
error = +0.18

Drone update:
  adjustment = 0.05 × 0.18 × [0.6, 0.8] = [0.0054, 0.0072]
  drone_0 vector: [0.800, 0.200] → [0.806, 0.207]

Target update:
  adjustment = 0.05 × 0.18 × [0.8, 0.2] = [0.0072, 0.0018]
  target_5 vector: [0.600, 0.800] → [0.601, 0.799]
```

**What happened?** Both vectors rotated slightly toward each other. The drone "learned" about the target, and the target vector now better represents "what kind of drone works well against me."

**Why update both?** This is symmetric learning — the target vector encodes vulnerability patterns just as the drone vector encodes weapon patterns. When multiple drones hit the same target, that target's vector gets refined from multiple perspectives.

### Step 4: The Opposite Case

If drone_0 had **overestimated** (predicted 0.9, got 0.5):

```
error = 0.5 - 0.9 = -0.4  (overestimated!)
adjustment = 0.05 × (-0.4) × target_vector = negative adjustment
```

Vectors move **apart** → future predictions for this pair will be lower.

### Why This Works

| Scenario | Error | Vector Movement | Future Prediction |
|----------|-------|-----------------|-------------------|
| Underestimated (good surprise) | + | Vectors align more | Higher |
| Overestimated (bad surprise) | − | Vectors separate | Lower |
| Perfect prediction | 0 | No change | Same |

Over many interactions:
- **Good pairings** (systems → Class C): Vectors align → high predictions → drone keeps choosing these
- **Bad pairings** (systems → Class A): Vectors separate → low predictions → drone avoids these

**Result:** Specialization emerges from simple vector arithmetic!

---

## ZK-MRTA Compliance

The policy strictly adheres to Zero-Knowledge constraints:

| Constraint | How It's Satisfied |
|------------|-------------------|
| No shared state | Each drone has its own policy instance with private vectors |
| No communication | Drones only observe rewards, not internal states |
| No central coordinator | Action selection is fully decentralized |
| Indirect observation only | Learning from reward signals, not from seeing attributes |

---

## The Scenario We'll Trace

Throughout this walkthrough, we'll follow **drone_0** through **Step 2** of Episode 1:

| Parameter | Value |
|-----------|-------|
| **Drones** | 6 (indices 0-5) |
| **Targets** | 25 (indices 0-24) |
| **Latent dimension** | 3 |
| **Learning rate** | 0.05 |
| **Epsilon (exploration)** | 0.3 (decays by 0.99 per action) |
| **Reward mode** | HP reduction (total damage / max weapon damage) |

### Weapon-Class Mappings (Hidden from Agents)

**Weapon damage profiles:**

| Weapon | structural_integrity | envelope_integrity | utilities_lifesafety |
|--------|---------------------|-------------------|---------------------|
| structural | 10 | 1 | 1 |
| breach | 1 | 10 | 1 |
| systems | 1 | 1 | 10 |

**Target class attributes (initial HP per attribute):**

| Class | structural_integrity | envelope_integrity | utilities_lifesafety |
|-------|---------------------|-------------------|---------------------|
| A | 150 | 15 | 15 |
| B | 15 | 150 | 15 |
| C | 15 | 15 | 150 |

**Optimal pairings** (what the policy should learn):
- `structural` weapon → Class A targets (dominant attribute = structural_integrity)
- `breach` weapon → Class B targets (dominant attribute = envelope_integrity)
- `systems` weapon → Class C targets (dominant attribute = utilities_lifesafety)

### drone_0's Assignment

- **Weapon**: `systems`
- **Optimal targets**: Class C (utilities_lifesafety = 150)
- **Expected high reward**: 12/12 = 1.0 (total damage / max total damage)
- **Reward mode**: HP reduction — rewards based on total HP reduced, not dominant attribute

---

## What's Next

In the following sections, we'll:

1. **[01_math_foundation.md](01_math_foundation.md)** — Derive the SGD update rules from first principles
2. **[02_initial_state.md](02_initial_state.md)** — Examine drone_0's latent vectors before Step 2
3. **[03_step2_full_cycle.md](03_step2_full_cycle.md)** — Trace the complete Observe → Learn → Select → Execute cycle
4. **[04_insights.md](04_insights.md)** — Understand what the vectors are learning and why it works

---

*Next: [01_math_foundation.md](01_math_foundation.md) — The Math Foundation*
