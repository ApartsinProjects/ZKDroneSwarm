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

### 1. The Latent Space: A "Matching" Map

Think of the latent space as a 2D map (though in the code, it's 3-dimensional).

- **The Drone's Position (`agent_lv`):** Represents its "Weapon Profile" (e.g., high structural damage, low utility damage).
- **The Target's Position (`target_lv`):** Represents its "Vulnerability Profile" (e.g., weak structural integrity).
- **The Dot Product:** Measures the "closeness" of these profiles. If the drone is at "12 o'clock" on the map and the target is also at "12 o'clock," the reward prediction is maximized (1.0).

### 2. The Flow: How Information Moves

The learning process is a **triangulation of data**. Here is how the three components interact:

#### A. Direct Experience (Self ↔ Target)

When you fire and get a reward, you perform a **Symmetric Update**.

- **The Logic:** You assume the error is caused by both an imperfect understanding of yourself and an imperfect understanding of the target.
- **The Result:** The drone vector and the target vector "tug" on each other. If the reward was high, they move closer. This creates a "gravity well" in the latent space where successful pairings live.

#### B. Observational Learning (Other ↔ Target)

This is where `other_agents_lv` becomes critical. It acts as a proxy or a "simulation" of your teammates.

- **The Flow:** Drone A watches Drone B hit Target X.
- **The Logic:** Drone A thinks: "I don't know exactly what Drone B's weapon is, but I saw the result. I will update my 'mental model' of Drone B (`other_agents_lv`) and my 'mental model' of Target X (`target_lv`) so they better explain that result."
- **The Importance:** This allows Drone A to learn about Target X's vulnerability even if it hasn't touched it yet.

#### C. The Indirect "Aha!" Moment (Self ↔ Other)

This is the "magic" of the latent space.

If Drone A and Drone B happen to have similar weapons, their `other_agents_lv` and `agent_lv` will eventually end up in the same "neighborhood" of the latent space.

Because the `target_lv` is being updated by everyone, it becomes an **emergent consensus** (each drone has its own private copy, but they converge toward similar values).

**The Chain Reaction:**
1. Drone B (Physical) hits Target X → Target X vector moves.
2. Drone A (Observer) sees this → Drone A updates its own copy of Target X's vector.
3. Drone A looks at its own `agent_lv` and sees it is now close to that updated Target X vector.
4. **Result:** Drone A now "knows" to target X in the next step.

### 3. Understanding the Learning Dynamics

To visualize why this leads to specialization, consider the "Push-Pull" mechanics:

| Action | Vector Interaction | Latent Space Effect |
|--------|-------------------|---------------------|
| High Reward | Attraction | The Drone and Target "collide" in the latent space, signaling a perfect match. |
| Low Reward | Repulsion | The Drone and Target "push" away from each other, creating a gap that prevents future selection. |
| Observation | Alignment | The Observer aligns the "Teammate Vector" with the "Target Vector," essentially "filling in" the map for free. |

### 4. Why This Leads to Specialization

The **Normalization** step in the math (`new_vector = normalize(...)`) is vital. Because the vectors are constrained to a fixed length (a unit sphere), they cannot simply grow to infinity to satisfy a reward.

**Instead, they must rotate.**

- If a drone is good at killing "Class A" targets, its vector will rotate toward "Class A" space.
- By rotating toward A, it naturally rotates **away** from "Class B."

Specialization is an inevitable byproduct of the geometry: **You cannot point in two opposite directions at once.** The drones are forced to "choose" a signature that yields the most consistent rewards.

### 5. Summary of the Flow

| Step | Action |
|------|--------|
| **Prediction** | Check the map (dot product) |
| **Selection** | Pick the target with highest predicted reward |
| **Action** | Test the prediction |
| **Correction** | If reality ≠ prediction, rotate the vectors to "fix" the map |
| **Scaling** | Use teammates' actions to "fix" the map faster without wasting your own ammo |

### Key Insight: Reward Patterns Reveal Hidden Structure

| What drone_0 experiences | What it means (hidden from drone) |
|--------------------------|-----------------------------------|
| "Target X keeps giving me 1.0" | X is Class C (matches systems weapon) |
| "Target Y gave 1.0, then 0.5, then 0.2" | Y is Class A/B (mismatched, HP depletes fast) |
| "drone_4 gets same rewards as me" | drone_4 has same weapon type |
| "drone_1 gets opposite rewards" | drone_1 has different weapon type |

---

## The Math (Gentle Introduction)

Now let's see how the learning actually works, with a simple example.

### Step 0: How the Environment Calculates Rewards

Before we can learn, we need to understand what we're learning *from*. The environment supports **3 reward modes**:

#### 1. HP_REDUCTION (default)

Rewards proportional to total HP reduced relative to HP before the shot.

$$\text{reward} = \frac{HP_{before} - HP_{after}}{HP_{before}}$$

#### 2. DOMINANT_ATTRIBUTE

Rewards damage dealt to the target's **currently highest** attribute. Encourages pivoting as attributes deplete.

$$\text{reward} = \frac{\text{damage to dominant attribute}}{\text{max single-attribute weapon damage}}$$

#### 3. ATTRIBUTE_ALIGNMENT

Rewards based on damage efficiency weighted by cosine similarity between weapon and target HP profiles.

$$\text{reward} = \text{DamageEfficiency} \times \text{CosineSimilarity}(\vec{weapon}, \vec{targetHP})$$

Where:

$$\text{DamageEfficiency} = \frac{\sum_i \min(weaponDmg_i, targetHP_i)}{\sum_i weaponDmg_i}$$

*(Actual damage dealt / maximum weapon potential — capped by remaining HP per attribute)*

#### ZK-MRTA Compliance

| Mode | Information used | ZK-MRTA compliant? |
|------|------------------|-------------------|
| **HP_REDUCTION** | Aggregate HP (scalars only) | **Yes** — pure black-box outcome |
| **DOMINANT_ATTRIBUTE** | Per-attribute HP + per-attribute damage | **No** — reward pattern leaks weapon profile |
| **ATTRIBUTE_ALIGNMENT** | Full weapon vector + full target HP vector | **No** — reward pattern leaks weapon profile |

**Purist ZK-MRTA** requires HP_REDUCTION. The other modes are valid reward shaping strategies but allow the drone to theoretically reverse-engineer its weapon profile from reward patterns.

---

#### Examples

**Configuration:**
- drone_1 weapon: `{structural: 2, envelope: 35, utilities: 5}` → total = 42
- drone_2 weapon: `{structural: 5, envelope: 35, utilities: 2}` → total = 42
- Target HP: `{structural: 10, envelope: 30, utilities: 25}` → total = 65

**HP_REDUCTION:**
```
drone_1: damage = min(2,10) + min(35,30) + min(5,25) = 37  →  reward = 37/65 = 0.569
drone_2: damage = min(5,10) + min(35,30) + min(2,25) = 37  →  reward = 37/65 = 0.569
```

**DOMINANT_ATTRIBUTE:** (dominant = envelope, HP=30)
```
drone_1: damage_to_dominant = 35  →  reward = 35/35 = 1.0
drone_2: damage_to_dominant = 35  →  reward = 35/35 = 1.0
```

**ATTRIBUTE_ALIGNMENT:**
```
drone_1: efficiency = 37/42 = 0.881, cosine([2,35,5], [10,30,25]) = 0.837  →  reward = 0.737
drone_2: efficiency = 37/42 = 0.881, cosine([5,35,2], [10,30,25]) = 0.806  →  reward = 0.710
```

| Mode | drone_1 | drone_2 | Differentiates? |
|------|---------|---------|-----------------|
| HP_REDUCTION | 0.569 | 0.569 | No |
| DOMINANT_ATTRIBUTE | 1.0 | 1.0 | No |
| ATTRIBUTE_ALIGNMENT | 0.737 | 0.710 | **Yes** |

**Key insight:** Only ATTRIBUTE_ALIGNMENT differentiates weapons with identical aggregate damage but different profile shapes.

### Step 1: Prediction

Each drone predicts how good a target will be using a **dot product** of two vectors:

$$\text{predicted reward} = \frac{1 + (\text{drone vector} \cdot \text{target vector})}{2}$$

**Predicted reward range:** [0, 1]

Since vectors are normalized (unit length), the dot product ranges from -1 (opposite directions) to +1 (same direction). The `(1 + x) / 2` transformation maps this to [0, 1] to match the reward scale.

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

We use **Stochastic Gradient Descent (SGD)** to minimize the **Mean Squared Error (MSE)** loss:

```
Loss = ½ × (actual - predicted)²
```

#### Deriving the Update Rule

**For the drone vector:**

**Step 1: Chain rule**
```
∂Loss/∂drone = ∂Loss/∂predicted × ∂predicted/∂drone
```

**Step 2: Compute ∂Loss/∂predicted**
```
Loss = ½ × (actual - predicted)²

∂Loss/∂predicted = ½ × 2 × (actual - predicted) × (-1)
                 = -(actual - predicted)
                 = -error
```

**Step 3: Compute ∂predicted/∂drone**
```
predicted = (1 + drone · target) / 2
          = (1 + Σᵢ droneᵢ × targetᵢ) / 2

∂predicted/∂droneᵢ = targetᵢ / 2

∂predicted/∂drone = target / 2    (as a vector)
```

**Step 4: Combine via chain rule**
```
∂Loss/∂drone = ∂Loss/∂predicted × ∂predicted/∂drone
             = (-error) × (target / 2)
             = -error × target / 2
```

SGD moves in the **negative gradient** direction:

```
drone_new = drone - η × ∂Loss/∂drone
          = drone + η × error × target / 2
```

Absorbing `/2` into learning rate:

```
drone_new = drone + η × error × target
```

**For the target vector:** Symmetric derivation yields:

```
target_new = target + η × error × drone
```

#### The Update Formulas

```
new_drone  = normalize(drone  + learning_rate × error × target)
new_target = normalize(target + learning_rate × error × drone)
```

#### Example (learning_rate = 0.1)

**1. Starting State — The Two Vectors**
```
drone_0 vector  = [1.0, 0.0]     (pointing right)
target_5 vector = [0.6, 0.8]     (pointing up-right, ~53° from drone)
```

**2. How Close Are They? (Before Learning)**
```
dot product = 1.0×0.6 + 0.0×0.8 = 0.6
angle = arccos(0.6) = 53.1°
```

**3. Prediction**
```
predicted = (1 + 0.6) / 2 = 0.8
```

**4. Actual Reward & Error**
```
actual = 1.0
error = 1.0 - 0.8 = +0.2  (underestimated by 20%!)
```

**5. Calculate Updates**
```
Drone adjustment  = 0.1 × 0.2 × [0.6, 0.8] = [0.012, 0.016]
Target adjustment = 0.1 × 0.2 × [1.0, 0.0] = [0.02, 0.0]

drone_raw  = [1.012, 0.016]  → normalized → [0.999, 0.016]
target_raw = [0.62, 0.8]     → normalized → [0.613, 0.790]
```

**6. How Close Are They? (After Learning)**
```
dot product = 0.999×0.613 + 0.016×0.790 = 0.624
angle = arccos(0.624) = 51.4°
```

**7. Summary**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Dot product | 0.600 | 0.624 | **+0.024** |
| Angle | 53.1° | 51.4° | **-1.7°** |
| Prediction | 0.80 | 0.812 | **+0.012** |

**What happened?** Both vectors rotated ~1.7° toward each other in a single step. With repeated interactions, they'll continue converging until prediction ≈ actual.

**Why update both?** Symmetric learning — the target vector encodes vulnerability patterns just as the drone vector encodes weapon patterns. When multiple drones hit the same target, that target's vector gets refined from multiple perspectives.

**Why normalize?** Constrains vectors to unit length, preventing unbounded growth and ensuring predictions stay in [0, 1].

### Step 4: Learning from Others

So far we've seen how drone_0 learns from its **own** action. But drone_0 also observes **other drones' rewards** and learns from them too.

#### The Prediction (for another agent)

drone_0 predicts what reward drone_4 should get using its **mental models**:

```
predicted = (1 + other_agents_lv[4] · target_lv[1]) / 2
```

This uses:
- `other_agents_lv[4]` — drone_0's estimate of drone_4's "weapon signature"
- `target_lv[1]` — drone_0's estimate of target_1's "vulnerability"

#### The Update Rule

Same SGD formula as Step 3, but updates different vector pairs:

```
new_other_agents_lv[4] = normalize(other_agents_lv[4] + η × error × target_lv[1])
new_target_lv[1]       = normalize(target_lv[1] + η × error × other_agents_lv[4])
```

#### Example (learning_rate = 0.1)

**1. Starting State — drone_0's Mental Models**
```
other_agents_lv[4] = [0.8, 0.6]     (drone_0's estimate of drone_4)
target_lv[1]       = [0.2, 0.98]    (drone_0's estimate of target_1)
```

**2. How Close Are They? (Before Learning)**
```
dot product = 0.8×0.2 + 0.6×0.98 = 0.16 + 0.588 = 0.748
angle = arccos(0.748) = 41.6°
```

**3. Prediction**
```
predicted = (1 + 0.748) / 2 = 0.874
```

**4. drone_4's Actual Reward (Observed by drone_0)**
```
drone_4 fires at target_1 → gets reward 1.0
drone_0 observes this

actual = 1.0
error = 1.0 - 0.874 = +0.126  (underestimated!)
```

**5. Calculate Updates**
```
other_agents_lv[4] adjustment = 0.1 × 0.126 × [0.2, 0.98] = [0.0025, 0.0123]
target_lv[1] adjustment       = 0.1 × 0.126 × [0.8, 0.6]  = [0.0101, 0.0076]

other_agents_lv[4]: [0.8, 0.6]    → [0.8025, 0.6123] → normalized → [0.795, 0.607]
target_lv[1]:       [0.2, 0.98]   → [0.2101, 0.9876] → normalized → [0.208, 0.978]
```

**6. How Close Are They? (After Learning)**
```
dot product = 0.795×0.208 + 0.607×0.978 = 0.759
angle = arccos(0.759) = 40.6°
```

**7. Summary**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Dot product | 0.748 | 0.759 | **+0.011** |
| Angle | 41.6° | 40.6° | **-1.0°** |
| Prediction | 0.874 | 0.880 | **+0.006** |

**What happened?** drone_0's mental model of drone_4 and target_1 improved — without drone_0 firing a single shot!

#### The Indirect Learning Chain

Over time, if drone_4 has the same weapon type as drone_0:
1. drone_4 gets similar rewards to drone_0 on similar targets
2. drone_0's `other_agents_lv[4]` converges toward drone_0's own `agent_lv`
3. When drone_4 gets high reward on target_1, drone_0's `target_lv[1]` aligns with `other_agents_lv[4]`
4. Since `other_agents_lv[4] ≈ agent_lv`, this means `target_lv[1]` aligns with drone_0's signature too
5. **Result:** drone_0 learns target_1 is good for it, without ever firing at it!

#### What Gets Updated Each Step

| Vector Type | Updated For |
|-------------|-------------|
| `agent_lv` | Only from own action (1 update max) |
| `target_lv[X]` | Targets hit by **any** drone this step |
| `other_agents_lv[i]` | Drones that fired this step |

**Important:** `other_agents_lv` is used **only for learning**, not for target selection. Action selection uses only:

```
predicted_reward = (1 + agent_lv · target_lv[target_idx]) / 2
```

So `other_agents_lv` affects decisions **indirectly** — by improving `target_lv` estimates through observed rewards from other drones.

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
