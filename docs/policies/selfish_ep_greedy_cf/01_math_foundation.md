# Part 2: The Math Foundation

This section provides the rigorous mathematical derivation of the Collaborative Filtering (CF) policy. Each formula is derived from first principles and connected to its role in the learning cycle.

**Document Structure:**

| Section | Content | Role |
|---------|---------|------|
| **0. Reward Calculation** | 3 reward modes | Environment input |
| **1. Vectors** | agent_lv, target_lv, other_agents_lv | Building blocks |
| **2. Prediction** | Dot product formula, scaling | Policy predicts |
| **3. SGD Update** | The learning step | Policy learns |
| ↳ 3.1 Loss Function | MSE definition | *Why we update* |
| ↳ 3.2 Gradient Derivation | Chain rule for u, v | *How to update* |
| ↳ 3.3 Update Formulas | Final formulas + normalization | *What to apply* |
| **4. Learning from Others** | Prediction for others, update rules | Collaborative aspect |
| **5. Summary** | Key equations table | Quick reference |

**Runtime Flow:**
```
[Reward Calculation] ──reward──► [Prediction] ──error──► [SGD Update]
    (Section 0)                   (Section 2)            (Section 3)
                                       ▲                      │
                                       │                      ▼
                                  [Vectors]            [Normalization]
                                 (Section 1)            (Section 3.3)
```

**SGD Update Derivation (Section 3):**
```
[Loss Function] ──────► [Gradient Derivation] ──────► [Update Formulas]
  (Section 3.1)            (Section 3.2)               (Section 3.3)
```

**Vector Data Flow:**
```
                    ┌─────────────────────────────────────────────┐
                    │           PREDICTION (Section 2)            │
                    │                                             │
  agent_lv ────────►│  predicted = (1 + agent_lv · target_lv) / 2 │
  target_lv[j] ────►│                                             │
                    └─────────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │   LEARNING - Own Action (Section 3)         │
                    │                                             │
  agent_lv ◄───────│  agent_lv += η × error × target_lv          │
  target_lv[j] ◄───│  target_lv += η × error × agent_lv          │
                    └─────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────┐
                    │   LEARNING - From Others (Section 4)        │
                    │                                             │
  other_agents_lv[m]◄──│ other_agents_lv[m] += η × error × target_lv  │
  target_lv[j] ◄───────│ target_lv[j] += η × error × other_agents_lv[m]│
                    └─────────────────────────────────────────────┘
```

---

## 0. Reward Calculation (Environment Side)

**Purpose:** Define what the agent learns *from*. The reward signal is the ground truth that drives all learning.

**Flow context:** This is the INPUT to the learning cycle — the environment computes a reward after each action.

### The Three Reward Modes

The environment supports three reward calculation modes. Each produces a scalar reward $r \in [0, 1]$.

#### 0.1 HP_REDUCTION (Default, ZK-MRTA Compliant)

Rewards proportional to total HP reduced relative to HP before the shot.

$$r = \frac{HP_{before} - HP_{after}}{HP_{before}}$$

**Properties:**
- Uses only aggregate HP (scalars)
- No per-attribute information leaks to the agent
- **ZK-MRTA compliant** — pure black-box outcome

#### 0.2 DOMINANT_ATTRIBUTE

Rewards damage dealt to the target's currently highest attribute.

$$r = \frac{d_{dominant}}{\max_k(D_k)}$$

Where:
- $d_{dominant}$ = weapon's damage to the dominant attribute
- $\max_k(D_k)$ = maximum single-attribute damage across all weapons

**Properties:**
- Encodes per-attribute weapon damage in reward signal
- **Not ZK-MRTA compliant** — reward pattern leaks weapon profile

#### 0.3 ATTRIBUTE_ALIGNMENT

Rewards based on damage efficiency weighted by cosine similarity.

$$r = \text{DamageEfficiency} \times \cos(\vec{w}, \vec{h})$$

Where:

$$\text{DamageEfficiency} = \frac{\sum_i \min(w_i, h_i)}{\sum_i w_i}$$

$$\cos(\vec{w}, \vec{h}) = \frac{\vec{w} \cdot \vec{h}}{\|\vec{w}\| \|\vec{h}\|}$$

- $\vec{w}$ = weapon damage profile vector
- $\vec{h}$ = target HP profile vector

**Properties:**
- Uses full weapon and target vectors
- **Not ZK-MRTA compliant** — reward pattern leaks weapon profile

---

## 1. Vectors (Building Blocks)

**Purpose:** Define the latent vectors that store learned knowledge and enable predictions.

**Flow context:** These vectors are the CORE DATA STRUCTURES. They're initialized at start, used for prediction, and updated during learning.

### Vector Definitions

Each agent maintains three sets of latent vectors:

| Vector | Notation | Dimension | Description |
|--------|----------|-----------|-------------|
| **Agent vector** | $\mathbf{u}_i$ | $k \times 1$ | This agent's latent representation (encodes learned "weapon signature") |
| **Target vectors** | $\mathbf{v}_j$ | $k \times 1$ each | This agent's estimate of each target's vulnerability |
| **Other agent vectors** | $\mathbf{u}_m^{\text{est}}$ | $k \times 1$ each | This agent's estimate of other agents' signatures |

Where $k$ is the latent dimension (typically 2-3).

### Initialization

All vectors are initialized as random unit vectors:

$$\mathbf{x} = \text{normalize}(\text{uniform}(-1, 1)^k)$$

### Vector Usage

| Vector | Used in Prediction? | Updated by Own Action? | Updated by Others' Actions? |
|--------|---------------------|------------------------|----------------------------|
| `agent_lv` | ✅ Yes | ✅ Yes | ❌ No |
| `target_lv[j]` | ✅ Yes | ✅ Yes | ✅ Yes |
| `other_agents_lv[m]` | ❌ No | ❌ No | ✅ Yes |

**Key insight:** `target_lv` is refined from MULTIPLE sources (own + others), while `agent_lv` only learns from own actions.

---

## 2. Prediction

**Purpose:** Define how the agent estimates expected rewards using latent vectors.

**Flow context:** This is the PREDICTION step. Before acting, the agent predicts which target will yield the highest reward.

### The Goal

We want to predict the reward $r_{ij}$ that agent $i$ will receive when engaging target $j$.

In a traditional recommender system, we'd have a **rating matrix** $R$ where $R_{ij}$ is user $i$'s rating of item $j$. Here, our "rating" is the reward.

### Latent Factor Model

We approximate the reward using **latent vectors**:

$$\hat{r}_{ij} = f(\mathbf{u}_i \cdot \mathbf{v}_j)$$

Where:
- $\mathbf{u}_i \in \mathbb{R}^k$ is agent $i$'s latent vector (encodes "weapon characteristics")
- $\mathbf{v}_j \in \mathbb{R}^k$ is target $j$'s latent vector (encodes "vulnerability characteristics")
- $k$ is the latent dimension (in our scenario, $k = 3$)
- $f(\cdot)$ is a scaling function to bound predictions

### The Scaling Function

The dot product $\mathbf{u}_i \cdot \mathbf{v}_j$ can range from $-\|\mathbf{u}_i\| \|\mathbf{v}_j\|$ to $+\|\mathbf{u}_i\| \|\mathbf{v}_j\|$.

Since we normalize vectors to unit length ($\|\mathbf{u}\| = \|\mathbf{v}\| = 1$), the dot product ranges from $-1$ to $+1$.

To map this to the reward range $[0, 1]$:

$$\hat{r}_{ij} = \frac{1 + \mathbf{u}_i \cdot \mathbf{v}_j}{2}$$

**Interpretation:**
- Dot product = +1 (perfectly aligned) → Predicted reward = 1.0
- Dot product = 0 (orthogonal) → Predicted reward = 0.5
- Dot product = -1 (opposite) → Predicted reward = 0.0

---

## 3. SGD Update

**Purpose:** Apply Stochastic Gradient Descent to adjust latent vectors, reducing future prediction errors.

**Flow context:** This is the LEARNING step. After observing an actual reward, we compute the error and update vectors to improve future predictions.

### 3.1 Loss Function

We use **squared error loss** for a single observation:

$$L = \frac{1}{2}(r_{ij} - \hat{r}_{ij})^2$$

The $\frac{1}{2}$ is a convenience factor that simplifies the gradient.

**Expanded form** (substituting prediction formula):

$$L = \frac{1}{2}\left(r_{ij} - \frac{1 + \mathbf{u}_i \cdot \mathbf{v}_j}{2}\right)^2$$

**Error definition:**

$$e_{ij} = r_{ij} - \hat{r}_{ij}$$

Then: $L = \frac{1}{2} e_{ij}^2$

### 3.2 Gradient Derivation

We want to minimize $L$ by adjusting $\mathbf{u}_i$ and $\mathbf{v}_j$. The gradient tells us the direction and magnitude of adjustment.

#### Gradient with Respect to $\mathbf{u}_i$

Using the chain rule:

$$\frac{\partial L}{\partial \mathbf{u}_i} = \frac{\partial L}{\partial e_{ij}} \cdot \frac{\partial e_{ij}}{\partial \hat{r}_{ij}} \cdot \frac{\partial \hat{r}_{ij}}{\partial \mathbf{u}_i}$$

**Term 1:** $\frac{\partial L}{\partial e_{ij}} = e_{ij}$

**Term 2:** $\frac{\partial e_{ij}}{\partial \hat{r}_{ij}} = -1$

**Term 3:** $\frac{\partial \hat{r}_{ij}}{\partial \mathbf{u}_i} = \frac{\partial}{\partial \mathbf{u}_i}\left(\frac{1 + \mathbf{u}_i \cdot \mathbf{v}_j}{2}\right) = \frac{\mathbf{v}_j}{2}$

**Combined:**
$$\frac{\partial L}{\partial \mathbf{u}_i} = e_{ij} \cdot (-1) \cdot \frac{\mathbf{v}_j}{2} = -\frac{e_{ij} \cdot \mathbf{v}_j}{2}$$

#### Gradient with Respect to $\mathbf{v}_j$

Using the chain rule (same structure):

$$\frac{\partial L}{\partial \mathbf{v}_j} = \frac{\partial L}{\partial e_{ij}} \cdot \frac{\partial e_{ij}}{\partial \hat{r}_{ij}} \cdot \frac{\partial \hat{r}_{ij}}{\partial \mathbf{v}_j}$$

**Term 1:** $\frac{\partial L}{\partial e_{ij}} = e_{ij}$

**Term 2:** $\frac{\partial e_{ij}}{\partial \hat{r}_{ij}} = -1$

**Term 3:** $\frac{\partial \hat{r}_{ij}}{\partial \mathbf{v}_j} = \frac{\partial}{\partial \mathbf{v}_j}\left(\frac{1 + \mathbf{u}_i \cdot \mathbf{v}_j}{2}\right) = \frac{\mathbf{u}_i}{2}$

**Combined:**
$$\frac{\partial L}{\partial \mathbf{v}_j} = e_{ij} \cdot (-1) \cdot \frac{\mathbf{u}_i}{2} = -\frac{e_{ij} \cdot \mathbf{u}_i}{2}$$

### 3.3 Update Formulas + Normalization

Gradient descent updates parameters in the **opposite direction** of the gradient:

$$\mathbf{u}_i \leftarrow \mathbf{u}_i - \eta \cdot \frac{\partial L}{\partial \mathbf{u}_i} = \mathbf{u}_i + \frac{\eta}{2} \cdot e_{ij} \cdot \mathbf{v}_j$$

The implementation absorbs the $\frac{1}{2}$ into the learning rate, yielding:

$$\mathbf{u}_i \leftarrow \mathbf{u}_i + \eta \cdot e_{ij} \cdot \mathbf{v}_j$$

#### Final Update Rules (with Normalization)

$$\mathbf{u}_i^{\text{new}} = \text{normalize}\left(\mathbf{u}_i + \eta \cdot e_{ij} \cdot \mathbf{v}_j\right)$$

$$\mathbf{v}_j^{\text{new}} = \text{normalize}\left(\mathbf{v}_j + \eta \cdot e_{ij} \cdot \mathbf{u}_i\right)$$

#### Why Normalize?

$$\text{normalize}(\mathbf{x}) = \frac{\mathbf{x}}{\|\mathbf{x}\|}$$

1. **Bounded predictions**: Keeps $\hat{r}_{ij} \in [0, 1]$
2. **Numerical stability**: Prevents vectors from growing unboundedly
3. **Geometric interpretation**: Vectors live on the unit hypersphere; learning adjusts their **direction**, not magnitude

#### Geometric Intuition

| Error | Meaning | Vector Movement | Future Prediction |
|-------|---------|-----------------|-------------------|
| $e > 0$ | Underestimated | Vectors rotate **toward** each other | Higher |
| $e < 0$ | Overestimated | Vectors rotate **away** from each other | Lower |
| $e = 0$ | Perfect | No change | Same |

---

## 4. Learning from Others

**Purpose:** Extend the update rules to learn from observed actions of other agents.

**Flow context:** When agent $i$ observes agent $m$ fire at target $j$ and receive reward $r_{mj}$, agent $i$ updates its **mental models** of both agent $m$ and target $j$.

### Prediction for Other Agent

Agent $i$ predicts what reward agent $m$ should get:

$$\hat{r}_{mj} = \frac{1 + \mathbf{u}_m^{\text{est}} \cdot \mathbf{v}_j}{2}$$

Where $\mathbf{u}_m^{\text{est}}$ is agent $i$'s **estimate** of agent $m$'s latent vector.

### Error

$$e_{mj} = r_{mj} - \hat{r}_{mj}$$

### Update Rules

$$\mathbf{u}_m^{\text{est, new}} = \text{normalize}\left(\mathbf{u}_m^{\text{est}} + \eta \cdot e_{mj} \cdot \mathbf{v}_j\right)$$

$$\mathbf{v}_j^{\text{new}} = \text{normalize}\left(\mathbf{v}_j + \eta \cdot e_{mj} \cdot \mathbf{u}_m^{\text{est}}\right)$$

### Key Insight

The agent's own latent vector $\mathbf{u}_i$ is **NOT** updated when learning from others. Only `other_agents_lv[m]` and `target_lv[j]` change.

This means:
- `target_lv` gets refined from MULTIPLE sources (own actions + all observed actions)
- `other_agents_lv` is used ONLY for learning, never for action selection

### The Indirect Learning Chain

If agent $m$ has a similar weapon to agent $i$:
1. Agent $m$ gets similar rewards to agent $i$ on similar targets
2. Agent $i$'s $\mathbf{u}_m^{\text{est}}$ converges toward $\mathbf{u}_i$
3. When agent $m$ gets high reward on target $j$, agent $i$'s $\mathbf{v}_j$ aligns with $\mathbf{u}_m^{\text{est}}$
4. Since $\mathbf{u}_m^{\text{est}} \approx \mathbf{u}_i$, this means $\mathbf{v}_j$ aligns with agent $i$'s signature too
5. **Result:** Agent $i$ learns target $j$ is good for it, without ever firing at it!

---

## 5. Summary of Key Equations

| Quantity | Formula |
|----------|---------|
| Predicted reward | $\hat{r}_{ij} = \frac{1 + \mathbf{u}_i \cdot \mathbf{v}_j}{2}$ |
| Error | $e_{ij} = r_{ij} - \hat{r}_{ij}$ |
| Agent update | $\mathbf{u}_i \leftarrow \text{norm}(\mathbf{u}_i + \eta \cdot e_{ij} \cdot \mathbf{v}_j)$ |
| Target update | $\mathbf{v}_j \leftarrow \text{norm}(\mathbf{v}_j + \eta \cdot e_{ij} \cdot \mathbf{u}_i)$ |
| Other agent update | $\mathbf{u}_m^{\text{est}} \leftarrow \text{norm}(\mathbf{u}_m^{\text{est}} + \eta \cdot e_{mj} \cdot \mathbf{v}_j)$ |
| Target update (from other) | $\mathbf{v}_j \leftarrow \text{norm}(\mathbf{v}_j + \eta \cdot e_{mj} \cdot \mathbf{u}_m^{\text{est}})$ |

---

*Previous: [00_overview.md](00_overview.md) — The Big Picture*

*Next: [02_initial_state.md](02_initial_state.md) — drone_0's Initial State*
