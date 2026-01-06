# Part 2: The Math Foundation

This section derives the SGD update rules from first principles. We'll start with the general matrix factorization framework, then show exactly how the code implements it.

---

## 1. Matrix Factorization Setup

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

## 2. Loss Function

We use **squared error loss** for a single observation:

$$L = \frac{1}{2}(r_{ij} - \hat{r}_{ij})^2$$

The $\frac{1}{2}$ is a convenience factor that simplifies the gradient.

### Expanded Form

Substituting the prediction formula:

$$L = \frac{1}{2}\left(r_{ij} - \frac{1 + \mathbf{u}_i \cdot \mathbf{v}_j}{2}\right)^2$$

---

## 3. Gradient Derivation

We want to minimize $L$ by adjusting $\mathbf{u}_i$ and $\mathbf{v}_j$ using gradient descent.

### Step 3.1: Define the Error

Let:
$$e_{ij} = r_{ij} - \hat{r}_{ij} = r_{ij} - \frac{1 + \mathbf{u}_i \cdot \mathbf{v}_j}{2}$$

Then:
$$L = \frac{1}{2} e_{ij}^2$$

### Step 3.2: Gradient with Respect to $\mathbf{u}_i$

Using the chain rule:

$$\frac{\partial L}{\partial \mathbf{u}_i} = \frac{\partial L}{\partial e_{ij}} \cdot \frac{\partial e_{ij}}{\partial \hat{r}_{ij}} \cdot \frac{\partial \hat{r}_{ij}}{\partial \mathbf{u}_i}$$

**Term 1:** $\frac{\partial L}{\partial e_{ij}} = e_{ij}$

**Term 2:** $\frac{\partial e_{ij}}{\partial \hat{r}_{ij}} = -1$

**Term 3:** $\frac{\partial \hat{r}_{ij}}{\partial \mathbf{u}_i} = \frac{\partial}{\partial \mathbf{u}_i}\left(\frac{1 + \mathbf{u}_i \cdot \mathbf{v}_j}{2}\right) = \frac{\mathbf{v}_j}{2}$

**Combined:**
$$\frac{\partial L}{\partial \mathbf{u}_i} = e_{ij} \cdot (-1) \cdot \frac{\mathbf{v}_j}{2} = -\frac{e_{ij} \cdot \mathbf{v}_j}{2}$$

### Step 3.3: Gradient with Respect to $\mathbf{v}_j$

By symmetry:

$$\frac{\partial L}{\partial \mathbf{v}_j} = -\frac{e_{ij} \cdot \mathbf{u}_i}{2}$$

---

## 4. SGD Update Rules

Gradient descent updates parameters in the **opposite direction** of the gradient:

$$\mathbf{u}_i \leftarrow \mathbf{u}_i - \eta \cdot \frac{\partial L}{\partial \mathbf{u}_i}$$

Substituting:

$$\mathbf{u}_i \leftarrow \mathbf{u}_i - \eta \cdot \left(-\frac{e_{ij} \cdot \mathbf{v}_j}{2}\right) = \mathbf{u}_i + \frac{\eta}{2} \cdot e_{ij} \cdot \mathbf{v}_j$$

### Simplification in Code

The implementation absorbs the $\frac{1}{2}$ factor into the learning rate. Effectively:

$$\mathbf{u}_i \leftarrow \mathbf{u}_i + \eta' \cdot e_{ij} \cdot \mathbf{v}_j$$

Where $\eta' = \frac{\eta}{2}$. In the code, `learning_rate = 0.05` already accounts for this.

### Final Update Rules

**For agent's own action:**

$$\mathbf{u}_i^{\text{new}} = \text{normalize}\left(\mathbf{u}_i + \eta \cdot e_{ij} \cdot \mathbf{v}_j\right)$$

$$\mathbf{v}_j^{\text{new}} = \text{normalize}\left(\mathbf{v}_j + \eta \cdot e_{ij} \cdot \mathbf{u}_i\right)$$

**For other agent's action** (agent $i$ observing agent $m$'s reward):

$$\mathbf{u}_m^{\text{est, new}} = \text{normalize}\left(\mathbf{u}_m^{\text{est}} + \eta \cdot e_{mj} \cdot \mathbf{v}_j\right)$$

$$\mathbf{v}_j^{\text{new}} = \text{normalize}\left(\mathbf{v}_j + \eta \cdot e_{mj} \cdot \mathbf{u}_m^{\text{est}}\right)$$

Where $\mathbf{u}_m^{\text{est}}$ is agent $i$'s **estimate** of agent $m$'s latent vector.

---

## 5. Normalization

After each update, vectors are normalized to unit length:

$$\text{normalize}(\mathbf{x}) = \frac{\mathbf{x}}{\|\mathbf{x}\|}$$

### Why Normalize?

1. **Bounded predictions**: Keeps $\hat{r}_{ij} \in [0, 1]$
2. **Numerical stability**: Prevents vectors from growing unboundedly
3. **Geometric interpretation**: Vectors live on the unit hypersphere; learning adjusts their **direction**, not magnitude

---

## 6. Code Mapping

Here's how the math maps to the actual implementation:

### Prediction (`BaseCFPolicy.predict_reward`)

```python
def predict_reward(self, target_idx: int) -> float:
    dot = np.dot(self.agent_lv, self.target_lv[target_idx])
    return (1 + dot) / 2  # Maps [-1, 1] to [0, 1]
```

**Math:** $\hat{r}_{ij} = \frac{1 + \mathbf{u}_i \cdot \mathbf{v}_j}{2}$

### Update for Own Action (`BaseCFPolicy.update`)

```python
def update(self, target_idx: int, observed_reward: float) -> None:
    if target_idx < 0 or observed_reward < 0:
        return  # Skip NoOp or wasted shots
    
    predicted = self.predict_reward(target_idx)
    error = observed_reward - predicted  # e_ij
    
    # Store copies before update (simultaneous update)
    agent_vec = self.agent_lv.copy()
    target_vec = self.target_lv[target_idx].copy()
    
    # SGD updates
    self.agent_lv += self.learning_rate * error * target_vec
    self.target_lv[target_idx] += self.learning_rate * error * agent_vec
    
    # Normalize to unit sphere
    self.agent_lv = normalize(self.agent_lv)
    self.target_lv[target_idx] = normalize(self.target_lv[target_idx])
```

**Math:**
- $e_{ij} = r_{ij} - \hat{r}_{ij}$
- $\mathbf{u}_i \leftarrow \text{normalize}(\mathbf{u}_i + \eta \cdot e_{ij} \cdot \mathbf{v}_j)$
- $\mathbf{v}_j \leftarrow \text{normalize}(\mathbf{v}_j + \eta \cdot e_{ij} \cdot \mathbf{u}_i)$

### Update from Other's Action (`BaseCFPolicy._update_from_other`)

```python
def _update_from_other(self, other_agent_idx: int, target_idx: int, observed_reward: float) -> None:
    if target_idx < 0 or observed_reward < 0:
        return
    
    predicted = self._predict_reward_for_other(other_agent_idx, target_idx)
    error = observed_reward - predicted
    
    other_agent_vec = self.other_agents_lv[other_agent_idx].copy()
    target_vec = self.target_lv[target_idx].copy()
    
    self.other_agents_lv[other_agent_idx] += self.learning_rate * error * target_vec
    self.target_lv[target_idx] += self.learning_rate * error * other_agent_vec
    
    self.other_agents_lv[other_agent_idx] = normalize(self.other_agents_lv[other_agent_idx])
    self.target_lv[target_idx] = normalize(self.target_lv[target_idx])
```

**Key difference:** Updates `other_agents_lv[m]` (this agent's estimate of agent $m$) instead of `agent_lv`.

---

## 7. Intuition: What Do the Updates Do?

### Positive Error ($r > \hat{r}$): "Underestimated"

The actual reward was **higher** than predicted. We should:
- Move $\mathbf{u}_i$ **toward** $\mathbf{v}_j$ (increase alignment)
- Move $\mathbf{v}_j$ **toward** $\mathbf{u}_i$

This increases the dot product, raising future predictions for this pair.

### Negative Error ($r < \hat{r}$): "Overestimated"

The actual reward was **lower** than predicted. We should:
- Move $\mathbf{u}_i$ **away from** $\mathbf{v}_j$ (decrease alignment)
- Move $\mathbf{v}_j$ **away from** $\mathbf{u}_i$

This decreases the dot product, lowering future predictions for this pair.

### Geometric View

On the unit hypersphere:
- **High reward** → Vectors rotate toward each other
- **Low reward** → Vectors rotate away from each other

Over time, agent vectors cluster with targets they're effective against.

---

## 8. Summary of Key Equations

| Quantity | Formula |
|----------|---------|
| Predicted reward | $\hat{r}_{ij} = \frac{1 + \mathbf{u}_i \cdot \mathbf{v}_j}{2}$ |
| Error | $e_{ij} = r_{ij} - \hat{r}_{ij}$ |
| Agent update | $\mathbf{u}_i \leftarrow \text{norm}(\mathbf{u}_i + \eta \cdot e_{ij} \cdot \mathbf{v}_j)$ |
| Target update | $\mathbf{v}_j \leftarrow \text{norm}(\mathbf{v}_j + \eta \cdot e_{ij} \cdot \mathbf{u}_i)$ |
| Other agent update | $\mathbf{u}_m^{\text{est}} \leftarrow \text{norm}(\mathbf{u}_m^{\text{est}} + \eta \cdot e_{mj} \cdot \mathbf{v}_j)$ |

---

*Previous: [00_overview.md](00_overview.md) — The Big Picture*

*Next: [02_initial_state.md](02_initial_state.md) — drone_0's Initial State*
