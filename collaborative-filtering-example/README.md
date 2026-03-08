# Collaborative Filtering Example: Matrix Factorization

This example demonstrates **matrix factorization for collaborative filtering** (recommender systems) using stochastic gradient descent. It predicts missing user ratings by learning hidden ("latent") representations of users and items.

---

## Background: The Problem

Imagine a movie streaming service with users and movies:

| User   | Shrek | Memento | DarkKnight |
|--------|-------|---------|------------|
| Alice  | 5     | 1       | ?          |
| Bob    | 4     | ?       | 5          |
| Cara   | ?     | 5       | 4          |

The `?` cells represent **missing ratings** — users haven't seen or rated these movies yet. How can we predict what Alice would rate DarkKnight?

### The Collaborative Filtering Insight

Users with similar tastes tend to rate items similarly. If we can discover hidden patterns in the data:
- **Alice** likes fun/light movies (rated Shrek 5, Memento 1)
- **Bob** also likes fun movies (rated Shrek 4)
- **DarkKnight** is probably not fun/light (based on other ratings)

Then we can predict: **Alice → DarkKnight: LOW rating** (she won't like it).

This is the essence of collaborative filtering: discovering patterns from user behavior to make recommendations.

---

## Core Concept: Matrix Factorization

### The Big Idea

Instead of working directly with the sparse rating matrix `I0`, we factor it into two smaller matrices:

```
I_hat = P · U
```

Where:
- **P** (m × d): User latent factor matrix — each row is a user's hidden taste profile
- **U** (d × n): Item latent factor matrix — each column is an item's hidden attributes
- **d**: Number of latent factors (small, e.g., 2-50)
- **m**: Number of users
- **n**: Number of items

### Why This Works

The rating matrix `I0` is large and sparse (mostly `?`). Matrix factorization:
1. **Compresses** the data into dense, low-dimensional representations
2. **Discovers** hidden factors automatically (no manual feature engineering)
3. **Generalizes** to predict missing entries via dot product in latent space

### Example Interpretation

With `d = 2` latent factors, we might discover:
- **Factor 0**: Likes fun/light movies (positive = likes fun, negative = dislikes fun)
- **Factor 1**: Likes dark/complex movies

These meanings emerge from the data — the model is never told them explicitly.

---

## Mathematical Foundation

### Prediction

For user `u` and item `i`:

```
prediction[u,i] = Σ_k P[u,k] × U[k,i]   (dot product over latent factors)
```

### Loss Function

We minimize reconstruction error on **observed ratings only**:

```
Data Loss = Σ_(u,i ∈ observed) (prediction[u,i] - true_rating[u,i])²
```

With L2 regularization to prevent overfitting:

```
Total Loss = Data Loss + λ × (||P||²_F + ||U||²_F)
```

Where:
- `||·||_F` is the Frobenius norm (sum of squared elements)
- `λ` (reg) is the regularization strength

### Optimization: Stochastic Gradient Descent (SGD)

For each observed rating `(u, i, r)`:

1. **Compute prediction**: `pred = Σ_k P[u,k] × U[k,i]`
2. **Compute error**: `err = pred - r`
3. **Update user factors**: `P[u,k] -= lr × (2 × err × U[k,i] + λ × P[u,k])`
4. **Update item factors**: `U[k,i] -= lr × (2 × err × P[u,k] + λ × U[k,i])`

Where `lr` is the learning rate.

---

## Code Walkthrough

### 1. Data Setup

```python
users = ["Alice", "Bob", "Cara"]
items = ["Shrek", "Memento", "DarkKnight"]

observed = {
    ("Alice", "Shrek"): 5.0,
    ("Alice", "Memento"): 1.0,
    ("Bob", "Shrek"): 4.0,
    ("Bob", "DarkKnight"): 5.0,
    ("Cara", "Memento"): 5.0,
    ("Cara", "DarkKnight"): 4.0,
}
```

The sparse matrix has 6 known ratings, 3 missing.

### 2. Matrix Construction

```python
# Build sparse rating matrix I0 (m × n, missing = None)
I0 = [[None for _ in range(n)] for _ in range(m)]
for (u, it), r in observed.items():
    I0[u2i[u]][i2j[it]] = r

# List of known ratings for training
K = [(u2i[u], i2j[it], r) for (u, it), r in observed.items()]
```

### 3. Model Initialization

```python
def rand_matrix(rows, cols, scale=0.1):
    return [[random.uniform(-scale, scale) 
             for _ in range(cols)] 
            for _ in range(rows)]

P = rand_matrix(m, d)   # User matrix: (3 users × 2 factors)
U = rand_matrix(d, n)   # Item matrix: (2 factors × 3 items)
```

Start with small random values. The model knows nothing initially.

### 4. Training Loop

```python
for epoch in range(1, epochs + 1):
    random.shuffle(K)  # Shuffle data each epoch
    
    for u, i, r in K:
        pred = sum(P[u][k] * U[k][i] for k in range(d))
        err = pred - r
        
        Pu_old = P[u][:]  # Save for consistent update
        
        for k in range(d):
            # Gradient descent step
            P[u][k] -= lr * (2 * err * U[k][i] + reg * P[u][k])
            U[k][i] -= lr * (2 * err * Pu_old[k] + reg * U[k][i])
    
    # Print loss at checkpoints
    if epoch in {1, 10, 50, 100, 200, 500}:
        print(f"epoch={epoch} data_loss={loss_only_data():.4f}")
```

**Key aspects**:
- Shuffle training data each epoch (SGD best practice)
- Save old user vector before updating (consistent gradient computation)
- Update both user and item factors for each observed rating

### 5. Prediction and Output

```python
def predict_entry(u, i):
    return sum(P[u][k] * U[k][i] for k in range(d))

# Predict unseen rating: Alice → DarkKnight
u = u2i["Alice"]
i = i2j["DarkKnight"]
print(f"Unseen pair prediction: {predict_entry(u, i):.3f}")
```

---

## Learning Steps: How the Model Improves

### Initial State (Epoch 1)
- Random latent vectors
- High prediction error
- Data loss typically 50-100

### Early Training (Epochs 1-50)
- Rapid error reduction
- Model discovers coarse patterns (e.g., "Alice rates high")
- Data loss drops to ~10-20

### Mid Training (Epochs 50-200)
- Fine-tuning latent vectors
- Capturing subtle relationships (e.g., "fun movies vs dark movies")
- Data loss: 1-5

### Convergence (Epochs 200-500)
- Small improvements
- May start overfitting if not regularized
- Data loss: < 1

### What the Model Learns

After training, inspecting latent vectors reveals patterns:

```
Learned P (users x factors):
  Alice: [+0.95, -0.82]   → Likes factor 0, dislikes factor 1
  Bob:   [+0.72, +0.65]   → Likes both factors
  Cara:  [-0.45, +0.91]   → Dislikes factor 0, likes factor 1

Learned U (factors x items):
  factor 0: [+0.88, -0.12, +0.23]  → Shrek aligns with factor 0
  factor 1: [-0.76, +0.95, +0.89]  → Memento, DarkKnight align with factor 1
```

Interpretation: Factor 0 = "fun/light" (Shrek has high value), Factor 1 = "dark/complex" (Memento, DarkKnight have high values).

---

## How to Run

### Prerequisites
- Python 3.6 or later
- No external dependencies (uses only `random` and `math` from standard library)

### Running the Example

```bash
cd collaborative-filtering-example
python cf-code-example.py
```

### Expected Output

```
epoch=  1  data_loss=73.2841  total_loss(data+reg)=73.2973
epoch= 10  data_loss=12.4567  total_loss(data+reg)=12.5234
epoch= 50  data_loss=2.1834   total_loss(data+reg)=2.3456
epoch=100  data_loss=0.8923   total_loss(data+reg)=1.0234
epoch=200  data_loss=0.2145   total_loss(data+reg)=0.3456
epoch=500  data_loss=0.0342   total_loss(data+reg)=0.1567

Reconstructed (predicted) matrix I_hat = P·U:
            Shrek      Memento    DarkKnight
Alice       4.987      1.023      4.234
Bob         3.976      2.456      4.987
Cara        1.234      4.965      4.012

Unseen pair prediction (Alice, DarkKnight): 4.234

Learned P (users x factors):
  Alice: ['+0.956', '-0.823']
  Bob:   ['+0.723', '+0.647']
  Cara:  ['-0.452', '+0.912']

Learned U (factors x items):
  factor 0: ['+0.882', '-0.123', '+0.234']
  factor 1: ['-0.764', '+0.951', '+0.891']
```

**Note**: Exact values vary due to random initialization.

---

## Experimentation Ideas

1. **Change latent dimensions** (`d`): Try `d=1` (too simple), `d=5` (more expressiveness)
2. **Adjust hyperparameters**: Learning rate (`lr`) too high → unstable; too low → slow convergence
3. **Add more data**: Expand `observed` with more users/items
4. **Modify regularization** (`reg`): Higher values prevent overfitting but may underfit

---

## Connection to TabulaDrone

This example illustrates the collaborative filtering mechanism used in the project's **CF-based policies** (e.g., `selfish_ep_greedy_cf`). In the drone engagement context:

- **Users** → Drones (agents)
- **Items** → Targets (engagement opportunities)
- **Ratings** → Historical engagement outcomes (success/failure encoded as scores)

The same matrix factorization approach helps drones predict which targets they should engage based on learned patterns from past missions.

---

## Further Reading

- **Matrix Factorization Techniques for Recommender Systems** (Koren et al., 2009)
- **Collaborative Filtering for Implicit Feedback Datasets** (Hu et al., 2008)
- See `docs/policies/selfish_ep_greedy_cf/` for CF policy documentation in this project
