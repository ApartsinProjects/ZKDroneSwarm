# Collaborative Filtering Example: Matrix Factorization

This example demonstrates **matrix factorization for collaborative filtering** (recommender systems) using stochastic gradient descent. It predicts missing user ratings by learning hidden (**latent**) representations of users and items.

A **latent vector** is a compact numeric representation that captures hidden preferences or hidden properties that are not written explicitly in the data.  
In this setting:

- each **user latent vector** represents that user's hidden taste profile
- each **item latent vector** represents that item's hidden characteristics
- these vectors are called **latent** because the model learns them from rating patterns rather than receiving them as labeled features

This is a **larger example** with train/test evaluation:

- 10 users, 12 movies
- 110 training ratings, 10 hidden test ratings
- Evaluation with MAE and RMSE metrics

---

## Background: The Problem

Imagine a movie streaming service with users and movies:

| User   | Shrek | ToyStory | FindingNemo | Memento | Se7en | DarkKnight | ... |
|--------|-------|----------|-------------|---------|-------|------------|-----|
| Alice  | 5     | **?**    | 5           | 1       | 1     | 2          | ... |
| Bob    | 5     | 4        | 5           | 1       | 2     | 2          | ... |
| ...    | ...   | ...      | ...         | ...     | ...   | ...        | ... |

The **?** cells represent **missing ratings** — users have not seen or rated these movies yet. The goal is to predict those missing values. For example: how can we estimate what Alice would rate **ToyStory**?

### The Collaborative Filtering Insight

Users with similar tastes tend to rate items similarly. In this toy dataset, there are clear preference patterns:

- **Alice, Bob**: family / animation lovers
- **Cara, Dan**: crime / thriller lovers
- **Eve, Frank**: sci-fi / action lovers
- **Grace, Hank, Ivy, Jake**: mixed profiles

By learning hidden structure from the ratings that are already known, the model can infer ratings that are still missing.

---

## Core Concept: Matrix Factorization

Matrix factorization approximates a sparse user-item rating matrix using two smaller dense matrices:

$$
\hat{I} = P U
$$

Where:

- $I_0 \in \mathbb{R}^{m \times n}$ is the original partially observed rating matrix
- $\hat{I} \in \mathbb{R}^{m \times n}$ is the model's reconstructed rating matrix
- $P \in \mathbb{R}^{m \times d}$ is the **user latent factor matrix**
- $U \in \mathbb{R}^{d \times n}$ is the **item latent factor matrix**
- $m$ = number of users
- $n$ = number of items
- $d$ = number of latent factors

---

### Latent Vectors

The key idea is that each user and each item is represented by a **latent vector**.

A latent vector is a learned numeric representation of hidden structure in the data:

- a **user latent vector** captures hidden preferences or taste
- an **item latent vector** captures hidden characteristics or attributes

For user $u$, the latent vector is the $u$-th row of $P$:

$$
P_u \in \mathbb{R}^d
$$

For item $i$, the latent vector is the $i$-th column of $U$:

$$
U_i \in \mathbb{R}^d
$$

These vectors are called **latent** because their meaning is not manually defined.  
The model learns them automatically from rating patterns.

---

### How a Prediction Is Made

The predicted rating for user $u$ on item $i$ is the dot product of their latent vectors:

$$
\hat{r}_{ui} = P_u^\top U_i
$$

Expanded form:

$$
\hat{r}_{ui} = \sum_{k=1}^{d} P_{u,k} U_{k,i}
$$

This means the prediction is built from the interaction of the user's hidden preferences and the item's hidden properties across all latent dimensions.

If a user's latent vector aligns well with an item's latent vector, the predicted rating is high.  
If the alignment is weak, the predicted rating is low.

---

### Observed Ratings Only

In collaborative filtering, many entries in $I_0$ are missing.  
We do **not** treat missing values as zeros.

Instead, we define the set of observed ratings as:

$$
\Omega = \{(u,i) \mid r_{ui}\text{ is observed}\}
$$

Only those entries contribute to training.

---

### Data Loss

The model learns $P$ and $U$ by minimizing prediction error on the observed entries.

The data loss is:

$$
\mathcal{L}_{\text{data}}(P,U)=\sum_{(u,i)\in\Omega}\left(P_u^\top U_i-r_{ui}\right)^2
$$

This is the sum of squared differences between:

- the predicted rating $P_u^\top U_i$
- the true observed rating $r_{ui}$

So the model tries to make predicted ratings as close as possible to the known ratings.

---

### Regularized Total Loss

To avoid overfitting, we add L2 regularization on the latent factors.

The full loss becomes:

$$
\mathcal{L}(P,U)=\sum_{(u,i)\in\Omega}\left(P_u^\top U_i-r_{ui}\right)^2+\lambda\left(\|P\|_F^2+\|U\|_F^2\right)
$$

Where:

- $\lambda$ is the regularization strength
- $\|P\|_F^2$ is the sum of squares of all entries in $P$
- $\|U\|_F^2$ is the sum of squares of all entries in $U$

This loss balances two goals:

- fit the observed ratings well
- keep the latent vectors from becoming unnecessarily large

---

### The Optimization Objective

Training means finding the matrices $P$ and $U$ that minimize the total loss:

$$
(P^{*},U^{*})=\arg\min_{P,U}\mathcal{L}(P,U)
$$

This is the central optimization problem in matrix factorization.

---

### Connection to the Matrix Form

At a high level, matrix factorization tries to make:

$$
P U \approx I_0
$$

But because $I_0$ is only partially observed, the precise data loss is:

$$
\mathcal{L}_{\text{data}}(P,U)=\sum_{(u,i)\in\Omega}\left((PU)_{ui}-(I_0)_{ui}\right)^2
$$

This is the sparse version of reconstruction error.

To write this cleanly in matrix form, we use a binary mask matrix $M$, where:

- $M_{ui}=1$ if rating $(u,i)$ is observed
- $M_{ui}=0$ otherwise

Then the full objective can be written as:

$$
\mathcal{L}(P,U)=\left\|M\odot(PU-I_0)\right\|_F^2+\lambda\left(\|P\|_F^2+\|U\|_F^2\right)
$$

Where:

- $\odot$ is elementwise multiplication
- the mask ensures that only observed entries contribute to the loss

---

### Why This Is the Core Idea

The full process can be understood through four connected ideas:

1. Represent users and items with latent vectors
2. Predict ratings with a dot product
3. Measure prediction error only on observed entries
4. Minimize that error, with regularization, to learn the latent factors

That is the essence of collaborative filtering with matrix factorization.

---

## Mathematical Foundation

### What the Model Is Trying to Minimize

We start from a partially observed rating matrix:

$$
I_0 \in \mathbb{R}^{m \times n}
$$

where:

- $m$ = number of users
- $n$ = number of items
- $I_0(u,i)$ = the true rating of user $u$ for item $i$, **only if that rating is observed**

Because many entries are missing, we do **not** try to fit the entire matrix directly.  
Instead, we learn two low-rank matrices:

$$
P \in \mathbb{R}^{m \times d}, \qquad U \in \mathbb{R}^{d \times n}
$$

so that their product approximates the observed ratings:

$$
\hat{I} = P U
$$

Here:

- row $P_u \in \mathbb{R}^d$ is the latent vector of user $u$
- column $U_i \in \mathbb{R}^d$ is the latent vector of item $i$
- $d$ is the latent dimension

The predicted rating for user $u$ and item $i$ is:

$$
\hat{r}_{ui} = \hat{I}_{ui} = P_u^\top U_i = \sum_{k=1}^{d} P_{u,k} U_{k,i}
$$

So the learning problem is:

> Find $P$ and $U$ such that the predicted ratings $\hat{r}_{ui}$ are as close as possible to the known ratings $r_{ui}$, for the entries we actually observed.

---

### Observed Entries Only

Let:

$$
\Omega = \{(u,i) \mid \text{rating } r_{ui} \text{ is known}\}
$$

This set $\Omega$ contains only the user-item pairs that appear in the training data.

That means the model does **not** minimize error over missing entries.  
It minimizes error only on the observed entries:

$$
(u,i) \in \Omega
$$

This is important because a missing value does **not** mean rating $0$; it means **unknown**.

---

### Reconstruction Objective

The matrix factorization model tries to reconstruct the known part of $I_0$ with the low-rank product $PU$.

If we write this idea informally, the goal is to make:

$$
PU \approx I_0
$$

But since $I_0$ is only partially observed, the precise objective is:

$$
\hat{r}_{ui} \approx r_{ui}
\qquad \text{for all } (u,i)\in\Omega
$$

So the quantity being minimized is the total squared discrepancy between prediction and truth on known entries.

---

### Data Loss

For one observed rating $(u,i)$, define the prediction error:

$$
e_{ui} = \hat{r}_{ui} - r_{ui}
$$

The squared error for that one example is:

$$
e_{ui}^2 = (\hat{r}_{ui} - r_{ui})^2
$$

Summing over all observed ratings gives the **data loss**:

$$
\mathcal{L}_{\text{data}}(P,U)
=
\sum_{(u,i)\in\Omega}
(\hat{r}_{ui} - r_{ui})^2
$$

Substituting $\hat{r}_{ui} = P_u^\top U_i$:

$$
\mathcal{L}_{\text{data}}(P,U)
=
\sum_{(u,i)\in\Omega}
\left(P_u^\top U_i - r_{ui}\right)^2
$$

or equivalently:

$$
\mathcal{L}_{\text{data}}(P,U)
=
\sum_{(u,i)\in\Omega}
\left(
\sum_{k=1}^{d} P_{u,k}U_{k,i} - r_{ui}
\right)^2
$$

This is the core quantity the model is trying to reduce.

---

### Why Squared Error?

The squared loss is used because it has several useful properties:

1. **Large mistakes are penalized more strongly**  
   If the error doubles, the penalty becomes four times larger.

2. **It is smooth and differentiable**  
   This makes gradient-based optimization straightforward.

3. **It measures reconstruction quality**  
   A lower value means the factorized model $PU$ better matches the observed ratings.

For example, if the true rating is $5$:

- prediction $4.5$ gives squared error:

$$
(4.5 - 5)^2 = 0.25
$$

- prediction $3$ gives squared error:

$$
(3 - 5)^2 = 4
$$

So the second mistake is penalized much more heavily.

---

### Regularization Term

If we only minimize reconstruction error, the model may overfit the training data by making the latent factors too large or too specialized.

To control this, we add **L2 regularization**:

$$
\mathcal{L}_{\text{reg}}(P,U)
=
\lambda
\left(
\|P\|_F^2 + \|U\|_F^2
\right)
$$

where:

$$
\|P\|_F^2 = \sum_{u=1}^{m}\sum_{k=1}^{d} P_{u,k}^2
\qquad\text{and}\qquad
\|U\|_F^2 = \sum_{k=1}^{d}\sum_{i=1}^{n} U_{k,i}^2
$$

The Frobenius norm here is just the sum of squares of all entries in the matrix.

Expanded fully, the regularization term is:

$$
\mathcal{L}_{\text{reg}}(P,U)
=
\lambda
\left(
\sum_{u=1}^{m}\sum_{k=1}^{d} P_{u,k}^2
+
\sum_{k=1}^{d}\sum_{i=1}^{n} U_{k,i}^2
\right)
$$

#### Intuition

This term discourages the model from using excessively large latent values.

Without regularization, the model might fit the training ratings very closely but generalize poorly.  
With regularization, the model is encouraged to find a simpler, more stable factorization.

---

### Total Loss Function

The full objective is the sum of:

- the reconstruction error on observed ratings
- the regularization penalty on the latent factors

So the total loss is:

$$
\mathcal{L}(P,U)
=
\sum_{(u,i)\in\Omega}
\left(P_u^\top U_i - r_{ui}\right)^2
+
\lambda
\left(
\|P\|_F^2 + \|U\|_F^2
\right)
$$

This is the exact function the training process minimizes.

Equivalently, in fully expanded form:

$$
\mathcal{L}(P,U)
=
\sum_{(u,i)\in\Omega}
\left(
\sum_{k=1}^{d} P_{u,k}U_{k,i} - r_{ui}
\right)^2
+
\lambda
\left(
\sum_{u=1}^{m}\sum_{k=1}^{d} P_{u,k}^2
+
\sum_{k=1}^{d}\sum_{i=1}^{n} U_{k,i}^2
\right)
$$

---

### The Optimization Problem

Training means solving:

$$
(P^{*}, U^{*}) = \arg\min_{P,U} \mathcal{L}(P,U)
$$

That is:

$$
(P^{*}, U^{*})
=
\arg\min_{P,U}
\left[
\sum_{(u,i)\in\Omega}
\left(P_u^\top U_i - r_{ui}\right)^2
+
\lambda
\left(
\|P\|_F^2 + \|U\|_F^2
\right)
\right]
$$

This expression answers the key question directly:

> We are minimizing the squared prediction error on observed ratings, plus a penalty that keeps the latent vectors from growing too large.

---

### Connection to the Matrix Form $ (PU - I_0)^2 $

A common shorthand is to write something like:

$$
(PU - I_0)^2
$$

or

$$
\|PU - I_0\|^2
$$

This captures the intuition of matrix reconstruction, but it is incomplete for sparse collaborative filtering, because $I_0$ has missing entries.

The precise sparse version is:

$$
\mathcal{L}_{\text{data}}(P,U)
=
\sum_{(u,i)\in\Omega}
\left((PU)_{ui} - (I_0)_{ui}\right)^2
$$

Sometimes this is written using a masking operator:

$$
\mathcal{L}_{\text{data}}(P,U)
=
\left\|
M \odot (PU - I_0)
\right\|_F^2
$$

where:

- $M$ is a binary mask matrix
- $M_{ui} = 1$ if $(u,i)\in\Omega$, else $0$
- $\odot$ is elementwise multiplication

This version makes the sparsity explicit: only observed entries contribute to the loss.

Then the full objective becomes:

$$
\mathcal{L}(P,U)
=
\left\|
M \odot (PU - I_0)
\right\|_F^2
+
\lambda
\left(
\|P\|_F^2 + \|U\|_F^2
\right)
$$

This is often the cleanest matrix-level way to write the optimization problem.

---

### Per-Example View Used in SGD

In stochastic gradient descent, training proceeds one observed rating at a time.

For a single sample $(u,i,r_{ui})$, define:

$$
\hat{r}_{ui} = P_u^\top U_i
$$

$$
e_{ui} = \hat{r}_{ui} - r_{ui}
$$

The per-example contribution to the loss is:

$$
\ell_{ui}
=
(\hat{r}_{ui} - r_{ui})^2
+
\lambda
\left(
\|P_u\|^2 + \|U_i\|^2
\right)
$$

In practice, implementations often apply the regularization during the update step rather than computing the full loss from scratch every time, but mathematically it comes from the same global objective above.

---

### What Minimization Means in Practice

As optimization progresses:

- if predictions are poor, the squared errors are large
- gradient descent adjusts $P$ and $U$
- predictions move closer to the known ratings
- the total loss decreases

So when training is working correctly, the model is gradually finding latent user vectors and item vectors such that:

$$
P_u^\top U_i \approx r_{ui}
\quad \text{for observed pairs}
$$

while still keeping $P$ and $U$ reasonably small through regularization.

That balance between **fit** and **simplicity** is exactly what the loss function enforces.

---

## Code Walkthrough

### 1. Data Setup (Full Ground Truth + Train/Test Split)

```python
users = ["Alice", "Bob", "Cara", "Dan", "Eve", "Frank", 
         "Grace", "Hank", "Ivy", "Jake"]

items = ["Shrek", "ToyStory", "FindingNemo", "Memento", 
         "Se7en", "DarkKnight", "Inception", "Matrix",
         "Interstellar", "Avengers", "Up", "PulpFiction"]

# Full ground-truth ratings (1-5 scale)
full_ratings = {
    "Alice": {"Shrek": 5, "ToyStory": 5, "FindingNemo": 5, ...},
    "Bob": {"Shrek": 5, "ToyStory": 4, "FindingNemo": 5, ...},
    # ... 10 users total
}

# Intentionally hide 10 entries for testing
hidden_pairs = [
    ("Alice", "ToyStory"),    # true rating = 5
    ("Bob", "Up"),            # true rating = 5
    ("Cara", "PulpFiction"),  # true rating = 5
    # ... 10 pairs total
]

# Build observed = all ratings EXCEPT hidden
observed = {}
for u in users:
    for it in items:
        if (u, it) not in hidden_truth:
            observed[(u, it)] = float(full_ratings[u][it])
```

This creates a proper ML train/test split: train on 110 ratings, test on 10 hidden ratings.

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

P = rand_matrix(m, d)   # User matrix: (10 users × 7 factors)
U = rand_matrix(d, n)   # Item matrix: (7 factors × 12 items)
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
    if epoch in {1, 10, 50, 100, 500, 1000, 2000, 5000}:
        print(f"epoch={epoch} data_loss={loss_only_data():.4f}")
```

**Key aspects**:
- Shuffle training data each epoch (SGD best practice)
- Save old user vector before updating (consistent gradient computation)
- Update both user and item factors for each observed rating

### 5. Hyperparameters

```python
d = 7           # latent factors (more expressive for larger data)
lr = 0.01       # learning rate (conservative for stability)
reg = 0.02      # regularization strength
epochs = 5000   # training iterations
```

### 6. Evaluation on Hidden Test Entries

```python
def evaluate_hidden():
    print("\nHidden-entry evaluation:")
    print(f"{'User':12s} {'Movie':15s} {'True':>8s} {'Pred':>10s} {'AbsErr':>10s}")

    abs_errors = []
    sq_errors = []

    for (u, it), true_r in hidden_truth.items():
        pred = predict_entry(u2i[u], i2j[it])
        abs_err = abs(pred - true_r)
        sq_err = (pred - true_r) ** 2

        abs_errors.append(abs_err)
        sq_errors.append(sq_err)

        print(f"{u:12s} {it:15s} {true_r:8.0f} {pred:10.3f} {abs_err:10.3f}")

    mae = sum(abs_errors) / len(abs_errors)
    rmse = (sum(sq_errors) / len(sq_errors)) ** 0.5

    print(f"\nMAE on hidden entries : {mae:.4f}")
    print(f"RMSE on hidden entries: {rmse:.4f}")
```

This is the key addition: **proper ML evaluation** comparing predictions against held-out true ratings.

---

## Learning Steps: How the Model Improves

### Initial State (Epoch 1)
- Random latent vectors
- High prediction error
- Data loss typically 1000-1200

### Early Training (Epochs 1-100)
- Rapid error reduction
- Model discovers coarse patterns (e.g., "Alice rates family movies high")
- Data loss drops to ~3-7

### Mid Training (Epochs 100-1000)
- Fine-tuning latent vectors
- Capturing subtle relationships between taste groups
- Data loss: 0.4-2

### Convergence (Epochs 1000-5000)
- Small improvements
- May start overfitting if not regularized
- Data loss: ~0.3-0.4

### What the Model Learns

After training, inspecting latent vectors reveals taste patterns:

```
Learned P (users x factors):
  Alice       : ['+0.13', '+1.25', '-0.23', ...]  → family lover
  Bob         : ['+0.03', '+1.38', '-0.25', ...]  → family lover
  Cara        : ['+0.41', '+0.99', '-1.21', ...]  → crime/thriller lover
  Dan         : ['-0.74', '+1.37', '-1.04', ...] → crime/thriller lover
  Eve         : ['+0.27', '+1.47', '-0.96', ...] → sci-fi lover
  ...

Learned item vectors from U (items as columns):
  Shrek       : ['+0.72', '+1.57', '-0.20', ...]  → aligns with family lovers
  ToyStory    : ['+0.69', '+1.08', '-0.20', ...]  → aligns with family lovers
  Memento     : ['-0.28', '+1.09', '-1.48', ...] → aligns with crime lovers
  Matrix      : ['-0.56', '+1.48', '-1.22', ...] → aligns with sci-fi lovers
  ...
```

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

---

## Experimentation Ideas

1. **Change latent dimensions** (`d`): Try `d=3` (may underfit), `d=10` (may overfit)
2. **Adjust learning rate** (`lr`): Higher → faster convergence but risk instability; lower → slower but safer
3. **Modify regularization** (`reg`): Higher values prevent overfitting but may underfit
4. **Change train/test split**: Hide different entries, see how robust predictions are
5. **Add noise to ratings**: Simulate real-world rating variance

---

## Actual Running Output

See [OUTPUT_EXAMPLE.md](OUTPUT_EXAMPLE.md) for a complete example of actual terminal output with detailed explanations of:
- Training progress and convergence
- Observed matrix (training vs. hidden data)
- Predicted matrix for all entries
- Evaluation metrics (MAE, RMSE) on hidden entries
- Learned latent vectors and their patterns

---

## Connection to TabulaDrone

This example illustrates the collaborative filtering mechanism used in the project's **CF-based policies** (e.g., `selfish_ep_greedy_cf`). In the drone engagement context:

- **Users** → Drones (agents)
- **Items** → Targets (engagement opportunities)
- **Ratings** → Historical engagement outcomes (success/failure encoded as scores)

The same matrix factorization approach helps drones predict which targets they should engage based on learned patterns from past missions. The train/test evaluation methodology shown here is essential for validating that the learned patterns actually generalize.

---

## Further Reading

- **Matrix Factorization Techniques for Recommender Systems** (Koren et al., 2009)
- **Collaborative Filtering for Implicit Feedback Datasets** (Hu et al., 2008)
- See `docs/policies/selfish_ep_greedy_cf/` for CF policy documentation in this project
