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