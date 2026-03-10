# Collaborative Filtering Example: Matrix Factorization

This example explains the **classical matrix factorization** approach for collaborative filtering in a compact way. The setting remains the standard **user–movie recommendation** problem, but the focus here is only on the critical ideas, formulas, and training steps.

---

## 1. Problem Setup

Assume we have a partially observed rating matrix:

$$
I_0 \in \mathbb{R}^{m \times n}
$$

where:

- $m$ = number of users
- $n$ = number of movies
- $I_0(u,i)$ = the rating user $u$ gave to movie $i$, when that rating is known

A toy example:

| User   | Shrek | ToyStory | FindingNemo | Memento | Se7en |
|--------|-------|----------|-------------|---------|-------|
| Alice  | 5     | ?        | 5           | 1       | 1     |
| Bob    | 5     | 4        | 5           | 1       | 2     |
| Cara   | 1     | 1        | 2           | 5       | 5     |

The missing entries (`?`) are unknown ratings that we want to predict.

---

## 2. Core Idea: Latent Factors

Matrix factorization assumes that each user and each movie can be represented by a small **latent vector**:

- a **user latent vector** captures hidden taste
- a **movie latent vector** captures hidden movie characteristics

We factorize the rating matrix into two smaller matrices:

$$
\hat{I} = P U
$$

where:

- $P \in \mathbb{R}^{m \times d}$ is the **user latent factor matrix**
- $U \in \mathbb{R}^{d \times n}$ is the **movie latent factor matrix**
- $d$ is the latent dimension

For user $u$, the latent vector is:

$$
P_u \in \mathbb{R}^d
$$

For movie $i$, the latent vector is:

$$
U_i \in \mathbb{R}^d
$$

---

## 3. Prediction Rule

The predicted rating of user $u$ for movie $i$ is the dot product of their latent vectors:

$$
\hat{r}_{ui} = P_u^\top U_i
$$

Equivalently:

$$
\hat{r}_{ui} = \sum_{k=1}^{d} P_{u,k} U_{k,i}
$$

Interpretation:

- if the user vector and movie vector align well, the predicted rating is high
- if they do not align, the predicted rating is low

---

## 4. Observed Ratings Only

In collaborative filtering, many ratings are missing. These missing entries are **not zeros**. They are simply unknown.

Define the set of observed ratings:

$$
\Omega = \{(u,i) \mid r_{ui}\text{ is observed}\}
$$

Training is performed **only** on entries in $\Omega$.

---

## 5. Loss Function

For one observed rating $(u,i)$, define the prediction error:

$$
e_{ui} = \hat{r}_{ui} - r_{ui}
$$

The data loss over all observed entries is:

$$
\mathcal{L}_{\text{data}}(P,U)
=
\sum_{(u,i)\in\Omega}
\left(P_u^\top U_i - r_{ui}\right)^2
$$

To reduce overfitting, add L2 regularization:

$$
\mathcal{L}_{\text{reg}}(P,U)
=
\lambda \left(\|P\|_F^2 + \|U\|_F^2\right)
$$

So the full objective becomes:

$$
\mathcal{L}(P,U)
=
\sum_{(u,i)\in\Omega}
\left(P_u^\top U_i - r_{ui}\right)^2
+
\lambda \left(\|P\|_F^2 + \|U\|_F^2\right)
$$

Training means solving:

$$
(P^{*},U^{*})=\arg\min_{P,U}\mathcal{L}(P,U)
$$

---

## 6. SGD Training Steps

The model is usually trained with **stochastic gradient descent (SGD)**.

### Step 1: Initialize
Start with small random values in $P$ and $U$.

### Step 2: Predict
For one observed rating $(u,i,r_{ui})$, compute:

$$
\hat{r}_{ui} = P_u^\top U_i
$$

### Step 3: Compute Error
$$
e_{ui} = \hat{r}_{ui} - r_{ui}
$$

### Step 4: Update the User Vector
$$
P_u \leftarrow P_u - \gamma \left(2 e_{ui} U_i + \lambda P_u\right)
$$

### Step 5: Update the Item Vector
$$
U_i \leftarrow U_i - \gamma \left(2 e_{ui} P_u + \lambda U_i\right)
$$

where:

- $\gamma$ = learning rate
- $\lambda$ = regularization strength

### Step 6: Repeat
Loop over all observed ratings, for many epochs, until the loss stabilizes.

---

## 7. Matrix View

At a high level, matrix factorization tries to reconstruct the known part of the rating matrix:

$$
P U \approx I_0
$$

Because $I_0$ is sparse, the precise matrix-form objective uses a mask matrix $M$:

$$
\mathcal{L}(P,U)
=
\left\|M \odot (PU - I_0)\right\|_F^2
+
\lambda \left(\|P\|_F^2 + \|U\|_F^2\right)
$$

where:

- $M_{ui}=1$ if rating $(u,i)$ is observed
- $M_{ui}=0$ otherwise
- $\odot$ is elementwise multiplication

This ensures that only known ratings affect training.

---

## 8. Intuition from the Toy User–Movie Example

In the movie setting, users often exhibit hidden taste structure:

- some users prefer animation and family films
- some prefer crime and thriller movies
- some prefer sci-fi and action

Matrix factorization does not require these categories to be labeled explicitly. Instead, it learns latent vectors from rating patterns alone.

So if Alice likes **Shrek** and **Finding Nemo**, and other users with similar patterns also like **Toy Story**, the model can infer that Alice will probably rate **Toy Story** highly even if that rating is missing.

---

## 9. Bottom Line

The essential idea of classical collaborative filtering with matrix factorization is:

1. represent each user and each movie by a latent vector  
2. predict ratings using a dot product  
3. train only on observed ratings  
4. minimize squared prediction error with regularization  
5. update the latent vectors iteratively with SGD  

In short:

> Matrix factorization learns hidden user preferences and hidden movie properties so that their interaction can explain the ratings we observe and predict the ratings we do not.

That is the classical collaborative-filtering baseline in its most compact form.