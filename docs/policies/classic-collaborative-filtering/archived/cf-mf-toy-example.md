# Classical Collaborative Filtering Example: Matrix Factorization

## Purpose of This Example

This example presents the **classical matrix-factorization formulation of collaborative filtering** in its standard recommender-system setting.

The goal is simple:

- users rate movies,
- many ratings are missing,
- and the model predicts the missing ratings from the observed ones.

The purpose of this example is not to show implementation details.  
It is to make the **core mathematical structure** of classical collaborative filtering clear before adapting the same logic to the ZK-MRTA setting.

---

## Background: The Recommendation Problem

Assume a movie recommendation setting with:

- a set of users,
- a set of movies,
- and a partially observed user-movie rating matrix.

A toy example is:

| User   | Shrek | ToyStory | FindingNemo | Memento | Se7en |
|--------|-------|----------|-------------|---------|-------|
| Alice  | 5     | ?        | 5           | 1       | 1     |
| Bob    | 5     | 4        | 5           | 1       | 2     |
| Cara   | 1     | 1        | 2           | 5       | 5     |

Here:

- rows correspond to users,
- columns correspond to movies,
- observed entries are known ratings,
- and `?` denotes a missing rating to be predicted.

The classical collaborative-filtering question is therefore:

> Given the ratings that are already known, how can we estimate the ratings that are still missing?

---

## Why Matrix Factorization Is Useful

A central difficulty in recommendation problems is that user preferences are only partially observed.

We do not directly know:

- a user's hidden taste profile,
- a movie's hidden characteristics,
- or the latent structure that explains why certain users tend to like certain movies.

Matrix factorization addresses this by learning a compact **latent representation** for both users and movies.

The key idea is:

- each user is represented by a latent vector,
- each movie is represented by a latent vector,
- and their interaction determines the predicted rating.

Thus, the model does not rely on explicit labels such as “family-movie lover” or “crime-thriller fan.”  
Instead, it learns these hidden structures implicitly from rating patterns.

---

# Step 1: Represent the Rating Matrix

Let:

- $m$ be the number of users,
- $n$ be the number of movies,
- $d$ be the latent dimension.

Let the partially observed rating matrix be:

$$
I_0 \in \mathbb{R}^{m \times n}
$$

where $(I_0)_{ui} = r_{ui}$ is the rating given by user $u$ to movie $i$ whenever that rating is observed.

The matrix is sparse, because in practice most users rate only a small subset of all movies.

---

# Step 2: Introduce Latent User and Movie Factors

Matrix factorization approximates the user-movie matrix using two smaller latent matrices:

$$
\hat{I} = P U
$$

where:

$$
P \in \mathbb{R}^{m \times d}
$$

$$
U \in \mathbb{R}^{d \times n}
$$

with interpretation:

- $P$ = user latent-factor matrix
- $U$ = movie latent-factor matrix
- $\hat{I}$ = reconstructed rating matrix

For user $u$, the row

$$
P_u \in \mathbb{R}^d
$$

is the latent vector representing that user's hidden taste profile.

For movie $i$, the column

$$
U_i \in \mathbb{R}^d
$$

is the latent vector representing that movie's hidden characteristics.

---

# Step 3: Predict Ratings from Latent Interactions

The predicted rating for user $u$ on movie $i$ is the dot product of their latent vectors:

$$
\hat{r}_{ui} = P_u^\top U_i
$$

Equivalently,

$$
\hat{r}_{ui} = \sum_{k=1}^{d} P_{u,k} U_{k,i}
$$

The interpretation is straightforward:

- if the user vector and movie vector align strongly, the predicted rating is high,
- if they align weakly, the predicted rating is low.

So the model explains ratings through **latent compatibility** between users and movies.

---

# Step 4: Learn Only from Observed Ratings

In collaborative filtering, missing ratings are not treated as zeros.  
They are simply unknown.

Define the set of observed user-movie pairs:

$$
\Omega = \{(u,i) \mid r_{ui} \text{ is observed}\}
$$

Training is performed only on entries in $\Omega$.

This is a critical point.  
The objective is not to reconstruct every matrix entry directly, but to fit the known ratings and use the learned latent structure to generalize to the missing ones.

---

# Step 5: Define the Learning Objective

For an observed rating $(u,i)$, define the prediction error:

$$
e_{ui} = \hat{r}_{ui} - r_{ui}
$$

The squared-error objective over observed entries is:

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
\lambda
\left(
\|P\|_F^2 + \|U\|_F^2
\right)
$$

Therefore, the full objective becomes:

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

Training means solving:

$$
(P^{*}, U^{*})
=
\arg\min_{P,U}
\mathcal{L}(P,U)
$$

This gives the core interpretation of matrix factorization:

> learn latent user vectors and latent movie vectors that explain the observed ratings while remaining sufficiently simple through regularization.

---

## Equivalent Matrix Form

A common matrix-level expression of the same objective uses a binary mask matrix $M$:

- $M_{ui} = 1$ if $(u,i)\in\Omega$
- $M_{ui} = 0$ otherwise

Then the objective can be written as:

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

where $\odot$ denotes elementwise multiplication.

This formulation makes the sparsity explicit: only observed ratings contribute to the loss.

---

# Step 6: Train with Stochastic Gradient Descent

A standard way to optimize the model is **stochastic gradient descent (SGD)**.

For one observed sample $(u,i,r_{ui})$:

### Prediction
$$
\hat{r}_{ui} = P_u^\top U_i
$$

### Error
$$
e_{ui} = \hat{r}_{ui} - r_{ui}
$$

### Per-sample loss
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

The corresponding SGD updates are:

$$
P_u
\leftarrow
P_u
-
\gamma
\left(
2 e_{ui} U_i + \lambda P_u
\right)
$$

$$
U_i
\leftarrow
U_i
-
\gamma
\left(
2 e_{ui} P_u + \lambda U_i
\right)
$$

where:

- $\gamma$ is the learning rate,
- $\lambda$ is the regularization coefficient.

Training then proceeds by repeating this update process over observed ratings for many passes through the data.

---

## Interpretation of Learning

As optimization progresses:

- poor predictions produce large errors,
- the latent vectors are adjusted,
- predicted ratings move closer to the observed ones,
- and the total loss decreases.

So, in practical terms, matrix factorization is learning a hidden geometric structure in which:

$$
P_u^\top U_i \approx r_{ui}
\quad \text{for observed user-movie pairs}
$$

Once this hidden structure has been learned, the model can predict ratings for missing entries as well.

---

## Intuition from the Toy User-Movie Example

The toy matrix above suggests a simple hidden pattern:

- Alice and Bob appear to prefer family or animation movies,
- Cara appears to prefer darker crime or thriller movies.

These semantic labels are not given to the model explicitly.  
Instead, matrix factorization infers them indirectly from the observed ratings.

For example:

- Alice rates **Shrek** and **Finding Nemo** highly,
- Bob shows a similar pattern,
- Bob also rates **Toy Story** highly,
- so the model may infer that Alice is also likely to rate **Toy Story** highly.

This is the essence of collaborative filtering:

> similar rating patterns imply similar latent structure, and that latent structure supports prediction of missing ratings.

---

# Bottom Line

The classical matrix-factorization baseline can be summarized in five steps:

1. represent users and movies with latent vectors,
2. predict ratings by dot products between those vectors,
3. train only on observed ratings,
4. minimize squared prediction error with regularization,
5. update latent vectors iteratively with SGD.

The core conclusion is:

> Matrix factorization learns hidden user preferences and hidden movie properties so that their interaction explains the ratings that are observed and predicts the ratings that are missing.

This is the classical collaborative-filtering mechanism in its cleanest form, and it is the conceptual starting point for later adaptations to the ZK-MRTA setting.