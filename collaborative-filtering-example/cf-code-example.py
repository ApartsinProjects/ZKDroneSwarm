import random
import math  # harmless to keep


# ============================================================
# MATRIX FACTORIZATION FOR EXPLICIT RATINGS (LARGER EXAMPLE)
# ------------------------------------------------------------
# We keep ratings in the range 1..5 for learning purposes.
#
# Core collaborative-filtering idea remains the same:
# - each user gets a latent vector
# - each item gets a latent vector
# - prediction = dot(user_vector, item_vector)
#
# We will:
# 1) define a FULL rating table (ground truth)
# 2) intentionally hide some entries
# 3) train only on the remaining observed ratings
# 4) compare predictions against the hidden true ratings
# ============================================================


# ============================================================
# 1) Users and items
# ------------------------------------------------------------
# We make a larger example:
# - 10 users
# - 12 movies
#
# The movie set is designed to have broad taste groups:
# - family / animation
# - crime / thriller
# - sci-fi / action
# - a few bridge movies
# ============================================================

users = [
    "Alice", "Bob", "Cara", "Dan", "Eve",
    "Frank", "Grace", "Hank", "Ivy", "Jake"
]

items = [
    "Shrek",
    "ToyStory",
    "FindingNemo",
    "Memento",
    "Se7en",
    "DarkKnight",
    "Inception",
    "Matrix",
    "Interstellar",
    "Avengers",
    "Up",
    "PulpFiction",
]


# ============================================================
# 2) Full ground-truth rating matrix
# ------------------------------------------------------------
# This is the COMPLETE table before hiding any test entries.
#
# Rows = users
# Cols = movies
#
# We handcraft it so the preference structure is visible:
#
# - Alice, Bob: family/animation lovers
# - Cara, Dan: crime/thriller lovers
# - Eve, Frank: sci-fi/action lovers
# - Grace, Hank, Ivy, Jake: mixed profiles
#
# Ratings are on a 1..5 scale.
# ============================================================

full_ratings = {
    "Alice": {
        "Shrek": 5, "ToyStory": 5, "FindingNemo": 5, "Memento": 1,
        "Se7en": 1, "DarkKnight": 2, "Inception": 2, "Matrix": 2,
        "Interstellar": 3, "Avengers": 3, "Up": 5, "PulpFiction": 1,
    },
    "Bob": {
        "Shrek": 5, "ToyStory": 4, "FindingNemo": 5, "Memento": 1,
        "Se7en": 2, "DarkKnight": 2, "Inception": 2, "Matrix": 1,
        "Interstellar": 2, "Avengers": 3, "Up": 5, "PulpFiction": 1,
    },
    "Cara": {
        "Shrek": 1, "ToyStory": 2, "FindingNemo": 1, "Memento": 5,
        "Se7en": 5, "DarkKnight": 4, "Inception": 4, "Matrix": 2,
        "Interstellar": 3, "Avengers": 2, "Up": 1, "PulpFiction": 5,
    },
    "Dan": {
        "Shrek": 1, "ToyStory": 1, "FindingNemo": 2, "Memento": 5,
        "Se7en": 5, "DarkKnight": 5, "Inception": 4, "Matrix": 3,
        "Interstellar": 3, "Avengers": 2, "Up": 1, "PulpFiction": 4,
    },
    "Eve": {
        "Shrek": 2, "ToyStory": 2, "FindingNemo": 2, "Memento": 3,
        "Se7en": 2, "DarkKnight": 4, "Inception": 5, "Matrix": 5,
        "Interstellar": 5, "Avengers": 4, "Up": 2, "PulpFiction": 2,
    },
    "Frank": {
        "Shrek": 2, "ToyStory": 1, "FindingNemo": 2, "Memento": 3,
        "Se7en": 2, "DarkKnight": 4, "Inception": 5, "Matrix": 5,
        "Interstellar": 4, "Avengers": 5, "Up": 1, "PulpFiction": 2,
    },
    "Grace": {
        "Shrek": 4, "ToyStory": 4, "FindingNemo": 4, "Memento": 2,
        "Se7en": 2, "DarkKnight": 3, "Inception": 3, "Matrix": 3,
        "Interstellar": 4, "Avengers": 4, "Up": 4, "PulpFiction": 2,
    },
    "Hank": {
        "Shrek": 2, "ToyStory": 2, "FindingNemo": 2, "Memento": 4,
        "Se7en": 4, "DarkKnight": 5, "Inception": 4, "Matrix": 4,
        "Interstellar": 4, "Avengers": 3, "Up": 2, "PulpFiction": 4,
    },
    "Ivy": {
        "Shrek": 3, "ToyStory": 3, "FindingNemo": 4, "Memento": 3,
        "Se7en": 2, "DarkKnight": 3, "Inception": 4, "Matrix": 4,
        "Interstellar": 5, "Avengers": 4, "Up": 4, "PulpFiction": 2,
    },
    "Jake": {
        "Shrek": 3, "ToyStory": 2, "FindingNemo": 3, "Memento": 4,
        "Se7en": 4, "DarkKnight": 4, "Inception": 4, "Matrix": 3,
        "Interstellar": 3, "Avengers": 3, "Up": 2, "PulpFiction": 5,
    },
}


# ============================================================
# 3) Choose entries to hide for evaluation
# ------------------------------------------------------------
# These are TRUE ratings that we intentionally hide from training.
#
# The point:
# - the model does not see them during training
# - after training, we compare prediction vs truth
#
# Try to hide values that are interesting but still somewhat guessable
# from the overall taste structure.
# ============================================================

hidden_pairs = [
    ("Alice", "ToyStory"),       # true rating should be high
    ("Bob", "Up"),               # true rating should be high
    ("Cara", "PulpFiction"),     # true rating should be high
    ("Dan", "DarkKnight"),       # true rating should be high
    ("Eve", "Matrix"),           # true rating should be high
    ("Frank", "Avengers"),       # true rating should be high
    ("Grace", "Memento"),        # moderate/low
    ("Hank", "Se7en"),           # high
    ("Ivy", "Interstellar"),     # high
    ("Jake", "ToyStory"),        # lower/moderate
]

hidden_truth = {(u, it): float(full_ratings[u][it]) for (u, it) in hidden_pairs}


# ============================================================
# 4) Build observed ratings
# ------------------------------------------------------------
# observed = all ratings EXCEPT the intentionally hidden test cells
# ============================================================

observed = {}

for u in users:
    for it in items:
        if (u, it) not in hidden_truth:
            observed[(u, it)] = float(full_ratings[u][it])


# ============================================================
# 5) Build index maps
# ============================================================

u2i = {u: r for r, u in enumerate(users)}
i2j = {it: c for c, it in enumerate(items)}

m = len(users)
n = len(items)


# ============================================================
# 6) Build sparse matrix I0
# ------------------------------------------------------------
# I0 contains:
# - observed ratings as floats
# - hidden test entries as None
# ============================================================

I0 = [[None for _ in range(n)] for _ in range(m)]

for (u, it), r in observed.items():
    I0[u2i[u]][i2j[it]] = r

K = [(u2i[u], i2j[it], r) for (u, it), r in observed.items()]


# ============================================================
# 7) Hyperparameters
# ------------------------------------------------------------
# d can be larger now because the matrix is larger.
# ============================================================

d = 7
lr = 0.01
reg = 0.02
epochs = 20000


# ============================================================
# 8) Random initialization
# ============================================================

def rand_matrix(rows, cols, scale=0.1):
    return [[random.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

P = rand_matrix(m, d)
U = rand_matrix(d, n)


# ============================================================
# 9) Prediction and loss
# ============================================================

def predict_entry(u, i):
    return sum(P[u][k] * U[k][i] for k in range(d))

def frob_sq(M):
    return sum(x * x for row in M for x in row)

def loss_only_data():
    return sum((predict_entry(u, i) - r) ** 2 for (u, i, r) in K)

def loss_with_reg():
    return loss_only_data() + reg * (frob_sq(P) + frob_sq(U))


# ============================================================
# 6) The Learning Loop (SGD Algorithm)
# ------------------------------------------------------------
# This section documents how the Stochastic Gradient Descent
# algorithm maps mathematical formulas to code.
#
# For each observed rating (u, i, r_ui):
#
# 1. PREDICT: Compute dot product of user and item latent vectors
#    Formula:    r_hat_ui = P_u^T * U_i = sum_k(P[u][k] * U[k][i])
#    Code:       pred = sum(P[u][k] * U[k][i] for k in range(d))
#
# 2. ERROR: Calculate prediction error
#    Formula:    e_ui = r_hat_ui - r_ui
#    Code:       err = pred - r
#
# 3. GRADIENT DESCENT UPDATE: Adjust vectors to reduce error
#    The gradient of squared error (r_hat - r)^2 gives the
#    update direction. The factor of 2 comes from differentiation.
#
#    Formula for P:    P_u <- P_u - gamma * (2 * e_ui * U_i + lambda * P_u)
#    Code for P:       P[u][k] -= lr * (2 * err * U[k][i] + reg * P[u][k])
#
#    Formula for U:    U_i <- U_i - gamma * (2 * e_ui * P_u + lambda * U_i)
#    Code for U:       U[k][i] -= lr * (2 * err * Pu_old[k] + reg * U[k][i])
#
#    Where:
#    - gamma  = lr  (learning rate = 0.01)
#    - lambda = reg (regularization = 0.02)
#    - Pu_old ensures we use pre-update P for consistent gradients
#
# 4. EPOCH: Repeat for all observed ratings, shuffle, repeat
#    Code:       for epoch in range(1, epochs + 1):
#                   random.shuffle(K)
#                   for u, i, r in K:
#                       ... update as above ...
#
# 5. CONVERGE: Continue until loss stabilizes at minimum
# ============================================================


# ============================================================
# 9.5) Print full truth & summary before training
# ============================================================

print("\n=== Full Ground-Truth Rating Matrix ===")
header = " " * 10 + " ".join(f"{it[:8]:8s}" for it in items)
print(header)
for u in users:
    row = [f"{full_ratings[u][it]:8.0f}" for it in items]
    print(f"{u:10s}" + " " + " ".join(row))

print("\n=== Taste Group Summary ===")
print("- Alice & Bob : Family / Animation lovers (Shrek, ToyStory, FindingNemo, Up)")
print("- Cara & Dan  : Crime / Thriller lovers   (Memento, Se7en, PulpFiction, DarkKnight)")
print("- Eve & Frank : Sci-Fi / Action lovers    (Inception, Matrix, Interstellar, Avengers)")
print("- Grace, Hank,")
print("  Ivy, Jake   : Mixed / Bridging profiles")
print("===========================================\n")

print("Starting training...")

# ============================================================
# 10) Training loop (SGD)
# ============================================================

for epoch in range(1, epochs + 1):
    random.shuffle(K)

    for u, i, r in K:
        pred = predict_entry(u, i)
        err = pred - r

        Pu_old = P[u][:]

        for k in range(d):
            P[u][k] -= lr * (2 * err * U[k][i] + reg * P[u][k])
            U[k][i] -= lr * (2 * err * Pu_old[k] + reg * U[k][i])

    if epoch % 1000 == 0:
        print(
            f"epoch={epoch:4d}  "
            f"data_loss={loss_only_data():.4f}  "
            f"total_loss={loss_with_reg():.4f}"
        )


# ============================================================
# 11) Print observed matrix with hidden cells marked
# ============================================================

def print_observed_matrix():
    print("\nObserved matrix I0 (None = intentionally hidden for testing):")
    header = " " * 10 + " ".join(f"{it[:8]:8s}" for it in items)
    print(header)

    for u in users:
        row = []
        for it in items:
            val = I0[u2i[u]][i2j[it]]
            if val is None:
                row.append(f"{'None':>8s}")
            else:
                row.append(f"{val:8.0f}")
        print(f"{u:10s}" + " " + " ".join(row))


# ============================================================
# 12) Print reconstructed matrix I_hat = P·U
# ============================================================

def print_matrix_hat():
    print("\nReconstructed (predicted) matrix I_hat = P·U:")
    header = " " * 10 + " ".join(f"{it[:8]:8s}" for it in items)
    print(header)

    for u in users:
        uu = u2i[u]
        row_preds = [predict_entry(uu, i2j[it]) for it in items]
        print(f"{u:10s}" + " " + " ".join(f"{x:8.1f}" for x in row_preds))


# ============================================================
# 13) Evaluate on hidden test entries
# ------------------------------------------------------------
# This is the key addition:
# compare predicted values to the true hidden ratings
# ============================================================

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


# ============================================================
# 14) Show some learned factors
# ============================================================

def print_latent_vectors():
    print("\nLearned P (matrix P: m×d, users as rows):")
    for u in users:
        print(f"  {u:12s}: {['%+.3f' % x for x in P[u2i[u]]]}")

    print("\nLearned U (matrix U: d×n, items as columns):")
    header = " " * 8 + " ".join(f"{it[:8]:8s}" for it in items)
    print(header)

    for k in range(d):
        row_vals = [U[k][i2j[it]] for it in items]
        print(f"U[{k}]     " + " ".join(f"{x:8.3f}" for x in row_vals))


# ============================================================
# 15) Run outputs
# ============================================================

print_observed_matrix()
print_matrix_hat()
evaluate_hidden()
print_latent_vectors()

# ============================================================
# 16) Visualize Embeddings with t-SNE
# ============================================================

from embedding_visualizer import EmbeddingVisualizer

print("\nGenerating t-SNE visualization of latent spaces...")
visualizer = EmbeddingVisualizer(perplexity=5)
visualizer.visualize(P, U, users, items, layout='joint', save_path='embeddings_tsne.png')