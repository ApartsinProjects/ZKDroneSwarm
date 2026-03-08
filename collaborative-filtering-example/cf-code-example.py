import random
import math  # not really used here, but harmless to keep


# ============================================================
# MATRIX FACTORIZATION FOR EXPLICIT RATINGS
# ------------------------------------------------------------
# Big idea:
# We start from a sparse rating matrix I0:
#
#        items →
# users   Shrek   Memento   DarkKnight
# Alice     5        1         ?
# Bob       4        ?         5
# Cara      ?        5         4
#
# ? means "missing / unknown"
#
# We want to learn:
#   I_hat = P · U
#
# where:
#   P = user latent-factor matrix
#   U = item latent-factor matrix
#
# "latent" = hidden
# "latent factors" = hidden dimensions that explain preferences
#
# Example interpretation (just intuition, not given to the model):
#   factor 1 = likes fun/light movies
#   factor 2 = likes dark/complex movies
#
# The model is NOT told these meanings.
# It learns useful hidden dimensions from the ratings.
# ============================================================


# ============================================================
# 1) Define users and items
# ------------------------------------------------------------
# Users are the people giving ratings.
# Items are the things being rated.
#
# In collaborative filtering:
# - rows usually represent users
# - columns usually represent items
# ============================================================

users = ["Alice", "Bob", "Cara"]
items = ["Shrek", "Memento", "DarkKnight"]


# ============================================================
# 2) Define the observed ratings
# ------------------------------------------------------------
# This is explicit feedback:
# users directly gave ratings (1..5)
#
# We store only the known ratings.
# This is a sparse representation because many ratings are missing.
# ============================================================

observed = {
    ("Alice", "Shrek"): 5.0,
    ("Alice", "Memento"): 1.0,
    ("Bob", "Shrek"): 4.0,
    ("Bob", "DarkKnight"): 5.0,
    ("Cara", "Memento"): 5.0,
    ("Cara", "DarkKnight"): 4.0,
}


# ============================================================
# 3) Build index maps
# ------------------------------------------------------------
# Matrices use integer row/column indices, not names.
#
# Example:
#   Alice -> row 0
#   Bob   -> row 1
#   Cara  -> row 2
#
#   Shrek      -> col 0
#   Memento    -> col 1
#   DarkKnight -> col 2
# ============================================================

u2i = {u: r for r, u in enumerate(users)}   # user name -> row index
i2j = {it: c for c, it in enumerate(items)} # item name -> column index

m = len(users)  # number of users = number of rows
n = len(items)  # number of items = number of columns


# ============================================================
# 4) Build the original sparse rating matrix I0
# ------------------------------------------------------------
# I0 is the observed rating matrix.
#
# Shape:
#   I0 is (m x n)
#   rows = users
#   cols = items
#
# Missing ratings are stored as None.
# ============================================================

I0 = [[None for _ in range(n)] for _ in range(m)]

for (u, it), r in observed.items():
    I0[u2i[u]][i2j[it]] = r

# K = list of known ratings only.
# Each entry is:
#   (user_index, item_index, true_rating)
#
# This matters because we train ONLY on observed entries.
# We do NOT try to match missing entries.
K = [(u2i[u], i2j[it], r) for (u, it), r in observed.items()]


# ============================================================
# 5) Hyperparameters
# ------------------------------------------------------------
# d      = number of latent factors
# lr     = learning rate (step size)
# reg    = regularization strength
# epochs = how many times to pass over the known ratings
#
# d = 2 means:
# every user is represented by 2 hidden numbers
# every item is represented by 2 hidden numbers
#
# This 2D hidden representation is the latent space.
# ============================================================

d = 2
lr = 0.05
reg = 0.02
epochs = 500


# ============================================================
# 6) Helper to create a random matrix
# ------------------------------------------------------------
# We start from random guesses for the latent factors.
# At the beginning, the model does not know anything.
# Learning gradually improves these values.
# ============================================================

def rand_matrix(rows, cols, scale=0.1):
    return [[random.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


# ============================================================
# 7) Create the factor matrices P and U
# ------------------------------------------------------------
# Goal:
#   I_hat = P · U
#
# Shapes:
#   P is (m x d) = users x latent_factors
#   U is (d x n) = latent_factors x items
#
# So:
#   (m x d) · (d x n) = (m x n)
#
# That matches the shape of the original rating matrix I0.
#
# Meaning of P:
#   Each row P[u] is the latent vector for user u
#   -> hidden taste / preference profile
#
# Meaning of U:
#   Each column U[:, i] is the latent vector for item i
#   -> hidden attribute / feature profile
#
# This is the "user space vs item space" idea:
# - users live in latent space
# - items live in latent space
# - ratings come from how well they match
# ============================================================

P = rand_matrix(m, d)   # user matrix: users x latent factors
U = rand_matrix(d, n)   # item matrix: latent factors x items


# ============================================================
# 8) Prediction function
# ------------------------------------------------------------
# For one user u and one item i:
#
#   prediction = (P·U)[u,i]
#              = sum_k P[u,k] * U[k,i]
#
# This is the dot product between:
# - the user's latent vector
# - the item's latent vector
#
# Intuition:
# - if user preferences align well with item properties -> high score
# - if they do not align -> low score
# ============================================================

def predict_entry(u, i):
    return sum(P[u][k] * U[k][i] for k in range(d))


# ============================================================
# 9) Loss functions
# ------------------------------------------------------------
# We want P and U such that:
#   P·U approximates I0
#
# But only on observed ratings.
#
# Data loss:
#   sum over known ratings of:
#       (predicted - true)^2
#
# That is:
#   Σ_(u,i in K) ( (P·U)[u,i] - I0[u,i] )^2
#
# This is the squared reconstruction error.
#
# Regularization is added to keep values from becoming too large.
# ============================================================

def frob_sq(M):
    # Frobenius norm squared:
    # sum of squares of all numbers in matrix M
    return sum(x * x for row in M for x in row)

def loss_only_data():
    # Sum of squared errors over OBSERVED entries only
    return sum((predict_entry(u, i) - r) ** 2 for (u, i, r) in K)

def loss_with_reg():
    # Full objective:
    # data loss + regularization
    return loss_only_data() + reg * (frob_sq(P) + frob_sq(U))


# ============================================================
# 10) Training loop (SGD = stochastic gradient descent)
# ------------------------------------------------------------
# Learning idea:
# For each known rating (u, i, r):
#
#   pred = current predicted rating
#   err  = pred - true_rating
#
# Then update:
#   P[u, :]   -> user vector
#   U[:, i]   -> item vector
#
# Why only these?
# Because only these values affect prediction (u, i).
#
# If prediction is too high:
#   move factors so prediction goes down
#
# If prediction is too low:
#   move factors so prediction goes up
#
# This is the "learning" in the model.
# ============================================================

for epoch in range(1, epochs + 1):
    # Shuffle the observed examples each epoch.
    # This is common in SGD and helps learning.
    random.shuffle(K)

    # Process one known rating at a time
    for u, i, r in K:
        # Current predicted rating for user u and item i
        pred = predict_entry(u, i)

        # Error = prediction - truth
        # positive -> predicted too high
        # negative -> predicted too low
        err = pred - r

        # Save old user vector before updating it
        # so the item update uses the old value consistently
        Pu_old = P[u][:]

        # Update each latent factor dimension separately
        for k in range(d):
            # Gradient step for the user latent factor P[u][k]
            #
            # Main error gradient part:
            #   2 * err * U[k][i]
            #
            # Regularization part:
            #   reg * P[u][k]
            #
            # Then gradient descent subtracts:
            #   lr * gradient
            P[u][k] -= lr * (2 * err * U[k][i] + reg * P[u][k])

            # Gradient step for the item latent factor U[k][i]
            #
            # Main error gradient part:
            #   2 * err * old_user_value
            #
            # Regularization part:
            #   reg * U[k][i]
            U[k][i] -= lr * (2 * err * Pu_old[k] + reg * U[k][i])

    # Print progress at a few checkpoints
    # As learning works, data_loss should usually go down.
    if epoch in {1, 10, 50, 100, 200, 500}:
        print(
            f"epoch={epoch:3d}  "
            f"data_loss={loss_only_data():.4f}  "
            f"total_loss(data+reg)={loss_with_reg():.4f}"
        )


# ============================================================
# 11) Print the reconstructed matrix I_hat = P·U
# ------------------------------------------------------------
# After training, P and U define a full predicted matrix:
#
#   I_hat = P·U
#
# This includes:
# - approximations of known ratings
# - predictions for missing ratings
#
# This is the core of recommendation:
# filling in the unknown cells of the user-item matrix.
# ============================================================

def print_matrix_hat():
    print("\nReconstructed (predicted) matrix I_hat = P·U:")
    header = " " * 10 + "  ".join(f"{it:10s}" for it in items)
    print(header)

    for u_name in users:
        u = u2i[u_name]
        row_preds = [predict_entry(u, i2j[it]) for it in items]
        print(f"{u_name:10s}" + "  " + "  ".join(f"{x:10.3f}" for x in row_preds))


print_matrix_hat()


# ============================================================
# 12) Example: predict one unseen rating
# ------------------------------------------------------------
# Alice never rated DarkKnight in the observed data.
#
# But now the model can estimate it using:
# - Alice's learned latent vector
# - DarkKnight's learned latent vector
#
# This is exactly how matrix factorization predicts missing entries.
# ============================================================

u = u2i["Alice"]
i = i2j["DarkKnight"]
print(f"\nUnseen pair prediction (Alice, DarkKnight): {predict_entry(u, i):.3f}")


# ============================================================
# 13) Show learned user latent vectors
# ------------------------------------------------------------
# Each row in P is a user representation in latent space.
#
# These numbers do NOT have explicit labels.
# They are hidden learned coordinates.
#
# Still, they summarize each user's taste in compressed form.
# ============================================================

print("\nLearned P (users x factors):")
for u_name in users:
    print(f"  {u_name:5s}: {['%+.3f' % x for x in P[u2i[u_name]]]}")


# ============================================================
# 14) Show learned item latent vectors
# ------------------------------------------------------------
# Each item is also represented in the same latent space.
#
# In U, each item corresponds to one column.
#
# These numbers summarize hidden item properties learned from data.
# ============================================================

print("\nLearned U (factors x items):")
for k in range(d):
    print(f"  factor {k}: {['%+.3f' % U[k][i2j[it]] for it in items]}")