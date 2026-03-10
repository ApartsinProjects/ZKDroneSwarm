# Decentralized Collaborative Matrix-Factorization Policy for ZK-MRTA

## Purpose of This Policy

This policy adapts **classical collaborative filtering with matrix factorization** to **ZK-MRTA** as a decentralized mechanism for learning **latent utility**.

The mapping is:

- **users** $\rightarrow$ **drones**
- **items** $\rightarrow$ **targets**
- **ratings** $\rightarrow$ **observed engagement outcomes**
- **predicted score** $\rightarrow$ **expected engagement effectiveness**

The goal is not to predict preference, but to estimate how effective a drone is likely to be on a target under zero-knowledge constraints.

Each drone runs this as an **online loop**: score targets, act, observe public outcomes, and update its local model.

---

# Step 1: Local Latent Model Initialization

Each drone $a$ maintains its own local factorization:

$$
P^{(a)} \in \mathbb{R}^{N_d \times d},
\qquad
U^{(a)} \in \mathbb{R}^{d \times N_t}
$$

where:

- $P^{(a)}_i$ is drone $a$’s local latent representation of drone $i$
- $U^{(a)}_t$ is drone $a$’s local latent representation of target $t$

All entries are initialized with small random values.

---

# Step 2: Local Prediction of Drone-Target Utility

Given an observed or candidate interaction between drone $i$ and target $t$, drone $a$ predicts its utility by:

$$
\hat{r}^{(a)}_{i,t} = \left(P^{(a)}_i\right)^\top U^{(a)}_t
$$

This dot product is drone $a$’s current estimate of the effectiveness of drone $i$ on target $t$.

---

# Step 3: Action Selection

When drone $a$ chooses its own action, it uses its own row $P^{(a)}_a$.

For each active target $t$:

$$
\hat{r}^{(a)}_{a,t} = \left(P^{(a)}_a\right)^\top U^{(a)}_t
$$

Drone $a$ then selects the valid target with the highest predicted score.

Learning may use public events from all drones, but action selection is always based on the acting drone’s own local scores.

---

# Step 4: Public Interaction Events

After actions are executed, drones observe public events of the form:

$$
(i, t, r^{(i)}_t)
$$

where:

- $i$ is the acting drone
- $t$ is the engaged target
- $r^{(i)}_t$ is the observed reward

Each such event is a decentralized analogue of an observed rating in classical collaborative filtering.

---

# Step 5: Collaborative Local Update / Learning

Drone $a$ uses each public event $(i,t,r^{(i)}_t)$ as a local training sample.

Prediction:

$$
\hat{r}^{(a)}_{i,t} = \left(P^{(a)}_i\right)^\top U^{(a)}_t
$$

Error:

$$
e^{(a)}_{i,t} = \hat{r}^{(a)}_{i,t} - r^{(i)}_t
$$

Update:

$$
P^{(a)}_i
\leftarrow
P^{(a)}_i
-
\eta
\left(
2 e^{(a)}_{i,t} U^{(a)}_t + \lambda P^{(a)}_i
\right)
$$

$$
U^{(a)}_t
\leftarrow
U^{(a)}_t
-
\eta
\left(
2 e^{(a)}_{i,t} P^{(a)}_i + \lambda U^{(a)}_t
\right)
$$

So each drone updates:

- the latent row of the drone that acted
- the latent vector of the target involved

Every drone performs this update locally from the same public evidence stream. Collaboration comes from shared observations, not shared parameters.

---

# Step 6: Online Learning and Control Loop

For each drone $a$:

1. observe the current state and public trace
2. score each active target $t$ using $P^{(a)}_a$ and $U^{(a)}_t$
3. choose one valid action
4. observe new public outcomes
5. convert public events into MF training samples
6. update $P^{(a)}$ and $U^{(a)}$
7. repeat

This makes the policy an **online decentralized latent-utility learner**, not a static recommender.

---

## Reward Interpretation

The policy does not define the reward; the environment does.

The only requirement is that reward reflects **engagement effectiveness**:

$$
r^{(a)}_t = \text{observed effectiveness of drone } a \text{ on target } t
$$

A natural example is:

$$
r^{(a)}_t =
\begin{cases}
-1, & \text{if drone } a \text{ fires at an already inactive target} \\[6pt]
0, & \text{if drone } a \text{ chooses NoOp} \\[6pt]
\displaystyle
\frac{\sum_{k} \min\!\left(h^{\text{before}}_{t,k},\, w_{a,k}\right)}
{\sum_{k} w_{a,k}},
& \text{if drone } a \text{ fires at an active target}
\end{cases}
$$

For a valid shot on an active target:

$$
r^{(a)}_t \in [0,1]
$$

with the interpretation that higher values mean more effective use of the drone’s weapon against that target.

Under this reward contract, the latent model learns expected **operational effectiveness**, not preference.