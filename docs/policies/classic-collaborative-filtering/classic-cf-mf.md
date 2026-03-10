# Decentralized Collaborative Matrix-Factorization Policy for ZK-MRTA

## Purpose of This Policy

This policy reinterprets **classical collaborative filtering with matrix factorization** as a decentralized latent-utility learning mechanism for Zero-Knowledge Multi-Robot Task Allocation (ZK-MRTA).

The classical recommendation analogy is:

- **users** correspond to **drones**
- **items** correspond to **targets**
- **ratings** correspond to **observed engagement outcomes**
- **predicted score** corresponds to **expected engagement effectiveness**

The objective is therefore not to predict preference, but to estimate the hidden utility of assigning a drone to a target under strict zero-knowledge constraints.

Unlike a standard recommender system, this policy is not used as an offline batch predictor.  
It is embedded inside an **online sequential decision loop**, where drones repeatedly:

1. observe the current environment,
2. score available targets,
3. choose an action,
4. receive publicly observable outcomes, and
5. update their local latent model.

---

## Why Matrix Factorization Is Appropriate Here

In this setting, drones do not know:

- the true class of each target,
- the hidden attributes of each target,
- the true compatibility between drone capabilities and target vulnerabilities,
- or the latent structure governing successful engagements.

Instead, they must infer this structure indirectly from sparse interaction outcomes.

This makes matrix factorization a natural modeling choice.  
It provides a compact latent representation of hidden drone-target compatibility that can be learned from observed experience, even when symbolic task features are unavailable.

---

## Decentralized Interpretation

The classical matrix-factorization model learns a single global user-item interaction structure.  
In the present setting, such a centralized shared model is not assumed.

Instead, each drone maintains its **own local estimate** of the hidden drone-target interaction matrix.

Thus, the policy is decentralized in the following sense:

- each drone stores its own private local latent model,
- each drone updates that model independently,
- no latent vectors are exchanged directly,
- no centralized optimizer is used,
- and collaboration arises through **shared public interaction events**, not parameter sharing.

So the policy is collaborative because all drones learn from the same observable swarm experience, but decentralized because each drone reconstructs the latent interaction structure locally.

---

# Step 1: Local Latent Model Initialization

Let:

- $N_d$ be the number of drones,
- $N_t$ be the number of targets,
- $d$ be the latent dimension.

Each drone $a$ maintains its own local matrix-factorization model:

$$
P^{(a)} \in \mathbb{R}^{N_d \times d}
$$

$$
U^{(a)} \in \mathbb{R}^{d \times N_t}
$$

where:

- $P^{(a)}$ is drone $a$'s local drone latent matrix,
- $U^{(a)}$ is drone $a$'s local target latent matrix.

For any drone $b$, the row

$$
P^{(a)}_b \in \mathbb{R}^d
$$

is drone $a$'s current local estimate of drone $b$'s latent profile.

For any target $t$, the column-vector

$$
U^{(a)}_t \in \mathbb{R}^d
$$

is drone $a$'s current local estimate of target $t$'s latent profile.

At initialization, all entries of $P^{(a)}$ and $U^{(a)}$ are set to small random values.

This initialization serves three purposes:

1. it provides a complete latent representation over all drones and all targets,
2. it encodes the fact that the true compatibility structure is initially unknown,
3. and it prepares the model for online learning through stochastic gradient updates.

---

# Step 2: Local Prediction of Drone-Target Utility

Inside drone $a$'s local model, the predicted utility of drone $b$ engaging target $t$ is:

$$
\hat{r}^{(a)}_{b,t} = \left(P^{(a)}_b\right)^\top U^{(a)}_t
$$

This is the direct decentralized analogue of the classical matrix-factorization prediction rule:

$$
\hat{r}_{ui} = P_u^\top U_i
$$

Interpretation:

- $P^{(a)}_b$ captures drone $a$'s current latent belief about drone $b$,
- $U^{(a)}_t$ captures drone $a$'s current latent belief about target $t$,
- the dot product gives drone $a$'s current estimate of the effectiveness of assigning drone $b$ to target $t$.

So each drone maintains a full local predictive model over the entire drone-target interaction space.

---

# Step 3: Action Selection

When drone $a$ must choose its own next action, it uses its own row in its local drone matrix:

$$
P^{(a)}_a
$$

For every currently active target $t$, drone $a$ computes:

$$
\hat{r}^{(a)}_{a,t} = \left(P^{(a)}_a\right)^\top U^{(a)}_t
$$

Drone $a$ then selects the target with the highest predicted score among the currently valid candidates.

Thus, decision-making is self-specific:

- learning may involve observations about many drones,
- but action selection uses only the acting drone's own row inside its local model.

This preserves independent decentralized control while still allowing collaborative latent learning.

---

# Step 4: Public Interaction Events

At each environment step, drones observe public interaction outcomes generated by the swarm.

A public event has the form:

$$
(b, t, r^{(b)}_t)
$$

where:

- $b$ is the drone that acted,
- $t$ is the selected target,
- $r^{(b)}_t$ is the observed engagement outcome or reward.

This is the direct analogue of an observed rating event $(u,i,r_{ui})$ in classical collaborative filtering.

The critical structural point is that the event belongs to the pair:

- drone $b$,
- target $t$.

Therefore, in matrix-factorization terms, the event should update:

- the row associated with drone $b$,
- the column associated with target $t$.

---

# Step 5: Collaborative Local Update / Learning

This is the core learning phase.

At each environment step, drone $a$ observes a set of newly available public interaction events generated by the swarm.

Each observed event has the form:

$$
(i, t, r^{(i)}_t)
$$

where:

- $i$ is the acting drone,
- $t$ is the engaged target,
- $r^{(i)}_t$ is the observed engagement outcome or reward.

Drone $a$ treats each such public event as a local supervised training sample.

Using its own local matrices, drone $a$ predicts the observed outcome as:

$$
\hat{r}^{(a)}_{i,t} = \left(P^{(a)}_i\right)^\top U^{(a)}_t
$$

and computes the local prediction error:

$$
e^{(a)}_{i,t} = \hat{r}^{(a)}_{i,t} - r^{(i)}_t
$$

This is the proper decentralized analogue of the classical MF learning rule:

- the observation concerns acting drone $i$ and target $t$,
- so drone $a$ updates its local row for drone $i$,
- and its local vector for target $t$.

This formulation includes both cases naturally:

- if $i=a$, the event corresponds to drone $a$'s own action,
- if $i \neq a$, the event corresponds to another drone's public action.

So there is only one update rule.  
It applies to **whoever acted**.

---

## Local Objective Function

For a single observed event $(i,t,r^{(i)}_t)$, drone $a$'s local per-sample loss is:

$$
\ell^{(a)}_{i,t}
=
\left(
\left(P^{(a)}_i\right)^\top U^{(a)}_t - r^{(i)}_t
\right)^2
+
\lambda
\left(
\|P^{(a)}_i\|^2 + \|U^{(a)}_t\|^2
\right)
$$

where:

- the first term is the squared prediction error,
- the second term is L2 regularization,
- and $\lambda > 0$ is the regularization strength.

Over the full set of public interaction events visible to drone $a$, the local training objective is:

$$
\mathcal{L}^{(a)}(P^{(a)},U^{(a)})
=
\sum_{(i,t)\in\Omega^{(a)}}
\left(
\left(P^{(a)}_i\right)^\top U^{(a)}_t - r^{(i)}_t
\right)^2
+
\lambda
\left(
\|P^{(a)}\|_F^2 + \|U^{(a)}\|_F^2
\right)
$$

where $\Omega^{(a)}$ denotes the set of public interaction events visible to drone $a$.

Thus, each drone minimizes local reconstruction error over the public swarm interaction stream available to it.

---

## SGD Update Rule

At a given environment step, drone $a$ first receives the newly revealed public event set from that step.

Only after those events are observed does drone $a$ update its local model.

This means the causal order is:

1. action selection uses the current local model,
2. the environment executes all drone actions,
3. public outcomes are revealed,
4. drone $a$ updates its local matrices from those outcomes,
5. the updated model is used at the next decision step.

So no event update is conceptually privileged for the current step's action.  
All newly observed events contribute to learning **after** the step is completed, and their effect appears in the **next** step.

For a single observed event $(i,t,r^{(i)}_t)$, drone $a$ computes:

$$
\hat{r}^{(a)}_{i,t} = \left(P^{(a)}_i\right)^\top U^{(a)}_t
$$

$$
e^{(a)}_{i,t} = \hat{r}^{(a)}_{i,t} - r^{(i)}_t
$$

It then updates the corresponding row and column of its local model:

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

where:

- $\eta$ is the learning rate,
- $\lambda$ is the regularization coefficient.

These are standard stochastic-gradient updates applied locally by each drone.

In practice, if multiple public events are revealed in the same step, drone $a$ may process them sequentially in any fixed order using SGD.

So the implementation may be sequential, but the policy interpretation remains:

- first observe the step's public outcomes,
- then learn from them,
- then use the updated model at the next step.

---

## Interpretation of Learning

The learning interpretation is therefore:

> When drone $a$ observes that acting drone $i$ engaged target $t$ and obtained reward $r^{(i)}_t$, drone $a$ uses that event to refine its own local belief about the latent interaction between drone $i$ and target $t$.

So drone $a$ updates:

- the local latent row $P^{(a)}_i$,
- the local latent vector $U^{(a)}_t$.

If $i=a$, this is drone $a$'s self-update.  
If $i \neq a$, this is drone $a$'s update based on another drone's public action.

This means that every drone is reconstructing its own local version of the hidden interaction matrix from the same public evidence stream.

Collaboration therefore occurs through **common observable experience**, while decentralization is preserved because no hidden states are exchanged.

---

# Step 6: Online Learning and Control Loop

The overall loop for drone $a$ is:

1. observe the current environment state and public swarm trace,
2. use $P^{(a)}_a$ and $U^{(a)}$ to score active targets,
3. select one valid action,
4. observe new public engagement outcomes,
5. convert each public event into a local MF training sample,
6. update $P^{(a)}$ and $U^{(a)}$ using SGD,
7. repeat at the next step.

Thus, the recommendation-style model is embedded inside a sequential decentralized control process.

This is not a static recommender.  
It is an **online decentralized latent-utility learner**.

---

# Reward Interpretation

The policy itself does not define the reward.  
The reward is defined by the environment.

However, for the learned latent scores to be meaningful, the reward should encode some notion of engagement effectiveness.

A general interpretation is:

$$
r^{(a)}_t = \text{observed effectiveness of drone } a \text{ on target } t
$$

If rewards reflect damage efficiency, target neutralization contribution, or another operational utility measure, then the latent predictions inherit that semantic meaning.

Thus, the model does not estimate preference; it estimates expected operational effectiveness under the environment's reward contract.


---

## Reward Interpretation

The policy itself does not define the reward. The reward is defined by the environment.

However, for the learned latent scores to be meaningful, the reward should encode some notion of engagement effectiveness.

A general interpretation is:

$$
r^{(a)}_t = \text{observed effectiveness of drone } a \text{ on target } t
$$

A natural reward contract consistent with this interpretation is:

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

where:

- $h^{\text{before}}_{t,k}$ = remaining health of target $t$ in attribute $k$ before the hit
- $w_{a,k}$ = drone $a$'s weapon damage in attribute $k$

The numerator measures the **actual effective damage dealt**, since damage beyond the remaining target health is wasted.  
The denominator measures the drone's **maximum possible weapon output**.

Therefore, for a valid shot on an active target:

$$
r^{(a)}_t \in [0,1]
$$

with interpretation:

- $r^{(a)}_t = 1$ = the drone used its full weapon potential effectively
- $r^{(a)}_t \approx 0$ = the engagement produced little useful effect
- intermediate values = partial effectiveness

Under this reward contract, the latent model learns to predict expected **damage efficiency** rather than abstract preference.


---

# How the Environment Connects to the Policy

The environment is responsible for supplying the information needed to drive the latent-learning cycle.

## Environment-Side Requirements

At minimum, the environment must define:

- the number of drones $N_d$,
- the number of targets $N_t$,
- the discrete action interface,
- and a public stream of observable interaction outcomes.

## Action Interface

Each drone returns one valid discrete action:

- `0` = NoOp
- `1..N_t` = fire at target index

## Learning Interface

After actions are executed, the environment must expose enough public information to construct interaction events of the form:

$$
(b,t,r^{(b)}_t)
$$

These events are the decentralized equivalent of observed ratings in collaborative filtering.

Once such events are available, each drone can locally perform the MF prediction-error update cycle.

---

# Final Interpretation

This policy can be summarized as follows:

> Each drone behaves like a decentralized collaborative matrix-factorization learner. It does not know the true target class, hidden target attributes, or true drone-target compatibility. Instead, it maintains a local latent model over all drones and all targets, updates that model from publicly observable swarm interaction events, and chooses actions using its own row within that local latent model.

This formulation preserves the core logic of classical matrix factorization while adapting it to the decentralized, sequential, and zero-knowledge constraints of ZK-MRTA.

Its main properties are:

- it preserves the user-item factorization structure,
- it replaces preference with operational utility,
- it learns online from sparse experience,
- it remains decentralized,
- it requires no direct communication,
- and it produces independent per-drone action selection from a collaboratively informed local model.

---

## Recommended Hyperparameters and Initialization Defaults

The core matrix-factorization policy is defined in symbolic form above.  
For implementation consistency, the following defaults are recommended as a practical baseline.

These values are not mathematically mandatory, but they provide a stable starting point for online decentralized learning in the present setting.

### Latent Dimension

Let:

$$
d \in \mathbb{N}
$$

denote the latent dimension used in each local factorization model.

A recommended default is:

$$
d = 8
$$

A practical tuning range is:

$$
d \in [4,16]
$$

with interpretation:

- smaller values may underfit the hidden compatibility structure,
- larger values may represent more nuanced structure,
- but overly large values may increase noise sensitivity and overfitting under sparse interaction data.

Thus, the default recommendation is to begin with a **small-to-moderate latent dimension** rather than a large expressive model.

### Learning Rate

Let:

$$
\eta > 0
$$

denote the SGD learning rate.

A recommended default is:

$$
\eta = 0.01
$$

A practical tuning range is:

$$
\eta \in [0.001,\,0.05]
$$

with interpretation:

- lower values are more stable but slower,
- higher values adapt more quickly,
- but values that are too large may cause oscillation or divergence in the online update process.

For this reason, a conservative learning rate is preferred as the baseline.

### Regularization Coefficient

Let:

$$
\lambda > 0
$$

denote the L2 regularization coefficient.

A recommended default is:

$$
\lambda = 0.02
$$

A practical tuning range is:

$$
\lambda \in [10^{-4},\,10^{-1}]
$$

with interpretation:

- smaller values allow more flexible fitting,
- larger values suppress large latent vectors more aggressively,
- and excessive regularization may cause underfitting.

In sparse online interaction settings, some regularization is strongly recommended in order to prevent unstable growth of latent factors.

### Exploration Rate

If an exploration policy is used, let:

$$
\varepsilon_s \in [0,1]
$$

denote the exploration probability at decision step $s$.

A recommended default schedule is:

$$
\varepsilon_{\text{start}} = 0.20
\qquad
\varepsilon_{\min} = 0.02
$$

with linear decay over the early phase of training.

This gives the practical rule:

- begin with moderate exploration,
- reduce exploration gradually as experience accumulates,
- and retain a small nonzero exploration floor.

This prevents the policy from becoming prematurely greedy before the latent estimates are informative.

---

## Initialization Strategy

At initialization, all entries of $P^{(a)}$ and $U^{(a)}$ should be drawn independently from a small zero-centered distribution.

A recommended default is:

$$
P^{(a)}_{b,k} \sim \mathcal{N}(0,\,0.01^2)
\qquad\text{and}\qquad
U^{(a)}_{k,t} \sim \mathcal{N}(0,\,0.01^2)
$$

for all valid indices.

Equivalently, an implementation may use a small symmetric uniform initialization such as:

$$
P^{(a)}_{b,k},\,U^{(a)}_{k,t} \sim \mathrm{Uniform}(-0.05,\,0.05)
$$

provided the scale remains small.

The purpose of this recommendation is:

- to break symmetry between latent factors,
- to keep early predicted scores near zero,
- and to avoid large unstable updates at the beginning of learning.

Thus, the policy should not use large random initial values.  
A **small zero-centered initialization** is the intended baseline.

---

## Recommended Exploration Schedule

At decision step $s$, drone $a$ selects its action as follows:

- with probability $\varepsilon_s$, choose a random valid target,
- with probability $1-\varepsilon_s$, choose the valid target with highest predicted score.

A recommended linear schedule is:

$$
\varepsilon_s
=
\max
\left(
\varepsilon_{\min},
\;
\varepsilon_{\text{start}}
-
\left(
\varepsilon_{\text{start}}-\varepsilon_{\min}
\right)
\frac{s}{S_{\text{decay}}}
\right)
$$

for

$$
s \le S_{\text{decay}}
$$

and

$$
\varepsilon_s = \varepsilon_{\min}
\qquad\text{for}\qquad
s > S_{\text{decay}}
$$

where:

- $\varepsilon_{\text{start}} = 0.20$,
- $\varepsilon_{\min} = 0.02$,
- and $S_{\text{decay}}$ is the number of steps over which exploration is reduced.

A practical default is to set $S_{\text{decay}}$ to approximately the first one-half of the expected training horizon.

The intended interpretation is:

- early in learning, exploration is useful because latent estimates are still unreliable,
- later in learning, the policy should increasingly exploit its learned utility model,
- but a small exploration floor is retained so that the policy can continue correcting mistaken beliefs.

If exploration is not desired, one may set:

$$
\varepsilon_s = 0
\qquad \forall s
$$

and recover the purely greedy baseline.

---

## Practical Default Summary

For a baseline implementation, the following default choices are recommended:

- $d = 8$
- $\eta = 0.01$
- $\lambda = 0.02$
- initialize all latent entries from $\mathcal{N}(0,\,0.01^2)$
- use $\varepsilon$-greedy exploration with:
  - $\varepsilon_{\text{start}} = 0.20$
  - $\varepsilon_{\min} = 0.02$
  - linear decay over the first half of training

These defaults are intended to make the policy **concrete, reproducible, and stable** while remaining fully consistent with the classical matrix-factorization interpretation developed above.