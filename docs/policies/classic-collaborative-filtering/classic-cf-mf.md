# High-Level Flow of the Decentralized Collaborative Filtering Matrix Factorization Policy

## Decentralized Policy Assumption

This policy is decentralized.

Each drone maintains its own private latent-factor model and selects actions independently.

For each drone $a$, the local model contains:

- a private drone latent vector
- a private target latent matrix

These latent parameters are not shared across drones.

Each drone updates its model only from information available through its own observation stream from the environment.

Therefore:

- there is no global shared latent matrix
- there is no centralized optimizer
- there is no joint swarm-level assignment step
- each drone acts according to its own learned local utility estimates

# Step 1: Latent Vector Initialization (Initial Phase)

The policy begins by assuming that hidden compatibility between drones and targets can be represented in a low-dimensional latent space.

For each drone $a$, the policy maintains a private local latent model:

$$
p^{(a)} \in \mathbb{R}^{d}
$$

$$
U^{(a)} \in \mathbb{R}^{d \times N_t}
$$

Where:

- $p^{(a)}$ = private latent vector of drone $a$
- $U^{(a)}$ = private local target latent matrix maintained by drone $a$
- $N_t$ = number of targets
- $d$ = number of latent factors

At initialization, the true latent vectors are unknown, so all entries in $p^{(a)}$ and $U^{(a)}$ are set to **small random values**.

Conceptually:

- each drone starts with its own random latent representation
- each drone starts with its own random local estimates of all targets
- these positions do not yet have learned meaning

Why small random values matter:

- **Randomness breaks symmetry** so all latent values do not start identically
- **Small values keep early predictions stable** and help learning begin smoothly

So the purpose of the initial phase is:

- create the private latent representation for each drone
- start with no prior knowledge
- prepare the model for learning from future interaction outcomes

# Step 2: Prediction / Scoring

After initialization, each drone uses its own private latent model to estimate how good each target is.

For drone $a$ and target $t$, the predicted score is:

$$
\hat{r}^{(a)}_{t} = \left(p^{(a)}\right)^\top u^{(a)}_{t}
$$

Where:

- $p^{(a)} \in \mathbb{R}^{d}$ = private latent vector of drone $a$
- $u^{(a)}_{t} \in \mathbb{R}^{d}$ = drone $a$'s private latent vector for target $t$
- $\hat{r}^{(a)}_{t}$ = drone $a$'s predicted score for target $t$

This score represents drone $a$'s current estimate of how effective or useful it would be to attack target $t$.

The purpose of this step is to turn the local latent representation into estimated utilities that can later be used for ranking and action selection.

# Step 3: Action Selection

After computing predicted scores, each drone must choose one target as its action.

For a given drone, the policy:

1. considers only the currently active targets
2. looks at their predicted scores
3. selects one target independently

In the most classic recommendation-style version, the action rule is:

$$
t_a^* = \arg\max_{t \in \mathcal{A}} \hat{r}^{(a)}_{t}
$$

Where:

- $\mathcal{A}$ = set of currently active targets
- $\hat{r}^{(a)}_{t}$ = drone $a$'s predicted score for target $t$
- $t_a^*$ = target selected by drone $a$

This means drone $a$ chooses the active target with the highest predicted score according to its own local model.

Optionally, the policy can use $\varepsilon$-greedy exploration:

- with probability $1-\varepsilon$, choose the highest-scoring active target
- with probability $\varepsilon$, choose a random active target

The purpose of this step is to convert local predicted utilities into an actual decision that the drone can execute in the environment.

# Step 4: Environment Feedback / Observed Outcome

After the drone selects and executes an action, the environment returns the actual result of that interaction.

## Reward Signal

For the decentralized classic MF-style policy, the recommended reward signal is:

$$
r^{(a)}_{t} =
\begin{cases}
-1 & \text{if drone } a \text{ fires at an already inactive target} \\
0 & \text{if drone } a \text{ chooses NoOp} \\
\frac{\text{actual damage dealt}}{\text{max weapon potential of drone } a} & \text{otherwise}
\end{cases}
$$

Where:

- $r^{(a)}_{t}$ = observed reward for drone $a$ on target $t$
- actual damage dealt = total damage truly absorbed by the target after the hit
- max weapon potential of drone $a$ = total damage the drone could deliver across all attributes in a single shot

For a valid shot on an active target, this can be written as:

$$
r^{(a)}_{t} =
\frac{\sum_k \min\left(h^{\text{before}}_{t,k},\; w_{a,k}\right)}
{\sum_k w_{a,k}}
$$

Where:

- $h^{\text{before}}_{t,k}$ = target $t$ remaining HP in attribute $k$ before the hit
- $w_{a,k}$ = drone $a$ weapon damage in attribute $k$

for drone $a$ and its chosen target $t$, the policy receives an observed outcome:

$$
r^{(a)}_{t}
$$

Where:

- $r^{(a)}_{t}$ = observed reward or outcome for drone $a$ interacting with target $t$

This value represents what actually happened in the environment after the action was taken.

Conceptually:

- the local model predicts a score
- the drone chooses a target
- the environment returns the real outcome

The purpose of this step is to provide the learning signal that will later be compared against the predicted score.

So this step turns the selected interaction into an observed training example.

# Step 5: Model Update / Learning

This is the phase where each drone learns its own private latent space.

For an observed interaction between drone $a$ and target $t$, the local model first predicts:

$$
\hat{r}^{(a)}_{t} = \left(p^{(a)}\right)^\top u^{(a)}_{t}
$$

Where:

- $p^{(a)} \in \mathbb{R}^{d}$ = private latent vector of drone $a$
- $u^{(a)}_{t} \in \mathbb{R}^{d}$ = drone $a$'s private latent vector for target $t$
- $\hat{r}^{(a)}_{t}$ = predicted outcome

After the environment returns the actual observed outcome $r^{(a)}_{t}$, the model computes the prediction error:

$$
e^{(a)}_{t} = \hat{r}^{(a)}_{t} - r^{(a)}_{t}
$$

## Loss Function

Drone $a$ learns by minimizing squared prediction error over its own observed interactions:

$$
\mathcal{L}^{(a)}_{\text{data}}=\sum_{t \in \Omega^{(a)}}\left(\left(p^{(a)}\right)^\top u^{(a)}_{t}-r^{(a)}_{t}\right)^2
$$

Where:

- $\Omega^{(a)}$ = set of drone-target interactions observed by drone $a$

To keep latent vectors from growing too large, regularization is added:

$$
\mathcal{L}^{(a)}=\sum_{t \in \Omega^{(a)}}\left(\left(p^{(a)}\right)^\top u^{(a)}_{t}-r^{(a)}_{t}\right)^2+\lambda\left(\|p^{(a)}\|^2+\|U^{(a)}\|_F^2\right)
$$

Where:

- $\lambda$ = regularization strength
- $\|p^{(a)}\|^2$ = squared norm of drone $a$'s latent vector
- $\|U^{(a)}\|_F^2$ = sum of squares of all entries in drone $a$'s private target matrix

## SGD Update Rule

The model uses Stochastic Gradient Descent (SGD), which means each drone updates its own latent vectors one observed interaction at a time.

For one observed pair $(a,t)$, the gradients are:

$$
\frac{\partial \mathcal{L}^{(a)}}{\partial p^{(a)}} = 2 e^{(a)}_{t} u^{(a)}_{t} + \lambda p^{(a)}
$$

$$
\frac{\partial \mathcal{L}^{(a)}}{\partial u^{(a)}_{t}} = 2 e^{(a)}_{t} p^{(a)} + \lambda u^{(a)}_{t}
$$

The gradient descent updates are:

$$
p^{(a)} \leftarrow p^{(a)} - \gamma \left(2 e^{(a)}_{t} u^{(a)}_{t} + \lambda p^{(a)}\right)
$$

$$
u^{(a)}_{t} \leftarrow u^{(a)}_{t} - \gamma \left(2 e^{(a)}_{t} p^{(a)} + \lambda u^{(a)}_{t}\right)
$$

Where:

- $\gamma$ = learning rate

## Intuition

- If the observed outcome is higher than predicted, the update increases compatibility between drone $a$ and target $t$ in drone $a$'s local latent space
- If the observed outcome is lower than predicted, the update decreases compatibility
- Repeating this over many observed interactions gradually organizes each drone's private latent space into meaningful patterns

So this step is the core learning mechanism: each drone reshapes its own latent vectors so future predictions better match observed outcomes.

## How the Environment Plugs Into the 5 Steps

### Step 1: Initialization

This step happens inside each drone's policy instance, not inside the environment.

The environment provides the structural information needed to initialize the local latent model:

- the number of drones
- the number of targets
- the action space size

This allows each drone to create:

- its own private latent vector
- its own private latent representation of all targets

So the environment defines the problem dimensions, and each drone initializes its own local latent model accordingly.

---

### Step 2: Prediction / Scoring

At each decision step, the environment provides an observation for each drone.

In collaborative mode, the observation contains:

- `targets`
- `selected_targets`
- `observed_rewards`

For a classic recommendation-style policy, the most important part for prediction is:

- `targets`

The `targets` field contains, for every target:

- x-position
- y-position
- active flag

The active flag is what matters most here, because each drone should score only targets that are still valid candidates.

So in this step, the environment supplies the current candidate set, and each drone computes local predicted scores for the active targets.

---

### Step 3: Action Selection

After scoring the active targets, each drone must return one valid environment action.

The environment expects a discrete action per drone:

- `0` = NoOp
- `1..N` = fire at target index (1-indexed)

So each drone takes its internal target choice and converts it into the environment action format.

This means:

- internally, the drone selects the best target according to its own local model
- externally, it returns the corresponding action expected by the environment

---

### Step 4: Environment Feedback / Observed Outcome

After receiving the actions, the environment executes them inside `step(actions)`.

The environment then returns:

- next observations
- rewards
- terminations
- truncations
- infos

For the learning cycle, the most important output for drone $a$ is:

- `rewards[agent_id]`

This is the actual scalar outcome of the action that was just taken by that drone. It is the direct feedback signal for the selected drone-target interaction.

So this step is where the environment provides the real result that can be compared against the local prediction.

---

### Step 5: Model Update / Learning

Drone $a$ uses the environment's returned reward to build an observed interaction:

$$
(a, t, r^{(a)}_{t})
$$

Where:

- $a$ = drone index
- $t$ = selected target index
- $r^{(a)}_{t}$ = reward returned by the environment for that drone

This observed interaction becomes the training sample used in drone $a$'s local latent-factor update step.

In a faithful decentralized classic-MF-style policy, the main learning signal should come from:

- the chosen target
- the direct reward returned by `step()`

The observation fields:

- `selected_targets`
- `observed_rewards`

can be treated as optional locally observed context, but they are not required for the most classical version of the policy.

---

## Summary Mapping

- **Step 1:** environment provides problem dimensions
- **Step 2:** environment provides current valid targets through `targets`
- **Step 3:** each drone returns a valid discrete action to the environment
- **Step 4:** environment returns the actual reward for the chosen action
- **Step 5:** each drone uses that reward as the observed signal for local learning

So the environment plugs into the decentralized CF cycle by providing:

- the candidate set for scoring
- the action interface for execution
- the reward signal for local learning
