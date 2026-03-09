# High-Level Flow of the Collaborative Filtering Matrix Factorization Policy

# Initial Phase: Latent Vector Initialization

The policy begins by assuming that hidden compatibility between drones and targets can be represented in a low-dimensional latent space.

We define two latent matrices:

$$
P \in \mathbb{R}^{N_d \times d}
$$

$$
U \in \mathbb{R}^{d \times N_t}
$$

Where:

- $N_d$ = number of drones
- $N_t$ = number of targets
- $d$ = number of latent factors
- $P$ = drone latent factor matrix
- $U$ = target latent factor matrix

The predicted interaction matrix is:

$$
\hat{I} = P U
$$

Where:

- $\hat{I} \in \mathbb{R}^{N_d \times N_t}$ is the predicted drone-target score matrix

At initialization, the true latent vectors are unknown, so all entries in $P$ and $U$ are set to **small random values**.

Conceptually:

- each drone starts at a random point in latent space
- each target starts at a random point in latent space
- these positions do not yet have learned meaning

Why small random values matter:

- **Randomness breaks symmetry** so all drones and targets do not start identically
- **Small values keep early predictions stable** and help learning begin smoothly

So the purpose of the initial phase is:

- create the latent representation
- start with no prior knowledge
- prepare the model for learning from future interaction outcomes

# Step 2: Prediction / Scoring

After initialization, the policy uses the latent vectors to estimate how good each drone-target interaction is.

For a drone $a$ and a target $t$, the predicted score is:

$$
\hat{r}_{a,t} = p_a^\top u_t
$$

Where:

- $p_a \in \mathbb{R}^{d}$ = latent vector of drone $a$
- $u_t \in \mathbb{R}^{d}$ = latent vector of target $t$
- $\hat{r}_{a,t}$ = predicted score for the interaction between drone $a$ and target $t$

This score represents the model’s current estimate of how effective or useful it would be for drone $a$ to attack target $t$.

If this is computed for all drones and all targets together, the result is the full predicted interaction matrix:

$$
\hat{I} = P U
$$

Where:

- $P \in \mathbb{R}^{N_d \times d}$ = drone latent matrix
- $U \in \mathbb{R}^{d \times N_t}$ = target latent matrix
- $\hat{I} \in \mathbb{R}^{N_d \times N_t}$ = predicted drone-target score matrix

Each entry $\hat{I}_{a,t}$ is the predicted score of drone $a$ for target $t$.

The purpose of this step is to turn the latent representation into estimated utilities that can later be used for ranking and action selection.

# Step 3: Action Selection

After computing predicted scores, the policy must choose one target as the action.

For a given drone, the policy:

1. considers only the currently active targets
2. looks at their predicted scores
3. selects one target

In the most classic recommendation-style version, the action rule is:

$$
a^* = \arg\max_{t \in \mathcal{A}} \hat{r}_{a,t}
$$

Where:

- $\mathcal{A}$ = set of currently active targets
- $\hat{r}_{a,t}$ = predicted score of drone $a$ for target $t$
- $a^*$ = selected target

This means the drone chooses the active target with the highest predicted score.

Optionally, the policy can use $\varepsilon$-greedy exploration:

- with probability $1-\varepsilon$, choose the highest-scoring active target
- with probability $\varepsilon$, choose a random active target

The purpose of this step is to convert predicted utilities into an actual decision that the drone can execute in the environment.

# Step 4: Environment Feedback / Observed Outcome

After the drone selects and executes an action, the environment returns the actual result of that interaction.

For a chosen drone-target pair, the policy receives an observed outcome:

$$
r_{a,t}
$$

Where:

- $r_{a,t}$ = observed reward or outcome for drone $a$ interacting with target $t$

This value represents what actually happened in the environment after the action was taken.

Conceptually:

- the model predicts a score
- the drone chooses a target
- the environment returns the real outcome

The purpose of this step is to provide the learning signal that will later be compared against the predicted score.

So this step turns the selected interaction into an observed training example.

# Step 5: Model Update / Learning

This is the phase where the latent space is learned.

For an observed interaction between drone $a$ and target $t$, the model first predicts:

$$
\hat{r}_{a,t} = p_a^\top u_t
$$

Where:

- $p_a \in \mathbb{R}^{d}$ = latent vector of drone $a$
- $u_t \in \mathbb{R}^{d}$ = latent vector of target $t$
- $\hat{r}_{a,t}$ = predicted outcome

After the environment returns the actual observed outcome $r_{a,t}$, the model computes the prediction error:

$$
e_{a,t} = \hat{r}_{a,t} - r_{a,t}
$$

## Loss Function

The model learns by minimizing squared prediction error over the observed interactions:

$$
\mathcal{L}_{\text{data}}(P,U)=\sum_{(a,t)\in\Omega}\left(p_a^\top u_t-r_{a,t}\right)^2
$$

Where:

- $\Omega$ = set of observed drone-target interactions

To keep latent vectors from growing too large, regularization is added:

$$
\mathcal{L}(P,U)=\sum_{(a,t)\in\Omega}\left(p_a^\top u_t-r_{a,t}\right)^2+\lambda\left(\|P\|_F^2+\|U\|_F^2\right)
$$

Where:

- $\lambda$ = regularization strength
- $\|P\|_F^2$ = sum of squares of all entries in $P$
- $\|U\|_F^2$ = sum of squares of all entries in $U$

## SGD Update Rule

The model uses Stochastic Gradient Descent (SGD), which means it updates the latent vectors one observed interaction at a time.

For one observed pair $(a,t)$, the gradients are:

$$
\frac{\partial \mathcal{L}}{\partial p_a} = 2 e_{a,t} u_t + \lambda p_a
$$

$$
\frac{\partial \mathcal{L}}{\partial u_t} = 2 e_{a,t} p_a + \lambda u_t
$$

The gradient descent updates are:

$$
p_a \leftarrow p_a - \gamma \left(2 e_{a,t} u_t + \lambda p_a\right)
$$

$$
u_t \leftarrow u_t - \gamma \left(2 e_{a,t} p_a + \lambda u_t\right)
$$

Where:

- $\gamma$ = learning rate

## Intuition

- If the observed outcome is higher than predicted, the update increases compatibility between the drone and target vectors
- If the observed outcome is lower than predicted, the update decreases compatibility
- Repeating this over many observed interactions gradually organizes the latent space into meaningful patterns

So this step is the core learning mechanism: it reshapes the latent vectors so future predictions better match observed outcomes.

## How the Environment Plugs Into the 5 Steps

### Step 1: Initialization

This step happens inside the policy, not inside the environment.

The environment provides the structural information needed to initialize the latent model:

- the number of drones
- the number of targets
- the action space size

This allows the policy to create:

- one latent vector per drone
- one latent vector per target

So the environment defines the problem dimensions, and the policy initializes the latent matrices accordingly. 

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

The active flag is what matters most here, because the policy should score only targets that are still valid candidates. 

So in this step, the environment supplies the current candidate set, and the policy computes predicted scores for the active targets.

---

### Step 3: Action Selection

After scoring the active targets, the policy must return one valid environment action.

The environment expects a discrete action per drone:

- `0` = NoOp
- `1..N` = fire at target index (1-indexed)

So the policy takes its internal target choice and converts it into the environment action format. 

This means:

- internally, the policy selects the best target
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

For the learning cycle, the most important output is:

- `rewards[agent_id]`

This is the actual scalar outcome of the action that was just taken. It is the direct feedback signal for the selected drone-target interaction. 

So this step is where the environment provides the real result that can be compared against the prediction.

---

### Step 5: Model Update / Learning

The policy uses the environment’s returned reward to build an observed interaction:

$$
(a, t, r_{a,t})
$$

Where:

- $a$ = drone index
- $t$ = selected target index
- $r_{a,t}$ = reward returned by the environment for that drone

This observed interaction becomes the training sample used in the latent-factor update step. 

In a faithful classic-MF-style policy, the main learning signal should come from:

- the chosen target
- the direct reward returned by `step()`

The observation fields:

- `selected_targets`
- `observed_rewards`

can be treated as optional collaborative context, but they are not required for the most classical version of the policy. 

---

## Summary Mapping

- **Step 1:** environment provides problem dimensions
- **Step 2:** environment provides current valid targets through `targets`
- **Step 3:** policy returns a valid discrete action to the environment
- **Step 4:** environment returns the actual reward for the chosen action
- **Step 5:** policy uses that reward as the observed signal for learning

So the environment plugs into the CF cycle by providing:

- the candidate set for scoring
- the action interface for execution
- the reward signal for learning

## Reward Signal

For the classic MF-style policy, the recommended reward signal is:

$$
r_{a,t} =
\begin{cases}
-1 & \text{if drone } a \text{ fires at an already inactive target} \\
0 & \text{if drone } a \text{ chooses NoOp} \\
\frac{\text{actual damage dealt}}{\text{max weapon potential of drone } a} & \text{otherwise}
\end{cases}
$$

Where:

- $r_{a,t}$ = observed reward for drone $a$ on target $t$
- actual damage dealt = total damage truly absorbed by the target after the hit
- max weapon potential of drone $a$ = total damage the drone could deliver across all attributes in a single shot

For a valid shot on an active target, this can be written as:

$$
r_{a,t} =
\frac{\sum_k \min\left(h^{\text{before}}_{t,k},\; w_{a,k}\right)}
{\sum_k w_{a,k}}
$$

Where:

- $h^{\text{before}}_{t,k}$ = target $t$ remaining HP in attribute $k$ before the hit
- $w_{a,k}$ = drone $a$ weapon damage in attribute $k$

This reward signal is chosen because it reflects the real outcome of the interaction while remaining normalized and easy to use as the observed utility value in a matrix-factorization learning policy.