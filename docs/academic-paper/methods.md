# Methods

This section presents the decision methods evaluated in the ZK-MRTA framework. Its purpose is to describe how agents transform limited observations into task-selection actions under the zero-knowledge assumptions, while leaving simulator configuration, parameter settings, and benchmark design to the Experiments section.

In this study, a strategy is understood as an algorithm that processes observations, maintains internal state when needed, and outputs task-allocation decisions. The methods considered here therefore differ primarily in two dimensions: the information used to estimate task utility, and the rule used to convert that estimate into an action.

A common interface is used for all policies. At each decision step, a policy receives per-agent observations and returns one action for each agent. Learning-based methods may additionally update internal state after observing the consequences of the previous step, whereas non-learning baselines leave this update stage empty. This unified interface is important because it allows different policies to be compared under identical information constraints and interaction mechanics.

---

## 1. Methodological Framing: Estimation and Decision

The methodological logic of this section is to separate each policy into an estimation component and a decision component. The estimation component determines what the agent knows, infers, or assumes about the utility of assigning itself to a target. The decision component then maps that information into a concrete action.

This distinction is especially useful in ZK-MRTA, because the problem is defined precisely by the absence of prior knowledge, explicit communication, and direct access to hidden compatibility structure. As a result, different methods can be interpreted as alternative responses to the same core difficulty: how to act effectively when the relevant agent-task relationships are latent and must be inferred indirectly.

Within this framing, the random baseline uses no estimation at all, the oracle benchmark assumes full privileged knowledge, and the collaborative-filtering method constructs local utility estimates online from observed interaction outcomes. This progression creates a meaningful spectrum between ignorance, idealized complete knowledge, and decentralized learning under strict ZK constraints.


## 2. Random Baseline

The random policy serves as the weakest baseline and lower-bound reference. It is fully ZK-compliant: it does not use target hit points, hidden target types, latent compatibility values, or any history of previous interactions. Instead, it only reads the visible active/inactive flag of each target from the observation and samples uniformly from the available actions.

Depending on configuration, the valid action set may include a no-operation action in addition to all currently active targets. Because the policy has no memory and performs no learning, it treats all active targets as equivalent at every step.

Formally, let $\mathcal{A}_t(i)$ denote the set of valid actions available to drone $i$ at time $t$, consisting of the indices of active targets and, optionally, a NoOp action. The random policy samples:

$$a_t(i) \sim \text{Uniform}(\mathcal{A}_t(i))$$

This baseline is intentionally naive. Its value lies not in performance, but in providing a clean reference for behavior in the absence of estimation, memory, or coordination.


## 3. Oracle Benchmark

The oracle policy provides an upper-bound benchmark by assuming privileged access to information unavailable under the ZK assumptions. In the implemented version, the oracle is an HP-aware marginal-allocation planner that uses true latent drone vectors, true target latent vectors, and current target hit points. It is therefore not ZK-compliant and is used only as an idealized comparison point.

Unlike a one-to-one assignment rule, the oracle explicitly allows multiple drones to focus fire on the same target when doing so is beneficial. Its scoring rule combines several factors: effective damage capped by remaining target HP, a bottleneck weight that prioritizes difficult targets, a finish bonus for assignments that eliminate a target in the current step, an overkill penalty that discourages waste, and a future-value term that favors starting progress on hard targets earlier. The planner then performs a greedy marginal-allocation pass, assigning one drone at a time to the currently best drone-target pair until no beneficial assignment remains.

Let $d_{ij}$ denote the damage drone $i$ could inflict on target $j$, and let $h_j$ denote the current remaining HP of target $j$. The oracle evaluates candidate assignments using a marginal score of the general form:

$$S(i,j) = \min(d_{ij}, h_j) + \text{future\_value}(j) + \text{finish\_bonus}(i,j) - \text{overkill\_penalty}(i,j) - \text{waste\_term}(i,j)$$

then repeatedly selects the best remaining pair. In this way, the oracle approximates globally efficient short-horizon coordination while retaining a computationally simple greedy structure. Because it relies on privileged state, its purpose is not to model realistic deployment but to indicate the performance ceiling against which decentralized methods can be interpreted.


## 4. Decentralized Collaborative-Filtering Policy

The main method investigated in this research is a decentralized matrix-factorization policy inspired by classical collaborative filtering. The key conceptual adaptation is that collaborative filtering is not used here to estimate user preference, but to estimate operational utility: the expected effectiveness of assigning a drone to a target. Drones play the role of users, targets play the role of items, and observed engagement outcomes replace ratings. Because the problem is sequential, decentralized, and observation-limited, the recommender mechanism is repurposed as an online decision policy rather than a centralized offline prediction model.

Classical matrix factorization represents users and items through latent vectors and predicts missing interactions via their dot product. In the present setting, this idea is adapted so that each drone independently maintains a private local model of the interaction space. There is no centralized shared factorization and no direct exchange of learned parameters across drones. Collaboration arises only through publicly observable interaction events. This distinction is essential: the method is collaborative in inspiration, but decentralized in implementation.

For agent $a$, the learned model consists of a drone embedding matrix:

$$P^{(a)} \in \mathbb{R}^{m \times d_f}$$

and a target embedding matrix:

$$U^{(a)} \in \mathbb{R}^{d_f \times n}$$

where $m$ is the number of drones, $n$ is the number of targets, and $d_f$ is the factorization dimension (a hyperparameter distinct from the environment's latent dimension $d$). The predicted utility assigned by agent $a$ to drone $i$ engaging target $j$ is:

$$\hat{r}_{ij}^{(a)} = (P_{i,:}^{(a)})^\top U_{:,j}^{(a)}$$

In the implemented policy, each drone maintains its own local matrices $P$ and $U$, initialized with entries drawn independently from $\mathcal{N}(0,\, 0.01^2)$ and updated online from observed swarm interaction events.

This method is particularly well suited to ZK-MRTA because the relevant interaction matrix is not given in advance. Agents begin with no knowledge of true compatibility and must build an interaction structure gradually from sparse experience. The matrix-factorization policy addresses this by learning latent representations from observed rewards, thereby estimating hidden structure that cannot be read directly from the environment.

---
## 5. Prediction, Action Selection, and Online Learning

At decision time, each drone uses only its own latent row $P_{a,:}^{(a)}$ to evaluate currently active targets. Action selection follows an $\varepsilon$-greedy mechanism. With probability $\varepsilon$, the agent explores by selecting a random active target. Otherwise, it exploits by selecting the target with the highest predicted utility. The implementation also supports an optional Boltzmann-style alternative in the exploitation branch: when selection noise is positive, targets are sampled according to a softmax distribution over predicted utilities rather than by pure argmax. After each action, the exploration rate decays multiplicatively until it reaches a predefined floor.

Formally, if $\mathcal{T}_t(a)$ is the set of active targets observed by agent $a$ at time $t$, then exploitation selects:

$$a_t(a) = \arg\max_{j \in \mathcal{T}_t(a)} \hat{r}_{aj}^{(a)}$$

while exploration samples uniformly from $\mathcal{T}_t(a)$. When softmax selection is enabled, the policy instead samples according to:

$$\Pr(a_t(a) = j) \propto \exp\left(\frac{\hat{r}_{aj}^{(a)}}{\tau}\right)$$

where $\tau > 0$ is the selection-noise temperature (corresponding to the `selection_noise` hyperparameter). As $\tau \to 0$ the distribution converges to argmax; as $\tau$ increases, selection becomes increasingly uniform. This option is useful because decentralized agents may otherwise collapse onto the same apparently best target, whereas a softer selection rule can reduce contention without introducing explicit communication.
### 5.1 Learning from Shared Interaction Data

Learning occurs after the environment reveals public information about the previous step. The technical design assumes that each agent can observe selected targets and the corresponding reward signals, even though the hidden compatibility structure remains inaccessible. Two supervision modes are supported:

- **Direct mode**: The policy compares its predicted utility for each observed drone-target event directly to the corresponding observed reward. All events — including those producing negative anti-signals from shots on inactive targets — are used for gradient updates, with negative rewards downweighted by the anti-signal coefficient.
- **Integration-matrix mode**: The policy first updates a running-mean interaction matrix and then uses that accumulated estimate as the supervision target. Crucially, in this mode the validity mask $\mathbf{w}_{t-1}$ acts as an eligibility gate: if the targeted task was already inactive at the time of engagement, the entire event is **skipped** (no accumulation, no SGD update), rather than being downweighted. Only interactions with active targets contribute to the running mean, producing a cleaner supervision signal. This is especially useful when rewards are noisy or when dead-target penalties would otherwise distort the learned compatibility estimates.

Let the prediction error for an observed event $(i,j)$ be denoted by:

$$e_{ij}^{(a)} = \hat{r}_{ij}^{(a)} - y_{ij}$$

where $y_{ij}$ is either the immediate observed reward or the current running-mean entry of the integration matrix. The local embeddings are then updated by stochastic gradient descent with $L_2$ regularization:

$$P_{i,:}^{(a)} \leftarrow P_{i,:}^{(a)} - \eta \left( 2 e_{ij}^{(a)} U_{:,j}^{(a)} + \lambda P_{i,:}^{(a)} \right)$$

$$U_{:,j}^{(a)} \leftarrow U_{:,j}^{(a)} - \eta \left( 2 e_{ij}^{(a)} P_{i,:}^{(a)} + \lambda U_{:,j}^{(a)} \right)$$

where $\eta$ is the learning rate and $\lambda$ is the regularization coefficient. Negative rewards may additionally be down-weighted through an anti-signal parameter, allowing the method to reduce the influence of wasted shots while still preserving their corrective effect.

An important property of the implemented method is that learning persists across episodes. The policy's soft reset preserves the learned embedding matrices rather than reinitializing them, so knowledge accumulated in earlier episodes continues to shape future decisions. This makes the training process cumulative, which is appropriate for a method intended to model gradual knowledge acquisition under repeated interaction.

---
## 6. Summary of the Compared Methods

The three methods examined here occupy distinct positions on the knowledge spectrum:

- **Random baseline**: Performs no estimation and represents behavior under complete strategic ignorance
- **Oracle benchmark**: Assumes privileged access to hidden structure and approximates an idealized upper bound on achievable coordination
- **Decentralized matrix-factorization policy**: Lies between these two extremes—it begins without prior knowledge, learns latent utility estimates from sparse public outcomes, and uses those estimates to guide future action selection

Together, these methods provide a coherent basis for evaluating whether collaborative-filtering-inspired utility learning can close part of the gap between naive decentralized behavior and an idealized planner.