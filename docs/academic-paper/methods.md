# 4. Methods

## 4.1 Methodological Framing: Estimation and Decision

The methods in this paper can be understood through two components: estimation and decision. The estimation component determines what an agent knows, infers, or assumes about the utility of assigning itself to a target. The decision component then maps that information into a concrete action.

This distinction is especially useful in ZK-MRTA, because the problem is defined precisely by the absence of prior knowledge, explicit communication, and direct access to hidden compatibility structure. As a result, different methods can be interpreted as alternative responses to the same core difficulty: how to act effectively when the relevant agent-task relationships are latent and must be inferred indirectly.

Within this framing, the random baseline uses no estimation at all, the oracle benchmark assumes full privileged knowledge, and the collaborative-filtering method constructs local utility estimates online from observed interaction outcomes. This progression creates a meaningful spectrum between ignorance, idealized complete knowledge, and decentralized learning under strict ZK constraints.

## 4.2 Random Baseline

The random policy serves as the weakest baseline and lower-bound reference. It is fully ZK-compliant: it does not use target hit points, hidden target types, latent compatibility values, or any history of previous interactions. Instead, it reads only the visible active/inactive status of each target and samples uniformly from the valid actions.

Depending on configuration, the valid action set may include a no-operation action in addition to all currently active targets. Because the policy has no memory and performs no learning, it treats all active targets as equivalent at every step.

Formally, let $\mathcal{U}_t(a_i)$ denote the set of valid actions available to agent $a_i$ at time $t$, consisting of the indices of active targets and, optionally, a NoOp action. The random policy samples:

$$a_t(a_i) \sim \text{Uniform}(\mathcal{U}_t(a_i))$$

This baseline is intentionally naive. Its value lies not in performance, but in providing a clean reference for behavior in the absence of estimation, memory, or coordination.

## 4.3 Oracle Benchmark

The oracle policy provides an upper-bound benchmark by assuming privileged access to information unavailable under the ZK assumptions. In the implemented version, the oracle is an HP-aware marginal-allocation planner that uses true latent drone vectors, true target latent vectors, and current target hit points. It is therefore not ZK-compliant and is used only as an idealized comparison point.

Unlike a one-to-one assignment rule, the oracle explicitly allows multiple drones to focus fire on the same target when doing so is beneficial. Its scoring rule combines several factors: effective damage capped by remaining target HP, a bottleneck weight that prioritizes difficult targets, a finish bonus for assignments that eliminate a target in the current step, an overkill penalty that discourages waste, and a future-value term that favors starting progress on hard targets earlier. The planner then performs a greedy marginal-allocation pass, assigning one drone at a time to the currently best drone-target pair until no beneficial assignment remains.

The oracle approximates globally efficient short-horizon coordination while retaining a computationally simple greedy structure: it evaluates all candidate drone-target pairs using a composite marginal score that combines these factors, then repeatedly selects the best remaining pair. Because it relies on privileged state, its purpose is not to model realistic deployment but to indicate the performance ceiling against which decentralized methods can be interpreted.

## 4.4 Decentralized Collaborative-Filtering Policy

The main method investigated in this paper is a decentralized matrix-factorization policy inspired by classical collaborative filtering. The key conceptual adaptation is that collaborative filtering is not used here to estimate user preference, but to estimate operational utility: the expected effectiveness of assigning a drone to a target. Drones play the role of users, targets play the role of items, and observed engagement outcomes replace ratings. Because the problem is sequential, decentralized, and observation-limited, the recommender mechanism is repurposed as an online decision policy rather than a centralized offline prediction model.

Classical matrix factorization represents users and items through latent vectors and predicts missing interactions via their dot product. In the present setting, this idea is adapted so that each drone independently maintains a private local model of the interaction space. There is no centralized shared factorization and no direct exchange of learned parameters across drones. Collaboration arises only through shared observations of interaction outcomes. This distinction is essential: the method is collaborative in inspiration, but decentralized in implementation.

For an agent $a_i$, the learned model consists of a drone embedding matrix:

$$P^{(a_i)} \in \mathbb{R}^{m \times d_f}$$

and a target embedding matrix:

$$U^{(a_i)} \in \mathbb{R}^{d_f \times n}$$

where $m$ is the number of drones, $n$ is the number of targets, and $d_f$ is the factorization dimension (a hyperparameter distinct from the environment's latent dimension $d$). Agent $a_i$'s predicted utility for the interaction between drone $a_k$ and target $t_j$ is:

$$\hat{r}_{kj}^{(a_i)} = (P_{k,:}^{(a_i)})^\top U_{:,j}^{(a_i)}$$

where $a_k$ denotes the acting drone whose interaction with target $t_j$ is being evaluated. In the implemented policy, each drone maintains its own local matrices $P^{(a_i)}$ and $U^{(a_i)}$, initialized with entries drawn independently from $\mathcal{N}(0,\, 0.01^2)$ and updated online from observed swarm interaction events.

This method is particularly well suited to ZK-MRTA because the relevant interaction matrix is not given in advance. Agents begin with no knowledge of true compatibility and must build an interaction structure gradually from sparse experience. The matrix-factorization policy addresses this by learning latent representations from observed rewards, thereby estimating hidden structure that cannot be read directly from the environment.

## 4.5 Action Selection and Online Update Rules

At decision time, each agent $a_i$ uses only its own latent row $P_{i,:}^{(a_i)}$ to evaluate currently active targets. Action selection follows an $\varepsilon$-greedy mechanism. With probability $\varepsilon$, the agent explores by selecting a random active target. Otherwise, it exploits by selecting the target with the highest predicted utility. After each action, the exploration rate decays multiplicatively until it reaches a predefined floor.

Formally, if $\mathcal{T}_t(a_i)$ is the set of active targets observed by agent $a_i$ at time $t$, then exploitation selects:

$$a_t(a_i) = \arg\max_{j \in \mathcal{T}_t(a_i)} \hat{r}_{ij}^{(a_i)}$$

while exploration samples uniformly from $\mathcal{T}_t(a_i)$.

### 4.5.1 Learning from Shared Interaction Data

Learning occurs after the environment reveals public information about the previous step. Each agent observes the selected targets and corresponding reward signals, even though the hidden compatibility structure remains inaccessible. Two supervision modes are supported:

- **Direct mode**: The policy compares its predicted utility for each observed drone-target event directly to the corresponding observed reward. All events — including those producing negative anti-signals from shots on inactive targets — are used for gradient updates, with negative rewards downweighted by the anti-signal coefficient.
- **Integration-matrix mode**: The policy first updates a running-mean interaction matrix and then uses that accumulated estimate as the supervision target. Crucially, in this mode the validity mask $\mathbf{w}_{t-1}$ acts as an eligibility gate: if the targeted task was already inactive at the time of engagement, the entire event is **skipped** (no accumulation, no SGD update), rather than being downweighted. Only interactions with active targets contribute to the running mean, producing a cleaner supervision signal. This is especially useful when rewards are noisy or when dead-target penalties would otherwise distort the learned compatibility estimates.

Because all interaction outcomes are publicly observable, each agent $a_k$ updates its local model for every observed drone-target event $(a_i, t_j)$, not only its own engagements. Let the prediction error for an observed event $(a_i, t_j)$, as computed by agent $a_k$, be denoted by:

$$e_{ij}^{(a_k)} = \hat{r}_{ij}^{(a_k)} - y_{ij}$$

where $y_{ij}$ is either the immediate observed reward or the current running-mean entry of the integration matrix. The local embeddings are then updated by stochastic gradient descent with $L_2$ regularization:

$$P_{i,:}^{(a_k)} \leftarrow P_{i,:}^{(a_k)} - \eta \left( 2 e_{ij}^{(a_k)} U_{:,j}^{(a_k)} + \lambda P_{i,:}^{(a_k)} \right)$$

$$U_{:,j}^{(a_k)} \leftarrow U_{:,j}^{(a_k)} - \eta \left( 2 e_{ij}^{(a_k)} P_{i,:}^{(a_k)} + \lambda U_{:,j}^{(a_k)} \right)$$

where $\eta$ is the learning rate and $\lambda$ is the regularization coefficient. In direct mode, negative rewards from wasted shots (engagements with inactive targets) are down-weighted by a fixed scalar factor before entering the gradient update, reducing their influence while still preserving their corrective effect.

An important property of the implemented method is that learning persists across episodes. Episode boundaries do not reinitialize the learned embedding matrices, so knowledge accumulated in earlier episodes continues to shape future decisions. The resulting training process is therefore cumulative, which is appropriate for a method intended to model gradual knowledge acquisition under repeated interaction.

## 4.6 Summary of the Compared Methods

The three methods examined here occupy distinct positions on the knowledge spectrum:

- **Random baseline**: Performs no estimation and represents behavior under complete strategic ignorance
- **Oracle benchmark**: Assumes privileged access to hidden structure and approximates an idealized upper bound on achievable coordination
- **Decentralized matrix-factorization policy**: Lies between these two extremes; it begins without prior knowledge, learns latent utility estimates from sparse public outcomes, and uses those estimates to guide future action selection

Together, these methods provide a coherent basis for evaluating whether collaborative-filtering-inspired utility learning can close part of the gap between naive decentralized behavior and an idealized planner.
