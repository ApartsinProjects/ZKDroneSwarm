# 4. Methods

This section presents the three methods compared in the paper and frames them through a single distinction: what an agent *knows or infers* about the utility of an assignment (estimation) versus how that information is turned into an action (decision). This framing is useful in ZK-MRTA because the problem is defined precisely by the absence of prior knowledge, explicit communication, and direct access to hidden compatibility structure; different methods are therefore best read as alternative responses to the same core difficulty.

## 4.1 Random Baseline

The random policy serves as the weakest baseline and lower-bound reference. It is fully ZK-compliant: it does not use target hit points, hidden target types, latent compatibility values, or any history of previous interactions. Instead, it reads only the visible active/inactive status of each target and samples uniformly from the valid actions.

Depending on configuration, the valid action set may include a no-operation action in addition to all currently active targets. Because the policy has no memory and performs no learning, it treats all active targets as equivalent at every step.

Formally, let $\mathcal{U}_t(a_i)$ denote the set of valid actions available to agent $a_i$ at time $t$, consisting of the indices of active targets and, optionally, a NoOp action. The random policy samples:

$$a_t(a_i) \sim \text{Uniform}(\mathcal{U}_t(a_i))$$

This baseline is intentionally naive. Its value lies not in performance, but in providing a clean reference for behavior in the absence of estimation, memory, or coordination.

## 4.2 Oracle Benchmark

The oracle policy provides a near-optimal greedy benchmark by assuming privileged access to information unavailable under the ZK assumptions. In the implemented version, the oracle is an HP-aware marginal-allocation planner that uses true latent drone vectors, true target latent vectors, and current target hit points. It is therefore not ZK-compliant and is used only as an idealized comparison point.

Unlike a one-to-one assignment rule, the oracle explicitly allows multiple drones to focus fire on the same target when doing so is beneficial. Its scoring rule combines several factors: effective damage capped by remaining target HP, a bottleneck weight that prioritizes difficult targets, a finish bonus for assignments that eliminate a target in the current step, an overkill penalty that discourages waste, and a future-value term that favors starting progress on hard targets earlier. The planner then performs a greedy marginal-allocation pass, assigning one drone at a time to the currently best drone-target pair until no beneficial assignment remains. This approximates globally efficient short-horizon coordination while retaining a computationally simple greedy structure. Because the oracle relies on privileged state, its purpose is not to model realistic deployment but to indicate the performance ceiling against which decentralized methods can be interpreted.

## 4.3 Decentralized Collaborative-Filtering Policy

The main method investigated in this paper is a decentralized matrix-factorization policy inspired by classical collaborative filtering. The key conceptual adaptation is that collaborative filtering is not used here to estimate user preference, but to estimate operational utility: the expected effectiveness of assigning a drone to a target. Drones play the role of users, targets play the role of items, and observed engagement outcomes replace ratings. Because the problem is sequential, decentralized, and observation-limited, the recommender mechanism is repurposed as an online decision policy rather than a centralized offline prediction model.

Classical matrix factorization represents users and items through latent vectors and predicts missing interactions via their dot product. In the present setting, this idea is adapted so that each drone independently maintains a private local model of the interaction space. There is no centralized shared factorization and no direct exchange of learned parameters across drones. Collaboration arises only through shared observations of interaction outcomes. This distinction is essential: the method is collaborative in inspiration, but decentralized in implementation.

For an agent $a_i$, the learned model consists of a drone embedding matrix:

$$P^{(a_i)} \in \mathbb{R}^{m \times d_f}$$

and a target embedding matrix:

$$U^{(a_i)} \in \mathbb{R}^{d_f \times n}$$

where $m$ is the number of drones, $n$ is the number of targets, and $d_f$ is the factorization dimension (a hyperparameter distinct from the environment's latent dimension $d$). Agent $a_i$'s predicted utility for the interaction between drone $a_k$ and target $t_j$ is:

$$\hat{r}_{kj}^{(a_i)} = (P_{k,:}^{(a_i)})^\top U_{:,j}^{(a_i)}$$

where $a_k$ denotes the acting drone whose interaction with target $t_j$ is being evaluated. In the implemented policy, each drone maintains its own local matrices $P^{(a_i)}$ and $U^{(a_i)}$, initialized with entries drawn independently from $\mathcal{N}(0,\, 0.01^2)$ and updated online from observed swarm interaction events.

Although the predictor $\hat{r}_{kj}^{(a_i)}$ is defined for any drone $a_k$, agent $a_i$ only uses its own row $P_{i,:}^{(a_i)}$ when choosing an action (§4.4). The remaining rows are nevertheless learned, and this is necessary rather than incidental: updating the shared target embedding $U^{(a_i)}_{:,j}$ from drone $a_m$'s observed engagement requires agent $a_i$'s current estimate of $P^{(a_i)}_{m,:}$, so maintaining the full drone-embedding matrix is what enables the target embeddings to absorb information from swarm-wide interactions. This is the classical collaborative-filtering transfer effect: updates driven by other drones refine the shared $U^{(a_i)}$ and thereby sharpen the self-row's predictions. Each agent's parameters remain private — collaboration flows through the shared observation stream, not through parameter exchange.

This method is particularly well suited to ZK-MRTA because the relevant interaction matrix is not given in advance. Agents begin with no knowledge of true compatibility and must build an interaction structure gradually from sparse experience. The matrix-factorization policy addresses this by learning latent representations from observed rewards, thereby estimating hidden structure that cannot be read directly from the environment.

The structural assumption underlying this approach — that agent-task effectiveness is governed by a low-rank compatibility matrix — has a formal basis in the emerging low-rank bandit and multi-user RL literature. Katariya et al. [40] showed that a rank-1 interaction structure is learnable with sublinear regret from bandit feedback, and Kang et al. [41] extended this to general low-rank matrix bandit problems. Nagaraj et al. [39] established that multiple users sharing a low-rank reward structure can achieve significantly better sample efficiency than users learning independently, which motivates why each drone in ZK-MRTA maintains a full model of the swarm interaction matrix rather than only its own row. The ZK-MRTA setting differs from these in that agents cannot share raw trajectories with a central learner; they exploit the shared public observation stream instead. This difference motivates the decentralized SGD update architecture (§4.4.1) rather than centralized joint optimization.

## 4.4 Action Selection

At decision time, each agent $a_i$ uses only its own latent row $P_{i,:}^{(a_i)}$ to evaluate currently active targets. Action selection follows an $\varepsilon$-greedy mechanism. With probability $\varepsilon$, the agent explores by selecting a random active target. Otherwise, it exploits by selecting the target with the highest predicted utility. After each action, the exploration rate decays multiplicatively until it reaches a predefined floor. The initial exploration rate, decay factor, and floor are configurable hyperparameters; their values for the reported experiments are listed in §5.

Formally, if $\mathcal{T}_t(a_i)$ is the set of active targets observed by agent $a_i$ at time $t$, then exploitation selects:

$$a_t(a_i) = \arg\max_{j \in \mathcal{T}_t(a_i)} \hat{r}_{ij}^{(a_i)}$$

while exploration samples uniformly from $\mathcal{T}_t(a_i)$.

### 4.4.1 Learning from Shared Interaction Data

Learning relies on the public-observation assumption introduced in §3: after each step, every agent sees which drone engaged which target and the resulting reward signal, even though the underlying compatibility structure that generated those rewards remains hidden. Concretely, the public step summary exposes two arrays that drive the learner: the joint action vector identifies the engaged drone-target pairs $(a_i, t_j)$; and the per-agent reward vector provides the supervision target $y_{ij}$ in direct mode, or a single sample feeding the running mean in integration-matrix mode. No privileged state is consumed — the learner sees only what the environment makes public.

This is what makes the method collaborative despite being decentralized: no parameters are exchanged, but the observation stream is shared. Two supervision modes are supported:

- **Direct mode**: The policy compares its predicted utility for each observed drone-target event directly to the corresponding observed reward. All events — including wasted shots on inactive targets — are used for gradient updates, because the reward stream always reflects compatibility rather than a separate invalid-action marker.
- **Integration-matrix mode**: The policy first updates a running-mean interaction matrix and then uses that accumulated estimate as the supervision target. Every observed reward contributes to that running mean, including rewards from shots that were task-ineffective because the target had already been neutralized earlier in the step. This preserves the interpretation that the learner models compatibility from the public reward stream alone, while coordination costs remain visible through episode-level efficiency metrics rather than reward gating.

The experiments reported in this paper use **integration-matrix mode**, so the running-mean interaction matrix is the primary supervision target in the reported runs.

Each agent $a_k$ therefore updates its local model for every observed event $(a_i, t_j)$, not only its own engagements. Let the prediction error for an observed event $(a_i, t_j)$, as computed by agent $a_k$, be denoted by:

$$e_{ij}^{(a_k)} = \hat{r}_{ij}^{(a_k)} - y_{ij}$$

where $y_{ij}$ is either the immediate observed reward or the current running-mean entry of the integration matrix. The local embeddings minimize the regularized squared-error loss:

$$\mathcal{L}_{ij}^{(a_k)} = \left( e_{ij}^{(a_k)} \right)^2 + \tfrac{\lambda}{2} \left( \| P_{i,:}^{(a_k)} \|^2 + \| U_{:,j}^{(a_k)} \|^2 \right)$$

**Note on loss convention.** The data term uses $(e)^2$ rather than the $\frac{1}{2}(e)^2$ convention common in the collaborative filtering literature (e.g., Koren et al. [22]). The resulting gradient carries an explicit factor of 2 in the data term, which effectively doubles the data-loss contribution relative to the regularizer for the same $\eta$. Practitioners comparing learning-rate settings to published CF implementations should account for this difference.

using stochastic gradient descent (SGD):

$$P_{i,:}^{(a_k)} \leftarrow P_{i,:}^{(a_k)} - \eta \left( 2 e_{ij}^{(a_k)} U_{:,j}^{(a_k)} + \lambda P_{i,:}^{(a_k)} \right)$$

$$U_{:,j}^{(a_k)} \leftarrow U_{:,j}^{(a_k)} - \eta \left( 2 e_{ij}^{(a_k)} P_{i,:}^{(a_k)} + \lambda U_{:,j}^{(a_k)} \right)$$

where $\eta$ is the learning rate and $\lambda$ is the regularization coefficient.

An important property of the implemented method is that learning persists across episodes. Episode boundaries do not reinitialize the learned embedding matrices, so knowledge accumulated in earlier episodes continues to shape future decisions. The resulting training process is therefore cumulative, which is appropriate for a method intended to model gradual knowledge acquisition under repeated interaction.

## 4.5 Summary of the Compared Methods

The three methods examined here occupy distinct positions on the knowledge spectrum. Their relationships are summarized by the following axis-by-axis comparison:

| Axis | Random | Oracle | Matrix-factorization |
|---|---|---|---|
| Knowledge of latent drone/target vectors | None | Full (privileged) | None |
| Access to target HP | None | Full (privileged) | None |
| Information used at decision time | Active/inactive flags | Latent vectors + HP | Own learned latent row |
| Learning | None | None (planner) | Online SGD on embeddings |
| Coordination mechanism | None | Centralized greedy planner | Implicit via shared observations |
| ZK-compliant | Yes | No | Yes |
| Memory across steps/episodes | None | None | Persistent embeddings |

Together, these methods provide a coherent basis for evaluating whether collaborative-filtering-inspired utility learning can close part of the gap between naive decentralized behavior and an idealized planner.
