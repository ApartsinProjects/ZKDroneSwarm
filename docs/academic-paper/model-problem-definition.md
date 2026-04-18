# 3. Model and Problem Definition

This section formalizes ZK-MRTA as a sequential multi-agent decision problem under strict informational constraints. It defines the core assumptions, interaction model, objectives, and notation used throughout the paper; notation is summarized in §3.10.

## 3.1 Problem Setting

We consider a decentralized multi-agent system composed of a set of agents and tasks. Let $\mathcal{A} = \{a_1, a_2, \ldots, a_m\}$ denote a finite set of agents, and $\mathcal{T} = \{t_1, t_2, \ldots, t_n\}$ denote a finite set of tasks.

The system evolves over discrete time steps $t = 1, 2, \ldots$, where agents interact with tasks through a shared environment. Each task $t_j \in \mathcal{T}$ is associated with an internal state (e.g., active/inactive or remaining workload), which evolves over time as a result of agent actions.

At each time step, agents select actions that influence task states, and the environment updates accordingly. The goal is to determine how agents should allocate themselves to tasks over time in order to optimize global system performance, subject to the informational constraints of the Zero-Knowledge setting.

## 3.2 Zero-Knowledge Assumptions

The ZK-MRTA framework is defined by strict informational constraints:

- **No prior knowledge of tasks**: Agents have no access to task attributes such as difficulty, type, or required resources
- **No self-knowledge**: Agents do not know their own capabilities or effectiveness
- **No knowledge of other agents**: Agents cannot access the capabilities or internal states of others
- **No communication**: Direct communication between agents is not permitted
- **Outcome-based learning only**: Agents can observe only:
  - the outcomes of their own actions, and
  - indirect evidence of other agents' actions through environmental changes

These assumptions eliminate access to explicit models of task-agent compatibility and require all decision-making to rely solely on observed interaction outcomes.

## 3.3 Relation to Classical MRTA

Classical Multi-Robot Task Allocation is typically formulated as an optimization problem defined by:

- a cost function $C(a_i, t_j)$, and
- a feasibility function $G(a_i, t_j) \in \{0, 1\}$

with the objective of finding an assignment $\Lambda \subseteq \mathcal{A} \times \mathcal{T}$ that minimizes total cost:

$$\min_{\Lambda} \sum_{(a_i, t_j) \in \Lambda} C(a_i, t_j)
\quad \text{subject to} \quad
G(a_i, t_j) = 1$$

This formulation assumes that both cost and feasibility are known or computable in advance.

In contrast, the ZK-MRTA framework removes access to both $C$ and $G$. Agents must therefore infer task suitability indirectly, based solely on observed outcomes, without explicit knowledge of task properties or their own capabilities.

## 3.4 Sequential Interaction Model

ZK-MRTA is inherently a sequential decision-making problem. At each time step:

1. Each agent receives a local observation
2. Each agent selects an action
3. The environment updates its internal state
4. Agents receive outcome signals

This interaction loop can be summarized as:

$$o_t(a_i) \to a_t(a_i) \to \text{environment update} \to \text{outcome}$$

Unlike static assignment formulations, task allocation unfolds over time, and agents must adapt their decisions based on accumulated experience and evolving system dynamics.

## 3.5 Problem Variants

The ZK-MRTA framework is defined abstractly, but particular benchmark instances may differ along a small number of structural dimensions. Two relevant axes are task dynamics and agent resource constraints.

### 3.5.1 Target (Task) Types

- **Passive targets**: Static tasks whose internal state evolves only as a consequence of incoming actions.
- **Active targets**: Dynamic tasks that may react, move, or otherwise change in response to interaction.

The benchmark studied in this paper instantiates passive targets with fixed resilience (HP), so neutralization is a cumulative-damage process.

### 3.5.2 Agent Resource Constraints

- **Infinite-capacity**: Agents are not subject to a hard resource budget over the episode horizon.
- **Finite-capacity**: Agents possess limited expendable resources, such as ammunition or energy.

The experiments reported in this paper use the **infinite-capacity** variant: no per-agent ammunition limit is enforced, although ammunition consumption is still tracked diagnostically.

## 3.6 Observation Model

At each time step $t$, each agent $a_i \in \mathcal{A}$ receives a local observation $o_t(a_i) \in O$, derived from the environment state. The observation is a structured tuple:

$$o_t(a_i) = \Big(\mathbf{p},\ \mathbf{b},\ \hat{\mathbf{s}}_{t-1},\ \tilde{\mathbf{r}}_{t-1},\ \mathbf{w}_{t-1}\Big)$$

where:

- $\mathbf{p} \in \mathbb{R}^{n \times 2}$ — spatial positions of all tasks
- $\mathbf{b} \in \{0,1\}^n$ — binary active/inactive status of each task
- $\hat{\mathbf{s}}_{t-1} \in \{0,\ldots,n\}^m$ — the (possibly noisy) joint action vector from the previous step (which task each agent selected)
- $\tilde{\mathbf{r}}_{t-1} \in \mathbb{R}^m$ — the (possibly noisy) reward received by each agent in the previous step
- $\mathbf{w}_{t-1} \in \{0,1\}^m$ — a binary mask indicating whether each agent's selected task was still active at the time of engagement

The $\mathbf{w}_{t-1}$ field is particularly important: it allows agents to distinguish informative interactions (shots on active targets) from wasted engagements (shots on already-neutralized targets), which affects how learning algorithms weight each observation.

Crucially, observations do not include:

- task attributes or types,
- agent capabilities or latent vectors,
- explicit compatibility information.

Thus, the structure governing agent-task effectiveness is entirely hidden and must be inferred from interaction history.

## 3.7 Action Space

At each time step, each agent selects an action:

$$a_t(a_i) \in A$$

where the action space $A$ consists of:

- selecting a task $t_j \in \mathcal{T}$, or
- performing a no-operation ($a_t(a_i) = 0$).

Actions correspond to attempted task engagements, and their effects are determined by the underlying environment dynamics. No assumptions are made at this stage regarding the mechanism used to select actions.

## 3.8 Hidden Environment Structure

The environment is governed by an underlying latent compatibility structure that determines the effectiveness of agent-task interactions.

Each agent $a_i$ and task $t_j$ is associated with hidden latent representations:

$$\mathbf{z}_i^{(a)} \in \mathbb{R}^d_{>0}, \quad \mathbf{z}_j^{(t)} \in \mathbb{R}^d_{>0}$$

The raw compatibility between agent $a_i$ and task $t_j$ is governed by the latent dot product:

$$g_{ij} = (\mathbf{z}_i^{(a)})^\top \mathbf{z}_j^{(t)}$$

Each task $t_j$ is initialized with a resilience budget $\text{HP}_j > 0$. An engagement by agent $a_i$ reduces the task's remaining HP by an amount proportional to $g_{ij}$, and the task is neutralized when its HP reaches zero.

### 3.8.1 Reward Signal

The reward received by agent $a_i$ upon engaging task $t_j$ is not equal to the damage applied. Instead, it is a function of latent compatibility, decoupling the learning signal from the task execution outcome. The framework supports two reward modes:

- **Cosine mode** (used in all experiments reported here): The reward is the angular alignment between the agent and task latent vectors:
$$r_{ij} = \frac{(\mathbf{z}_i^{(a)})^\top \mathbf{z}_j^{(t)}}{\|\mathbf{z}_i^{(a)}\| \cdot \|\mathbf{z}_j^{(t)}\|}$$

- **Damage mode**: The reward equals the effective HP reduction applied to the task.

In both modes, the observed reward $\tilde{r}_{ij}$ may be corrupted by additive Gaussian noise with configurable standard deviation $\sigma_r \geq 0$:
$$\tilde{r}_{ij} = r_{ij} + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma_r^2)$$

If the targeted task is already inactive at the time of engagement, the agent receives a negative anti-signal rather than a compatibility-based reward.

These latent variables are not observable to agents and serve only as an internal mechanism generating interaction outcomes. As a result, agents must operate without direct access to the true compatibility structure.

## 3.9 Objective and Performance Metrics

The goal of ZK-MRTA is to optimize global system performance over time under the informational constraints defined above. To preserve comparability across decision methods, performance is evaluated using strategy-independent metrics computed from logged interaction traces.

At an abstract level, these metrics fall into four categories:

- **Time-based metrics**: measures of how quickly the system reaches a desired level of task completion
- **Completion or progress metrics**: measures of how much of the task set has been completed over the episode horizon
- **Efficiency metrics**: measures of resource use relative to achieved task progress
- **Coordination metrics**: measures of swarm-level redundancy, contention, and wasted effort

The specific metrics used in the empirical evaluation are:

- **Steps to completion** and **total ammo consumed** (time-based / efficiency)
- **Shots per target** and **targets neutralized** (efficiency / completion)
- **Total collisions** and **total overkill** (coordination)
- **Average latent match quality** and **latent mismatch** (learning quality)

These metrics constitute the set $M$ in the formal definition below. Their operational computation is specified in the experiments section (§6), while their role in the formalism is to define the system-level objectives against which any policy is assessed.

## 3.10 Notation Summary

| Symbol | Description |
|--------|-------------|
| $\mathcal{A} = \{a_1, \ldots, a_m\}$ | Set of agents |
| $\mathcal{T} = \{t_1, \ldots, t_n\}$ | Set of tasks |
| $m$ | Number of agents |
| $n$ | Number of tasks |
| $t$ | Discrete time step |
| $d$ | Latent vector dimension |
| $o_t(a_i)$ | Observation of agent $a_i$ at time $t$ |
| $a_t(a_i)$ | Action of agent $a_i$ at time $t$ |
| $A$ | Action space |
| $O$ | Observation space |
| $\mathbf{a}_t$ | Joint action of all agents |
| $\mathbf{p}$ | Task position matrix |
| $\mathbf{b}$ | Task active/inactive status vector |
| $\hat{\mathbf{s}}_{t-1}$ | Previous joint action vector (public, possibly noisy) |
| $\tilde{\mathbf{r}}_{t-1}$ | Previous observed reward vector (public) |
| $\mathbf{w}_{t-1}$ | Target-was-active-at-engagement mask |
| $\mathbf{z}_i^{(a)} \in \mathbb{R}^d_{>0}$ | Hidden latent vector of agent $a_i$ |
| $\mathbf{z}_j^{(t)} \in \mathbb{R}^d_{>0}$ | Hidden latent vector of task $t_j$ |
| $g_{ij}$ | Latent dot-product compatibility |
| $r_{ij}$ | Reward signal for agent $a_i$ engaging task $t_j$ |
| $\tilde{r}_{ij}$ | Observed (noisy) reward |
| $\sigma_r$ | Reward noise standard deviation |
| $\text{HP}_j$ | Task resilience (hit points) |
| $M$ | Set of performance metrics |
| $C(a_i, t_j)$ | Cost function (classical MRTA reference) |
| $G(a_i, t_j)$ | Feasibility function (classical MRTA reference) |

**Implementation note.** The benchmark implementation uses $N$ for the number of agents and $M$ for the number of tasks (matching PettingZoo conventions), and denotes the environment latent dimension as $d_z$ to distinguish it explicitly from the policy factorization dimension $d_f$. Throughout this paper the symbols $m$, $n$, and $d$ are used in place of $N$, $M$, and $d_z$ respectively to keep notation concise and consistent with the MRTA literature.

## 3.11 Formal Definition of ZK-MRTA

**Definition 1** (Zero-Knowledge Multi-Robot Task Allocation).

A ZK-MRTA problem is defined by the tuple:

$$\mathcal{Z} = (\mathcal{A}, \mathcal{T}, A, O, E, M)$$

where:

- $\mathcal{A} = \{a_1, \ldots, a_m\}$ is a finite set of agents, each associated with a hidden latent vector $\mathbf{z}_i^{(a)} \in \mathbb{R}^d_{>0}$
- $\mathcal{T} = \{t_1, \ldots, t_n\}$ is a finite set of tasks, each associated with a hidden latent vector $\mathbf{z}_j^{(t)} \in \mathbb{R}^d_{>0}$ and resilience $\text{HP}_j > 0$
- $A$ is the discrete action space: $\{0\} \cup \{1, \ldots, n\}$ (no-op or task selection)
- $O$ is the observation space: structured tuples $o_t(a_i) = (\mathbf{p}, \mathbf{b}, \hat{\mathbf{s}}_{t-1}, \tilde{\mathbf{r}}_{t-1}, \mathbf{w}_{t-1})$ as defined in §3.6
- $E$ is the environment transition function mapping $(\text{world state}_t, \mathbf{a}_t) \to (\text{world state}_{t+1}, \tilde{\mathbf{r}}_t, \mathbf{w}_t, \text{done}_t)$, where rewards are determined by the latent compatibility structure and reward mode (§3.8)
- $M$ is a set of strategy-independent performance metrics as defined in §3.9

The system evolves over discrete time steps. At each time step, each agent receives a local observation $o_t(a_i)$ and selects an action $a_t(a_i)$. The environment applies the joint action $\mathbf{a}_t$, updates task states (HP), produces noisy reward signals $\tilde{\mathbf{r}}_t$ and validity flags $\mathbf{w}_t$, and terminates when all tasks are neutralized or the maximum step limit is reached.

The problem is subject to the Zero-Knowledge constraints defined in §3.2: agents have no prior knowledge of tasks or their own capabilities, cannot communicate, and must rely solely on $O$ at each step.

The objective is to design decentralized agent behaviors — algorithms that map observations and private memory to actions — that maximize system-level performance according to the metrics in $M$.
