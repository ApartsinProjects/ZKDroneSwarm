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
  - a shared *public summary* of the previous step — the joint action vector and the per-agent reward vector — which carries the actions and outcomes of other agents but contains no capability or identity information (see §3.6 for the exact structure).

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

1. Each agent receives a per-agent observation
2. Each agent selects an action
3. The environment updates its internal state
4. Agents receive outcome signals

This interaction loop can be summarized as:

$$o_t(a_i) \to a_t(a_i) \to \text{environment update} \to \text{outcome}$$

Unlike static assignment formulations, task allocation unfolds over time, and agents must adapt their decisions based on accumulated experience and evolving system dynamics.

Within a single time step, individual agents' actions are resolved in a uniformly random order. This ordering governs the sequential resolution of contending engagements — for example, when multiple agents target the same task, an earlier-processed agent may neutralize it before a later-processed agent's shot lands.

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

At each time step $t$, each agent $a_i \in \mathcal{A}$ receives an observation $o_t(a_i) \in O$, derived from the environment state. The term "local" is avoided here because the observation is not purely private: it combines environment state (task positions and activity) with a *shared public summary* of the previous step's joint actions and outcomes. No agent receives privileged information beyond what this public summary provides. The observation is a structured tuple:

$$o_t(a_i) = \Big(\mathbf{p},\ \mathbf{b},\ \hat{\mathbf{s}}_{t-1},\ \tilde{\mathbf{r}}_{t-1}\Big)$$

where:

- $\mathbf{p} \in \mathbb{R}^{n \times 2}$ — spatial positions of all tasks
- $\mathbf{b} \in \{0,1\}^n$ — binary active/inactive status of each task
- $\hat{\mathbf{s}}_{t-1} \in \{0,\ldots,n\}^m$ — the (possibly noisy) joint action vector from the previous step (which task each agent selected)
- $\tilde{\mathbf{r}}_{t-1} \in \mathbb{R}^m$ — the (possibly noisy) reward received by each agent in the previous step

A central feature of this observation model is that the two per-agent fields $\hat{\mathbf{s}}_{t-1}$ and $\tilde{\mathbf{r}}_{t-1}$ are **public**: each agent observes the *full joint vector* across all $m$ agents, not only its own entry. This public broadcast is the sole channel through which an agent can indirectly observe other agents' behavior, and it replaces the direct communication that classical MRTA approaches assume.

The noise on $\hat{\mathbf{s}}_{t-1}$ corrupts each entry independently with probability $\pi_s \in [0,1]$, replacing it with a uniformly sampled task index; the noise on $\tilde{\mathbf{r}}_{t-1}$ is additive Gaussian with standard deviation $\sigma_r \geq 0$ (see §3.8.1). Setting $\pi_s = 0$ and $\sigma_r = 0$ recovers the fully-observed public-history regime.

In the implementation, $\mathbf{p}$ and $\mathbf{b}$ are transmitted together as a single packed vector of length $3n$ (one $(x, y, \text{active})$ triple per task); the decomposition above is conceptual.

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

Each task $t_j$ is initialized with a resilience budget $\text{HP}_j > 0$. An engagement by agent $a_i$ applies damage

$$d_{ij} = \max(0,\ g_{ij})$$

to the task, reducing its remaining HP by $d_{ij}$. Negative dot products contribute zero damage rather than healing, and the task is neutralized when its HP reaches zero.

### 3.8.1 Reward Signal

The reward received by agent $a_i$ upon engaging task $t_j$ is not equal to the damage applied. Instead, it is a function of latent compatibility, decoupling the learning signal from the task execution outcome. The framework defines two reward modes; all experiments in this paper use the cosine mode:

- **Cosine mode** (used in all experiments reported here): The reward is the angular alignment between the agent and task latent vectors:
$$r_{ij} = \frac{(\mathbf{z}_i^{(a)})^\top \mathbf{z}_j^{(t)}}{\|\mathbf{z}_i^{(a)}\| \cdot \|\mathbf{z}_j^{(t)}\|}$$

- **Damage mode**: The reward equals the effective HP reduction applied to the task.

In both modes, the observed reward $\tilde{r}_{ij}$ may be corrupted by additive Gaussian noise with configurable standard deviation $\sigma_r \geq 0$:
$$\tilde{r}_{ij} = r_{ij} + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma_r^2)$$

If the targeted task is already inactive at the time of engagement, the agent still receives the compatibility-based reward generated by the latent vectors; however, the shot applies no damage and therefore affects performance only indirectly through wasted effort.

These latent variables are not observable to agents and serve only as an internal mechanism generating interaction outcomes. As a result, agents must operate without direct access to the true compatibility structure.

### 3.8.2 Collision Events

Because multiple agents may independently select the same task within a single time step, the environment tracks *collisions* as a coordination signal. A collision is registered whenever two or more agents target the same task in the same step: for each such task, every agent beyond the first (in the random processing order of §3.4) contributes one collision to the step count. Collisions are a diagnostic of coordination failure only; they do not alter task dynamics or affect rewards directly, but engagements against a task that has already been neutralized earlier in the step produce wasted shots whose rewards still reflect latent compatibility while contributing no damage.

## 3.9 Objective and Performance Metrics

The goal of ZK-MRTA is to optimize global system performance over time under the informational constraints defined above. To preserve comparability across decision methods, performance is evaluated using strategy-independent metrics computed from logged interaction traces.

Formally, a decentralized policy is a tuple $\pi = (\pi_1, \ldots, \pi_m)$ in which each $\pi_i$ maps the observation-action history of agent $a_i$ to a next action in $A$. For a given metric $k \in M$, let $\mu_k(\pi)$ denote its expected value under $\pi$, where the expectation is taken over random scenario instantiations (hidden latent configuration, task resilience, initial positions) and observation noise. The ZK-MRTA objective is to design $\pi$ so as to improve $\{\mu_k(\pi)\}_{k \in M}$ relative to reference baselines. Because the elements of $M$ span distinct operational axes (time, efficiency, coordination, and learning quality), this paper evaluates policies along all elements of $M$ rather than collapsing them into a single scalar; policies are compared empirically across sampled scenarios.

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
| $\mathbf{z}_i^{(a)} \in \mathbb{R}^d_{>0}$ | Hidden latent vector of agent $a_i$ |
| $\mathbf{z}_j^{(t)} \in \mathbb{R}^d_{>0}$ | Hidden latent vector of task $t_j$ |
| $g_{ij}$ | Latent dot-product compatibility |
| $r_{ij}$ | Reward signal for agent $a_i$ engaging task $t_j$ |
| $\tilde{r}_{ij}$ | Observed (noisy) reward |
| $\sigma_r$ | Reward noise standard deviation |
| $\pi_s$ | Action-observation corruption probability |
| $\text{HP}_j$ | Task resilience (hit points) |
| $M$ | Set of performance metrics |
| $\pi = (\pi_1, \ldots, \pi_m)$ | Joint decentralized policy |
| $\mu_k(\pi)$ | Expected value of metric $k \in M$ under policy $\pi$ |
| $C(a_i, t_j)$ | Cost function (classical MRTA reference) |
| $G(a_i, t_j)$ | Feasibility function (classical MRTA reference) |

**Implementation note.** In the accompanying benchmark implementation, the symbols $m$, $n$, and $d$ used throughout this paper correspond to the code variables `num_drones`, `num_targets`, and `latent_dim` respectively. The policy-side factorization dimension (used in §4) is tracked separately in code under a distinct variable to avoid conflating the environment's generative latent dimension with the learner's chosen approximation rank. The shorter symbols $m$, $n$, $d$ are preferred in the paper for consistency with the MRTA literature.

## 3.11 Formal Definition of ZK-MRTA

**Definition 1** (Zero-Knowledge Multi-Robot Task Allocation — general problem class).

A ZK-MRTA problem is defined by the tuple:

$$\mathcal{Z} = (\mathcal{A}, \mathcal{T}, A, O, E, M)$$

where:

- $\mathcal{A} = \{a_1, \ldots, a_m\}$ is a finite set of agents
- $\mathcal{T} = \{t_1, \ldots, t_n\}$ is a finite set of tasks
- $A$ is the discrete action space: $\{0\} \cup \{1, \ldots, n\}$ (no-op or task selection)
- $O$ is the observation space of structured tuples $o_t(a_i) = (\mathbf{p}, \mathbf{b}, \hat{\mathbf{s}}_{t-1}, \tilde{\mathbf{r}}_{t-1})$ as defined in §3.6
- $E$ is the environment transition function mapping $(\text{world state}_t, \mathbf{a}_t) \to (\text{world state}_{t+1}, \tilde{\mathbf{r}}_t, \text{done}_t)$, parameterized by a hidden effectiveness structure that governs task outcomes. The hidden structure is sampled per episode, held fixed for the duration of that episode, and is never observable to the agents.
- $M$ is a set of strategy-independent performance metrics as defined in §3.9

The system evolves over discrete time steps. At each time step, each agent receives a per-agent observation $o_t(a_i)$ and selects an action $a_t(a_i)$. The environment applies the joint action $\mathbf{a}_t$, updates task states, produces noisy reward signals $\tilde{\mathbf{r}}_t$, and terminates when all tasks are neutralized or the maximum step limit is reached.

The problem is subject to the Zero-Knowledge constraints defined in §3.2: agents have no prior knowledge of tasks or their own capabilities, cannot communicate, and must rely solely on $O$ at each step. The objective is to design decentralized agent behaviors — algorithms that map observations and private memory to actions — that improve the performance metrics $M$ (§3.9).

Definition 1 intentionally leaves the hidden effectiveness structure abstract, so that ZK-MRTA names a *class* of problems rather than a single instantiation. Concrete instantiations differ in how the hidden structure generates damage and reward.

**Definition 2** (Latent-compatibility benchmark instantiation).

The benchmark studied in this paper instantiates Definition 1 with the following hidden structure (§3.8):

- each agent $a_i$ has a hidden latent vector $\mathbf{z}_i^{(a)} \in \mathbb{R}^d_{>0}$, and each task $t_j$ has a hidden latent vector $\mathbf{z}_j^{(t)} \in \mathbb{R}^d_{>0}$ and a resilience budget $\text{HP}_j > 0$;
- raw compatibility is the dot product $g_{ij} = (\mathbf{z}_i^{(a)})^\top \mathbf{z}_j^{(t)}$;
- damage applied per engagement is $d_{ij} = \max(0, g_{ij})$, and the reward signal is determined by the cosine or damage mode of §3.8.1.

Together with the passive-target, infinite-capacity choices of §3.5, this specifies the concrete environment used in all experiments. Other instantiations of Definition 1 (e.g., categorical compatibility structures, or active-target dynamics) are consistent with the general framework but are outside the scope of this paper.
