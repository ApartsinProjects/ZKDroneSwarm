# 3. Model and Problem Definition

We formalize ZK-MRTA as a sequential multi-agent decision problem subject to strict informational constraints. Notation is summarized in §3.12.

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

$$o_t(a) \to a_t(a) \to \text{environment update} \to \text{outcome}$$

Unlike static assignment formulations, task allocation unfolds over time, and agents must adapt their decisions based on accumulated experience and evolving system dynamics.

## 3.5 Problem Variants

The ZK-MRTA framework is defined abstractly, but two structural dimensions determine the specific variant being studied:

### 3.5.1 Target (Task) Types

- **Passive targets**: Static tasks that accumulate damage from agent interactions until neutralized. They do not react to engagement and their state evolves only as a function of incoming actions.
- **Active targets**: Dynamic or retaliatory tasks that may adapt, move, or respond to engagement. These introduce additional uncertainty into the agent's planning problem.

The current benchmark instantiates passive targets with fixed resilience (HP), making neutralization a cumulative-damage process.

### 3.5.2 Agent Resource Constraints

- **Infinite-capacity**: Agents can perform an unlimited number of interactions. This provides an idealized baseline for evaluating the pure learning and coordination challenge.
- **Finite-capacity**: Agents possess limited resources (e.g., ammunition, energy), restricting the total number of interactions before the agent is exhausted. This introduces an additional planning dimension: selecting not only which tasks to engage, but how to allocate limited resources across the mission horizon.

Both variants are supported by the framework. The experiments reported in this paper use the **infinite-capacity** variant: no per-agent ammunition limit is enforced, and episodes terminate either when all targets are neutralized or the step budget is exhausted. Per-agent ammunition consumption is nonetheless tracked as a diagnostic metric throughout execution.

## 3.6 Observation Model

At each time step $t$, each agent $a \in \mathcal{A}$ receives a local observation $o_t(a) \in O$, derived from the environment state. The observation is a structured tuple:

$$o_t(a) = \Big(\mathbf{p},\ \mathbf{b},\ \hat{\mathbf{s}}_{t-1},\ \tilde{\mathbf{r}}_{t-1},\ \mathbf{w}_{t-1}\Big)$$

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

$$a_t(a) \in A$$

where the action space $A$ consists of:

- selecting a task $t_j \in \mathcal{T}$, or
- performing a no-operation ($a_t(a) = 0$).

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

- **Cosine mode** (current default): The reward is the angular alignment between the agent and task latent vectors:
$$r_{ij} = \frac{(\mathbf{z}_i^{(a)})^\top \mathbf{z}_j^{(t)}}{\|\mathbf{z}_i^{(a)}\| \cdot \|\mathbf{z}_j^{(t)}\|}$$

- **Damage mode**: The reward equals the effective HP reduction applied to the task.

In both modes, the observed reward $\tilde{r}_{ij}$ may be corrupted by additive Gaussian noise with standard deviation $\eta \geq 0$:
$$\tilde{r}_{ij} = r_{ij} + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \eta^2)$$

If the targeted task is already inactive at the time of engagement, the agent receives a negative anti-signal rather than a compatibility-based reward.

These latent variables are not observable to agents and serve only as an internal mechanism generating interaction outcomes. As a result, agents must operate without direct access to the true compatibility structure.

## 3.9 Objective and Performance Metrics

The goal of ZK-MRTA is to optimize global system performance over time. Performance is evaluated using metrics defined independently of any specific strategy:

**Time-based metrics** measure the duration required to achieve a specified level of task completion — for example, the number of steps to neutralize all tasks, or the time at which a given fraction of tasks is completed.

**Ratio-based metrics** evaluate performance relative to progress — for example, the proportion of tasks neutralized after a fixed number of steps, or the ratio of remaining task resilience to total initial resilience.

**Efficiency metrics** measure resource utilization relative to outcomes:
- *Damage efficiency*: net damage applied to targets as a fraction of total potential damage
- *Gross vs. net damage*: gross damage counts all shots fired (including overkill and shots on inactive targets); net damage counts only HP reduction that contributed to neutralization
- *Actions per neutralized task*: the average number of engagements required to complete each task

**Coordination metrics** capture swarm-level behavior:
- *Overkill*: damage accumulated on a task beyond its initial HP
- *Collisions*: the number of time steps on which multiple agents selected the same active task simultaneously
- *Coordination score*: a normalized measure of target-spread across the swarm

All metrics are computed by the environment from logged interaction data and remain independent of the specific strategy used.

## 3.10 Baseline Strategies

The ZK-MRTA framework is evaluated by comparing strategies that represent distinct levels of knowledge and decision-making capability. Three canonical baselines bracket the performance space:

- **Random**: Agents select tasks uniformly at random from the set of active tasks at each step. This represents a lower bound on performance — the expected outcome with no learning or inference.

- **Greedy**: Agents choose tasks based on observable features (e.g., proximity or last observed state). Although computationally simple, this heuristic produces locally optimal but globally suboptimal outcomes.

- **Oracle**: Agents possess perfect knowledge of task properties and their own latent vectors, enabling optimal allocation decisions at each step. This represents a theoretical upper bound and serves as the reference against which learning-based strategies are evaluated.

Performance metrics defined in §3.9 are computed identically for all strategies, enabling direct comparison under identical conditions.

## 3.11 Input–Output Formalization

The ZK-MRTA problem can be summarized as follows:

- At each time step $t$, each agent $a \in \mathcal{A}$ receives an observation $o_t(a) \in O$
- Each agent produces an action $a_t(a) \in A$
- The environment maps the joint action $\mathbf{a}_t = (a_t(a_1), \ldots, a_t(a_m))$ together with the current world state to updated task states, reward signals $\tilde{r}_{ij}$, engagement validity flags $\mathbf{w}_t$, and a termination signal

The objective is to design decentralized agent behaviors that maximize global performance metrics in $M$ over the full mission horizon.

## 3.12 Notation Summary

| Symbol | Description |
|--------|-------------|
| $\mathcal{A} = \{a_1, \ldots, a_m\}$ | Set of agents |
| $\mathcal{T} = \{t_1, \ldots, t_n\}$ | Set of tasks |
| $m$ | Number of agents |
| $n$ | Number of tasks |
| $t$ | Discrete time step |
| $d$ | Latent vector dimension |
| $o_t(a)$ | Observation of agent $a$ at time $t$ |
| $a_t(a)$ | Action of agent $a$ at time $t$ |
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
| $r_{ij}$ | Reward signal for agent $i$ engaging task $j$ |
| $\tilde{r}_{ij}$ | Observed (noisy) reward |
| $\eta$ | Reward noise standard deviation |
| $\text{HP}_j$ | Task resilience (hit points) |
| $M$ | Set of performance metrics |
| $C(a_i, t_j)$ | Cost function (classical MRTA reference) |
| $G(a_i, t_j)$ | Feasibility function (classical MRTA reference) |

**Implementation note.** The benchmark implementation uses $N$ for the number of agents and $M$ for the number of tasks (matching PettingZoo conventions), and denotes the environment latent dimension as $d_z$ to distinguish it explicitly from the policy factorization dimension $d_f$. Throughout this paper the symbols $m$, $n$, and $d$ are used in place of $N$, $M$, and $d_z$ respectively to keep notation concise and consistent with the MRTA literature.

## 3.13 Formal Definition of ZK-MRTA

**Definition 1** (Zero-Knowledge Multi-Robot Task Allocation).

A ZK-MRTA problem is defined by the tuple:

$$\mathcal{Z} = (\mathcal{A}, \mathcal{T}, A, O, E, M)$$

where:

- $\mathcal{A} = \{a_1, \ldots, a_m\}$ is a finite set of agents, each associated with a hidden latent vector $\mathbf{z}_i^{(a)} \in \mathbb{R}^d_{>0}$
- $\mathcal{T} = \{t_1, \ldots, t_n\}$ is a finite set of tasks, each associated with a hidden latent vector $\mathbf{z}_j^{(t)} \in \mathbb{R}^d_{>0}$ and resilience $\text{HP}_j > 0$
- $A$ is the discrete action space: $\{0\} \cup \{1, \ldots, n\}$ (no-op or task selection)
- $O$ is the observation space: structured tuples $o_t(a) = (\mathbf{p}, \mathbf{b}, \hat{\mathbf{s}}_{t-1}, \tilde{\mathbf{r}}_{t-1}, \mathbf{w}_{t-1})$ as defined in §3.6
- $E$ is the environment transition function mapping $(\text{world state}_t, \mathbf{a}_t) \to (\text{world state}_{t+1}, \tilde{\mathbf{r}}_t, \mathbf{w}_t, \text{done}_t)$, where rewards are determined by the latent compatibility structure and reward mode (§3.8)
- $M$ is a set of strategy-independent performance metrics as defined in §3.9

The system evolves over discrete time steps. At each time step, each agent receives a local observation $o_t(a)$ and selects an action $a_t(a)$. The environment applies the joint action $\mathbf{a}_t$, updates task states (HP), produces noisy reward signals $\tilde{\mathbf{r}}_t$ and validity flags $\mathbf{w}_t$, and terminates when all tasks are neutralized or the maximum step limit is reached.

The problem is subject to the Zero-Knowledge constraints defined in §3.2: agents have no prior knowledge of tasks or their own capabilities, cannot communicate, and must rely solely on $O$ at each step.

The objective is to design decentralized agent behaviors — algorithms that map observations and private memory to actions — that maximize system-level performance according to the metrics in $M$.
