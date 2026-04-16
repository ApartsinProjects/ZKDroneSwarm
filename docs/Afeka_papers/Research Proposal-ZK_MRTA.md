School of Software Engineering: Intelligent Systems

# Decentralized Multi-Robot Task Allocation under the Zero-Knowledge Assumption using a Collaborative-Filtering Approach (ZK-MRTA)

Final Project Proposal

Student Name: Yigal Meshulam

Supervisor: Dr. Sasha Apartsin

Advisor: Dr. Yehudit Aperstein

Date: November 2025

For the Degree of: Master of Science (M.Sc.) in Intelligent Systems

Submitted to: Afeka College of Engineering

# Introduction

Swarm robotics, inspired by the collective behaviors of social animals, offers a robust and scalable framework for addressing distributed coordination challenges. The rapid advancement of autonomous systems, including aerial, ground, and sea drones, has enabled large robotic swarms to perform complex tasks such as reconnaissance, disaster relief, and battlefield operations. These swarms, particularly when heterogeneous, integrate agents with varying capabilities, such as different munitions or sensors, to operate effectively in dynamic and challenging environments. A compelling example is a battlefield scenario, where drones must autonomously prioritize and execute tasks like reconnaissance or target neutralization. As tasks grow in complexity and scale, ensuring optimal task allocation within such swarms becomes a significant challenge.

## Classical MRTA

Multi-Robot Task Allocation (MRTA) addresses the fundamental question of how to assign tasks to multiple agents in a way that optimizes overall system performance. Classical MRTA research considers factors such as task types (single vs. multi-robot), assignment modes (instantaneous vs. time-extended), and robot capabilities (homogeneous vs. heterogeneous). Existing methods span optimization-based approaches, market-based mechanisms, and reinforcement learning, each relying, explicitly or implicitly, on **prior knowledge** of tasks, agent abilities, and often on **communication between agents**. While effective under structured conditions, these methods face limitations when deployed in dynamic, uncertain, and communication-constrained environments.

## Introducing ZK-MRTA

To address these challenges, we define a new problem type: Zero-Knowledge Multi-Robot Task Allocation (ZK-MRTA). In ZK-MRTA, agents operate under stringent assumptions:

* They have no prior knowledge of task attributes.  
* They have no knowledge of their own capabilities.  
* They have no knowledge of the capabilities of other agents.  
* They cannot communicate with one another.  
* They can only observe the outcomes of their own actions.  
* They can only indirectly observe the outcomes of others’ actions.

The task of ZK-MRTA is therefore to develop agent strategies \- algorithms that process observations, incorporate memory, and select tasks, to optimize performance metrics. Importantly, these performance metrics are defined independently of any specific strategy, enabling systematic evaluation across different approaches. Examples include minimizing task completion time or maximizing the utility achieved by the swarm. By formalizing ZK-MRTA and establishing a structured framework for its evaluation, this work seeks to expand the theoretical and practical understanding of decentralized swarm robotics in real-world environments.

## Proposed Solution

As an example strategy for the ZK-MRTA task, we propose a Collaborative Filtering (CF)-based approach. CF, widely used in recommendation systems, can infer relationships between agents and tasks from sparse observational data without requiring explicit task or agent modeling. This is only one of many possible strategies, and its effectiveness must therefore be assessed empirically. To this end, we will develop a simulation framework that enables rigorous evaluation of the CF-based approach against baseline solutions such as random, greedy and oracle strategies. The framework will support testing under diverse conditions, including varying task complexities, environmental noise, and levels of observability, allowing us to systematically analyze performance. Through this process, we aim to demonstrate the scalability and adaptability of decentralized solutions to ZK-MRTA.

# Literature Review and Existing Approaches

## Inspiration and Background

The study of Multi-Robot Task Allocation (**MRTA**) draws inspiration from the collective behaviors of social organisms such as ants, bees, and birds, whose decentralized coordination enables them to accomplish complex objectives without centralized control. Swarm systems embody three foundational principles:   
**robustness** \- the ability to maintain functionality despite failures or environmental changes, **scalability** \- consistent performance as the number of agents or tasks increases, **flexibility** \- adaptation to unforeseen tasks or dynamic conditions. These qualities make swarm robotics particularly valuable in high-risk environments such as disaster response, hazardous exploration, and adversarial military operations.

Swarm systems are inherently decentralized, relying on local sensing and, in some cases, limited communication between agents. Their behavior can be analyzed through **microscopic models**, which focus on individual interactions, or **macroscopic models**, which capture the collective performance of the swarm. A central distinction lies between **homogeneous swarms**, composed of identical agents, and **heterogeneous swarms**, integrating agents with diverse capabilities. Heterogeneous swarms offer greater flexibility and robustness, as they can handle a broader range of tasks and compensate for individual failures. However, this diversity also introduces greater coordination challenges.

Early MRTA research produced diverse task-specific algorithms and bio-inspired approaches, such as ant colony optimization and distributed coordination methods. Reviews by Brambilla et al. \[3\] and Cheraghi et al. \[5\], emphasize how these early methodologies prioritized implementation but lacked generalizable frameworks. More recently, the field has shifted toward standardized models and systematic performance evaluation, laying the groundwork for scalable and reliable real-world deployment \[2\].

## Taxonomy of Task Allocation Problems

A major step in structuring MRTA research was the taxonomy introduced by Gerkey and Mataric \[7\], which categorizes problems along three dimensions:

* Task-Type Characteristics: Single-Task Robots (ST) vs. Multi-Task Robots (MT)  
* Assignment Characteristics: Instantaneous Assignment (IA) vs. Time-Extended Assignment (TA)  
* Robot Capabilities: Single-Robot Tasks (SR) vs. Multi-Robot Tasks (MR)

These dimensions yield six possible categories, including ST-IA-SR and MT-TA-MR. Of particular relevance is **MT-TA-MR**, which involves multi-task robots making time-extended assignments for multi-robot tasks in dynamic environments. This category captures scenarios where heterogeneous agents must collaborate under uncertainty and evolving constraints.

Subsequent extensions expanded the taxonomy to account for additional complexities. Korsah et al. \[8\] introduced interdependencies such as synchronization and precedence constraints, while the TAMER model \[9\] unified these perspectives into an entity-relationship framework for multi-mission scenarios. Together, these models provide a structured basis for analyzing MRTA problems and evaluating solution methods.

## Positioning of This Research

Our research falls within the **MT-TA-MR category**, focusing on multi-task robots operating in dynamic environments that demand collaborative execution. However, it extends the paradigm by introducing **Zero-Knowledge** **assumptions**: agents possess no prior knowledge of tasks, their own capabilities, or those of others, and cannot communicate. This situates the proposed Zero-Knowledge MRTA (ZK-MRTA) as a novel problem type, pushing beyond existing MRTA frameworks and highlighting the need for new decentralized, adaptive strategies.

## Heterogeneous Task Allocation Formal Definition

In robotics, Multi-Robot Task Allocation (MRTA) addresses the fundamental question of how to assign tasks to multiple agents in a way that optimizes overall system performance. This is typically framed as an optimization problem that balances efficiency, resource usage, and mission objectives.  
Formally, let:

1.  A set of agents A \= {a1, a2, …, am};​  
2.  And a set of tasks T= {t1,t2, …,tn}​;    
3. A cost function C \= (ai,tj), representing the cost of assigning an agent ai to a task tj;  
4. A constraint function G \= (ai,tj), indicates whether the agent ai can perform a task tj based on its capabilities.

The goal is to find an assignment   ⊆ A × T that minimizes the total cost: min C (ai,tj)   
 subject to:  Gai,tj=1     ∀ai,tj ∈ Λ  

This classical formulation, as formalized in earlier works \[13-14\], captures the essence of MRTA: finding an optimal mapping between agents and tasks under resource and capability constraints. It applies across a wide range of domains, from logistics and industrial automation to surveillance and military operations. The complexity increases significantly in heterogeneous systems, where agents differ in their abilities, capacities, and resource limitations. In such cases, the cost and feasibility functions must account for variations in agent performance and task requirements, making the assignment problem considerably more challenging.

Importantly, these traditional MRTA formulations rest on a critical assumption that both the cost function 𝐶  
and the feasibility function 𝐺 are known in advance. 

This reliance on predefined models of tasks and agent capabilities enables optimization-based, market-based, and learning-based approaches to function, but it also limits their applicability in real-world, uncertain environments. In practice, agents may not know in advance how difficult a task will be, whether they themselves are capable of performing it, or whether another agent is better suited. Communication failures and environmental uncertainty further exacerbate this limitation.

Our research addresses this gap by relaxing these assumptions, introducing the Zero-Knowledge MRTA (ZK-MRTA) framework, in which neither costs nor feasibility are explicitly available to the agents. Instead, agents must learn task suitability indirectly, relying only on observations of their own actions and those of others.

## Prior Work Relevant to Our Approach

Traditional task allocation methods have been widely studied, with most approaches falling into four main categories: market-based mechanisms, optimization-based methods, reinforcement learning (RL), and coalition formation. Each operates under specific assumptions and is suited to particular environments, but all rely, explicitly or implicitly, on prior knowledge of tasks and agent capabilities.

### Market-Based Mechanisms

Market-based mechanisms model task allocation as a decentralized bidding process where agents compute utilities for tasks and submit bids to determine allocations \[4\]. These methods assume reliable inter-agent communication to exchange bids and utility information and require prior knowledge about tasks and agent capabilities to calculate meaningful utilities. For example, agents need explicit models to evaluate task difficulty or their own suitability for a task. While effective in dynamic environments, these methods face scalability challenges as the number of agents and tasks grows, primarily due to their reliance on communication and centralized auctioneers for bid resolution. Additionally, their dependence on prior knowledge limits their applicability in scenarios with uncertain or incomplete information.

### Optimization-Based Approaches

Optimization-based methods employ techniques such as linear programming and metaheuristics (e.g., genetic algorithms) to compute near-optimal task assignments \[10\]. These approaches typically require centralized knowledge of task attributes, agent capabilities, and environmental conditions to formulate the optimization problem. The assumption of a centralized decision-maker and complete system knowledge makes them unsuitable for decentralized systems or environments characterized by uncertainty or partial observability. While these methods can provide high-quality solutions, their computational complexity and reliance on global information significantly limit their scalability in real-time, large-scale systems.

### Reinforcement Learning (RL)

Reinforcement Learning (RL) allows agents to learn task allocation strategies through trial-and-error interactions with the environment. In single-robot task allocation strategies, RL agents focus on optimizing their own performance by learning policies based on individual experiences. While effective for single-agent systems, these strategies do not account for the coordination challenges inherent in multi-robot scenarios.

In multi-robot systems, Multi-Agent Reinforcement Learning (MARL) extends RL to address collaborative task allocation \[11\]. MARL assumes that agents can observe or communicate with one another to some extent, enabling them to learn cooperative strategies. However, MARL faces significant challenges, including non-stationarity (as other agents’ policies continuously evolve), scalability (due to the exponential growth of the state-action space with the number of agents), and high computational complexity. Moreover, both single-robot and MARL approaches often assume access to environmental feedback and require significant training time to converge to effective policies, which may not be feasible in dynamic, real-time scenarios.

### Coalition Formation

Coalition formation methods focus on dividing agents into dynamic groups or teams to handle tasks that require collaboration \[12\]. These methods assume partial inter-agent communication and typically require agents to negotiate or share information about their capabilities and task requirements. Coalition formation is particularly suitable for tasks requiring joint effort (e.g., lifting heavy objects or executing coordinated attacks) and emphasizes optimizing team utility functions. However, these methods depend on agents having prior knowledge of their own capabilities and those of others’ capabilities, as well as task requirements, to form effective coalitions. This reliance on explicit modeling and communication restricts their applicability in Zero-Knowledge or communication-free scenarios.

## Our Approach: Key Differences

We introduce a novel framework for task allocation under **Zero-Knowledge (ZK) assumptions**, termed **ZK-MRTA**. Unlike traditional approaches such as market-based mechanisms or coalition formation, our framework does not rely on explicit communication between agents or prior knowledge of task attributes and agent capabilities. Instead, ZK-MRTA defines a new MRTA problem type, characterized by the following assumptions:

* Agents have no prior knowledge of task properties, their own capabilities, or those of others.  
* Agents cannot communicate with one another.  
* Agents can only observe the outcomes of their own actions and, indirectly, the actions of others.

Within this framework:

* A **strategy** is defined as an algorithm that processes agent observations, incorporates memory, and selects tasks. Each strategy represents a potential solution to the ZK-MRTA problem.  
* **Performance metrics** are defined independently of any specific strategy, providing a universal basis for evaluating effectiveness. Examples include minimizing task completion time or maximizing swarm utility.

As an example candidate solution, we propose a **Collaborative Filtering (CF)-based strategy**, adapted from recommendation systems to infer agent-task relationships from sparse observational data. This is one possible approach, its relative effectiveness must be tested empirically.

To enable such evaluation, we will develop a **simulation framework** that supports systematic comparison of strategies under ZK-MRTA assumptions. This framework will allow rigorous testing of the CF-based approach against baseline strategies, including **random** selection, **greedy** heuristics, and **oracle** benchmarks, across diverse conditions such as varying task complexities, environmental noise, and degrees of observability.

## Theoretical Background: CF, MF and SVD

### Collaborative Filtering (CF)

Collaborative Filtering (CF) is a widely used technique in recommendation systems that predicts interactions, such as preferences or suitability, based on historical data. The central idea is that entities with similar interaction patterns in the past are likely to exhibit similar patterns in the future. Unlike content-based approaches, which rely on explicit features of entities, CF leverages only historical interaction data \[18\].

In the context of ZK-MRTA, CF is adapted to infer hidden relationships between agents and tasks. Here, tasks are analogous to items, agents are analogous to users, and observed outcomes (e.g., whether a task was successfully completed) form the entries of a task-agent interaction matrix. By analyzing these outcomes, agents can estimate their compatibility with tasks, learning patterns from experience rather than relying on explicit task or capability models.

CF methods can be broadly categorized into two families. **Memory-based** **approaches** directly use historical data to compute similarities between tasks or agents, relying on observed outcomes to guide predictions. In contrast, **model-based approache**s employ machine learning techniques, such as Matrix Factorization (MF), to uncover latent factors underlying the interaction patterns between agents and tasks.

#### Memory-Based Approaches

Memory-based collaborative filtering methods directly use historical data to estimate similarities between tasks or agents. For example, similarity measures such as cosine similarity or Pearson correlation can be applied to the task-agent interaction matrix to identify agents with comparable performance profiles or tasks with similar outcome patterns. Predictions are then generated by aggregating the observed outcomes of the most similar entities.

#### Model-based approaches

While conceptually simple and effective in settings with dense interaction data, memory-based methods are less suitable for ZK-MRTA, where observations are sparse, noisy, and highly dynamic. In such cases, similarity-based measures may be unreliable, and the lack of sufficient overlap in observed outcomes limits predictive accuracy. This limitation motivates the use of model-based approaches, such as Matrix Factorization and SVD, which can generalize beyond sparse data by uncovering latent structures in the interaction space. \[19-20\]

### Matrix Factorization (MF)

Matrix Factorization (MF) is a foundational model-based collaborative filtering approach. It decomposes the user-item interaction matrix R  into lower-dimensional representations that capture the latent factors driving observed interactions \[21\]. Formally, the decomposition can be expressed as: RUVT

where:

* 𝑈 is a matrix representing the latent features of users (agents), and  
* 𝑉 is a matrix representing the latent features of items (tasks).

This mapping embeds both agents and tasks into a shared latent space. The suitability of an unobserved agent-task interaction can then be estimated by computing the dot product of their corresponding latent vectors.

MF is particularly well suited to ZK-MRTA because it enables agents to infer hidden task-agent relationships dynamically, even when the available data are sparse and noisy. This property makes MF a powerful tool for decentralized environments where explicit models of task attributes and agent capabilities are unavailable.

### Singular Value Decomposition (SVD)

Singular Value Decomposition (SVD) is a specific type of MF that decomposes the user-item matrix R into three matrices: R=UΣVT

* U: An orthogonal matrix representing user latent factors.          
* : A diagonal matrix containing singular values representing the weights of the latent factors.                         
* VT: An orthogonal matrix representing item latent factors.

In the context of CF, SVD performs a **low-rank approximation** of interaction-matrix R by retaining only the top k singular values in . This reduces dimensionality, allowing the model to focus on the most significant latent factors while discarding noise \[22\]. This makes it particularly effective in sparse data scenarios, such as ZK-MRTA, where agents have limited observations of task outcomes.

### Handling Sparse and Noisy Data

In real-world scenarios like ZK-MRTA, the task-agent matrix is often sparse and noisy due to limited observations. Extensions to MF and SVD address these challenges:

1. **SVD++**: Incorporates implicit feedback, such as whether an agent attempted a task, to improve predictions. This is relevant to ZK-MRTA, where agents learn not only from their own outcomes but also from indirect observations of other agents’ actions \[23\]\].  
2. **Probabilistic Matrix Factorization (PMF)**: Introduces Gaussian priors over latent factors, making the model robust to noise and sparsity. PMF is particularly useful for ZK-MRTA as it provides probabilistic estimates of task suitability. \[22\].

## Relevance to ZK-MRTA

CF techniques, particularly SVD and PMF, address several key challenges in ZK-MRTA:

1. **Latent Relationship Inference**: Agents discover hidden relationships between tasks and their capabilities from sparse and noisy observations.  
2. **Decentralized Decision-Making**: Each agent independently maintains and updates its local matrix, enabling autonomous task selection without requiring communication or centralized coordination.  
3. **Scalability and Adaptability**: Low-rank approximations ensure computational efficiency, while extensions like PMF enhance robustness to uncertainty and evolving task environments.

By leveraging CF and its advanced techniques, ZK-MRTA provides a scalable and communication-free framework for decentralized task allocation, offering a novel solution for heterogeneous swarms operating in dynamic and uncertain environments.

# Research Statement

## Research Question

How can the ZK-MRTA problem be formally defined and addressed, and to what extent can decentralized strategies, such as a Collaborative Filtering-based approach, optimize swarm performance? 

## Methodology

This section outlines the formal framework, strategies, simulation design, and experimental procedures that will be used to address the proposed research problem. The methodology is structured as follows:

### Formal Definition of the ZK-MRTA Task

The Zero-Knowledge Multi-Robot Task Allocation (ZK-MRTA) problem is defined by two fundamental entities, agents and tasks. The set of **agents** is denoted as A \= {a1, a2, …, am}  where each agent is heterogeneous but its specific capabilities are unknown. Similarly, the set of **tasks** is defined as T= {t1,t2, …,tn} with each task characterized by unknown properties such as complexity, required resources, or resilience.

The ZK-MRTA framework operates under three key assumptions. First, agents have no prior knowledge of task attributes, their own capabilities, or the capabilities of other agents. Second, communication between agents is not permitted. Third, agents may only observe the outcomes of their own actions and, indirectly, the outcomes of actions performed by other agents.

The objective of ZK-MRTA is to develop strategies that enable agents to autonomously allocate tasks in a decentralized manner. Each strategy is an algorithm that processes observations, incorporates memory, and guides task selection, with effectiveness evaluated through performance metrics defined independently of any specific strategy.

### Variants of the Task

The ZK-MRTA framework can be extended to incorporate several variants that reflect realistic mission conditions. Two key dimensions are considered: target types and agent resource constraints.

**Target types** capture the dynamic nature of the environment. Passive targets represent static tasks that require no reaction once engaged, whereas active targets denote dynamic or retaliatory tasks, such as mobile or adaptive threats.

**Agent resource constraints** determine the operational limits of each agent. In the finite-capacity setting, agents possess limited weapons, energy, or endurance, restricting the number of tasks they can execute before requiring resupply. In contrast, the infinite-capacity case assumes that agents can perform an unlimited number of tasks, providing an idealized baseline for evaluating performance under unconstrained conditions.

### Success Criteria

The effectiveness of any strategy in ZK-MRTA is assessed using performance metrics that are defined independently of the specific strategy employed. These metrics provide a consistent basis for comparing alternative approaches under varying conditions. Three main categories of metrics are considered:

**Time-based metrics** measure the duration required to achieve a specified level of task completion, such as the time taken to destroy a given percentage of targets.

**Ratio-based metrics** evaluate performance relative to progress or remaining resources, for example, the proportion of total destruction achieved after a fixed number of rounds, or the ratio between the maximum remaining health of all targets and their initial health.

**Weighted performance metrics** combine multiple factors, such as destruction rate and time efficiency, into a single composite score to enable holistic evaluation of strategy performance.

### Baseline Strategies

To evaluate the performance of the proposed Collaborative Filtering (CF)-based strategy, a set of baseline strategies will be implemented for comparative analysis. These baselines provide reference points that represent distinct levels of knowledge and decision-making capability within the ZK-MRTA framework.

**Random Strategy**: Agents select tasks randomly, without considering previous outcomes or environmental context. This strategy serves as a lower bound for performance, reflecting behavior in the absence of any learning or inference mechanism.

**Greedy Strategy**: Agents choose tasks based on immediate and observable features, such as proximity or visible target state. Although computationally simple, this heuristic often leads to locally optimal but globally suboptimal outcomes.

**Oracle Strategy**: Agents possess perfect knowledge of task properties and their own capabilities, enabling optimal allocation decisions. This represents an upper bound on performance and serves as a theoretical benchmark for evaluating decentralized approaches.

By comparing the CF-based strategy against these baselines, the evaluation will quantify not only its absolute effectiveness but also its relative improvement over naive and idealized solutions.

### Proposed Collaborative Filtering (CF)-Based Strategy

Building on the formal framework of ZK-MRTA, we propose a Collaborative Filtering (CF)-based strategy as an example decentralized approach for task allocation under Zero-Knowledge assumptions. The central idea is to enable each agent to infer its potential effectiveness on unobserved tasks by learning from the outcomes of previous interactions, both its own and those of other agents observed in the environment.

In this approach, agents estimate a utility matrix U where each element  Uij represents the expected utility of assigning agent ai to task tj. Since agents initially have no prior knowledge, this matrix is populated and updated dynamically based on observed task outcomes. Using matrix factorization techniques, such as Singular Value Decomposition (SVD) or Probabilistic Matrix Factorization (PMF), agents learn latent representations of both tasks and agents. These latent factors capture underlying relationships that allow each agent to predict the expected success of future task assignments, even when direct data are sparse.

At each decision step, agents select the task that maximizes their predicted utility according to the learned matrix. The strategy is designed to be fully decentralized: each agent independently maintains and updates its own model without requiring communication or global synchronization. This structure ensures scalability and robustness, even in environments characterized by uncertainty and limited observability.

By leveraging CF principles, the proposed strategy aims to replicate the adaptive learning behavior of recommendation systems within a swarm robotics context, allowing agents to progressively refine their task selection policies through experience. Its performance will be empirically evaluated against the random, greedy, and oracle baselines to assess its effectiveness, scalability, and adaptability across varying operational conditions.

### Simulation Framework.

A dedicated simulation environment will be developed using the standard Gymnasium API and a PettingZoo ParallelEnv multi-agent interface. This environment serves as the executable realization of the ZK-MRTA problem: multiple static drones (agents) engage multiple static targets (tasks) in a 2D world while operating under strict Zero-Knowledge constraints: no prior information about task attributes, no knowledge of their own or others’ capabilities, and no explicit communication. The simulation is designed as a reusable testbed for comparing different decision policies, starting from a Random baseline and later extending to more informed strategies (e.g., Greedy, Oracle, Collaborative Filtering)

At its core, the framework models a discrete-time multi-agent world. In each time step, the PettingZoo ParallelEnv presents every drone with a local observation that respects the ZK-MRTA assumptions (e.g., target locations and active/inactive status only). Each strategy maps these observations into per-drone actions, such as selecting a target to engage. The environment aggregates all actions, updates target states and world time, and records performance metrics and episode traces for post-hoc analysis. This design, grounded in well-established RL interfaces, enables controlled and reproducible experiments while supporting direct, apples-to-apples comparison between different decentralized task allocation strategies under identical simulated conditions

#### Configuration Parameters

1. The simulation is instantiated from a compact configuration that defines both the **scenario** and how it is exposed through the **Gymnasium \- compatible PettingZoo ParallelEnv**. These parameters shape the decision space while preserving ZK-MRTA assumptions (zero prior knowledge, no explicit communication, decentralized per-drone decisions).  
   * **Seed.**  
      A global random seed controls all stochastic components of the environment and policies, ensuring deterministic reproducibility and fair comparison between strategies under identical conditions.  
   * **Time Step and Mission Duration.**  
      A fixed time step (`dt_s`) and mission limit (`mission_time_limit_s`) define the temporal horizon. Each environment step advances the world by one unit, and the mission limit bounds time-based metrics such as completion time or residual threat at timeout.  
   * **Number and Characteristics of Targets.**  
      The configuration specifies the number of targets and their characteristics (e.g., type, size, zone, or other scenario-specific attributes), which together determine task load and diversity.  
   * **Target Resilience.**  
      Each target is assigned an effective resilience or hit-point value derived from its characteristics. These values determine how many successful strikes are required to neutralize a target and define a hidden utility landscape that is fully known to the environment but not revealed to agents.  
   * **Number and Types of Drones.**  
      The swarm configuration includes the number of drones and, optionally, their roles or profiles (e.g., light vs. heavy strikers). Profile differences induce latent variation in drone-target effectiveness that strategies must cope with under Zero-Knowledge.  
   * **Missile Damage Score.**  
      Each drone has a fixed damage score representing its per-shot effectiveness. The environment uses these scores to update target resilience, but they are not exposed in observations, maintaining the Zero-Knowledge constraint.  
   * **Environment and Interface Parameters.**  
      Additional flags bind the scenario to the RL API: an env\_id, an observation\_mode (default: Zero-Knowledge spatial view with target positions and active/inactive flags), an action\_mode (discrete choice over NoOp and all targets), and a reward\_mode (e.g., shared reward on neutralization). Logging options control whether detailed per-step traces are recorded for post-hoc analysis of performance and emergent coordination, while a policy\_type parameter selects the decision policy (here: a uniform Random baseline). All performance metrics are computed by the environment and remain independent of the chosen policy.  
2. **Strategy Interface**  
   The simulation framework defines a standardized interface through which each drone-level strategy interacts with the PettingZoo ParallelEnv. This interface abstracts away environment implementation details while ensuring that all evaluated policies operate under identical information constraints.  
   * **Input.**  
     At each time step, a strategy receives, for each drone, its current observation vector together with any internal memory it maintains. The observation is derived from the environment’s Zero-Knowledge observation model and typically includes only spatial and activity information for targets (e.g., positions and active/inactive status), along with implicit evidence of past outcomes encoded in the evolving world state. Any richer history, such as past actions or inferred utility estimates, is stored and managed by the strategy itself as an internal memory state.  
   * **Output.**  
     Given the current observation and its internal memory, the strategy produces a per-drone action, such as selecting a target to engage or performing a no-operation. More sophisticated strategies (e.g., Collaborative Filtering-based approaches) may internally compute utility scores or task preferences, but only the chosen action is passed back to the environment through the Gymnasium/PettingZoo API.

This interface accommodates both simple baselines (such as a uniform Random policy) and learning-based or model-based strategies, while preserving the core ZK-MRTA assumptions: no prior knowledge of task or agent parameters, no explicit communication, and strictly decentralized decision-making based solely on local observations and self-maintained memory.

3. **Simulation Schema**  
   Each simulation run follows a structured sequence of steps that defines the interaction cycle between strategies and the PettingZoo ParallelEnv. This schema ensures consistent execution across all policies and supports systematic performance measurement under the Zero-Knowledge assumptions.  
   * **Initialization**.  
     The environment is instantiated according to the configuration parameters (scenario, targets, drones, mission duration, seed). Drone and target states are initialized, including target resilience and drone damage profiles, which are known to the environment but not exposed to agents. The chosen strategy (e.g., Random, Greedy, Oracle, CF-based) is bound to the per-drone observation-action interface.  
   * **Observation and Decision.**  
     At each discrete time step, the ParallelEnv produces a per-drone observation that respects the Zero-Knowledge model (e.g., target positions and active/inactive status). The strategy consults these observations together with any internal memory it maintains and selects an action for each drone (e.g., target index or NoOp), yielding a joint action dictionary for the environment.  
   * **Action and Update.**  
     The environment applies the joint action, updates target states based on accumulated damage and resilience, advances the time step, and determines whether any targets are neutralized or the episode has terminated (all targets inactive or mission time exceeded). Strategies can update their internal memory using only information derivable from subsequent observations and rewards.  
   * **Measurement and Logging.**  
     At each step, the environment records performance and state information, such as time-based progress, the fraction of neutralized targets, per-step rewards, and episode traces capturing actions and target state changes. These logs enable post-hoc computation of aggregate utility, efficiency, and coordination-related metrics without altering what agents observe.

	This structured simulation cycle provides a repeatable experimental process that isolates strategic  
behavior from environmental randomness and API details, enabling robust comparison of different task allocation strategies under identical conditions..

# Data Description

All datasets used in this research are generated by the ZK-MRTA simulation framework, implemented as a Gymnasium-compatible PettingZoo ParallelEnv. The simulation operates in discrete time steps and records complete system state transitions. Each run produces structured JSON logs describing three core components: Drones (agents), Targets (tasks), and the Environment. These logs form the basis for evaluating performance, efficiency, and emergent coordination under the Zero-Knowledge assumptions.

#### Drone Attributes.

Each drone record includes a unique identifier (drone\_id), an optional role or type label (e.g., striker\_light), its static 2D position, and its inherent damage score (used internally by the environment but not exposed to the agent policy). Per-step logs also capture the action chosen by the strategy (e.g., target index or NoOp), whether a shot was fired, the damage applied in that step, and cumulative shot counts. More advanced strategies may additionally log strategy-specific quantities (such as internal utility estimates), but these are not required by the environment and are treated as optional analysis fields rather than part of the core API.

#### Target Attributes.

Each target entry maintains a unique identifier (target\_id), its characteristics (e.g., type, zone), an effective resilience value (initial hit points derived from these characteristics), the current health, and a destruction or active flag indicating whether it has been neutralized. These attributes evolve over time as drones act on targets and collectively describe task state and progress throughout the simulation.

#### Environment Attributes.

Environment-level metadata capture the global simulation configuration, including the scenario identifier, random seed, time step (dt\_s), mission time limit (mission\_time\_limit\_s), total number of drones and targets, and summary statistics over target resilience and types. These attributes define the boundary conditions and allow runs to be reproduced and compared systematically, but remain static during a given execution.

At each simulation tick, the framework records an interaction event for analysis, such as  
⟨tick, drone\_id, action, target\_id, damage\_applied, target\_health\_post, destroyed\_flag, destruction\_ratio⟩.  
This event-level view supports fine-grained evaluation of per-drone behavior, damage allocation, and coordination patterns (e.g., overlap on the same target).

All data are persistently stored as structured JSON snapshot files, typically one snapshot per simulation tick and one summary file per run. Each snapshot captures the full environment state in a hierarchical format.

These snapshots provide a policy-agnostic, machine-readable record of each episode, enabling downstream computation of aggregate performance metrics and emergent coordination measures without modifying the underlying environment or agent interfaces.

# Work Plan: Key Milestones, Expected Deliverables, and Estimated Timeline

| Milestone | Main Activities | Expected Deliverables | Estimated Timeline |
| ----- | ----- | ----- | ----- |
| **1\. Theoretical Framework and Literature Review** | Conduct comprehensive review of existing MRTA and swarm robotics frameworks. Analyze taxonomies (Gerkey & Matarić, Korsah, TAMER) and identify the conceptual gap leading to the Zero-Knowledge (ZK) extension. Formalize ZK-MRTA definition, assumptions, and metrics. | Formal definition of ZK-MRTA. Updated literature matrix (papers, categories, methods, metrics). Draft of the “Theoretical Background” and “Formal Framework” sections | **Month 1** |
| **2\. Gymnasium & PettingZoo Simulation Environment Development** | Design and implement a discrete-time ZK-MRTA simulation environment as a Gymnasium-compatible PettingZoo ParallelEnv (no communication, zero prior knowledge, limited observability). Implement the configuration schema, per-drone strategy interface, and JSON-based episode trace logging, and validate deterministic reproducibility. | Functional multi-agent simulation engine with configurable parameters (env\_id, seed, mission time, drone/target setup, observation/reward modes) and a verified JSON snapshot / episode-trace format for data logging. | **Month 2** |
| **3\. Agent Strategy Implementation and Experimentation** | Implement baseline strategies (Random, Greedy, Oracle) and the proposed Collaborative Filtering (CF)-based strategy. Conduct comparative experiments under varying target types, agent counts, and resource assumptions. Collect and analyze metrics such as time-to-destruction and destruction ratio. | Implemented baseline and CF-based strategies. Experimental results (tables, plots, metrics). Evaluation report comparing performance under ZK conditions | **Month 3-6** |
| **4\. Analysis, Discussion, and Final Report** | Perform in-depth analysis of empirical findings. Evaluate scalability, robustness, and learning behavior of CF vs. baselines. Summarize results and draw theoretical insights for decentralized MRTA. Prepare final documentation and presentation materials. | Full written thesis and presentation slides. Discussion of implications, limitations, and future work. Final dataset and analysis scripts | **Month 7-9** |

# Bibliography

1. Sahin, E. (2005). Swarm robotics: From sources of inspiration to domains of application. In Swarm robotics (pp. 10-20). Springer, Berlin, Heidelberg.  
2. Dorigo, M., Theraulaz, G., & Trianni, V. (2020). Reflections on the future of swarm robotics. Science Robotics, 5(49), eabe4385.  
3. Brambilla, M., Ferrante, E., Birattari, M., & Dorigo, M. (2013). Swarm robotics: a review from the swarm engineering perspective. Swarm Intelligence, 7(1), 1-41.  
4. Khamis, A., Hussein, A., & Elmogy, A. (2015). Multi-robot task allocation: A review of the state-of-the-art. In Cooperative Robots and Sensor Networks 2015 (pp. 31-51). Springer, Cham.  
5. Cheraghi, A. R., Shahzad, S., & Graffi, K. (2022). Past, present, and future of swarm robotics. In Intelligent Systems and Applications: Proceedings of the 2021 Intelligent Systems Conference (IntelliSys) Volume 3 (pp. 190-233). Springer International Publishing.‏  
6. Rossi, F., Bandyopadhyay, S., Wolf, M., & Pavone, M. (2018). Review of multi-agent algorithms for collective behavior: a structural taxonomy. IFAC-PapersOnLine, 51(12), 112-117.  
7. Gerkey, B. P., & Mataric, M. J. (2004). A formal analysis and taxonomy of task allocation in multi-robot systems. The International journal of robotics research, 23(9), 939-954.‏  
8. Korsah, G. A., Stentz, A., & Dias, M. B. (2013). A comprehensive taxonomy for multi-robot task allocation. The International Journal of Robotics Research, 32(12), 1495-1512.‏  
9. Miloradović, B., Frasheri, M., Cürüklü, B., Ekström, M., & Papadopoulos, A. V. (2019, October). TAMER: Task allocation in multi-robot systems through an entity-relationship model. In International Conference on Principles and Practice of Multi-Agent Systems (pp. 478-486). Cham: Springer International Publishing.‏  
10. Mosteo, A. R., & Montano, L. (2010). A survey of multi-robot task allocation. Instituto de Investigación en Ingenierıa de Aragón, University of Zaragoza, Zaragoza, Spain, Technical Report No. AMI-009-10-TEC.‏  
11. Prorok, A., Hsieh, M. A., & Kumar, V. (2017). The impact of diversity on optimal control policies for heterogeneous robot swarms. IEEE 

12. Transactions on Robotics, 33(2), 346-358.  
13. Wu, H., Peng, Q., Shi, M., & Xing, L. (2022). A survey of UAV swarm task allocation based on the perspective of coalition formation. International Journal of Swarm Intelligence Research (IJSIR), 13(1), 1-22.‏  
14. Tang, J., Duan, H., & Lao, S. (2023). Swarm intelligence algorithms for multiple unmanned aerial vehicles collaboration: A comprehensive review. Artificial Intelligence Review, 56(5), 4295-4327.‏  
15. Vandana Dabass, et al. (2024), Swarm Based Optimization Algorithms For Task Allocation In Multi-Robot Systems: A Comprehensive Review, Educational Administration: Theory and Practice, 30(4), 2689-2695  
16. Skaltsis, G. M., Shin, H. S., & Tsourdos, A. (2023). A review of task allocation methods for UAVs. Journal of 

17. Intelligent & Robotic Systems, 109(4), 76.‏  
18. Kolling, A., Walker, P., Chakraborty, N., Sycara, K., & Lewis, M. (2016). Human interaction with robot swarms: A survey. IEEE Transactions on Human-Machine Systems, 46(1), 9-26.  
19. Prisadnikov, N. (2014). Exploration-exploitation trade-offs via probabilistic matrix factorization (Master's thesis, ETH-Zürich, Department of Computer Science).  
20. Schafer, J. B., Frankowski, D., Herlocker, J., & Sen, S. (2007). Collaborative filtering recommender systems. In The adaptive web (pp. 291-324). Springer, Berlin, Heidelberg.‏  
21. Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering recommendation algorithms. In Proceedings of the 10th International Conference on World Wide Web (pp. 285-295).  
22. Billsus, D., & Pazzani, M. J. (1998). Learning collaborative information filters. In Proceedings of the Fifteenth International Conference on Machine Learning (pp. 46-54).  
23. Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2000). Application of dimensionality reduction in recommender system-a case study. Minnesota Univ Minneapolis Dept of Computer Science.  
24. Mnih, A., & Salakhutdinov, R. R. (2007). Probabilistic matrix factorization. Advances in neural information processing systems (pp. 1257-1264).  
25. Koren, Y. (2008). Factorization meets the neighborhood: a multifaceted collaborative filtering model. Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 426-434).  
26. Salakhutdinov, R., & Mnih, A. (2008). Bayesian probabilistic matrix factorization using Markov chain Monte Carlo. In Proceedings of the 25th International Conference on Machine Learning (pp. 880-887).  
27. Brand, M. (2003). Fast online SVD revisions for lightweight recommender systems. In Proceedings of the 2003 SIAM International Conference on Data Mining (pp. 37-46).  
28. He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural collaborative filtering. In Proceedings of the 26th International Conference on World Wide Web (pp. 173-182).  
29. Rendle, S. (2010). Factorization machines. In 2010 IEEE International Conference on Data Mining (pp. 995-1000).  
30. Yu, K., & Tresp, V. (2005). Learning to learn and collaborative filtering. In Neural Information Processing Systems Workshop on Inductive Transfer (Vol. 10).‏  
31. Zhang, K., et al. (2025). Agent Learning via Early Experience. *arXiv preprint* arXiv:2510.08558.

