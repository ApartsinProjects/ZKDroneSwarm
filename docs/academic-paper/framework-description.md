# 5. Framework Description

This section describes the architectural design of the ZK-MRTA benchmark framework as a reusable multi-agent environment. It explains how the framework instantiates the formal problem definition from Section 3 while preserving hidden compatibility structure and enforcing the zero-knowledge observability constraints that define the problem.

## 5.1 Overview of the Framework

The proposed framework is a latent Zero-Knowledge Multi-Robot Task Allocation (ZK-MRTA) benchmark implemented as a PettingZoo `ParallelEnv`, with action and observation spaces defined through Gymnasium `spaces`. It serves as a reusable multi-agent evaluation environment in which all agents act simultaneously at each discrete time step. Its main purpose is to provide a controlled benchmark for studying decentralized task-allocation strategies under strict information constraints while keeping the hidden compatibility structure of the environment inaccessible to the agents.

A central design principle is the separation between the **true hidden environment structure** and the **internal representations learned by policies**. The environment contains latent drone and target factors that govern actual compatibility and task outcomes, but these hidden variables are never directly observable. Instead, policies must infer useful structure indirectly from the limited public interaction history generated during execution.

## 5.2 Latent Benchmark Construction

Benchmark instances are generated in a hidden latent space whose structure is controlled by a set of configurable parameters. These include the latent dimension, the number and arrangement of latent mode centers, the structural relation between drone modes and target modes, and the sampling variance used to generate latent vectors around those centers. This design allows the environment to produce controlled low-rank compatibility structure while preserving the zero-knowledge assumption at the observation level.

The benchmark supports different structural modes of latent-world generation. In one mode, drones and targets may be generated from a shared family of latent centers; in another, they may be generated from separate mode families, allowing asymmetric hidden structure between agents and tasks. In addition, the center-generation mechanism itself is configurable, enabling more axis-aligned, orthogonal, or less structured latent geometries. As a result, the environment can represent a spectrum of hidden worlds, from relatively clean and separable compatibility patterns to more difficult and ambiguous ones.

Importantly, the framework does not expose symbolic labels such as latent mode identities to the agents. The benchmark therefore contains meaningful hidden structure, but that structure can only be approached indirectly through interaction outcomes rather than read off from explicit features.

## 5.3 Zero-Knowledge Observation Model

The observation model enforces the zero-knowledge constraint by construction: the environment internally maintains latent vectors for drones and targets, but these are excluded from the observation space. Each agent receives only the restricted tuple defined in §3.6, ensuring access to indirect evidence about hidden compatibility but not to the compatibility structure itself.

In addition to this structural constraint, the observation channel can be perturbed through configurable **observation noise**. Under this mechanism, the observed action identity of other agents may be corrupted before reaching a learner. This makes the inferred public history less reliable and enables controlled evaluation of how sensitive decentralized strategies are to imperfect perception of swarm behavior.

## 5.4 Engagement Dynamics and Task State Evolution

The framework models task execution as a persistent sequential process rather than as a one-shot assignment problem. Each target is associated with a configurable health budget, and agent actions reduce target health over time according to the hidden compatibility between drone and target latent vectors. Specifically, damage is strictly nonnegative and corresponds to the maximum of zero and the latent dot product ($d_{ij} = \max(0, g_{ij})$). A target remains active until its health reaches zero, after which it becomes inactive.

Because all agents act in parallel, simultaneous engagements must also be resolved. To do so, the environment processes joint actions in randomized order within each step. This mechanism naturally generates coordination phenomena such as **contention**, when multiple drones select the same target, and **overkill**, when total inflicted damage exceeds the remaining target health. These effects are not externally imposed penalties but emergent consequences of decentralized action selection under shared task dynamics.

This design is important because it turns the benchmark into a genuine coordination problem. A policy is not evaluated merely on whether it produces damage, but on whether it can allocate effort efficiently over time under hidden compatibility and partial information.

## 5.5 Reward Model

The reward model is designed to expose local learning signals without directly revealing the full global objective. The reward is derived from the hidden drone-target interaction but is not required to be identical to effective damage, allowing the framework to distinguish between the signal used for local learning and the actual physical progress of the swarm in neutralizing targets. The supported reward modes and noise mechanism are defined in §3.8.1.

Together with observation noise, reward noise gives the framework two separate mechanisms for controlling uncertainty: one affects the identity of observed public events, and the other affects the value attached to those events. In addition, the environment includes an explicit anti-signal for wasted actions, such as firing at already inactive targets. This makes locally uninformative or wasteful behavior visible to learning policies without converting the environment into a centrally shaped coordination objective.

## 5.6 Logging and Evaluation Support

The framework records rich execution traces that enable detailed post-hoc analysis without changing what agents observe during interaction. Logged data include environment state transitions, target status evolution, action histories, reward histories, and other diagnostic signals derived from the execution process. These records provide a policy-agnostic basis for computing performance and coordination metrics.

This logging capability is essential because many important properties of decentralized coordination, such as redundant targeting, wasted damage, or delayed task completion, cannot be understood from local rewards alone. By preserving full episode traces, the framework supports both quantitative evaluation and offline enrichment, including t-SNE latent-space visualizations that track policy embeddings over time and provide qualitative interpretation of emergent swarm behavior.

## 5.7 Summary

The framework should be understood as a **configurable latent benchmark environment** for decentralized ZK-MRTA rather than as one fixed scenario. Its architecture is defined by choices over latent-world generation, observability, task dynamics, reward design, and post-hoc analysis support. The next section specifies the particular parameter values and policy configurations used in the experiments reported in this paper.
