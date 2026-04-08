# From Classical Collaborative Filtering to ZK-MRTA: Conceptual Bridge

## Purpose of This Section

Before introducing a classical recommendation-style policy into the ZK-MRTA project, a conceptual bridge is required. The reason is that **classical collaborative filtering** and **Zero-Knowledge Multi-Agent Reinforcement Task Allocation (ZK-MRTA)** appear, at first glance, to solve different kinds of problems.

In a standard recommendation system, users express preferences over items, and the objective is to predict missing preferences. In the current project, however, drones do not express preferences. Instead, they interact with targets in a decentralized environment, receive outcome-based feedback, and must gradually learn which targets are likely to be effective choices.

Therefore, the first clarification that must be established is the following:

> In this project, collaborative filtering is not used for **preference prediction**. It is reinterpreted as a mechanism for **latent utility estimation** under decentralized, zero-knowledge constraints.

---

## 1. Analogy Between Classical Recommendation and ZK-MRTA

The first bridge is to define the structural analogy between the two domains.

In a classical recommender system:

- **users** interact with **items**
- some user-item scores are observed
- missing scores are predicted

In the ZK-MRTA setting, this analogy becomes:

- **drones / agents** correspond to **users**
- **targets / tasks** correspond to **items**
- **observed engagement outcomes or rewards** correspond to **ratings**
- **predicted score** corresponds to the **expected utility** of assigning drone `a` to target `t`

This analogy is essential because it explains how a recommendation mechanism can be meaningfully transferred into a decentralized task-allocation problem.

---

## 2. Preference Must Be Replaced by Utility

The second clarification concerns the meaning of the predicted score.

In a movie recommender, a high predicted score means:

> “This user is likely to like this item.”

In the ZK-MRTA project, a high predicted score must instead mean:

> “This drone is expected to be effective on this target.”

This is the key semantic reinterpretation.

The mechanism may still resemble classical matrix factorization or collaborative filtering, but the quantity being estimated is no longer **preference**. It is **operational utility** or **expected effectiveness**.

Thus, the policy does not recommend what an agent *likes*; it estimates what an agent is likely to perform well against.

---

## 3. Static Recommendation Becomes Sequential Decision-Making

A third clarification is required because classical recommendation is often framed as a static or offline prediction problem.

In contrast, the current project is inherently **sequential**:

1. the drone observes the current environment
2. it selects a target
3. the environment returns a reward or outcome
4. the drone updates its internal belief
5. the process repeats

Therefore, collaborative filtering is not being used here as a one-time batch predictor. Instead, it is adapted into an **online decision mechanism**.

This means the recommendation-style model becomes part of a control loop:

- estimate utility
- choose an action
- observe the result
- update the latent model

So the bridge must explicitly state that the classical recommendation mechanism is being repurposed as an **online policy component** inside a sequential task-allocation system.

---

## 4. The Interaction Matrix Is Not Given — It Is Built Through Experience

In many standard recommender problems, a global user-item interaction matrix is already partially available.

In the ZK-MRTA setting, this is not the case.

Agents begin with **no prior knowledge** about:

- target type
- hidden compatibility
- weapon effectiveness
- true task suitability

The only information available comes from interaction outcomes observed over time.

As a result, the equivalent of the user-item matrix is not pre-existing. It must be **constructed online from sparse experience**.

This is an important bridge statement because it explains why latent-factor reasoning is useful in the first place: the system must infer hidden structure from incomplete and gradually accumulated evidence.

---

## 5. Decentralization Changes the Meaning of “Collaborative”

This is one of the most important conceptual clarifications.

In many classical collaborative filtering systems, there is a **shared model** over all users and items. For example, matrix factorization is often built over a single global interaction matrix.

In the current project, however, the policy must respect the decentralized zero-knowledge setting. This means that the practical baseline should be understood as follows:

- each drone maintains a **private local latent model**
- each drone maintains a **local estimate of drone latent factors**
- each drone maintains a **local estimate of target latent factors**
- **no centralized optimizer** is used
- **no latent vectors are exchanged directly**
- collaboration arises through **shared public interaction events**, not parameter sharing

This remains inspired by classical collaborative filtering, but it is not identical to the strongest classical shared-matrix interpretation.

Therefore, the bridge should state clearly:

> The proposed policy is best understood as a **decentralized, local matrix-factorization baseline inspired by collaborative filtering**, rather than a fully centralized collaborative filtering engine.

This wording is important because it keeps the method faithful to the research proposal while remaining honest about the constraints imposed by the larger project.

---

## 6. Observation Limits Justify the Use of Latent Models

Another bridge that must be stated explicitly is the relationship between **limited observability** and **latent utility learning**.

The policy does not have access to hidden environment internals such as:

- true target class
- hidden target attributes
- true weapon-target compatibility
- full environment state beyond what is exposed through observation

Because of this restriction, the agent cannot rely on explicit symbolic matching rules.

Instead, it must infer hidden structure indirectly from:

- chosen actions
- observed rewards
- public outcomes generated by other agents

This is exactly the kind of setting in which latent-factor models become meaningful. They provide a compact internal representation for hidden compatibility patterns that cannot be read directly from the environment.

---

## 7. Recommended Framing for the Project

To align the classical recommendation mechanism with the ZK-MRTA research framing, the following interpretation should be introduced before presenting the policy itself:

> We reinterpret collaborative filtering from recommendation systems as a decentralized latent-utility learning mechanism for ZK-MRTA. Drones play the role of users, targets play the role of items, and observed engagement outcomes replace ratings. The objective is not to predict preference, but to estimate hidden drone-target effectiveness under strict zero-knowledge constraints. Because the problem is sequential, decentralized, and observation-limited, the recommender mechanism is adapted into an online per-agent decision policy rather than a standard centralized recommendation engine.

This paragraph can serve as the core bridge statement in the thesis or design documentation.

---

## 8. Positioning the Current Classical-MF Policy

Given the broader research proposal, the current `classic-cf-mf` policy should be introduced carefully.

The most accurate way to position it is:

- it is a **classical-matrix-factorization-inspired baseline**
- it adopts the **agent-task analogy** from collaborative filtering
- it learns from **reward-based interactions rather than explicit ratings**
- it operates as an **online decentralized policy**
- it uses **full local latent estimates over the drone-target interaction space**
- it is collaborative through **shared public evidence**, not through shared parameters

A useful concrete clarification is the following:

> In the concrete baseline instantiated here, each drone maintains a full local estimate of latent drone factors and latent target factors, updates that estimate from publicly observable swarm interaction events, and chooses actions using its own row within that local latent model.

This clarification is valuable because it prevents conceptual confusion. It tells the reader that the policy is part of the recommendation-inspired family, while also making the handoff to the actual implementation precise.

---

## Final Summary

Before introducing the classical recommendation-style policy into the project, the following conceptual clearance is needed:

1. define the mapping between recommendation terms and ZK-MRTA entities
2. replace the notion of **preference** with **operational utility**
3. explain that recommendation becomes an **online sequential decision mechanism**
4. clarify that the interaction matrix is **built from sparse experience**, not given in advance
5. explain that decentralization changes the meaning of **collaborative**
6. justify latent-factor learning through the project’s **strict observation limits**
7. state explicitly that the concrete baseline uses **per-drone full local latent estimates** updated from **public swarm interaction events**

Only after these points are established does the classical collaborative filtering mechanism fit naturally into the larger research narrative.