## Interpretation of the Local Latent Matrices

Let:

- $N_d$ be the number of drones,
- $N_t$ be the number of targets,
- $d$ be the latent dimension.

Each drone $a$ maintains its own local matrix-factorization model:

$$
P^{(a)} \in \mathbb{R}^{N_d \times d}
$$

$$
U^{(a)} \in \mathbb{R}^{d \times N_t}
$$

where:

- $P^{(a)}$ is drone $a$'s local drone latent matrix,
- $U^{(a)}$ is drone $a$'s local target latent matrix,
- $P^{(a)}_i \in \mathbb{R}^{d}$ is drone $a$'s local latent estimate of drone $i$,
- $U^{(a)}_t \in \mathbb{R}^{d}$ is drone $a$'s local latent estimate of target $t$.

The predicted utility of assigning drone $i$ to target $t$ inside drone $a$'s local model is

$$
\hat{r}^{(a)}_{i,t}
=
\left(P^{(a)}_i\right)^\top U^{(a)}_t
$$

When drone $a$ selects its own action, it uses only its own row $P^{(a)}_a$ against the available target columns $U^{(a)}_t$.

### What the Drone Matrix Represents

The matrix $P^{(a)}$ should be interpreted as drone $a$'s private latent belief over drone-side engagement capability.

Its rows do not represent symbolic knowledge such as weapon-type labels or explicit damage vectors. Those quantities are hidden from the agents in the environment. Instead, each row $P^{(a)}_i$ is a compressed latent summary of how drone $i$ tends to perform across targets, as inferred from observable interaction outcomes.

Thus, the drone matrix is best understood as encoding latent capability structure such as:

- how effective a drone tends to be in general,
- which hidden target profiles it tends to match well,
- and how its historical engagement outcomes compare across targets.

For the acting drone, the most operationally important row is

$$
P^{(a)}_a
$$

because this is the row used for self-specific target scoring at decision time. The remaining rows $P^{(a)}_i$ for $i \neq a$ represent drone $a$'s local beliefs about the other drones.

### What the Target Matrix Represents

The matrix $U^{(a)}$ should be interpreted as drone $a$'s private latent belief over target-side engagement suitability.

Its columns do not represent symbolic target classes or explicit attribute vectors. Those target properties exist internally in the environment, but they are hidden from the drones. Instead, each column $U^{(a)}_t$ is a compressed latent summary of how target $t$ tends to respond to different drones, as inferred only through observed public engagement events and rewards.

In practical terms, a target latent vector may come to encode hidden regularities such as:

- whether this target tends to yield high or low reward for certain drones,
- whether it is currently a promising or poor engagement choice,
- and what kind of hidden compatibility pattern it appears to have.

So the target matrix should be interpreted as a latent opportunity model rather than an explicit target descriptor.

### Why Both Matrices Are Needed

The two-matrix structure is meaningful because the environment contains hidden structure on both sides of the interaction.

On the drone side, each drone has a fixed internal weapon damage profile. On the target side, each target has a hidden class and hidden per-attribute health structure. Since neither side is directly known to the agents, the factorization

$$
\hat{r}^{(a)}_{i,t}
=
\left(P^{(a)}_i\right)^\top U^{(a)}_t
$$

lets each drone approximate the hidden compatibility between drone-side capability and target-side vulnerability through learned latent geometry rather than symbolic reasoning.

In this sense:

- $P^{(a)}$ captures **who is engaging** in latent form,
- $U^{(a)}$ captures **what is being engaged** in latent form,
- and their dot product captures **how effective that engagement is expected to be**.

### Stability Difference Between the Two Matrices

Although both matrices are necessary, they are not equally stable in meaning.

The drone matrix $P^{(a)}$ is relatively stable because drone capability is mostly fixed within an episode. A drone's weapon type and damage profile are assigned at initialization and remain constant during the episode. This makes a persistent latent row per drone conceptually natural.

The target matrix $U^{(a)}$, however, carries a heavier burden. This is because target usefulness changes over time. The reward for firing at a target depends on the target's remaining per-attribute health before the hit, and those hidden target attributes are reduced after each engagement. As a result, the effective utility of a target is not completely stationary.

So while $U^{(a)}_t$ is written as one latent vector per target, in practice that vector must absorb both:

- hidden target type structure, and
- the evolving engagement state of that target over time.

This means the target matrix is partly a latent model of target identity and partly a compressed memory of target condition.

### Role of Public Interaction Traces

The policy states that collaboration arises through shared public interaction events rather than parameter exchange.

In the current environment implementation, this is operationally supported because each agent's observation includes not only target activity information, but also arrays of the last selected targets and the last observed rewards for all drones.

This matters directly for the latent matrices.

Because drone $a$ can observe public action-reward traces from the swarm, it can update:

- $P^{(a)}_a$ from its own actions,
- $P^{(a)}_i$ for other drones from their publicly visible actions and outcomes,
- and $U^{(a)}_t$ from whichever targets were publicly engaged.

Accordingly, the full matrix $P^{(a)}$ is justified in the present implementation. It is not merely a self-row plus unused placeholders. It is a genuine local reconstruction of multi-drone capability structure from public swarm evidence.

### Final Interpretation

The most precise interpretation of the local latent matrices is therefore the following:

$$
P^{(a)}
=
\text{drone } a\text{'s private latent belief over drone capabilities}
$$

$$
U^{(a)}
=
\text{drone } a\text{'s private latent belief over target engagement suitability}
$$

and

$$
\left(P^{(a)}_i\right)^\top U^{(a)}_t
=
\text{drone } a\text{'s predicted engagement effectiveness for pair }(i,t)
$$

Under this interpretation:

- the drone matrix is primarily a latent capability model,
- the target matrix is primarily a latent target-opportunity model,
- the acting drone uses only its own row for decision-making,
- and learning updates both matrices from public interaction events.

The main conceptual caution is that the target matrix is being asked to summarize a partially non-stationary object, because target condition changes during the episode. By contrast, the drone matrix is modeling a more stable latent structure. For that reason, the local target matrix is the more approximate of the two, while the local drone matrix is the more structurally stable one.