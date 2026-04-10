# Latent ZK-MRTA Environment: Technical Overview

This document provides a precise technical specification of the latent zero-knowledge multi-robot task allocation (ZK-MRTA) benchmark environment and the decentralized matrix factorization learning mechanism.

The environment is implemented as a PettingZoo `ParallelEnv`, where all agents act simultaneously at each step, with action and observation spaces defined using Gymnasium `spaces`. This provides a standard multi-agent reinforcement learning interface while keeping the environment internals (latent vectors, compatibility structure) hidden from the agents.

**Notation conventions**: We distinguish between the **true hidden environment structure** and the **learned policy representations**. Environment latent vectors are denoted $\mathbf{z}^{(d)}_i$ (drone $i$) and $\mathbf{z}^{(t)}_j$ (target $j$), with dimension $d_z$. Policy embeddings are denoted $P^{(a)}$ (drone matrix) and $U^{(a)}$ (target matrix) for agent $a$, with factorization dimension $d_f$. This distinction is critical: agents never observe the true $\mathbf{z}$ vectors and must learn their own approximations through interaction.

---

## 1. Latent Benchmark Construction

The latent benchmark is constructed by defining a hidden latent space of dimension $d_z$, together with a small set of latent mode centers $\{\mathbf{c}_1,\dots,\mathbf{c}_K\}$. These mode centers act as the underlying factors that govern compatibility between drones and targets. Both drones and targets are sampled from log-normal distributions around these same centers, creating a consistent but unobserved low-rank interaction structure.

The benchmark supports two structural modes. In **common mode**, drones and targets share a single set of mode centers. In **independent mode**, drones and targets each have their own set of mode centers (potentially with different counts), allowing asymmetric latent structures. The current setup uses independent mode:

```json
"latent_world": {
  "mode": "independent",
  "center_mode": "one_hot",
  "epsilon": 0.1,
  "latent_dim": 3,
  "independent": {
    "drones": { "num_modes": 3, "drone_variance": 0.2 },
    "targets": { "num_modes": 3, "target_variance": 0.2 }
  }
}
```

Formally, for each drone $i$ and target $j$, a mode $m \in \{1,\dots,K\}$ is first sampled, and then a latent vector is drawn via log-normal sampling around the corresponding center. The sampling operates in log-space and then exponentiates, which guarantees strictly positive latent vectors:

$$\log \mathbf{z}^{(d)}_i \sim \mathcal{N}(\log \mathbf{c}_{m_i}, \sigma_d^2 I),
\qquad
\log \mathbf{z}^{(t)}_j \sim \mathcal{N}(\log \mathbf{c}_{m_j}, \sigma_t^2 I)$$

where $\mathbf{z}^{(d)}_i \in \mathbb{R}_{>0}^{d_z}$ is the hidden latent vector of drone $i$, and $\mathbf{z}^{(t)}_j \in \mathbb{R}_{>0}^{d_z}$ is the hidden latent vector of target $j$.

This is implemented directly in the latent scenario builder:

```python
mode_id = int(self._rng.randint(0, num_modes))
log_center = np.log(mode_centers[mode_id])
log_sample = self._rng.normal(
    loc=log_center,
    scale=np.sqrt(variance),
    size=latent_dim,
)
sample = np.exp(log_sample)
```

The sampled vectors are then embedded into the generated drone and target configurations:

```python
{
    "position": position,
    "mode_id": mode_id,
    "latent_vector": latent_vector,
}
```

The benchmark contains a **true hidden structure**, but not necessarily a perfectly recoverable symbolic structure. The agents do not learn mode IDs directly; they only interact with outcomes induced by these continuous vectors. The mode center generation is controlled by `"center_mode"`, which supports three options:

- **`"one_hot"`**: Creates label-smoothed near-axis-aligned centers controlled by an `epsilon` parameter. With `epsilon = 0.1`, each center has a dominant component of $1 - \epsilon = 0.9$ on one axis and $\epsilon / (d_z - 1) \approx 0.05$ on others, making hidden factors clean and nearly separable while avoiding exact sparsity
- **`"orthogonal"`**: Generates orthogonal mode centers via QR decomposition of a random matrix, preserving separability while removing axis alignment
- **`"random"`**: Samples mode centers in log-space and exponentiates, producing strictly positive centers with a more challenging latent structure

In the present configuration, `"center_mode": "one_hot"` is used with `epsilon = 0.1`. Combined with the log-normal sampling, this keeps latent vectors concentrated near the smoothed axis-aligned centers. This gives the benchmark a controlled low-rank geometry without exposing it explicitly to the agents.

---

## 2. Zero-Knowledge Observation Model

The zero-knowledge observation model enforces a strict separation between the hidden structure that governs interactions and the information actually available to the agents. Although the environment internally represents drones and targets by their latent vectors $\mathbf{z}^{(d)}_i$ and $\mathbf{z}^{(t)}_j$, these quantities are never exposed in the observation space.

Instead, at each step, each agent observes only the visible world state, consisting of target positions and whether each target is still active:

```python
target_obs.extend([x, y, 1.0 if target.is_active else 0.0])
```

In addition, every agent observes the most recent joint action profile of all drones:

```python
selected_targets = np.array(
    [self.last_actions.get(aid, 0) for aid in self.possible_agents],
    dtype=np.int32,
)
```

and the most recent observed reward signals:

```python
observed_rewards = np.array(
    [self._compute_observed_reward(aid) for aid in self.possible_agents],
    dtype=np.float32,
)
```

Thus, the observation available to agent $a$ at time $t$ can be summarized as:

$$o_t^{(a)} =
\Big(
\text{target positions},
\text{target active flags},
\mathbf{s}_{t-1},
\tilde{\mathbf{r}}_{t-1},
\mathbf{w}_{t-1}
\Big)$$

where $\mathbf{s}_{t-1}$ is the vector of previously selected targets by all drones, $\tilde{\mathbf{r}}_{t-1}$ is the vector of observed rewards, and $\mathbf{w}_{t-1}$ is a binary mask indicating whether each drone's target was still active at the time of engagement. This last field is used by the integration-matrix learning mode (§5) to distinguish informative interactions from wasted shots on already-neutralized targets.

The setting is not merely partially observable in a generic sense; it is specifically designed so that the **entire compatibility structure is hidden**. The agents know where targets are and whether they remain active, but they do not know why a given drone matches a given target well. Learning therefore has to emerge from inference over public interaction history rather than access to privileged features. This makes the benchmark a direct analogue of collaborative filtering: the latent factors exist, but only sparse interaction outcomes are observable.

---

## 3. Target HP and Engagement Mechanics

The environment introduces a persistent sequential interaction dynamic through target health (HP), converting what could have been a one-shot matching problem into a multi-step coordination problem. Each target is initialized with a fixed HP budget:

```json
"target_hp": 10.0
```

which is assigned at reset:

```python
hp=self.target_hp,
hp_initial=self.target_hp,
is_active=True,
```

When drone $i$ engages target $j$, the environment first computes a latent compatibility score using the hidden vectors:

$$g_{ij} = (\mathbf{z}^{(d)}_i)^\top \mathbf{z}^{(t)}_j$$

Damage is then defined as the nonnegative part of this score:

$$d_{ij} = \max(0, g_{ij})$$

and the target HP is updated accordingly:

```python
raw_dot = self._dot_product_reward(drone, target)
damage = max(0.0, raw_dot)
target.hp -= damage
```

A target remains active until its HP reaches zero:

```python
if target.hp <= 0:
    target.hp = 0.0
    target.is_active = False
```

The environment also processes simultaneous selections in a randomized order:

```python
processing_order = list(self.agents)
self.rng.shuffle(processing_order)
```

and records how many drones selected the same target:

```python
target_selections.setdefault(target_idx, []).append(agent_id)
if len(target_selections[target_idx]) > 1:
    collisions += 1
```

This creates two important coordination effects. First, **overkill** occurs when the applied damage exceeds the target's remaining HP, causing part of the effort to be wasted. Second, **contention** arises when several agents select the same target, especially because only some of those shots may contribute useful damage once the target is depleted.

Note that when a drone fires at a target that is already inactive, the potential damage (based on the dot product) is still counted toward gross damage, even though no HP is actually removed. This affects the Damage Efficiency metric (§6).

These are primarily **task-level inefficiencies**, not standalone reward terms. The environment tracks collisions, overkill, net damage, and gross damage as diagnostics, but the learning signal itself is still local. This distinction matters because the benchmark separates the global execution objective from the immediate scalar reward observed by each agent. In other words, good coordination is necessary for task efficiency, but it is not handed to the policy as a direct centralized score.

---

## 4. Reward Signal Design

The reward signal is designed to reveal latent compatibility while remaining local, incomplete, and potentially noisy. Importantly, the reward is **not** equal to the actual damage applied to the target. The environment supports two reward modes, controlled by an internal `reward_mode` selector (currently set to `"cosine"`):

- **`"cosine"`** (default): After computing the latent dot product $g_{ij}$, the environment converts it into a direction-only alignment score using cosine similarity:

$$r_{ij} = \frac{(\mathbf{z}^{(d)}_i)^\top \mathbf{z}^{(t)}_j}
{|\mathbf{z}^{(d)}_i| \cdot |\mathbf{z}^{(t)}_j|}$$

- **`"damage"`**: The reward equals the effective damage applied to the target, i.e. $\min(d_{ij}, \text{HP}_j^{\text{before}})$.

The cosine mode is implemented as:

```python
cosine_similarity = raw_dot / (drone_norm * target_norm)
reward = float(cosine_similarity)
```

The cosine design choice is important because it decouples **task impact** from **learning signal**. Damage depends on the nonnegative magnitude of the dot product, while reward depends on angular similarity. As a result, a drone is encouraged to learn which targets are aligned with it, not necessarily which targets will produce the greatest immediate HP reduction in a globally efficient plan.

The observed reward may then be corrupted by additive Gaussian noise:

$$\tilde{r}_{ij} = r_{ij} + \epsilon,
\qquad
\epsilon \sim \mathcal{N}(0, \sigma_r^2)$$

through:

```python
base_reward = self.last_rewards.get(source_agent_id, 0.0)
noise = self.rng.normal(0, self.reward_noise)
```

with the current configuration using:

```json
"reward_noise": 0.0
```

When `reward_noise` is set to 0.0, rewards are passed through without corruption. Nonzero values introduce stochasticity that makes the learning problem harder.

There is also a special anti-signal for wasted actions: if a drone fires at a target that is already inactive, it receives a negative reward:

```python
rewards[agent_id] = -1.0
```

Importantly, **collisions themselves are not directly penalized by a dedicated collision reward term**. Instead, poor coordination becomes costly indirectly: agents may waste shots on already neutralized targets, create overkill, or fail to distribute effort efficiently. Thus, the reward signal is informative but incomplete. It carries local compatibility information, while broader notions of team efficiency remain embedded in the environment dynamics and evaluation metrics rather than in the scalar reward alone.

---

## 5. Decentralized Matrix Factorization: Learning and Prediction

The learning mechanism is based on decentralized matrix factorization, where each drone independently maintains a local latent model of the interaction space. To keep the notation precise, it is useful to distinguish between the **true hidden environment vectors** $\mathbf{z}^{(d)}_i$, $\mathbf{z}^{(t)}_j$ and the **learned policy embeddings**. For agent $a$, the learned model consists of:

* a drone embedding matrix $P^{(a)} \in \mathbb{R}^{N \times d_f}$
* a target embedding matrix $U^{(a)} \in \mathbb{R}^{d_f \times M}$

where $N$ is the number of drones, $M$ is the number of targets, and $d_f$ is the factorization dimension used by the policy.

The predicted utility assigned by agent $a$ to drone $i$ engaging target $j$ is:

$$\hat{r}^{(a)}_{ij} = \big(P^{(a)}_{i,:}\big)^\top U^{(a)}_{:,j}$$

This is implemented in the policy as:

```python
return float(self.P[drone_idx] @ self.U[:, target_idx])
```

### Prediction and action selection

At decision time, agent $a$ uses only its own row $P^{(a)}_{a,:}$ to score active targets. Action selection follows an $\varepsilon$-greedy strategy:

* with probability $\varepsilon$, the agent explores by choosing a random active target;
* otherwise, it exploits by selecting the target with the highest predicted score.

```python
if self.rng.random() < self.epsilon:
    valid_actions = [t + 1 for t in active_targets]
    action = int(self.rng.choice(valid_actions))
else:
    scores = np.array([self.predict_reward(t_idx) for t_idx in active_targets])
    best_idx = np.argmax(scores)
    action = active_targets[best_idx] + 1
```

The exploit branch also supports an optional **Boltzmann (softmax) selection** mode, controlled by the `selection_noise` parameter. When `selection_noise > 0`, instead of argmax the agent samples targets proportionally to $\exp(\text{score} / \tau)$ where $\tau$ is the selection noise temperature. This provides a softer form of exploitation that can help de-conflict decentralized agents. In the current configuration, `selection_noise = 0.0`, so pure greedy is used.

After each action, exploration decays multiplicatively:

```python
self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

The exploration-exploitation transition is implemented through a decaying exploration schedule, which gradually reduces random exploration as learning progresses.

### Learning from shared interaction data

Although the policy is decentralized, each drone observes the public swarm interaction stream through the observation vector. The policy supports two supervision modes for computing the prediction error.

**Direct mode** (when `use_integration_matrix` is false): For every drone-target-reward event $(i,j,\tilde r_{ij})$, the local model computes a prediction error directly against the observed reward:

$$e^{(a)}_{ij} = \hat{r}^{(a)}_{ij} - \tilde r_{ij}$$

**Integration-matrix mode** (when `use_integration_matrix` is true, which is the current default): Instead of learning from raw observed rewards, the policy accumulates a running mean $\bar{M}_{ij}$ for each (drone, target) pair, updating only on events where the target was still active at the time of engagement (using the $\mathbf{w}$ mask from the observation). The prediction error is then computed against this running mean:

$$\bar{M}^{(a)}_{ij} = \frac{\sum_k \tilde r^{(k)}_{ij}}{n_{ij}},
\qquad
e^{(a)}_{ij} = \hat{r}^{(a)}_{ij} - \bar{M}^{(a)}_{ij}$$

This filters out dead-target penalties from the supervision signal and provides a more stable learning target as observations accumulate.

In both modes, the embeddings are updated using SGD with $L_2$ regularization:

$$P^{(a)}_{i,:}
\leftarrow
P^{(a)}_{i,:}
- \eta \left( 2 e^{(a)}_{ij} U^{(a)}_{:,j} + \lambda P^{(a)}_{i,:} \right)$$

$$U^{(a)}_{:,j}
\leftarrow
U^{(a)}_{:,j}
- \eta \left( 2 e^{(a)}_{ij} P^{(a)}_{i,:} + \lambda U^{(a)}_{:,j} \right)$$

implemented as:

```python
predicted = self._predict_for_drone(drone_idx, target_idx)
error = predicted - float(reward)  # or predicted - m_avg in integration-matrix mode

self.P[drone_idx] -= self.learning_rate * (
    2.0 * error * u_t + self.lambda_reg * p_i
)
self.U[:, target_idx] -= self.learning_rate * (
    2.0 * error * p_i + self.lambda_reg * u_t
)
```

Negative rewards are optionally reweighted:

```python
if reward < 0:
    error *= self.anti_signal_weight
```

### Decentralization and training horizon

A crucial point to sharpen is that decentralization here means **no parameter sharing**: each drone owns its own $P^{(a)}$ and $U^{(a)}$, and these parameters are never synchronized directly across agents. Coordination emerges only because all agents observe the same public action-reward history and update their private models accordingly.

Another important detail is that learning persists across episodes. In the current episodic setup, the environment runs for 35 episodes, but the matrix factorization policy does not reset its learned parameters between episodes:

```python
if not is_deterministic and episode_num > 1:
    policy.soft_reset()
```

while:

```python
def soft_reset(self, episode_count: Optional[int] = None) -> None:
    pass
```

Thus, the learned matrices $P^{(a)}$ and $U^{(a)}$ are retained across episodes, and $\varepsilon$ continues to decay throughout training. This means the training process is effectively cumulative, even though evaluation is organized episodically.

The current configuration uses a **factorization dimension of 3** for the policy, matching the true latent world dimension of **3**. This dimension is configurable via the `collaborative_filtering.matrix_factorization_cf.latent_dim` parameter. When the policy dimension matches the world dimension, the learner has the capacity to recover representations that align with the true latent structure. However, the policy can also be configured with a different dimension—either smaller (forcing compression) or larger (providing additional representational flexibility)—allowing the learner to discover an internal predictive representation that may differ from the ground-truth coordinates while still effectively modeling the interaction structure from observations alone.

---

## 6. Evaluation and Metrics

The environment tracks several metrics to evaluate policy performance and coordination efficiency. These metrics capture both the effectiveness of task completion and the efficiency of resource utilization.

### Total Ammo Used

Total ammo used measures the cumulative number of shots fired by all drones across an episode. Each engagement action counts as one shot, regardless of the damage dealt or whether the target was already inactive. This metric provides a direct measure of resource consumption:

$$\text{Total Ammo} = \sum_{t=1}^{T} \sum_{i=1}^{N} \mathbb{1}[\text{drone } i \text{ fired at step } t]$$

where $T$ is the episode length and $N$ is the number of drones. Lower values indicate more efficient resource usage, though this must be balanced against task completion requirements.

### Shots per Target

Shots per target measures the average number of shots required to neutralize each target. It is computed as the ratio of total ammo used to the number of targets destroyed:

$$\text{Shots per Target} = \frac{\text{Total Ammo Used}}{\text{Number of Targets Destroyed}}$$

This metric directly captures coordination efficiency. The theoretical minimum is determined by the target HP and the damage potential of optimally matched drones. Values significantly above this minimum indicate coordination failures such as:

* **Overkill**: Multiple drones firing at a target that is nearly depleted
* **Mismatched assignments**: Drones engaging targets with low compatibility scores
* **Sequential inefficiency**: Targets being engaged by poorly matched drones before well-matched ones

Since each target has HP of 10.0 and damage values depend on latent compatibility, an ideal policy would minimize shots per target by learning the compatibility structure and coordinating assignments accordingly.

### Total Gross Damage

Total gross damage measures the sum of all nonnegative damage produced by the swarm's shots, including damage that is later wasted by overkill. For each engagement where drone $i$ fires at target $j$, the damage is computed as:

$$d_{ij} = \max(0, (\mathbf{z}^{(d)}_i)^\top \mathbf{z}^{(t)}_j)$$

Total gross damage accumulates all such values across the episode:

$$\text{Total Gross Damage} = \sum_{t=1}^{T} \sum_{i=1}^{N} d_{ij}(t)$$

This metric captures the total damage potential exercised by the swarm, regardless of whether that damage was useful. It includes damage applied to targets that were already nearly depleted, as well as shots fired at inactive targets.

### Total Net Damage

Total net damage measures the sum of effective damage actually applied to target HP, capped by each target's remaining HP at the time of engagement. For each shot, the effective damage is:

$$d^{\text{eff}}_{ij} = \min(d_{ij}, \text{HP}_j^{\text{before}})$$

where $\text{HP}_j^{\text{before}}$ is the target's HP before the shot. Total net damage is:

$$\text{Total Net Damage} = \sum_{t=1}^{T} \sum_{i=1}^{N} d^{\text{eff}}_{ij}(t)$$

For fully successful episodes where all targets are neutralized, this equals $\text{target\_hp} \times \text{targets\_count}$. Values below this theoretical maximum indicate that some targets were not fully destroyed.

### Damage Efficiency

Damage efficiency measures the ratio of useful damage to attempted damage:

$$\text{Dmg Eff} = \frac{\text{Total Net Damage}}{\text{Total Gross Damage}}$$

This metric quantifies how much inflicted damage was not wasted. A value of 1.0 indicates perfect efficiency—every point of damage contributed to neutralizing targets. Lower values indicate coordination failures where damage was wasted through overkill or poor target selection. This metric is particularly sensitive to the timing and coordination of engagements, as it penalizes scenarios where multiple drones fire at nearly-depleted targets.

### Total Collisions

Total collisions measures the number of same-step events in which multiple drones select the same target. At each step, if $k$ drones select the same target, this contributes $k-1$ collisions:

$$\text{Collisions}_t = \sum_{j=1}^{M} \max(0, |\{i : a_i(t) = j\}| - 1)$$

where $a_i(t)$ is the action (target selection) of drone $i$ at step $t$. Total collisions accumulates these across the episode:

$$\text{Total Collisions} = \sum_{t=1}^{T} \text{Collisions}_t$$

This metric serves as a direct indicator of redundant decentralized assignment. High collision counts suggest that the swarm is not effectively coordinating target selection, leading to wasted effort and reduced throughput. However, collisions are not inherently penalized in the reward signal—they manifest as inefficiency only through their downstream effects on overkill and ammo usage.

### Total Overkill

Total overkill measures the cumulative excess damage applied beyond a target's remaining HP after neutralization. When a shot reduces a target's HP below zero, the absolute value of the negative HP represents wasted damage:

$$\text{Overkill}_{ij} = \max(0, -\text{HP}_j^{\text{after}})$$

where $\text{HP}_j^{\text{after}}$ is the target's HP after drone $i$'s shot. Total overkill accumulates all such excess damage:

$$\text{Total Overkill} = \sum_{t=1}^{T} \sum_{i=1}^{N} \text{Overkill}_{ij}(t)$$

This metric captures wasted fire caused by redundant or poorly timed engagements. Overkill occurs when multiple drones engage the same target in quick succession, or when a single drone applies more damage than needed to neutralize a target. Unlike collisions, which only count simultaneous selections, overkill quantifies the actual damage waste resulting from poor coordination across both simultaneous and sequential engagements.
