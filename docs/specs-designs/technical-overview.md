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

### Observation noise

The observed action profile $\mathbf{s}_{t-1}$ may be corrupted by **observation noise**, controlled by the `observation_noise` parameter (currently 0.2). For each non-NoOp entry in the action vector, there is a probability `observation_noise` that the observed target selection is replaced with a uniformly random valid target ID:

```python
if self.observation_noise > 0:
    for i in range(len(selected_targets)):
        if selected_targets[i] > 0 and self.rng.random() < self.observation_noise:
            selected_targets[i] = self.rng.randint(1, self.num_targets + 1)
```

This models imperfect perception of other agents' actions — a natural constraint in decentralized settings where observing precisely which target another drone engaged may not be reliably possible. When observation noise is active, the learning signal derived from other agents' actions is corrupted at the source, making the MF supervision problem harder: the policy may attribute a reward to the wrong target, creating spurious entries in its local model.

Thus, the observation available to agent $a$ at time $t$ can be summarized as:

$$o_t^{(a)} =
\Big(
\text{target positions},
\text{target active flags},
\hat{\mathbf{s}}_{t-1},
\tilde{\mathbf{r}}_{t-1},
\mathbf{w}_{t-1}
\Big)$$

where $\hat{\mathbf{s}}_{t-1}$ is the (possibly corrupted) vector of previously selected targets by all drones, $\tilde{\mathbf{r}}_{t-1}$ is the vector of observed rewards, and $\mathbf{w}_{t-1}$ is a binary mask indicating whether each drone's target was still active at the time of engagement. This last field is used by the integration-matrix learning mode (§5) to distinguish informative interactions from wasted shots on already-neutralized targets.

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

The reward signal is designed to reveal latent compatibility while remaining local, incomplete, and potentially noisy. Importantly, the reward is **not** equal to the actual damage applied to the target. The environment supports two reward modes. The active mode is set by the `reward_mode` local variable inside `step()`, currently hardcoded to `"cosine"` (the `"damage"` branch exists in the code but is not activated):

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
"reward_noise": 0.2
```

When `reward_noise` is set to 0.0, rewards are passed through without corruption. The current value of 0.2 introduces moderate stochasticity that makes the learning problem harder and tests the robustness of the MF policy's utility estimates.

Note that the environment applies two distinct noise channels: **reward noise** ($\sigma_r$) corrupts the scalar reward signal, while **observation noise** (§2) corrupts the action identity in the observed interaction stream. Both affect the quality of the supervision data available to learning policies, but through different mechanisms — reward noise distorts the *value* of an interaction, while observation noise distorts the *identity* of the interaction partner.

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
* otherwise, it exploits by selecting the target with the highest predicted score (pure greedy argmax).

```python
if self.rng.random() < self.epsilon:
    valid_actions = [t + 1 for t in active_targets]
    action = int(self.rng.choice(valid_actions))
else:
    scores = np.array([self.predict_reward(t_idx) for t_idx in active_targets])
    best_idx = np.argmax(scores)
    action = active_targets[best_idx] + 1
```

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

In both modes, the embeddings are updated using stochastic gradient descent:

$$P^{(a)}_{i,:}
\leftarrow
P^{(a)}_{i,:}
- \eta \left( 2 e^{(a)}_{ij} U^{(a)}_{:,j} + \lambda P^{(a)}_{i,:} \right)$$

$$U^{(a)}_{:,j}
\leftarrow
U^{(a)}_{:,j}
- \eta \left( 2 e^{(a)}_{ij} P^{(a)}_{i,:} + \lambda U^{(a)}_{:,j} \right)$$

where $\eta$ is the learning rate and $\lambda$ is the regularization coefficient. These gradient updates are derived from minimizing the regularized squared-error loss:

$$\mathcal{L}^{(a)}_{ij} = \left( e^{(a)}_{ij} \right)^2 + \tfrac{\lambda}{2} \left( \| P^{(a)}_{i,:} \|^2 + \| U^{(a)}_{:,j} \|^2 \right)$$

Note that the loss value itself is never computed in the implementation—only the gradient is applied directly.

The gradient update implementation:

```python
# Compute prediction error
predicted = self._predict_for_drone(drone_idx, target_idx)
error = predicted - float(reward)  # or predicted - m_avg in integration-matrix mode

# Snapshot vectors before update (for simultaneous update)
p_i = self.P[drone_idx].copy()
u_t = self.U[:, target_idx].copy()

# SGD update with L2 regularization
# Gradient of L = error² + (λ/2)(||P||² + ||U||²)
self.P[drone_idx] -= self.learning_rate * (
    2.0 * error * u_t + self.lambda_reg * p_i
)
self.U[:, target_idx] -= self.learning_rate * (
    2.0 * error * p_i + self.lambda_reg * u_t
)
```

Negative rewards are optionally reweighted in **direct mode only** (in integration-matrix mode, dead-target shots are already excluded by the `target_was_active_at_engagement` guard, so this path is never reached for negative-reward events):

```python
if not self.use_integration_matrix and reward < 0:
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

### Total Latent Mismatch

Total latent mismatch measures the cumulative damage shortfall due to suboptimal drone-target pairing. For each shot at an **active** target, the environment computes the maximum damage any drone could deal to that target (based on the precomputed optimal dot products) and tracks the shortfall:

$$\text{Latent Mismatch}_{ij} = \max\!\Big(0,\;\max_{k} d_{kj} - d_{ij}\Big)$$

where $d_{ij} = \max(0,\, (\mathbf{z}^{(d)}_i)^\top \mathbf{z}^{(t)}_j)$ is the actual damage dealt by drone $i$ to target $j$, and $\max_k d_{kj}$ is the best damage any drone could deal to target $j$. Total latent mismatch sums this across all shots at active targets:

$$\text{Total Latent Mismatch} = \sum_{t=1}^{T}\sum_{i:\,\text{target active}} \text{Latent Mismatch}_{ij}(t)$$

The optimal damage per target is precomputed at episode reset:

```python
def _precompute_max_damage_per_target(self) -> List[float]:
    max_damages = []
    for target in self.targets:
        target_vec = np.array(target.latent_vector, dtype=np.float64)
        best = 0.0
        for drone in self.drones:
            drone_vec = np.array(drone.latent_vector, dtype=np.float64)
            dot = float(np.dot(drone_vec, target_vec))
            best = max(best, max(0.0, dot))
        max_damages.append(best)
    return max_damages
```

During step execution, latent mismatch is accumulated for each engagement with an active target:

```python
optimal_damage = self._max_damage_per_target[target_idx]
step_latent_mismatch += max(0.0, optimal_damage - damage)
```

This metric isolates assignment quality from other sources of inefficiency. Unlike overkill (which measures actual waste from shooting nearly-dead targets) and collisions (which measure redundant simultaneous selection), latent mismatch captures the **opportunity cost** of suboptimal pairing. A random policy will have high latent mismatch because it ignores latent compatibility entirely; a well-trained MF policy should reduce it as it learns which drone-target pairs produce the strongest dot products.

### Average Latent Match Quality

Average latent match quality measures the fraction of optimal damage achieved per shot through drone-target pairing decisions:

$$\text{Avg Latent Match Quality} = \frac{\text{Total Gross Damage}}{\text{Total Optimal Potential}}$$

where Total Optimal Potential is the sum of optimal damages for all shots actually fired, computed using the precomputed maximum damage per target from `_precompute_max_damage_per_target()`. This is mathematically equivalent to computing the mean of $(d_{ij} / \max_k d_{kj})$ across all shots at active targets, where $d_{ij}$ is the actual damage dealt and $\max_k d_{kj}$ is the optimal damage for that target. The metric is bounded in $(0, 1]$: a value of 1.0 means every shot was fired by the optimally matched drone, while lower values indicate suboptimal pairing. It is computed in `EpisodeMetrics.__post_init__()`:

```python
self.avg_latent_match_quality = (
    self.total_gross_damage / self.total_optimal_potential 
    if self.total_optimal_potential > 0 else 0.0
)
```

The `total_optimal_potential` is accumulated during the episode by summing the optimal damage for each shot fired at an active target, using the values from `_max_damage_per_target`.

### Episode Metrics Serialization

All metrics are computed in the `EpisodeMetrics` dataclass and serialized to JSON after each episode. The episode JSON files (`logs/run_*/policy/episodes/episode_ep*.json`) contain a `metrics` field with the following structure:

```json
{
  "metrics": {
    "episode": 1,
    "steps": 73,
    "done_reason": "all_targets_neutralized",
    "targets_neutralized": 27,
    "total_ammo_used": 73,
    "total_overkill": 8.5,
    "total_net_damage": 270.0,
    "total_gross_damage": 278.5,
    "total_collisions": 12,
    "total_latent_mismatch": 45.2,
    "shots_per_target": 2.7,
    "avg_latent_match_quality": 0.86,
    "agent_rewards": { "drone_0": 42.1, "drone_1": 38.5, ... },
    "weapon_damage_profile_mapping": { ... }
  }
}
```

The derived metrics (`shots_per_target`, `avg_latent_match_quality`) are computed automatically in `EpisodeMetrics.__post_init__()` and included in the serialized output. This ensures all episode files contain the complete metric set for downstream analysis and reporting.

---

## 7. Environment Implementation

This section specifies the implementation details required to reconstruct the environment as a functioning PettingZoo `ParallelEnv`. All previous sections describe the conceptual and mathematical structure; this section provides the engineering blueprint.

### 7.1 PettingZoo ParallelEnv Interface

The environment is implemented as a subclass of `pettingzoo.utils.env.ParallelEnv`, with action and observation spaces defined using `gymnasium.spaces`. The required interface methods are:

| Method / Property | Signature | Purpose |
|---|---|---|
| `reset(seed, options)` | `→ (observations, infos)` | Initialize episode, return initial observations |
| `step(actions)` | `→ (observations, rewards, terminations, truncations, infos)` | Process one simultaneous action step |
| `agents` | `→ List[str]` | Currently active agent IDs |
| `possible_agents` | `→ List[str]` | All agent IDs (fixed at construction) |
| `action_space(agent)` | `→ spaces.Space` | Per-agent action space |
| `observation_space(agent)` | `→ spaces.Space` | Per-agent observation space |
| `state()` | `→ Dict` | Full environment state (for logging/serialization) |

Agent IDs follow the naming convention `drone_0`, `drone_1`, ..., `drone_{N-1}`.

The environment metadata is:

```python
metadata = {"name": "drone_engage_latent_mrta"}
```

### 7.2 State Dataclasses

The environment uses two internal dataclasses for entity state and one shared dataclass for world-level bookkeeping.

**LatentDroneState**:

```python
@dataclass
class LatentDroneState:
    id: str                          # "drone_0", "drone_1", ...
    position: Tuple[float, float]    # (x, y) in world coordinates
    mode_id: int                     # latent mode assignment
    latent_vector: Tuple[float, ...]  # hidden latent vector z^(d)
    ammo_used: int = 0               # cumulative shots fired
```

**LatentTargetState**:

```python
@dataclass
class LatentTargetState:
    id: str                          # "target_0", "target_1", ...
    position: Tuple[float, float]    # (x, y) in world coordinates
    mode_id: int                     # latent mode assignment
    latent_vector: Tuple[float, ...]  # hidden latent vector z^(t)
    hp: float = 1.0                  # current hit points
    hp_initial: float = 1.0          # initial hit points (immutable reference)
    is_active: bool = True           # False once HP reaches 0
```

**WorldState** (shared across environment variants):

```python
@dataclass
class WorldState:
    world_size: Tuple[float, float]  # (width, height) bounds
    time_step: int                   # current step index
    max_steps: int                   # maximum allowed steps per episode
    scenario_id: str                 # scenario configuration identifier
    seed: Optional[int] = None       # random seed for reproducibility
```

### 7.3 Constructor and Configuration

The environment constructor accepts the following parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `world_size` | `Tuple[float, float]` | `(1000.0, 1000.0)` | 2D world bounds |
| `max_steps` | `int` | `100` | Maximum steps per episode |
| `drones_config` | `List[Dict]` | required | Drone specs from scenario builder |
| `targets_config` | `List[Dict]` | required | Target specs from scenario builder |
| `scenario_id` | `str` | `"latent_mrta_benchmark"` | Identifier for this scenario |
| `reward_noise` | `float` | `0.0` | Std dev of additive Gaussian reward noise |
| `observation_noise` | `float` | `0.0` | Probability of corrupting each observed action |
| `builder` | `Optional[Any]` | `None` | Reference to the scenario builder (for re-generation) |
| `latent_world` | `Optional[Dict]` | `None` | Latent world configuration metadata |
| `target_hp` | `float` | `1.0` | Initial HP for all targets |

Each entry in `drones_config` and `targets_config` is a dict with the structure shown in §1:

```python
{"position": [x, y], "mode_id": int, "latent_vector": [float, ...]}
```

The constructor also derives two compatibility metadata structures used by the visualization frontend:

- **`class_attribute_mapping`**: Maps each `mode_{id}` to `{"latent_reward": target_hp}`, providing a uniform schema for frontend display.
- **`weapon_damage_profile_mapping`**: Maps each drone mode to an estimated max-damage-per-shot, computed as the mean drone-vector norm for that mode multiplied by the global mean target-vector norm. This serves as a Cauchy–Schwarz upper-bound estimate for the mode's achievable dot-product damage.

### 7.4 Action and Observation Spaces

**Action space** — `spaces.Discrete(num_targets + 1)`:

- Action `0` = NoOp (do nothing)
- Action `k` for $k \in \{1, \dots, M\}$ = fire at target $k-1$ (0-indexed internally)

**Observation space** — `spaces.Dict` with four keys:

| Key | Space | Shape | Dtype | Description |
|---|---|---|---|---|
| `targets` | `Box(0, +∞)` | `(3 × M,)` | `float32` | Flattened `[x, y, is_active]` per target |
| `selected_targets` | `Box(0, M)` | `(N,)` | `int32` | Last action per drone (possibly noised) |
| `observed_rewards` | `Box(-∞, +∞)` | `(N,)` | `float32` | Last reward per drone (possibly noised) |
| `target_was_active_at_engagement` | `Box(0, 1)` | `(N,)` | `int8` | Whether target was active when drone fired |

The `targets` array is built by iterating over all targets in order:

```python
for target in self.targets:
    target_obs.extend([x, y, 1.0 if target.is_active else 0.0])
```

### 7.5 Reset

On `reset(seed, options)`:

1. Initialize `self.rng = np.random.RandomState(seed)`
2. Create a fresh `WorldState` with `time_step=0`
3. Build `LatentDroneState` list from `drones_config`, each with `ammo_used=0`
4. Build `LatentTargetState` list from `targets_config`, each with `hp=target_hp`, `hp_initial=target_hp`, `is_active=True`
5. Reset `last_actions` to all `0`, `last_rewards` to all `0.0`, `last_target_was_active_at_engagement` to all `0`
6. Reset `cumulative_neutralizations` to `0`
7. Compute and return initial observations (no noise applied since all last-actions are NoOp) and per-agent info dicts

### 7.6 Step Execution Flow

The `step(actions)` method processes one simultaneous action step. The execution flow is:

**Phase 1 — Validation**:

```python
missing_agents = set(self.agents) - set(actions.keys())
if missing_agents:
    raise ValueError(...)
for agent_id, action in actions.items():
    if not self.action_spaces[agent_id].contains(action):
        raise ValueError(...)
```

**Phase 2 — Randomize processing order**:

```python
processing_order = list(self.agents)
self.rng.shuffle(processing_order)
```

**Phase 3 — Process engagements** (in shuffled order):

For each `agent_id` in `processing_order`:

1. Record `last_actions[agent_id] = action`
2. If action is `0` (NoOp): set reward to `0.0`, `target_was_active = 0`, continue
3. Extract `drone_idx` from agent ID, get drone and target references
4. Increment `drone.ammo_used`
5. Record target selection for collision tracking
6. **If target is inactive**: compute gross damage (dot product, clamped ≥ 0) for metrics, assign reward `−1.0`, set `target_was_active = 0`, continue
7. **If target is active**:
   - Compute raw dot product $g_{ij}$, damage $d_{ij} = \max(0, g_{ij})$
   - Accumulate gross damage
   - Deduct damage from `target.hp`
   - If `target.hp ≤ 0`: record overkill as $|\text{hp}|$, set `hp = 0`, `is_active = False`, increment neutralizations
   - Compute reward using the active reward mode (§4)
   - Record effective damage as $\min(d_{ij}, \text{hp\_before})$

**Phase 4 — Update world state**:

```python
self.cumulative_neutralizations += neutralizations_this_step
self.world.time_step += 1
```

**Phase 5 — Determine termination**:

- `all_targets_done`: all targets have `is_active = False`
- `max_steps_reached`: `world.time_step >= max_steps`
- `done_reason`: `"all_targets_neutralized"` or `"max_steps_reached"` or `None`

Termination and truncation dicts follow PettingZoo conventions:

```python
terminations = {agent_id: all_targets_done for agent_id in self.agents}
truncations = {agent_id: max_steps_reached and not all_targets_done for agent_id in self.agents}
```

**Phase 6 — Compute observations and diagnostics**, then return the 5-tuple.

### 7.7 Observation Construction

`_compute_observations()` is called at the end of both `reset()` and `step()`. It:

1. Builds the `targets` array (positions + active flags)
2. Builds the `selected_targets` array from `last_actions`
3. Applies **observation noise** to `selected_targets` (§2): for each non-NoOp entry, with probability `observation_noise`, replace the target ID with a uniform random sample from $\{1, \dots, M\}$
4. For each agent, computes `observed_rewards` by calling `_compute_observed_reward()` per drone, which adds Gaussian noise $\mathcal{N}(0, \sigma_r^2)$ to each `last_rewards` entry when `reward_noise > 0`
5. Copies the `target_was_active_at_engagement` mask from the step's records

Note that `selected_targets` is built once and shared across all agents (all agents see the same noised action profile), while `observed_rewards` is computed per-observing-agent (each agent gets an independent noise draw on the reward values).

### 7.8 Diagnostics Snapshot

After each step (and at reset), the environment builds an `EnvDiagnosticsSnapshot` dataclass capturing per-step telemetry:

```python
@dataclass
class EnvDiagnosticsSnapshot:
    step_index: int
    scenario_id: str
    actions: Dict[str, int]
    ammo_used: Dict[str, int]
    weapon_types: List[str]          # ["mode_0", "mode_1", ...] per drone
    target_hps: List[float]          # current HP per target
    target_classes: List[str]        # ["mode_0", "mode_1", ...] per target
    target_active: List[bool]        # active flag per target
    processing_order: Optional[List[str]] = None
    net_damage: Optional[float] = None
    neutralizations_this_step: Optional[int] = None
    cumulative_neutralizations: Optional[int] = None
    collisions: Optional[int] = None
    target_selections: Optional[Dict[int, List[str]]] = None
    overkill: Optional[Dict[int, float]] = None
    done_reason: Optional[str] = None
    total_gross_damage: Optional[float] = None
    latent_mismatch: Optional[float] = None
    optimal_potential: Optional[float] = None
```

The snapshot is serialized via `to_dict()` which deep-copies all mutable fields and includes optional fields only when non-`None`. The snapshot is stored on the environment as `self._latest_diagnostics` and is accessible via a `diagnostics` property.

The PettingZoo `infos` dict returned to agents is intentionally empty per agent (`{agent_id: {} for agent_id in self.agents}`). The diagnostics snapshot is an internal telemetry channel, not part of the agent observation contract.

### 7.9 State Serialization

The `state()` method returns a full environment snapshot as a plain dict, structured as:

```python
{
    "world": {
        "world_size": [width, height],
        "time_step": int,
        "max_steps": int,
        "scenario_id": str,
        "seed": Optional[int],
    },
    "drones": [
        {"id": str, "position": [x, y], "ammo_used": int,
         "mode_id": int, "latent_vector": [float, ...]},
        ...
    ],
    "targets": [
        {"id": str, "position": [x, y], "mode_id": int,
         "latent_vector": [float, ...], "hp": float,
         "hp_initial": float, "is_active": bool},
        ...
    ],
}
```

This includes the hidden latent vectors and is intended for logging and offline analysis, not for agent consumption.
