# Matrix Factorization Policy: Decentralized Collaborative Filtering

This document provide a comprehensive overview of the `MatrixFactorizationPolicy`, explaining how it adapts classical collaborative filtering techniques to the **Zero-Knowledge Multi-Robot Task Allocation (ZK-MRTA)** problem.

---

## 1. The Big Picture: Collaborative Filtering for Robots

In a typical ZK-MRTA scenario, drones are dropped into an environment without knowing their own capabilities (missile damage) or their targets' vulnerabilities. The `MatrixFactorizationPolicy` treats this as a **Recommendation Problem**:

*   **Users** = Drones (Agents)
*   **Items** = Targets
*   **Ratings** = Rewards (Observed Effectiveness)

The goal of the policy is to learn a hidden structure that predicts the utility of any drone-target pairing based on sparse, noisy interactions observed by the entire swarm.

### The Collaborative Insight
Even if **Drone A** has never fired at **Target X**, it can learn about Target X by watching **Drone B**'s success or failure. If Drone A knows (from other interactions) that it has a similar weapon profile to Drone B, it can infer that Target X will be a good (or bad) match for itself as well.

---

## 2. Environment-Policy Relationship

The policy operates within a tight feedback loop with the `DroneEngageZKMRTA` environment.

### The Full Interaction Cycle (Code Trace)

This is how the **Main Loop**, the **Environment**, and the **Policy** interact at every time step.

1. **Observe & Update (The Swarm Intelligence)**
   Drones learn from the *previous* step's public results stored in the `obs` object.
   - **Ownership**: The **Environment** owns and generates `obs` at every step. It is the only authoritative source of truth.
   - **Content**: In Collaborative Mode, the `obs` dictionary contains:
     - `targets`: Positions and active/inactive status of all targets.
     - `selected_targets`: The action taken by **every** drone in the swarm during the last step.
     - `observed_rewards`: The reward obtained by **every** drone in the swarm during the last step (subject to noise).
   - **`main.py`**: Triggers the update for all agents.
     ```python
     policy.update(obs) # MultiAgentPolicy fans out to all drones
     ```
   - **`policy.py`**: Each drone's `update_from_observation()` processes the swarm trace.
     ```python
     for agent_i in range(num_agents):
         # Error = Predicted vs. Observed Reward from swarm
         error = self._predict(agent_i, target_t) - observed_rewards[agent_i]
         # SGD Update local P and U matrices
         self.P[i] -= η * (2*e*u_t + λ*p_i)
         self.U[:, t] -= η * (2*e*p_i + λ*u_t)
     ```

2. **Select Action (The Decision)**
   - **`main.py`**: Requests actions from the policy wrapper.
     ```python
     actions = policy.select_actions(obs, info)
     ```
   - **`policy.py`**: Each drone uses its local $P$ and $U$ to score targets in `select_action()`.
     ```python
     # Predicted Reward = My latent row @ Target latent column
     predicted_reward = self.P[self.agent_idx] @ self.U[:, target_t]
     # Pick best or explore (epsilon-greedy)
     action = np.argmax(scores) if rng.random() > epsilon else random_target
     ```

3. **Execute (The Environment Step)**
   - **`main.py`**: Sends actions to the PettingZoo environment.
     ```python
     obs, rewards, terminations, truncations, info = env.step(actions)
     ```
   - **`env.py`**: Processes damage and prepares the ground truth in `step()`.
     ```python
     # Apply damage per drone weapon profile
     target.apply_damage(drone.damage_profile)
     # Compute reward (e.g. Damage Efficiency)
     rewards[drone_i] = actual_damage / max_weapon_potential
     ```

### The Interaction Loop

### The Interaction Loop

```text
+---------------------------+                    +-------------------------------+
|        Environment        |                    |            Policy             |
|                           |                    |                               |
|  +---------------------+  |                    |  +-------------------------+  |
|  | Observation:        |  |                    |  | Predict Utility: P * U  |  |
|  | Active Targets      |  +------------------->|  |                         |  |
|  +---------------------+  |                    |  +-------------------------+  |
|                           |                    |               |               |
|  +---------------------+  |                    |               v               |
|  | Public Interaction  |  |                    |  +-------------------------+  |
|  | Events: Drone,      |  |                    |  | Action Selection:       |  |
|  | Target, Reward      |  |                    |  | epsilon-Greedy          |  |
|  +---------------------+  |                    |  +-------------------------+  |
+---------------------------+                    |               |               |
                                                 |               | Action        |
                                                 +---------------|---------------+
                                                                 |
                                                                 v
                                                        +-------------------+
                                                        |    Environment    |
                                                        +-------------------+
                                                                 |
                                                                 | Reward Signal
                                                                 v
+---------------------------+                    +-------------------------------+
|        Environment        |                    |            Policy             |
|                           |                    |                               |
|  +---------------------+  |                    |  +-------------------------+  |
|  | Public Interaction  |  +------------------->|  | SGD Training:          |   |
|  | Events: Drone,      |  |                    |  | Local Latent Model     |   |
|  | Target, Reward      |  |                    |  | Update                 |   |
|  +---------------------+  |                    |  +-------------------------+  |
+---------------------------+                    |               |               |
                                                 |               | Refined Model |
                                                 |               v               |
                                                 |  +-------------------------+  |
                                                 |  | Predict Utility: P * U  |  |
                                                 |  +-------------------------+  |
                                                 +-------------------------------+
```

### 1. Information Flow
*   **Observations**: The environment provides a list of currently active targets. 
*   **Collaborative Mode**: Crucially, the policy relies on the **Collaborative Observation Mode** (OBS-C), where each drone sees every other drone's actions and rewards from the previous step.
*   **Rewards**: The environment calculates scalar rewards (e.g., HP reduction). The policy consumes these as ground truth for training.

---

## 3. The Mathematical Model

The policy represents the environment's complexity using two **Latent Matrices** stored locally in each drone:

1.  **Drone Latent Matrix ($P$):** A matrix of size $N_{agents} \times d$, where each row represents a drone's "weapon profile" in a $d$-dimensional hidden space.
2.  **Target Latent Matrix ($U$):** A matrix of size $d \times N_{targets}$, where each column represents a target's "vulnerability profile".

### Predicted Utility ($\hat{r}$)
The predicted utility of drone $i$ engaging target $t$ is calculated as the dot product of their latent vectors:

$$\hat{r}_{i,t} = P_i \cdot U_t = \sum_{k=1}^{d} P_{i,k} \cdot U_{k,t}$$

*   **Self-Prediction**: When a drone chooses its own action, it uses its own row in $P$: $\hat{r}_{self,t} = P_{self} \cdot U_t$.
*   **Social-Prediction**: When learning from others, it uses the rows corresponding to other drones in its local $P$ matrix.

---

## 4. The Training Process (Math)

The "learning" happens by adjusting the entries in $P$ and $U$ to minimize the difference between predicted utility ($\hat{r}$) and the actual reward ($r$) received from the environment.

### Objective Function
We minimize the squared error with **$L_2$ Regularization** to prevent the latent vectors from growing too large (which would cause over-confidence and instability):

$$\mathcal{L} = (r_{i,t} - \hat{r}_{i,t})^2 + \lambda (\|P_i\|^2 + \|U_t\|^2)$$

Where $\lambda$ is the regularization coefficient (`lambda_reg`).

### Stochastic Gradient Descent (SGD) Update
For every interaction event $(i, t, r)$ observed in the swarm, the drone updates its local matrices using the following rules:

1.  **Compute Error**: 
    $$e = \hat{r}_{i,t} - r$$
    *(Note: If $r < 0$ (wasted shot), the error is weighted by `anti_signal_weight` to avoid over-reacting to negative noise.)*

2.  **Update Drone Vector**: 
    $$P_i \leftarrow P_i - \eta \cdot (2 \cdot e \cdot U_t + \lambda \cdot P_i)$$

3.  **Update Target Vector**:
    $$U_t \leftarrow U_t - \eta \cdot (2 \cdot e \cdot P_i + \lambda \cdot U_t)$$

Where $\eta$ is the `learning_rate`.

---

## 5. Decision Making: Exploration vs. Exploitation

The policy uses an **$\epsilon$-Greedy Strategy** with a multiplicative decay schedule:

1.  **Exploration**: With probability $\epsilon$, the drone picks a random active target. This ensures the swarm discovers new weapon-target compatibilities.
2.  **Exploitation**: With probability $1 - \epsilon$, the drone picks the target $t$ that maximizes $P_{self} \cdot U_t$.
3.  **Decay**: After every step, $\epsilon$ is reduced:
    $$\epsilon \leftarrow \max(\epsilon_{min}, \epsilon \cdot \epsilon_{decay})$$

---

## 6. Summary of Properties

| Feature | Implementation | Purpose |
| :--- | :--- | :--- |
| **Decentralized** | Every drone has its own $P$ and $U$ | Compliance with ZK-MRTA (no shared memory). |
| **Collaborative** | Learns from all swarm interaction events | Accelerates discovery of target profiles. |
| **Latent Space** | $d=8$ (default) | Compresses complex physics into simple matching logic. |
| **Amnesia-Free** | Retains matrices across episodes | Enables "training" over a long horizon. |

---

> [!NOTE]
> For more details on the underlying requirements and environment setup, refer to the [Project Requirements](../specs/project-requirements.md).
