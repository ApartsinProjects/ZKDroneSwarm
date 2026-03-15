# Matrix Factorization Policy: Decentralized Collaborative Filtering

This document provide a comprehensive overview of the `MatrixFactorizationPolicy`, explaining how it adapts classical collaborative filtering techniques to the **Zero-Knowledge Multi-Robot Task Allocation (ZK-MRTA)** problem.

---

## 1. The Mechanism: Matrix Factorization

The goal of the policy is to learn a hidden structure that predicts the utility of any drone-target pairing based on sparse, noisy interactions observed by the entire swarm.

### 1.1 The Classic Concept: Industrial Motivation
Giant platforms like **Netflix**, **Amazon**, and **Google** face a massive challenge: how do you recommend the perfect movie, product, or app when each user has only interacted with a tiny fraction of the millions of available options? 

To solve this, they need a system that can "guess" unknown interests by identifying hidden patterns in the behavior of millions of other users. **Matrix Factorization** is the engine behind these systems. It works by mathematically decomposing a giant, empty table of interactions into two smaller, dense ones that distill complex human behaviors into simple, predictable profiles:

**The Two Hidden Matrices**  
The math assumes that every rating is the result of hidden (**latent**) matches. It breaks the giant empty table into two smaller, dense ones:
*   **The User Matrix ($P$):** Every user gets a row representing their **Taste Profile** (e.g., how much they like *Action*, *Romance*, or *Sci-Fi*).
*   **The Movie Matrix ($U$):** Every movie gets a column representing its **Trait Profile** (e.g., how much that movie *is* *Action*, *Romance*, or *Sci-Fi*).

### 1.2 The Mechanics (The Math in Simple Terms)
The magic of this approach lies in how it uses these profiles to predict and learn.

*   **The Prediction (The Alignment):**  
    To guess a rating, the computer aligns a user’s tastes with a movie’s traits and calculates their "overlap" using a **Dot Product**.
    > *Example:* If Alice's "Sci-Fi" taste is high (0.9) and the movie *The Matrix* is highly "Sci-Fi" (1.0), the resulting match is high (0.9 x 1.0 = 0.9). We sum these scores across all categories to get the final predicted rating.
*   **The Learning (The Nudge):**  
    When a real rating is provided, the computer compares the **Result** to its **Prediction**.
    *   **If the prediction was too low**, it "nudges" the User's taste and the Movie's traits slightly closer together.
    *   **If it was too high**, it pushes them apart.
    After thousands of these tiny nudges (using **Stochastic Gradient Descent**), the matrices eventually "solve" the hidden structure of the movies and the users' preferences.

## 2. Application to Drones: ZK-MRTA

### 2.1 The Recommendation Problem 
In a typical ZK-MRTA scenario, drones are dropped into an environment without knowing their own capabilities (missile damage) or their targets' vulnerabilities. The `MatrixFactorizationPolicy` treats this as a **Recommendation Problem**:

*   **Users** = Drones (with hidden weapon profiles in Matrix $P$).
*   **Items** = Targets (with hidden vulnerability profiles in Matrix $U$).
*   **Ratings** = Rewards (observed damage efficiency).

**Why it works:** By observing how every drone in the swarm hits every target, each individual drone can "triangulate" the hidden physics of the world. Even if you have never hit Target X before, you can look at the feedback from other drones and "nudge" your local understanding of Target X until your predictions become accurate.

### 2.2 The Collaborative Insight
Even if **Drone A** has never fired at **Target X**, it can learn about Target X by watching **Drone B**'s success or failure. If Drone A knows (from other interactions) that it has a similar weapon profile to Drone B, it can infer that Target X will be a good (or bad) match for itself as well.

---

## 3. System Workflow: The Policy in Action

The theory of Matrix Factorization is realized through a continuous, high-speed interaction loop between the drones and their environment. This workflow is the "heartbeat" of the system, where local latent models are incrementally updated based on real-world outcomes, bridging the gap between mathematical prediction and physical execution.

### 3.1 The Full Interaction Cycle (Code Trace)

This is how the **Main Loop**, the **Environment**, and the **Policy** interact at every time step.

1. **Observe & Update (The Swarm Intelligence)**
   Drones learn from the *previous* step's public results stored in the `obs` object.
   - **Ownership**: The **Environment** owns and generates `obs` at every step. It is the only authoritative source of truth.
   - **Content**: In Collaborative Mode, the `obs` dictionary contains:
     - `targets`: Positions and active/inactive status of all targets.
     - `selected_targets`: The action taken by **every** drone in the swarm during the last step.
     - `observed_rewards`: The noisy "labels" for the last step. These are distorted by the environment to challenge the policy's ability to filter truth from noise.
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
   - **`env.py`**: Processes damage and prepares the feedback signal in `step()`.
     ```python
     # Inside env.step()
     reward = self._calculate_reward(drone, target)
     self.last_rewards[drone_id] = reward # Ground Truth
     ```

4. **Noise Injection (The Fog of War)**
   Numerical truth in the real world is rarely pure. To simulate physical reality, the environment injects uncertainty into the feedback loop. This "Fog of War" serves a dual purpose: it forces the policy to become resilient against sensor inaccuracies and creates a fundamental distinction between high-confidence **Direct Experience** (own reward) and noisier **Social Observation** (observing others). This separation is critical for testing the swarm's ability to maintain "Social Trust" and collective intelligence even when distributed data is distorted by distance or transmission.

   Before the cycle repeats, the environment prepares the `observed_rewards` for the next `obs` packet. Noise is applied differently depending on who is being observed:

   - **A. Internal Reward Noise**: When a drone observes its **own** result, a baseline Gaussian noise ($\sigma_{reward}$) is applied. This simulates local sensor uncertainty in measuring weapon impact.
   - **B. Social Observation Noise**: When a drone observes **other** drones' results, an additional layer of distortion is added ($\sigma_{obs}$). The total noise follows the Root-Sum-Square (RSS) law: $\sigma_{total} = \sqrt{\sigma_{reward}^2 + \sigma_{observation}^2}$. This simulates the difficulty of gauging distant events versus local ones.
   - **Semantic Clipping**: For hits ($R \geq 0$), the noisy reward is clipped to $[0, 1]$ to maintain mathematical sanity.
   - **The Anti-Signal Exception**: For wasted shots ($R < 0$), noise is **skipped**. This ensures the "wasted shot" penalty remains a crisp, reliable teaching signal.

   - **`env.py` logic**:
     ```python
     if other_agent_id == agent_id:
         # Own reward (Sensor Noise)
         noise = rng.normal(0, reward_noise)
     else:
         # Other's reward (Sensor + Distance Noise)
         total_noise_std = (reward_noise**2 + observation_noise**2)**0.5
         noise = rng.normal(0, total_noise_std)
     ```

5. **Reward Modes (The Feedback Signal):**
   The environment computes the "Label" for the SGD update using one of four modes:

   *   **Damage Efficiency (Default)**: $R = \frac{\sum \min(h_k, w_k)}{\sum w_k}$
       *Measures how much of the drone's weapon potential ($w$) was actually effective against the target's remaining HP ($h$) across all attributes ($k$).*
   *   **HP Reduction**: $R = \frac{HP_{before} - HP_{after}}{HP_{before}}$
       *Measures the percentage of total target health removed by the shot, focusing on the target's survival rather than weapon potential.*
   *   **Dominant Attribute**: $R = \frac{w_{dominant}}{\max(w_{global})}$
       *Rewards damaging the target's currently healthiest attribute, encouraging drones to pivot as the target's vulnerability profile changes.*
   *   **Attribute Alignment**: $R = \text{Efficiency} \times \text{CosineSimilarity}(\vec{w}, \vec{h})$
       *Multiplies damage efficiency by the geometric alignment of the weapon profile and target HP profile, rewarding "surgical" compatibility.*

### 3.2 How Information Flows
The environment acts as the authoritative source of truth, providing **Observations** of active targets. In **Collaborative Mode**, it additionally reveals a public swarm trace (actions and rewards of all drones). The policy treats these rewards as the **Ground Truth** (the "Label") used to compute prediction errors and refine its latent model via SGD.

### 3.3 The Interaction Loop

```text
+---------------------------+                       +-------------------------------+
|        Environment        |                       |            Policy             |
|                           |                       |                               |
|  +---------------------+  |   1. Feedback (obs)   |  +-------------------------+  |
|  | Public Swarm Trace  |------------------------->|  | SGD: Predict vs. Reward |  |
|  | (Last Action+Reward)|  |   (The "Label")       |  | (Local Model Update)    |  |
|  +---------------------+  |                       |  +-------------------------+  |
|             ^             |                       |               |               |
|             |             |                       |               v               |
|             |             |                       |  +-------------------------+  |
|  +---------------------+  |                       |  | Target Selection:       |  |
|  | Physics Engine:     |  |   2. Action Selection |  | Scoring & epsilon-greedy|  |
|  | Hit -> Calc Reward  | |<-----------------------|  |                         |  |
|  +---------------------+  |                       |  +-------------------------+  |
+---------------------------+                       +-------------------------------+
```

---

### 3.4 The Mathematical Model

The policy represents the environment's complexity using two **Latent Matrices** stored locally in each drone:

1.  **Drone Latent Matrix ($P$):** A matrix of size $N_{agents} \times d$, where each row represents a drone's "weapon profile" in a $d$-dimensional hidden space.
2.  **Target Latent Matrix ($U$):** A matrix of size $d \times N_{targets}$, where each column represents a target's "vulnerability profile".

### Predicted Utility ($\hat{r}$)
The predicted utility of drone $i$ engaging target $t$ is calculated as the dot product of their latent vectors:

$$\hat{r}_{i,t} = P_i \cdot U_t = \sum_{k=1}^{d} P_{i,k} \cdot U_{k,t}$$

*   **Self-Prediction**: When a drone chooses its own action, it uses its own row in $P$: $\hat{r}_{self,t} = P_{self} \cdot U_t$.
*   **Social-Prediction**: When learning from others, it uses the rows corresponding to other drones in its local $P$ matrix.

---

### 3.5 The Training Process (Math)

The "learning" happens by adjusting the entries in $P$ and $U$ to minimize the difference between predicted utility ($\hat{r}$) and the actual reward ($r$) received from the environment.

### Objective Function
We minimize the squared error with **$L_2$ Regularization** to prevent the latent vectors from growing too large (which would cause over-confidence and instability):

$$\mathcal{L} = (r_{i,t} - \hat{r}_{i,t})^2 + \lambda (\|P_i\|^2 + \|U_t\|^2)$$

**Mechanics of the Regularization Term:**
The term $\lambda (\|P_i\|^2 + \|U_t\|^2)$ serves to constrain the growth of latent vectors, ensuring the model generalizes weapon-target physics rather than overfitting to noisy individual interactions. This specific formula is utilized for three critical design reasons:

1.  **System Symmetry & Balance**: Because the prediction is a product ($P \cdot U$), the model is sensitive to scale imbalances. The dual-penalty structure prevents states where one matrix shrinks toward zero while the other "balloons" to infinity to maintain the same product. This forces both matrices to remain at a stable, comparable scale.
2.  **Convergence to Minimal Representations**: By penalizing the squared Euclidean distance from the origin, the system is forced to find the simplest possible latent profile that explains the reward. Large parameter values are only sustained when the observed reward signal is strong enough to outweigh the regularization penalty.
3.  **Optimization Stability**: The squared magnitude is chosen for its smooth derivative ($2x$). In the SGD update rule, this results in a direct linear penalty (decay) proportional to the parameter's current value. This ensures continuous, stable updates and avoids the numerical oscillations or "jitter" associated with non-differentiable forms.

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

### 3.6 Decision Making: Exploration vs. Exploitation

The policy uses an **$\epsilon$-Greedy Strategy** with a multiplicative decay schedule:

1.  **Exploration**: With probability $\epsilon$, the drone picks a random active target. This ensures the swarm discovers new weapon-target compatibilities.
2.  **Exploitation**: With probability $1 - \epsilon$, the drone picks the target $t$ that maximizes $P_{self} \cdot U_t$.
3.  **Decay**: After every step, $\epsilon$ is reduced:
    $$\epsilon \leftarrow \max(\epsilon_{min}, \epsilon \cdot \epsilon_{decay})$$

---

## 4. Summary of Properties

| Feature | Implementation | Purpose |
| :--- | :--- | :--- |
| **Decentralized** | Every drone has its own $P$ and $U$ | Compliance with ZK-MRTA (no shared memory). |
| **Collaborative** | Learns from all swarm interaction events | Accelerates discovery of target profiles. |
| **Latent Space** | $d=8$ (default) | Compresses complex physics into simple matching logic. |
| **Amnesia-Free** | Retains matrices across episodes | Enables "training" over a long horizon. |

---

> [!NOTE]
> For more details on the underlying requirements and environment setup, refer to the [Project Requirements](../specs/project-requirements.md).
