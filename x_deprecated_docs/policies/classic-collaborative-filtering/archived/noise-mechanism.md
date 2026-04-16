## Noise Mechanisms for Decentralized Local Matrix-Factorization in ZK-MRTA

In the present decentralized ZK-MRTA policy, all drones maintain private local latent matrices and update them online from public interaction events. If local models become too similar over time, controlled noise can be used to preserve exploration diversity, reduce premature behavioral collapse, and improve learning robustness.

The following four noise mechanisms are listed in descending order of practical priority.

### 1. Exploration Score Noise

**Priority:** Highest

**Motivation:**  
This is the most direct and useful form of noise for the present policy. Its purpose is to prevent all drones from repeatedly choosing the same currently attractive target when their learned scores become too similar. This increases swarm-level exploration diversity and improves coverage of drone-target interactions.

**Policy Step:**  
This fits at the **target scoring and action-selection step**, after latent utility prediction and before target choice.

If drone $a$ predicts the utility of assigning itself to target $t$ as

$$
\hat{r}^{(a)}_{a,t} = \left(P^{(a)}_a\right)^\top U^{(a)}_t
$$

then score-noise action selection becomes

$$
\tilde{r}^{(a)}_{a,t} = \hat{r}^{(a)}_{a,t} + \xi^{(a)}_{t,s}
$$

with

$$
\xi^{(a)}_{t,s} \sim \mathcal{N}(0,\sigma_s^2)
$$

and drone $a$ selects

$$
t_a^\star = \arg\max_{t \in \mathcal{A}_s} \tilde{r}^{(a)}_{a,t}
$$

where $\mathcal{A}_s$ is the set of active targets at decision step $s$.

### 2. Parameter Noise

**Priority:** High

**Motivation:**  
Parameter noise perturbs the acting drone's local latent representation before scoring. Unlike pure score noise, it changes the decision process at the representation level, producing more coherent variation in behavior. This is useful when one wants each drone to preserve a slightly distinct local decision profile even under similar training histories.

**Policy Step:**  
This fits immediately **before local target scoring**, by perturbing the acting self-row.

Define a noisy acting row

$$
\tilde{P}^{(a)}_a = P^{(a)}_a + \epsilon^{(a)}_s
$$

with

$$
\epsilon^{(a)}_s \sim \mathcal{N}(0,\sigma_s^2 I_d)
$$

Then the noisy predicted utility becomes

$$
\tilde{r}^{(a)}_{a,t} = \left(\tilde{P}^{(a)}_a\right)^\top U^{(a)}_t
$$

and action selection proceeds from these perturbed scores.

### 3. Update Noise

**Priority:** Medium

**Motivation:**  
This noise acts during online learning itself. Its purpose is not primarily exploration, but rather to reduce brittle convergence and avoid premature lock-in to a particular local factorization trajectory. Since local matrix factorization is non-convex, small update noise can help maintain flexibility during learning.

**Policy Step:**  
This fits at the **latent update step**, after the reward prediction error is computed.

Given prediction error

$$
e^{(a)}_{i,t} = r_t^{(i)} - \left(P^{(a)}_i\right)^\top U^{(a)}_t
$$

the standard local updates are

$$
P^{(a)}_i \leftarrow P^{(a)}_i + \gamma \left( e^{(a)}_{i,t} U^{(a)}_t - \lambda P^{(a)}_i \right)
$$

$$
U^{(a)}_t \leftarrow U^{(a)}_t + \gamma \left( e^{(a)}_{i,t} P^{(a)}_i - \lambda U^{(a)}_t \right)
$$

A noisy update variant becomes

$$
P^{(a)}_i \leftarrow P^{(a)}_i + \gamma \left( e^{(a)}_{i,t} U^{(a)}_t - \lambda P^{(a)}_i \right) + \eta^{(a)}_{P,s}
$$

$$
U^{(a)}_t \leftarrow U^{(a)}_t + \gamma \left( e^{(a)}_{i,t} P^{(a)}_i - \lambda U^{(a)}_t \right) + \eta^{(a)}_{U,s}
$$

with

$$
\eta^{(a)}_{P,s} \sim \mathcal{N}(0,\tau_s^2 I_d),
\qquad
\eta^{(a)}_{U,s} \sim \mathcal{N}(0,\tau_s^2 I_d)
$$

where $\tau_s$ should typically be small and decayed over time.

### 4. Observation or Reward Noise

**Priority:** Lowest

**Motivation:**  
This form of noise is mainly useful for robustness testing rather than as a primary learning mechanism. It simulates imperfect sensing or imperfect public reward signals and tests whether the decentralized policy remains effective under corrupted observations.

**Policy Step:**  
This fits at the **public event observation step**, before local learning updates are applied.

If the true public event is $(i,t,r_t^{(i)})$, a noisy observation model may be written as

$$
\tilde{r}_t^{(i)} = r_t^{(i)} + \zeta_{r,s}
\qquad\text{with}\qquad
\zeta_{r,s} \sim \mathcal{N}(0,\nu_r^2)
$$

and similarly a noisy target observation may be written as

$$
\tilde{x}_{t,s} = x_{t,s} + \zeta_{x,s}
\qquad\text{with}\qquad
\zeta_{x,s} \sim \mathcal{N}(0,\nu_x^2 I)
$$

The local learner then updates from noisy public traces rather than exact ones.

### Recommended Interpretation

Among the four mechanisms, the most natural choice for the present policy is **exploration score noise**, followed by **parameter noise**. These directly improve action diversity without corrupting the public learning signal. **Update noise** is a secondary optimization-level mechanism, while **observation/reward noise** should mainly be treated as a robustness or stress-testing tool.

Accordingly, if only one noise mechanism is introduced into the policy, the preferred first choice is small decaying noise at the action-selection stage.