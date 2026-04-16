# Training and Performance Metrics Requirements

## Overview
This document defines the core metrics required to monitor and evaluate the learning progress and operational performance of the Matrix Factorization (MF) policy within the TabulaDrone environment. These metrics provide three distinct views of the system: actual performance, model accuracy, and decision reliability.

---

## 1. Rolling Average Reward

### Motivation
The primary objective of the reinforcement learning process is to maximize cumulative reward. The Rolling Average Reward tracks whether the drone’s decision quality is improving over time in practice, using the same reward signal that drives the learning process.

### Calculation
For a fixed logging window of $W$ steps, the mean observed reward is computed as:

$$
\bar{r}_{t}^{(W)} = \frac{1}{W} \sum_{k=t-W+1}^{t} r_k
$$

Where:
* $r_k$: Reward observed at step $k$.
* $W$: The window size (e.g., last 100 or 1000 engagements).
* $\bar{r}_{t}^{(W)}$: Rolling average reward at step $t$.

### Why it matters
A successful learning algorithm should show an upward trend in this metric, eventually stabilizing at a higher level than the initial random or baseline performance. Stagnation or decline indicates issues with the reward function, exploration strategy, or model capacity.

---

## 2. Prediction Error (Accuracy)

### Motivation
Matrix Factorization models the "usefulness" of a drone-target assignment as a latent relationship. This metric tracks whether the model is becoming better at estimating the true utility (reward) of an assignment.

### Calculation
For each engagement, the predicted score for the selected target is compared against the actual observed reward:

$$
e_k = \hat{r}_k - r_k
$$

The system tracks the **Mean Absolute Error (MAE)** over a rolling window:

$$
\text{MAE}_{t}^{(W)} = \frac{1}{W} \sum_{k=t-W+1}^{t} | \hat{r}_k - r_k |
$$

*Optional: Squared Error (MSE) may be used to heavily penalize large misestimates:*
$$
\text{MSE}_{t}^{(W)} = \frac{1}{W} \sum_{k=t-W+1}^{t} (\hat{r}_k - r_k)^2
$$

Where:
* $\hat{r}_k$: Predicted reward (score) at step $k$.
* $r_k$: Observed (true) reward at step $k$.

### Why it matters
As the MF model learns the latent structure of the environment, the prediction error should decrease. If rewards are improving but prediction error remains high, the model may be "stumbling" onto good actions without a solid internal representation, which can lead to instability.

---

## 3. Top-Choice Hit Rate (Reliability)

### Motivation
A model might be "accurate" on average but fail on its most confident predictions. This metric measures whether the drones' highest-ranked targets are consistently resulting in "good" outcomes.

### Calculation
At each step, the policy selects its top-ranked target. A "hit" is defined as a step where the chosen target produces a reward at or above a predefined quality threshold $\tau$.

**Binary Hit Metric:**
$$
h_k = 
\begin{cases} 
1 & \text{if } r_k \ge \tau \\
0 & \text{otherwise}
\end{cases}
$$

**Rolling Hit Rate:**
$$
\text{HitRate}_{t}^{(W)} = \frac{1}{W} \sum_{k=t-W+1}^{t} h_k
$$

Where:
* $r_k$: Observed reward from the selected top-ranked target.
* $\tau$: Predefined "good outcome" threshold.
* $h_k$: Indicator of a successful top choice.

*Note: A relative version can also be tracked, where a hit is defined as the chosen target being among the top $N$ performing targets available at that step.*

### Why it matters
The Hit Rate directly measures operational usefulness. A high hit rate means the policy's ranking is reliable for deployment. It ensures that the model isn't just "averaging out" but is actually making the best possible decisions in specific contexts.

---

## Summary of Metric Roles

| Metric | Role | Key Question |
| :--- | :--- | :--- |
| **Rolling Average Reward** | Performance | "Is actual performance improving?" |
| **Prediction Error** | Learning Quality | "Is the model understanding the environment better?" |
| **Top-Choice Hit Rate** | Operational Reliability | "Are the drone’s top decisions becoming more reliable?" |

## Part 2: Operational Scorecard (Continuous Mode Outcomes)

While Part 1 focuses on monitoring the *process* of learning, the Operational Scorecard evaluates the *outcome* of the mission. In Continuous Mode, "success" is defined by throughput, responsiveness, and dominance.

---

### 1. Throughput (Kills per 100 Steps)

**Motivation:**
Measures the raw productivity of the policy in a world without a terminal "board clear" event.

**Calculation:**
If $K$ targets were neutralized during a run of $S$ steps:

$$
\text{Throughput} = \frac{K}{S} \times 100
$$

**Interpretation:**
* **Higher is better.**
* Provides a normalized "Rate of Success" that can be compared across runs of different lengths.

---

### 2. Mean Target Life-Cycle (Response Time)

**Motivation:**
Measures how quickly the swarm reacts to and neutralizes new threats. This is the primary metric for operational responsiveness.

**Calculation:**
For each neutralized target instance $i$, let $L_i$ be the lifetime:
$L_i = t_i^{\text{death}} - t_i^{\text{spawn}}$

The Mean Target Life is the average across all $N$ neutralized targets:

$$
\text{Mean Target Life} = \frac{1}{N} \sum_{i=1}^{N} L_i
$$

**Interpretation:**
* **Lower is better.**
* A low value indicates that the swarm does not allow targets to persist on the board, representing high reactivity.

---

### 3. Steady-State Efficiency

**Motivation:**
Prevents the "weak" early learning phase from dragging down the evaluation of the "mature" policy.

**Calculation:**
Defined over a late-run evaluation window (e.g., the last 10% of steps or the last 3 logging intervals).

$$
\text{SteadyStateEff} = \frac{\text{net damage in window}}{\text{gross damage in window}}
$$

**Interpretation:**
* **Higher is better.**
* Allows for a fair comparison between an Oracle (strong from Step 1) and an MF Policy (which may only become strong after Step 500).

---

### 4. Environment Saturation (Dominance)

**Motivation:**
Measures whether the swarm is controlling the board or being overwhelmed by incoming targets.

**Calculation:**
Let $A_t$ be the number of active targets at step $t$, and $C$ be the maximum target capacity:

$$
\text{Saturation Ratio} = \frac{1}{S} \sum_{t=1}^{S} \frac{A_t}{C}
$$

**Interpretation:**
* **Lower is better.**
* **High Saturation:** Targets are accumulating faster than they can be removed.
* **Low Saturation:** The swarm is dominating the environment, keeping the board mostly clear.

---

## Comparison Summary Table (Target State)

| Policy | Throughput (K/100s) | Mean Target Life | Steady-State Eff | Saturation Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Random** | Low | High | Low | High |
| **Oracle** | High | Low | High | Low |
| **MF Policy** | Improving | Decreasing | High (Final) | Low (Final) |

## Implementation Notes
* **Part 1 (Rolling)** should be updated per-step or per-window for the **Live Dashboard**.
* **Part 2 (Scorecard)** should be calculated once per logging interval and summarized in the **Final Run Report**.
