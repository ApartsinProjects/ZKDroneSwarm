# Selfish ε-Greedy Collaborative Filtering Policy

This folder contains comprehensive documentation for the `SelfishEpGreedyCFPolicy`, a decentralized learning policy for Zero-Knowledge Multi-Robot Task Allocation (ZK-MRTA).

---

## Documentation Structure

### Walkthrough (Start Here)

A step-by-step deep dive following **drone_0** through **Step 2** of Episode 1:

| File | Description |
|------|-------------|
| [00_overview.md](00_overview.md) | **The Big Picture** — What the policy does and why |
| [01_math_foundation.md](01_math_foundation.md) | **The Math** — Full SGD derivation from first principles |
| [02_initial_state.md](02_initial_state.md) | **Initial State** — drone_0's latent vectors before learning |
| [03_step2_full_cycle.md](03_step2_full_cycle.md) | **The Full Cycle** — Observe → Learn → Select → Execute |
| [04_insights.md](04_insights.md) | **Key Insights** — What the vectors learn, emergent coordination |

### Reference

| File | Description |
|------|-------------|
| [api_reference.md](api_reference.md) | API documentation, hyperparameters, edge cases |

---

## Quick Links

### Key Concepts

- **Matrix Factorization**: [01_math_foundation.md#1-matrix-factorization-setup](01_math_foundation.md#1-matrix-factorization-setup)
- **SGD Update Rules**: [01_math_foundation.md#4-sgd-update-rules](01_math_foundation.md#4-sgd-update-rules)
- **ε-Greedy Selection**: [03_step2_full_cycle.md#phase-3-action-selection](03_step2_full_cycle.md#phase-3-action-selection)
- **Collaborative Learning**: [04_insights.md#2-why-collaborative-learning-works](04_insights.md#2-why-collaborative-learning-works)

### Key Equations

| Equation | Location |
|----------|----------|
| Predicted reward: $\hat{r} = \frac{1 + \mathbf{u} \cdot \mathbf{v}}{2}$ | [01_math_foundation.md](01_math_foundation.md) |
| SGD update: $\mathbf{u} \leftarrow \text{norm}(\mathbf{u} + \eta \cdot e \cdot \mathbf{v})$ | [01_math_foundation.md](01_math_foundation.md) |
| ε-decay: $\epsilon \leftarrow \max(\epsilon_{\min}, \epsilon \cdot \epsilon_{\text{decay}})$ | [03_step2_full_cycle.md](03_step2_full_cycle.md) |

---

## Source Code

| Component | File |
|-----------|------|
| Policy implementation | `tabula_drone/policies/selfish_ep_greedy_cf_policy.py` |
| Base class (SGD logic) | `tabula_drone/policies/base_cf_policy.py` |
| Environment | `tabula_drone/envs/drone_engage_zk_mrta_v0.py` |
| Episode runner | `main_zk_mrta.py` |

---

## Data Sources

The walkthrough uses real data from:

- **Episode log**: `logs/run_20260106_122459/selfish_ep_greedy_cf/episodes/episode_first_ep01.json`
- **Learning state**: `logs/run_20260106_122459/selfish_ep_greedy_cf/learning_state/learning_state_ep01.json`

### Configuration Used

```json
{
  "latent_dim": 3,
  "learning_rate": 0.05,
  "epsilon": 0.3,
  "epsilon_decay": 0.99,
  "epsilon_min": 0.05
}
```

**Reward mode**: `REWARD_DOMINANT_ATTRIBUTE=False` (HP reduction)

---

## Reading Order

**For understanding the policy end-to-end:**

```
00_overview.md → 01_math_foundation.md → 02_initial_state.md → 03_step2_full_cycle.md → 04_insights.md
```

**For quick reference:**

```
api_reference.md
```

**For specific topics:**

- "How does prediction work?" → [01_math_foundation.md](01_math_foundation.md)
- "What happens in a single step?" → [03_step2_full_cycle.md](03_step2_full_cycle.md)
- "Why does it work?" → [04_insights.md](04_insights.md)
