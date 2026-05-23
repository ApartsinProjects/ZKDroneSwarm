# Operational metrics (mission language)

Re-analysis of existing runs (no new simulation): the headline skill numbers, restated as metrics a robotics reviewer remembers. ESTR is a centralized explore-then-commit method (reference, not directly comparable); every other method is decentralized and online.

## (3) Time-to-competence: rounds to reach 25%% of oracle dispatch (partial broadcast)

| method | rounds to 25%% oracle | seeds reaching |
|---|---|---|
| Random | not reached | 0% |
| UCBIndep | not reached | 0% |
| UCBHomo | not reached | 0% |
| Tabular | 46 | 75% |
| MFSGD | not reached | 0% |
| ESTR (centralized, reference) | not reached | 0% |
| PTF | not reached | 38% |
| BPMF | not reached | 0% |
| **RewardCF** | 35 | 100% |
| **BothCF** | 36 | 100% |
| **HybridCF** | 38 | 100% |

**Read:** our online CF reaches a quarter of oracle dispatch in ~35 rounds (every seed); the structure-free and batch competitors mostly never reach it within the mission.

## (6) Cumulative regret = sum_t (oracle - earned) = lost mission value (lower is better)

| method | regret (rho=1.0) | regret (rho=0.25) |
|---|---|---|
| Random | 50.3 | 50.0 |
| UCBIndep | 50.2 | 50.4 |
| UCBHomo | 49.3 | 50.3 |
| Tabular | 43.5 | 43.0 |
| MFSGD | 46.2 | 46.5 |
| ESTR (centralized, reference) | 45.9 | 46.4 |
| PTF | 45.1 | 46.0 |
| BPMF | 48.8 | 49.9 |
| **RewardCF** | 38.8 | 41.2 |
| **BothCF** | 38.9 | 41.2 |
| **HybridCF** | 42.0 | 43.1 |

**Read:** our methods accumulate the least lost mission value over the run at both broadcast rates.

## (4) New-asset readiness latency: engagements for a newcomer to become effective

| broadcast | newcomer (CF) asymptote | engagements to 50%% | to 90%% | structure-free (tabular) |
|---|---|---|---|---|
| rho=1.00 | 0.57 | 1 | 16 | asym 0.02 (never ready) |
| rho=0.50 | 0.50 | 3 | 30 | asym 0.02 (never ready) |
| rho=0.25 | 0.35 | 0 | 2 | asym 0.02 (never ready) |

**Read:** a freshly added drone becomes mission-effective on targets it never tried after a handful of engagements (order of the latent rank), independent of the total target count; a structure-free newcomer never does.

## (5) Resilience to attrition: skill retained under continuous turnover

| method | active-set skill | recent-arrivals skill |
|---|---|---|
| **RewardCF** | 0.632 | 0.074 |
| **ActiveCFconv** | 0.697 | 0.363 |
| **EMCF** | 0.842 | 0.371 |
| Tabular | 0.448 | 0.062 |
| UCBIndep | 0.619 | 0.132 |

**Read:** under 5-round turnover of 8 assets, confidence-directed CF keeps the highest skill on both the active set and (especially) recent arrivals; structure-free collapses on newcomers.

## (7) Redundancy / collision rate (honest: must be read with earned reward)

| method | coll@240 | coll@60 | coll@30 | coll@15 |
|---|---|---|---|---|
| **ContentionAdaCF** | 0.126 | 0.392 | 0.465 | 0.581 |
| **ContentionCF** | 0.245 | 0.466 | 0.535 | 0.625 |
| CBBAlite | 0.091 | 0.356 | 0.507 | 0.658 |
| MusicalChairs | 0.391 | 0.555 | 0.625 | 0.697 |
| **ActiveCFconv** | 0.207 | 0.376 | 0.503 | 0.685 |
| RewardCFconv | 0.147 | 0.371 | 0.526 | 0.660 |
| PTF | 0.187 | 0.323 | 0.448 | 0.624 |
| UCBIndep | 0.054 | 0.209 | 0.363 | 0.561 |
| Random | 0.061 | 0.211 | 0.369 | 0.563 |

**Honest caveat:** collision rate alone is NOT a clean win: random/independent dispatch has the fewest collisions simply by spreading (and earns nothing), the same spreading-vs-effectiveness tension as coverage. Among the reward-seeking de-confliction methods our private-offset sits on the efficient frontier (low collisions WHILE earning the most under severe contention); we report it for completeness, not as a standalone headline.

## (8) Information efficiency: unseen-pair skill at a low observation budget (rho=0.10)

| method | unseen skill @ low budget | per unit budget |
|---|---|---|
| **RewardCF** | 0.169 | 1.69 |
| PTF | 0.185 | 1.85 |
| UCBIndep | 0.007 | 0.07 |
| Tabular | 0.001 | 0.01 |

**Honest caveat:** low-rank methods extract real skill from a tiny observation budget while structure-free extract essentially zero; this is a structure-vs-no-structure point. Among low-rank methods the batch-refit PTF is marginally more budget-efficient here, so we do not claim this as an ours-specific win.

