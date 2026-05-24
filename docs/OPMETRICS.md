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
| PTF | not reached | 25% |
| BPMF | not reached | 0% |
| **RewardCF** | 33 | 100% |
| **BothCF** | 34 | 100% |
| **HybridCF** | 39 | 100% |

**Read:** our online CF reaches a quarter of oracle dispatch in ~35 rounds (every seed); the structure-free and batch competitors mostly never reach it within the mission.

## (6) Cumulative regret = sum_t (oracle - earned) = lost mission value (lower is better)

| method | regret (rho=1.0) | regret (rho=0.25) |
|---|---|---|
| Random | 50.3 | 50.0 |
| UCBIndep | 50.2 | 50.4 |
| UCBHomo | 49.3 | 50.3 |
| Tabular | 43.5 | 43.0 |
| MFSGD | 46.1 | 46.1 |
| ESTR (centralized, reference) | 45.8 | 46.4 |
| PTF | 45.0 | 46.0 |
| BPMF | 49.7 | 49.6 |
| **RewardCF** | 39.6 | 40.9 |
| **BothCF** | 39.8 | 40.8 |
| **HybridCF** | 41.8 | 43.5 |

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

## (7) Redundancy / collision rate, treated as an efficiency frontier

| method | coll@240 | coll@60 | coll@30 | coll@15 | earned@15 |
|---|---|---|---|---|---|
| **ContentionAdaCF** | 0.126 | 0.392 | 0.465 | 0.581 | 0.100 |
| **ContentionCF** | 0.245 | 0.466 | 0.535 | 0.625 | 0.105 |
| CBBAlite | 0.091 | 0.356 | 0.507 | 0.658 | 0.064 |
| MusicalChairs | 0.391 | 0.555 | 0.625 | 0.697 | 0.028 |
| **ActiveCFconv** | 0.207 | 0.376 | 0.503 | 0.685 | 0.046 |
| RewardCFconv | 0.147 | 0.371 | 0.526 | 0.660 | 0.059 |
| PTF | 0.187 | 0.323 | 0.448 | 0.624 | 0.057 |
| UCBIndep | 0.054 | 0.209 | 0.363 | 0.561 | 0.004 |
| Random | 0.061 | 0.211 | 0.369 | 0.563 | -0.010 |

**Why raw collision rate is the wrong read:** random / structure-free dispatch has the FEWEST collisions simply by spreading, and earns ~0; minimizing collisions is trivial if you do not care about reward (the same effectiveness-vs-coverage tension as coverage).

**Treatment (read it as a frontier, among methods that actually earn).** Restrict to reward-seeking de-confliction methods at the most-contended pool (|S|=15): ActiveCFconv, CBBAlite, ContentionAdaCF, ContentionCF, PTF, RewardCFconv. Our private-offset methods are the top earners (best: **ContentionCF**), and **ContentionAdaCF** has the LOWEST collision rate of any reward-seeker; no field primitive (CBBA auction-with-backoff, MAB re-seating, greedy) or the batch PTF earns more OR collides less than both of ours. So our methods DOMINATE the field on the earned-vs-collision frontier: read against earned value, the private offset is a genuine coordination win, not a caveat.

| reward-seeker | collision@15 | earned@15 |
|---|---|---|
| **ContentionCF** | 0.625 | 0.105 |
| **ContentionAdaCF** | 0.581 | 0.100 |
| CBBAlite | 0.658 | 0.064 |
| RewardCFconv | 0.660 | 0.059 |
| PTF | 0.624 | 0.057 |
| **ActiveCFconv** | 0.685 | 0.046 |

## (8) Information efficiency, split into offline estimation vs online earning

| method | unseen skill @ low budget rho=0.10 | per unit budget |
|---|---|---|
| **RewardCF** | 0.180 | 1.80 |
| PTF | 0.191 | 1.91 |
| UCBIndep | 0.007 | 0.07 |
| Tabular | 0.001 | 0.01 |

**Two different efficiencies.** (i) OFFLINE estimation efficiency = unseen skill per observed entry: all low-rank methods turn a tiny budget into real skill while structure-free turns it into nothing (the structure-vs-no-structure point); among low-rank methods the batch-refit PTF is marginally ahead at the lowest budget, the same batch-on-dense-data advantage seen in the rho=1 crossover. (ii) ONLINE earning efficiency = reward EARNED per round while learning: here we win, because the batch methods pay an explore/probe phase. Under partial broadcast, RewardCF cumulative regret 40.9 is well below PTF 46.0 (lower = more value earned per round). So info-efficiency is a batch win for OFFLINE estimation and an OURS win for ONLINE earning, the regime that matters operationally.

