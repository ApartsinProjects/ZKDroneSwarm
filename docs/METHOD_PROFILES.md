# Method operating profiles, paradigms, and performance scorecard

Single source of truth (experiments/method_profiles.py). The ZK-MRTA setting is itself new, so this is a CONTROLLED SWEEP across the low-rank design space we instantiate in the setting, against the external structure-free paradigm and full-information reference ceilings, not 'our method vs rivals'. Provenance is explicit: PTF and the CF family are ours; MFSGD/ESTR/BPMF/SoftImpute/KNNCF/CLUB are standard estimators we adapt (cited); UCB/Tabular/Random are the structure-free paradigm; Oracle/CTDE are upper bounds.

**Harness fact:** run_masked builds ONE estimator per drone, so every bake-off method is decentralized and communication-free in-harness; ESTR is spectral/centralized only in ORIGIN and runs here as a per-drone explore-then-commit reduction. Genuine full information applies only to Oracle and the CTDE ceiling.

**Notation** `[dist | comm | obs | prior | compute]`.

## A. Method operating profiles

| Method | Provenance | Dist | Comm | Observability | Prior | Compute | Profile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **RewardCF** | ours | D | 0 | rho,sig | dhat | online | `D|0|rho,sig|dhat|online` |
| **EMCF** | ours | D | 0 | rho,sig | dhat | online | `D|0|rho,sig|dhat|online` |
| **BothCF** | ours | D | 0 | rho,sig | dhat | online | `D|0|rho,sig|dhat|online` |
| **ChoiceCF** | ours | D | 0 | rho,sig | dhat | online | `D|0|rho,sig|dhat|online` |
| **ContentionAdaCF** | ours | D | 0 | rho,sig | dhat | online | `D|0|rho,sig|dhat|online` |
| **UnifiedCF** | ours | D | 0 | rho,sig | dhat | online | `D|0|rho,sig|dhat|online` |
| **HybridCF** | ours | D | 0 | rho,sig | dhat | online | `D|0|rho,sig|dhat|online` |
| **PTF** | ours (hybrid) | D | 0 | rho,sig | dhat | batch | `D|0|rho,sig|dhat|batch` |
| MFSGD | standard, adapted | D | 0 | rho,sig | dhat | online | `D|0|rho,sig|dhat|online` |
| KNNCF | standard, adapted | D | 0 | rho,sig | none | memory | `D|0|rho,sig|none|memory` |
| BiasModel | standard, adapted | D | 0 | rho,sig | none | online | `D|0|rho,sig|none|online` |
| BPMF | standard, adapted | D | 0 | rho,sig | dhat | batch | `D|0|rho,sig|dhat|batch` |
| SoftImpute | standard, adapted | D | 0 | rho,sig | dhat | batch | `D|0|rho,sig|dhat|batch` |
| CLUB | standard, adapted | D | 0 | rho,sig | none | batch | `D|0|rho,sig|none|batch` |
| ESTR | standard, adapted | D | 0 | rho | dhat | ETC | `D|0|rho|dhat|ETC` |
| UCBIndep | structure-free baseline | D | 0 | rho,sig | none | online | `D|0|rho,sig|none|online` |
| UCBHomo | structure-free baseline | D | 0 | rho,sig | none | online | `D|0|rho,sig|none|online` |
| Tabular | structure-free baseline | D | 0 | rho,sig | none | online | `D|0|rho,sig|none|online` |
| Random | structure-free baseline | D | 0 | - | none | - | `D|0|-|none|-` |
| CTDE-ceiling | reference (upper bound) | C | full | full | dhat | batch | `C|full|full|dhat|batch` |
| Oracle | reference (upper bound) | C | full | full | U* | - | `C|full|full|U*|-` |

## B. MRTA / decentralized-learning paradigms in context

| Paradigm | Prior knowledge | Communication | Distribution | Observability | Note |
| --- | --- | --- | --- | --- | --- |
| Auction / CBBA (consensus bundle) | known task values/costs | message-passing (bids) | decentralized | full task info | needs communication AND known utilities |
| DCOP / consensus MRTA | known constraints/utilities | message-passing | decentralized | full | needs communication AND a known objective |
| Cooperative MARL (CTDE: MAPPO/QMIX/VDN) | none (learned) | centralized training | centralized-train / decentralized-exec | full (in training) | central critic; not comms-free |
| Learned-communication MARL (CommNet/TarMAC) | none (learned) | learned messages | decentralized | partial + messages | broadcast is learned message-passing, not passive |
| No-comms multiplayer bandits (SIC-MMAB, musical chairs) | none (per-arm) | none | decentralized | own pulls + collisions | comms-free but STRUCTURE-FREE (no unseen generalization) |
| Matrix completion (nuclear norm, spectral) | low-rank | centralized (one matrix) | centralized | partial (uniform) | centralized estimation, not online/decision |
| Low-rank / bilinear bandits (ESTR, etc.) | low-rank | centralized | centralized | partial | centralized and/or explore-then-commit |
| Federated / gossip CF | low-rank | broadcast of factors/gradients | decentralized | partial | shares PARAMETERS, not a passive stream |
| Multi-user RL, low-rank rewards (Nagaraj-Agarwal) | low-rank | centralized aggregation | centralized | partial | closest prior; centralizes trajectory aggregation |
| Trait-based MRTA (Prorok et al.) | KNOWN traits | varies | decentralized | full | capability/requirement traits are GIVEN, not learned |
| **OURS (broadcast CF for ZK-MRTA)** | low-rank, GUESSED rank only | none (passive sensing) | decentralized | masked + noisy | the hardest cell: no comms, no known utilities/traits, guessed rank |

## C. Our methods by mechanism

| Method | Signal channel | Exploration | Confidence | Contention | Rank | Coordination |
| --- | --- | --- | --- | --- | --- | --- |
| **RewardCF** | reward | eps-greedy | none | none | fixed d-hat | implicit |
| **ChoiceCF** | choice | eps-greedy | none | none | fixed d-hat | implicit |
| **BothCF** | reward+choice | eps-greedy | competence-weight | none | fixed d-hat | implicit |
| **EMCF** | reward | collective-UCB | Bayesian posterior | none | fixed d-hat | implicit |
| **ActiveCF** | reward | collective-UCB | Bayesian posterior | none | fixed d-hat | explicit (exploration division) |
| **CoordCF** | reward | neg-correlated-UCB | Bayesian posterior | none | fixed d-hat | explicit (no-comms division of labor) |
| **ContentionCF** | reward | eps-greedy | none | fixed private offset | fixed d-hat | explicit de-confliction (no comms) |
| **ContentionAdaCF** | reward | eps-greedy | none | scarcity-gated offset | fixed d-hat | explicit de-confliction (no comms) |
| **ARD-EMCF** | reward | collective-UCB | Bayesian posterior | none | ARD (self-tuned) | implicit |
| **HybridCF** | reward | probe-then-exploit | none | none | fixed d-hat | implicit |
| **PTF** | reward | probe-then-exploit | none | none | fixed d-hat | implicit (batch refit) |
| **UnifiedCF** | reward | gated collective-UCB | Bayesian posterior | gated offset | fixed d-hat | both (conditionally) |

## D. Performance scorecard (one canonical masked harness)

| Method | Provenance | unseen@rho=0.25 | unseen@rho=1.0 | regret@0.25 | rounds-to-25%-oracle | profile |
| --- | --- | --- | --- | --- | --- | --- |
| **RewardCF** | ours | 0.336 | 0.376 | 41.2 | 35 | `D|0|rho,sig|dhat|online` |
| **BothCF** | ours | 0.349 | 0.372 | 41.2 | 36 | `D|0|rho,sig|dhat|online` |
| **PTF** | ours (hybrid) | 0.280 | 0.516 | 46.0 | never | `D|0|rho,sig|dhat|batch` |
| MFSGD | standard, adapted | -0.019 | 0.042 | 46.5 | never | `D|0|rho,sig|dhat|online` |
| BPMF | standard, adapted | 0.126 | 0.233 | 49.9 | never | `D|0|rho,sig|dhat|batch` |
| ESTR | standard, adapted | 0.058 | 0.232 | 46.4 | never | `D|0|rho|dhat|ETC` |
| UCBHomo | structure-free baseline | 0.070 | 0.167 | 50.3 | never | `D|0|rho,sig|none|online` |
| UCBIndep | structure-free baseline | 0.003 | 0.004 | 50.4 | never | `D|0|rho,sig|none|online` |
| Tabular | structure-free baseline | 0.003 | -0.001 | 43.0 | 46 | `D|0|rho,sig|none|online` |
| Random | structure-free baseline | 0.004 | 0.007 | 50.0 | never | `D|0|-|none|-` |
