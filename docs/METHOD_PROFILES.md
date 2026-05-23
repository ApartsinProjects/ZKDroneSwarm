# Method operating profiles and comparison tables

Single source of truth (experiments/method_profiles.py) for how every method and paradigm sits on the axes that matter, so our wins read honestly: who gets more power is explicit.

**Notation** `[dist | comm | obs | prior | compute]`: dist D=decentralized / C=centralized; comm 0=none(passive) / B=broadcast params / full=message-passing; obs full / rho=masked / rho,sig=masked+noisy / self; prior none / d-hat=guessed rank / d=true rank / U*=oracle factors; compute online / batch / ETC=explore-then-commit / mem. Our flagship is the hardest cell `[D | 0 | rho,sig | d-hat | online]`.

## A. Method operating profiles

| Method | Class | Dist | Comm | Observability | Prior | Compute | Profile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **RewardCF** | ours | D | 0 | rho,sig | dhat | online | `D|0|rho,sig|dhat|online` |
| **EMCF** | ours | D | 0 | rho,sig | dhat | online | `D|0|rho,sig|dhat|online` |
| **BothCF** | ours | D | 0 | rho,sig | dhat | online | `D|0|rho,sig|dhat|online` |
| **ChoiceCF** | ours | D | 0 | rho,sig | dhat | online | `D|0|rho,sig|dhat|online` |
| **ContentionAdaCF** | ours | D | 0 | rho,sig | dhat | online | `D|0|rho,sig|dhat|online` |
| **UnifiedCF** | ours | D | 0 | rho,sig | dhat | online | `D|0|rho,sig|dhat|online` |
| **HybridCF** | ours | D | 0 | rho,sig | dhat | online | `D|0|rho,sig|dhat|online` |
| MFSGD | low-rank (online) | D | 0 | rho,sig | dhat | online | `D|0|rho,sig|dhat|online` |
| KNNCF | low-rank (online) | D | 0 | rho,sig | none | memory | `D|0|rho,sig|none|memory` |
| BiasModel | low-rank (online) | D | 0 | rho,sig | none | online | `D|0|rho,sig|none|online` |
| PTF | low-rank (batch) | D | 0 | rho,sig | dhat | batch | `D|0|rho,sig|dhat|batch` |
| BPMF | low-rank (batch) | D | 0 | rho,sig | dhat | batch | `D|0|rho,sig|dhat|batch` |
| SoftImpute | low-rank (batch) | D | 0 | rho,sig | dhat | batch | `D|0|rho,sig|dhat|batch` |
| CLUB | low-rank (batch) | D | 0 | rho,sig | none | batch | `D|0|rho,sig|none|batch` |
| ESTR | centralized reference | C | full | rho | dhat | ETC | `C|full|rho|dhat|ETC` |
| UCBIndep | structure-free | D | 0 | rho,sig | none | online | `D|0|rho,sig|none|online` |
| UCBHomo | structure-free | D | 0 | rho,sig | none | online | `D|0|rho,sig|none|online` |
| Tabular | structure-free | D | 0 | rho,sig | none | online | `D|0|rho,sig|none|online` |
| Random | structure-free | D | 0 | - | none | - | `D|0|-|none|-` |
| CTDE-ceiling | reference ceiling | C | full | full | dhat | batch | `C|full|full|dhat|batch` |
| Oracle | reference ceiling | C | full | full | U* | - | `C|full|full|U*|-` |

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
| **UnifiedCF** | reward | gated collective-UCB | Bayesian posterior | gated offset | fixed d-hat | both (conditionally) |
