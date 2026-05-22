# Zero-Knowledge / decentralization compliance audit

Goal: verify our approach falls strictly inside the stated setting:
ZERO PRIOR KNOWLEDGE, ZERO COMMUNICATION, PARTIAL + NOISY OBSERVATION, with
decentralized decisions arising from the absence of communication. This document
audits the generative model, the observation channel, every method, and the
evaluation, and states the one item that needs an explicit modeling convention.

## The three assumptions, made precise

1. ZERO PRIOR KNOWLEDGE. No learner may use the latent factors P, U, the true rank
   d, the cluster/type assignments, or any reward labels. A learner may know only
   the OBSERVABLE action-space dimensions (how many drones m, how many targets n,
   which targets are offered this round) and a GUESSED rank d_hat for its own
   factorization.
2. ZERO COMMUNICATION. No agent transmits any message or shares any parameter with
   any other agent; there is no coordinator, no consensus/gossip, no joint policy.
   Agents only PASSIVELY OBSERVE the public consequences of actions in the shared
   environment (an engagement and its outcome are publicly visible), subject to
   partial and noisy sensing. Observation of a shared environment is NOT
   communication: no agent chooses to send information, there is no protocol, no
   addressing, no parameter exchange.
3. PARTIAL + NOISY OBSERVATION. Each agent senses its own outcome cleanly (small
   sigma_own) and a per-agent-limited, noisy subset of other agents' public
   outcomes (masking rho, noise sigma_obs). Masking models limited
   detection/sensing (range, orientation, dropout), NOT radio packet loss (which
   would presuppose transmission, i.e. communication).

Decentralization is then a CONSEQUENCE: with no communication and heterogeneous
partial sensing, each agent forms a different internal state and decides
independently (formalized in THEORY_FORMAL.md Theorem 4).

## Component-by-component audit

### Generative world (experiments/core.py make_world)
Returns (P, U, R, meta). These are the GROUND TRUTH used ONLY by the experiment
harness to (i) generate observed rewards R[i, a] + noise and (ii) compute the skill
metric. They are NEVER passed to any learner. PASS.

### Observation channel (run loops in pilot_*.py)
Each round: drone i selects a_i from its offered set; the harness computes the true
reward R[i, a_i], adds noise, applies the per-agent mask, and delivers to drone i:
its own (a_i, R[i,a_i]+noise) and, for each unmasked teammate k, (a_k,
R[k,a_k]+noise). This is passive public-outcome sensing. No learner sends anything.
PASS (with the masking = sensing convention of assumption 3).

### Methods
| method | observes (cross-agent) | uses true d / P / U ? | shares params ? | strict ZK |
|---|---|---|---|---|
| Tabular | own outcome only | no (d_hat) | no | YES |
| UCBIndep | own outcome only | no | no | YES |
| UCBHomo | pooled outcomes (action+reward) | no | no | YES |
| MFSGD | action+reward | no (d_hat) | no | YES |
| ESTR | action+reward | no (d_hat) | no | YES |
| BPMF | action+reward | no (d_hat) | no | YES |
| RewardCF (ours) | teammates' action+reward | no (d_hat) | no | YES |
| HybridCF (ours) | teammates' action+reward | no (d_hat) | no | YES |
| ChoiceCF (ours) | teammates' action + OFFERED MENU | no (d_hat) | no | see note |
| BothCF (ours) | action+reward + OFFERED MENU | no (d_hat) | no | see note |

Key points verified in code:
- Every learner is constructed as Cls(m, n, d_hat, idx, seed, ...): it receives the
  GUESSED rank d_hat (= 8), never the true d (= 5), and never P, U, R, or meta.
- Factor initialisations are random (rng.normal); there is no warm start from
  ground truth. (HybridCF/PTF warm-start from an SVD of their OWN observed
  empirical matrix, not from the truth.)
- Each drone holds its OWN learner instance with its OWN P, U estimates. No method
  reads, averages, or receives another drone's parameters. There is no coordinator.
- RewardCF / HybridCF (our HEADLINE methods) consume ONLY teammates' (action,
  reward) outcomes. They are strictly ZK and strictly communication-free.

### Evaluation
skill = (method - random)/(oracle - random) uses the true R, but only in the
experimenter's metric, never inside any learner. Oracle is a CEILING baseline
(centralized + complete information); it is reported for normalisation, never used
as a method. PASS.

## The one item: ChoiceCF / BothCF observe the offered menu

ChoiceCF and BothCF use teammates' OFFERED candidate set (cand_sets[k]) to sample
"implicit negatives" (targets a teammate could have engaged but did not), the
standard exposure/MNL debiasing of implicit feedback. This requires observing not
just what teammate k did, but what k could have done. Two clean ways to keep this
strictly inside the setting:

(a) PUBLIC-ACTIVE-SET convention: treat the offered menu as the set of currently
    ACTIVE targets, which is publicly observable (everyone can see which targets
    exist / are alive). Under this convention the menu is part of passive
    observation and ChoiceCF/BothCF are ZK. (Our pilot draws per-drone random
    offers as a stand-in for per-drone availability; the convention treats the
    union/active set as public.)
(b) STRICT-ZK relaxation: sample negatives GLOBALLY from all n targets
    (within=False) using ONLY the observed chosen target a_k. Then ChoiceCF/BothCF
    observe nothing beyond teammates' actions, identical to RewardCF's observation
    footprint. This drops the exposure-debiasing refinement.

RECOMMENDATION: report RewardCF and HybridCF (strictly ZK, no menu needed) as the
headline methods; present ChoiceCF/BothCF as the choice-channel variant under
convention (a), and VERIFY in the choice-only ablation (E13) that their benefit
survives the strict-ZK relaxation (b). If it does, the choice channel's value is
not an artifact of menu observation.

## Conclusion

- ZERO PRIOR KNOWLEDGE: satisfied by all methods (guessed rank, random init, no
  ground-truth factors/types/labels). PASS.
- ZERO COMMUNICATION: satisfied. The broadcast is passive public-outcome sensing,
  not message passing or parameter sharing; each agent decides independently with
  no coordinator. PASS, under the masking-as-sensing convention (assumption 3).
- PARTIAL + NOISY OBSERVATION: satisfied by masking rho and noise sigma. PASS.
- The only assumption beyond pure action+outcome observation is the OFFERED MENU
  used by ChoiceCF/BothCF; the headline methods (RewardCF, HybridCF) do not need
  it, and E13 tests that the choice channel's value holds under the strict-ZK
  global-negative relaxation.

ACTION ITEMS (tracked): (1) standardise the wording "limited/noisy SENSING of
public outcomes" (not "lossy radio") across docs so zero-communication is airtight;
(2) E13 choice-only ablation including the strict-ZK ChoiceCF variant.
