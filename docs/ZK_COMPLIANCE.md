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

### The setting holds UNIFORMLY for ALL methods (apples-to-apples)

The answer to "does the setting hold for our AND all baseline methods?" is YES, by
construction of the harness, which treats every method identically:
- ZERO PRIOR KNOWLEDGE: no method receives P, U, R, the true rank d, or type/labels.
  Structured methods (MFSGD, ESTR, PTF, BPMF, RewardCF, HybridCF) all get the SAME
  guessed rank d_hat=8; tabular methods (Random, UCBIndep, UCBHomo, Tabular) carry
  no rank at all. None is warm-started from ground truth.
- ZERO COMMUNICATION / FULLY DISTRIBUTED: every method (baselines included) is
  instantiated as one INDEPENDENT per-drone learner; the run loop never shares,
  averages, or routes one drone's parameters to another, and there is no
  coordinator. (ESTR's literature default is a single centralized estimator; our
  port runs it PER DRONE so the comparison is fully distributed and fair.)
- PARTIAL + NOISY BROADCAST ONLY: every method receives the identical per-drone
  masked, noisy outcome stream (own clean-ish, teammates masked at rho and noised
  at sigma); no method gets a privileged or denoised view.
- The ORACLE is the only centralized/complete-information object, used solely to
  NORMALISE skill; it is never a competing method.
So all reported gaps are within one setting: zero prior knowledge, zero
communication, partial+noisy passive observation, fully distributed.

### Evaluation
skill = (method - random)/(oracle - random) uses the true R, but only in the
experimenter's metric, never inside any learner. Oracle is a CEILING baseline
(centralized + complete information); it is reported for normalisation, never used
as a method. PASS.

## The offered menu: RESOLVED (every drone may choose any active target)

Earlier we worried that ChoiceCF/BothCF read teammates' per-drone offered menus
(cand_sets[k]) for exposure-debiased negative sampling. We RESOLVE this by adopting
the natural model: there is NO per-drone private menu. Every drone may choose ANY
currently ACTIVE target, and the active-target set is PUBLIC (everyone sees which
targets exist). Consequences:
- The "menu" a teammate chose from is the public active set, so observing it is
  passive public observation, not communication. The exposure debiasing samples
  negatives from this public set. ZK holds.
- Equivalently, the choice channel can sample negatives GLOBALLY from all targets
  using only the observed chosen action a_k (within=False). This is the canonical
  choice channel we now use (ChoiceZK; StackCF's choice sub-estimator). It observes
  NOTHING beyond teammates' actions, identical to RewardCF's footprint.

EVIDENCE it costs nothing: the choice-only ablation (E13) shows ChoiceZK (global
negatives, no per-drone menu) matches ChoiceCF (per-drone menu) on every metric at
every rho (gap <= 0.03, within noise). So the choice channel's value is NOT an
artifact of menu observation. (The pilot still draws per-round size-c offers as a
stand-in for limited per-round availability; a drone always knows its OWN offer,
which is unproblematic, and never needs a teammate's private menu.)

CANONICAL METHODS are therefore all strictly ZK with an action+outcome observation
footprint: RewardCF, HybridCF (rewards only), ChoiceZK (actions only), StackCF
(adaptively selects between them by self-validation; global negatives).

## Conclusion

- ZERO PRIOR KNOWLEDGE: satisfied by all methods (guessed rank, random init, no
  ground-truth factors/types/labels). PASS.
- ZERO COMMUNICATION: satisfied. The broadcast is passive public-outcome sensing,
  not message passing or parameter sharing; each agent decides independently with
  no coordinator. PASS, under the masking-as-sensing convention (assumption 3).
- PARTIAL + NOISY OBSERVATION: satisfied by masking rho and noise sigma. PASS.
- OFFERED MENU: RESOLVED. Every drone may choose any active target; the active set
  is public; the canonical choice channel (ChoiceZK / StackCF) uses global negatives
  and observes only teammates' actions. E13 confirms this costs nothing (ChoiceZK ~=
  ChoiceCF). No method needs a teammate's private menu. PASS.

CONCLUSION: all canonical methods (RewardCF, HybridCF, ChoiceZK, StackCF) are
strictly ZK and communication-free: guessed rank, random init, independent per-drone
estimators, no parameter sharing, no coordinator, and an action+outcome observation
footprint over a passively-sensed public outcome stream.
