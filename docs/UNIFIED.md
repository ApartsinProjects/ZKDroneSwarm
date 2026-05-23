# H3: UnifiedCF capstone validation (one method, best-or-tied across regimes)

UnifiedCF (EMCF + loss-self-gating de-confliction offset + loss-gated exploration anneal) vs the per-regime specialist, SAME 8 seeds, bootstrap 95% CI. rho=1.0.

UnifiedCF = the committed H3 method; UnifiedCF+ab = same method with the abundance gate ON (abundance_k=4: damp UCB when |offer| > 4m). The gate fires ONLY at no-contention pool=240 (offer=240>120); the anytime/churn/pool<=60 regimes use offer=20<120 so they take the identical code path (ties must be preserved by construction).

## STANDARD (no contention): anytime skill

| UnifiedCF | UnifiedCF+ab | EMCF (specialist) |
|---|---|---|
| 0.437 [0.417, 0.457] | 0.437 [0.418, 0.457] | 0.433 [0.417, 0.447] |

## CHURN: active-set / recent-arrival skill

| metric | UnifiedCF | UnifiedCF+ab | EMCF (specialist) |
|---|---|---|---|
| active | 0.851 [0.840, 0.862] | 0.851 [0.840, 0.862] | 0.842 [0.830, 0.853] |
| recent | 0.347 [0.274, 0.417] | 0.347 [0.276, 0.417] | 0.371 [0.291, 0.439] |

## CONTENTION: earned-reward skill

| pool | UnifiedCF | UnifiedCF+ab | ContentionAdaCF (specialist) | greedy RewardCFconv |
|---|---|---|---|---|
| 15 | 0.104 [0.086, 0.121] | 0.104 [0.085, 0.120] | 0.100 [0.086, 0.113] | 0.059 [0.044, 0.078] |
| 240 | 0.344 [0.317, 0.367] | 0.425 [0.406, 0.444] | 0.448 [0.422, 0.472] | 0.439 [0.418, 0.462] |

WIN check (abundance gate): pool=240 earned UnifiedCF 0.344 -> UnifiedCF+ab 0.425 (greedy 0.439); pool=15 0.104 -> 0.104; anytime 0.437 -> 0.437. RESULT: the gate closes the no-contention residual while holding the other regimes -> UnifiedCF+ab is best-or-tied EVERYWHERE.

Read: the committed UnifiedCF ties the specialists in 4/5 regimes; the abundance gate targets the one residual (no-contention earned reward) by exploiting when targets are plentiful, without touching the small-offer regimes where exploration earns its keep.

