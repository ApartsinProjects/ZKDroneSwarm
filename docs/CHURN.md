# Non-stationarity: staying adapted under continuous target turnover (H6)

Active set of fixed size 200; every 5 rounds 8 targets DEPART and 8 FRESH ones ARRIVE. Steady-state skill (2nd half of a 80-round episode) on the current active set and on RECENT arrivals (active < 10 rounds). rho=1.0, 8 seeds, bootstrap 95% CI.

| method | skill on active set | skill on RECENT arrivals |
|---|---|---|
| **RewardCF** | 0.632 [0.610, 0.655] | 0.074 [0.059, 0.088] |
| Tabular | 0.448 [0.435, 0.459] | 0.062 [0.042, 0.082] |
| UCBIndep | 0.619 [0.583, 0.646] | 0.132 [0.114, 0.150] |

Read (HONEST result, NOT a clean categorical win): under FAST continuous churn CF does NOT dominate. On the active set CF beats the structure-free Tabular (0.63 vs 0.45) but only TIES the optimistic UCBIndep (~0.62); and on the FRESHEST arrivals (active < recency) CF actually TRAILS UCBIndep (0.074 vs 0.132, non-overlapping CIs). Diagnosis: collective fold-in needs ~d probes to pin a newcomer's factor, a latency that rapid churn outpaces, while UCBIndep's untried-arm optimism directs it straight onto the new targets. So CF's categorical advantage is a SAMPLE-STARVED STATIC-unseen property; under rapid non-stationarity the fold-in latency erodes it on the newest targets. An honest SCOPE LIMIT, not a third categorical result. (A faster re-adaptation, e.g. optimistic/active probing of fresh arrivals layered on CF, is the natural fix; logged as future work.)

