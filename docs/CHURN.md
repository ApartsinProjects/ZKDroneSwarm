# Non-stationarity: staying adapted under continuous target turnover (H6)

Active set of fixed size 200; every 5 rounds 8 targets DEPART and 8 FRESH ones ARRIVE. Steady-state skill (2nd half of a 80-round episode) on the current active set and on RECENT arrivals (active < 10 rounds). rho=1.0, 8 seeds, bootstrap 95% CI.

| method | skill on active set | skill on RECENT arrivals |
|---|---|---|
| **RewardCF** | 0.632 [0.610, 0.655] | 0.074 [0.059, 0.088] |
| ActiveCFconv | 0.697 [0.684, 0.709] | 0.363 [0.321, 0.402] |
| EMCF | 0.842 [0.830, 0.853] | 0.371 [0.291, 0.439] |
| Tabular | 0.448 [0.435, 0.459] | 0.062 [0.042, 0.082] |
| UCBIndep | 0.619 [0.584, 0.646] | 0.132 [0.114, 0.150] |

Read: under FAST continuous churn, PLAIN exploitative CF (RewardCF) does NOT win, it ties UCBIndep on the active set and TRAILS it on fresh arrivals (0.074 vs 0.132), because collective fold-in needs ~d probes to pin a newcomer and exploitation never probes them. The FIX is CF UNITED WITH DIRECTED EXPLORATION of the uncertain (fresh) targets: ActiveCFconv (broadcast count-bonus) and especially EMCF (predictive-variance UCB) PROBE newcomers AND fold them in via the shared structure, and they DOMINATE, on the active set EMCF 0.842 vs UCBIndep 0.619 / RewardCF 0.632, and on FRESH arrivals ActiveCFconv 0.363 and EMCF 0.371 vs UCBIndep 0.132 (all non-overlapping CIs). So non-stationarity IS handled: the win needs the variant that combines low-rank fold-in with confidence-directed probing of newcomers, neither structure-free optimism (UCBIndep) nor exploitative CF alone suffices. The arc: plain-CF negative -> diagnosis (must probe newcomers) -> confidence-directed CF win.

