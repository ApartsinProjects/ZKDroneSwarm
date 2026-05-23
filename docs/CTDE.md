# Centralized ceilings: the price of zero communication and of observation noise (E-CTDE / Rank5)

Earned-reward skill (matching-normalized; oracle = 1) under capacity-1 contention. Two centralized CEILINGS (NOT competitors), bracketing our comms-free methods from above: **CentralClean-ceiling** = one centralized low-rank model + Hungarian assignment that observes ALL effects with NO noise and NO masking (no priors, just the low-rank assumption; the strongest a centralized low-rank system can do short of being handed the true factors); **CTDE-ceiling** = the same but with realistic observation noise. 8 seeds, bootstrap 95%% CI.

| method | pool=240 | pool=60 | pool=30 | pool=15 |
|---|---|---|---|---|
| _CentralClean-ceiling (ceiling)_ | 0.520 [0.482, 0.558] | 0.501 [0.465, 0.534] | 0.475 [0.450, 0.493] | 0.275 [0.246, 0.303] |
| _CTDE-ceiling (ceiling)_ | 0.553 [0.525, 0.580] | 0.489 [0.464, 0.513] | 0.434 [0.405, 0.456] | 0.271 [0.239, 0.298] |
| **ContentionAdaCF** | 0.448 [0.422, 0.473] | 0.205 [0.186, 0.224] | 0.153 [0.133, 0.173] | 0.100 [0.086, 0.113] |
| **RewardCFconv** | 0.439 [0.419, 0.462] | 0.199 [0.186, 0.214] | 0.121 [0.102, 0.138] | 0.059 [0.044, 0.078] |
| UCBIndep | 0.005 [-0.003, 0.014] | -0.002 [-0.009, 0.005] | -0.002 [-0.009, 0.006] | 0.004 [-0.005, 0.013] |

Price of zero communication (CTDE-ceiling - ContentionAdaCF): pool=240: +0.105;  pool=60: +0.284;  pool=30: +0.281;  pool=15: +0.170

Price of observation noise (CentralClean-ceiling - CTDE-ceiling): pool=240: -0.032;  pool=60: +0.012;  pool=30: +0.041;  pool=15: +0.004

Read: CTDE (full comms) is a CEILING above our comms-free methods and below the oracle. A SMALL gap to ContentionAdaCF means communication buys little here, our comms-free de-confliction recovers most of the coordination value; the gap widens where within-round coordination matters most (severe contention).

