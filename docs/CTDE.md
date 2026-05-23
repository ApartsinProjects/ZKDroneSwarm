# CTDE comms-full ceiling: the price of zero communication (E-CTDE / Rank5)

Earned-reward skill (matching-normalized; oracle = 1) under capacity-1 contention. CTDE-ceiling = ONE centralized CF model + Hungarian assignment (full communication, coordinated to distinct targets); a CEILING, not a competitor. 8 seeds, bootstrap 95%% CI.

| method | pool=240 | pool=60 | pool=30 | pool=15 |
|---|---|---|---|---|
| _CTDE-ceiling (ceiling)_ | 0.553 [0.525, 0.581] | 0.489 [0.464, 0.512] | 0.434 [0.406, 0.456] | 0.271 [0.240, 0.299] |
| **ContentionAdaCF** | 0.448 [0.423, 0.473] | 0.205 [0.186, 0.224] | 0.153 [0.132, 0.173] | 0.100 [0.086, 0.113] |
| **RewardCFconv** | 0.439 [0.418, 0.462] | 0.199 [0.186, 0.214] | 0.121 [0.102, 0.138] | 0.059 [0.044, 0.078] |
| UCBIndep | 0.005 [-0.004, 0.014] | -0.002 [-0.009, 0.005] | -0.002 [-0.009, 0.006] | 0.004 [-0.005, 0.013] |

Price of zero communication (CTDE-ceiling - ContentionAdaCF): pool=240: +0.105;  pool=60: +0.284;  pool=30: +0.281;  pool=15: +0.170

Read: CTDE (full comms) is a CEILING above our comms-free methods and below the oracle. A SMALL gap to ContentionAdaCF means communication buys little here, our comms-free de-confliction recovers most of the coordination value; the gap widens where within-round coordination matters most (severe contention).

