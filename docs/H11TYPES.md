# De-confliction vs drone-type homogeneity (E-H11types)

Earned-reward skill at pool=30 (capacity-1), sweeping the number of DRONE TYPES K1 (K1=1 = all drones identical/rank-1 = maximal overlap; K1=30 = all distinct). 8 seeds, bootstrap 95% CI.

| method | K1=1 | K1=2 | K1=5 | K1=30 |
|---|---|---|---|---|
| **ContentionAdaCF** | 0.652 [0.531, 0.775] | 0.208 [0.181, 0.238] | 0.178 [0.162, 0.193] | 0.145 [0.134, 0.155] |
| CBBAlite | 0.503 [0.419, 0.583] | 0.165 [0.134, 0.195] | 0.144 [0.125, 0.167] | 0.113 [0.088, 0.135] |
| MusicalChairs | 0.460 [0.342, 0.580] | 0.090 [0.067, 0.114] | 0.075 [0.044, 0.116] | 0.055 [0.035, 0.071] |
| RewardCFconv | 0.540 [0.417, 0.656] | 0.159 [0.135, 0.186] | 0.144 [0.125, 0.166] | 0.110 [0.091, 0.130] |

Gap ContentionAdaCF - greedy RewardCFconv (de-confliction value) by K1:
  K1=1: +0.112;  K1=2: +0.049;  K1=5: +0.034;  K1=30: +0.035

Read: if the gap SHRINKS as K1 grows, de-confliction matters most under TYPE HOMOGENEITY (identical drones fight over the same targets, so proactive private spreading pays off); with diverse types drones naturally spread and the offset adds little.

