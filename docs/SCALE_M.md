# Positive scaling with swarm size: does the swarm get smarter as it grows? (rho=0.50)

Unseen-pair skill vs number of drones m (fixed n=240, T=50, partial broadcast). 8 seeds, bootstrap 95% CI.

| method | m=5 | m=10 | m=20 | m=40 | m=80 |
|---|---|---|---|---|---|
| **RewardCF** | 0.084 [0.053, 0.114] | 0.217 [0.203, 0.230] | 0.360 [0.336, 0.387] | 0.412 [0.382, 0.441] | 0.430 [0.409, 0.450] |
| UCBIndep | 0.002 [-0.015, 0.021] | -0.010 [-0.021, 0.003] | 0.003 [-0.005, 0.011] | 0.003 [-0.003, 0.008] | -0.004 [-0.008, -0.001] |
| Tabular | -0.001 [-0.015, 0.014] | 0.004 [-0.010, 0.017] | 0.007 [-0.002, 0.014] | -0.002 [-0.006, 0.002] | 0.002 [-0.003, 0.007] |

**Result:** CF unseen skill rises from 0.084 (m=5) to 0.430 (m=80) as the swarm grows -- the swarm gets SMARTER as it gets bigger, because more drones contribute more broadcast observations to the SHARED low-rank structure (collective recovery, T11). The structure-free learner is FLAT (0.002 -> -0.004): m does not help an agent that only learns its own row (Thm 1). Positive scaling with team size is unique to the structure-sharing swarm.

