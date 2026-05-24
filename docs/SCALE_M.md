# Positive scaling with swarm size: does the swarm get smarter as it grows? (rho=0.50)

Unseen-pair skill vs number of drones m (fixed n=240, T=50, partial broadcast). 8 seeds, bootstrap 95% CI.

| method | m=5 | m=10 | m=20 | m=40 | m=80 |
|---|---|---|---|---|---|
| **RewardCF** | 0.124 [0.089, 0.159] | 0.248 [0.232, 0.265] | 0.345 [0.303, 0.374] | 0.390 [0.350, 0.429] | 0.430 [0.402, 0.455] |
| PTF | 0.110 [0.055, 0.172] | 0.184 [0.139, 0.242] | 0.275 [0.249, 0.304] | 0.423 [0.398, 0.443] | 0.580 [0.558, 0.598] |
| UCBIndep | 0.002 [-0.014, 0.021] | -0.010 [-0.021, 0.003] | 0.003 [-0.005, 0.012] | 0.003 [-0.003, 0.008] | -0.004 [-0.008, -0.001] |
| Tabular | -0.001 [-0.016, 0.014] | 0.004 [-0.010, 0.017] | 0.007 [-0.002, 0.014] | -0.002 [-0.006, 0.002] | 0.002 [-0.003, 0.007] |

**Result:** CF unseen skill rises from 0.124 (m=5) to 0.430 (m=80) as the swarm grows -- the swarm gets SMARTER as it gets bigger, because more drones contribute more broadcast observations to the SHARED low-rank structure (collective recovery, T11). The structure-free learner is FLAT (0.002 -> -0.004): m does not help an agent that only learns its own row (Thm 1). Positive scaling with team size is unique to the structure-sharing swarm.

