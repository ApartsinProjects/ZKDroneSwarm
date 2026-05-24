# Positive scaling with swarm size: does the swarm get smarter as it grows? (rho=0.50)

Unseen-pair skill vs number of drones m (fixed n=240, T=50, partial broadcast). 16 seeds, bootstrap 95% CI.

| method | m=5 | m=10 | m=20 | m=40 | m=80 |
|---|---|---|---|---|---|
| **RewardCF** | 0.133 [0.108, 0.157] | 0.239 [0.221, 0.255] | 0.322 [0.296, 0.347] | 0.387 [0.361, 0.414] | 0.428 [0.409, 0.447] |
| PTF | 0.129 [0.081, 0.178] | 0.203 [0.174, 0.234] | 0.277 [0.249, 0.305] | 0.415 [0.384, 0.444] | 0.570 [0.544, 0.592] |
| UCBIndep | -0.004 [-0.014, 0.008] | -0.003 [-0.011, 0.004] | 0.005 [-0.000, 0.010] | 0.000 [-0.004, 0.004] | -0.004 [-0.006, -0.002] |
| Tabular | 0.000 [-0.011, 0.011] | -0.004 [-0.012, 0.005] | 0.004 [-0.002, 0.009] | -0.001 [-0.005, 0.003] | 0.002 [-0.001, 0.006] |

**Result:** CF unseen skill rises from 0.133 (m=5) to 0.428 (m=80) as the swarm grows -- the swarm gets SMARTER as it gets bigger, because more drones contribute more broadcast observations to the SHARED low-rank structure (collective recovery, T11). The structure-free learner is FLAT (-0.004 -> -0.004): m does not help an agent that only learns its own row (Thm 1). Positive scaling with team size is unique to the structure-sharing swarm.

