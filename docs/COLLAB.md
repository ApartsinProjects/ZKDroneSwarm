# Collaboration value: what is the broadcast worth? (rho=0 isolated -> rho=1 full)

Same masked harness, rho=0 = each drone sees ONLY its own outcomes (isolated), rho=1 = full passive broadcast. Unseen-pair skill, 16 seeds, bootstrap 95% CI. m=30, n=240.

| method | rho=0.00 | rho=0.10 | rho=0.25 | rho=0.50 | rho=1.00 | COLLAB VALUE (rho1 - rho0) |
|---|---|---|---|---|---|---|
| **RewardCF** | -0.003 [-0.007, 0.002] | 0.171 [0.150, 0.191] | 0.316 [0.287, 0.346] | 0.353 [0.321, 0.385] | 0.386 [0.370, 0.405] | **+0.389** |
| PTF | 0.003 [-0.002, 0.008] | 0.188 [0.169, 0.207] | 0.267 [0.240, 0.294] | 0.346 [0.320, 0.370] | 0.490 [0.455, 0.524] | **+0.488** |
| UCBIndep | 0.003 [-0.001, 0.008] | 0.006 [0.002, 0.010] | 0.000 [-0.004, 0.005] | 0.003 [-0.000, 0.007] | 0.001 [-0.004, 0.005] | **-0.003** |
| Tabular | -0.002 [-0.006, 0.003] | 0.001 [-0.003, 0.006] | 0.001 [-0.004, 0.006] | 0.002 [-0.003, 0.007] | -0.000 [-0.005, 0.004] | **+0.001** |

**Result:** the broadcast is worth +0.389 unseen skill to CF but only -0.003 to a structure-free learner. Collaboration is the crux: a lone agent cannot recover the shared low-rank structure from its single matrix row (isolated unseen ~ 0), so the broadcast UNLOCKS generalization for CF (T11, Thm 1); a tabular learner cannot use the broadcast at all (it has no model linking targets), so sharing buys it nothing. This is the operational 'why a swarm' answer: communication-FREE sharing turns m isolated, near-useless learners into one effective swarm.

