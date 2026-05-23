# Collaboration value: what is the broadcast worth? (rho=0 isolated -> rho=1 full)

Same masked harness, rho=0 = each drone sees ONLY its own outcomes (isolated), rho=1 = full passive broadcast. Unseen-pair skill, 8 seeds, bootstrap 95% CI. m=30, n=240.

| method | rho=0.00 | rho=0.10 | rho=0.25 | rho=0.50 | rho=1.00 | COLLAB VALUE (rho1 - rho0) |
|---|---|---|---|---|---|---|
| **RewardCF** | 0.001 [-0.005, 0.007] | 0.169 [0.150, 0.185] | 0.337 [0.308, 0.377] | 0.410 [0.386, 0.434] | 0.388 [0.361, 0.414] | **+0.387** |
| PTF | -0.002 [-0.008, 0.006] | 0.185 [0.162, 0.208] | 0.275 [0.240, 0.312] | 0.366 [0.345, 0.388] | 0.505 [0.483, 0.530] | **+0.507** |
| UCBIndep | 0.004 [-0.005, 0.012] | 0.007 [0.001, 0.013] | 0.005 [-0.002, 0.011] | 0.003 [-0.004, 0.009] | 0.002 [-0.002, 0.006] | **-0.002** |
| Tabular | -0.004 [-0.009, 0.003] | 0.001 [-0.004, 0.005] | 0.005 [-0.000, 0.011] | 0.002 [-0.005, 0.009] | 0.001 [-0.005, 0.007] | **+0.005** |

**Result:** the broadcast is worth +0.387 unseen skill to CF but only -0.002 to a structure-free learner. Collaboration is the crux: a lone agent cannot recover the shared low-rank structure from its single matrix row (isolated unseen ~ 0), so the broadcast UNLOCKS generalization for CF (T11, Thm 1); a tabular learner cannot use the broadcast at all (it has no model linking targets), so sharing buys it nothing. This is the operational 'why a swarm' answer: communication-FREE sharing turns m isolated, near-useless learners into one effective swarm.

