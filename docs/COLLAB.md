# Collaboration value: what is the broadcast worth? (rho=0 isolated -> rho=1 full)

Same masked harness, rho=0 = each drone sees ONLY its own outcomes (isolated), rho=1 = full passive broadcast. Unseen-pair skill, 8 seeds, bootstrap 95% CI. m=30, n=240.

| method | rho=0.00 | rho=0.10 | rho=0.25 | rho=0.50 | rho=1.00 | COLLAB VALUE (rho1 - rho0) |
|---|---|---|---|---|---|---|
| **RewardCF** | -0.005 [-0.011, 0.001] | 0.180 [0.155, 0.206] | 0.346 [0.316, 0.384] | 0.372 [0.345, 0.402] | 0.385 [0.361, 0.408] | **+0.390** |
| PTF | -0.001 [-0.007, 0.006] | 0.191 [0.174, 0.210] | 0.280 [0.241, 0.317] | 0.362 [0.338, 0.382] | 0.511 [0.488, 0.536] | **+0.512** |
| UCBIndep | 0.004 [-0.005, 0.012] | 0.007 [0.001, 0.013] | 0.005 [-0.002, 0.011] | 0.003 [-0.004, 0.009] | 0.002 [-0.002, 0.006] | **-0.002** |
| Tabular | -0.004 [-0.009, 0.003] | 0.001 [-0.004, 0.005] | 0.005 [-0.000, 0.011] | 0.002 [-0.005, 0.009] | 0.001 [-0.005, 0.007] | **+0.005** |

**Result:** the broadcast is worth +0.390 unseen skill to CF but only -0.002 to a structure-free learner. Collaboration is the crux: a lone agent cannot recover the shared low-rank structure from its single matrix row (isolated unseen ~ 0), so the broadcast UNLOCKS generalization for CF (T11, Thm 1); a tabular learner cannot use the broadcast at all (it has no model linking targets), so sharing buys it nothing. This is the operational 'why a swarm' answer: communication-FREE sharing turns m isolated, near-useless learners into one effective swarm.

