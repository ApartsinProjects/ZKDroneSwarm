# Operational target-servicing mission: dispatch the right asset (a clear, full-field CF win)

Same OBJECTIVE and METRIC as our earned-reward / anytime result, narrated as a mission: each round each drone services an offered target and earns the STANDARD reward R=<capability,requirement>; SERVICING SKILL = (earned - random-dispatch)/(oracle-dispatch - random-dispatch), 0 = no better than dispatching at random, 1 = oracle dispatch. Sample-starved (T=50 << n=240), passive masked broadcast (no comms/priors). OURS (RewardCF/EMCF) vs the STRUCTURED low-rank field (PTF/ESTR/BPMF/SoftImpute/MFSGD) vs structure-free (UCBIndep/Random). 8 seeds, bootstrap 95% CI.

| method | servicing skill (rho=1.00) | servicing skill (rho=0.25) |
|---|---|---|
| **RewardCF** | 0.402 [0.388, 0.417] | 0.348 [0.333, 0.363] |
| **EMCF** | 0.485 [0.475, 0.494] | 0.360 [0.339, 0.382] |
| PTF | 0.278 [0.258, 0.301] | 0.234 [0.213, 0.252] |
| ESTR | 0.216 [0.195, 0.239] | 0.181 [0.165, 0.199] |
| BPMF | 0.046 [0.038, 0.055] | 0.010 [-0.002, 0.021] |
| SoftImpute | 0.406 [0.390, 0.421] | 0.289 [0.275, 0.303] |
| MFSGD | 0.128 [0.111, 0.145] | 0.121 [0.100, 0.146] |
| UCBIndep | 0.001 [-0.011, 0.012] | -0.006 [-0.017, 0.005] |
| Random | -0.009 [-0.020, 0.003] | 0.000 [-0.013, 0.014] |

**WIN (full field, limited observability rho=0.25):** ours = 0.360 servicing skill vs the best of the entire competing field (structured low-rank AND structure-free) = 0.289. The mission objective rewards DISPATCHING WELL (earning high match-reward per engagement), so the generalization advantage is decisive: the swarm recovers the shared capability-requirement structure from the masked broadcast and dispatches the right asset to targets it never personally serviced, while the batch low-rank methods degrade under masking and the structure-free learners sit near the random-dispatch floor. At full broadcast (rho=1.0) the structured field is competitive; the separation opens under the LIMITED OBSERVABILITY that defines the operational regime. This is the applicative form of our headline result, on our standard reward and skill metric.

