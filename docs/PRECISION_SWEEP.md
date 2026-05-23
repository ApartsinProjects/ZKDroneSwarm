# Precision weighting vs broadcast noise (12 seeds, bootstrap 95% CI)

RewardCF (converged online weighted-ALS) with precision weighting 1/sigma^2 ON vs OFF (uniform obs weights), sweeping the broadcast-observation noise sigma_obs at rho=1.0 (full broadcast; own noise fixed at sigma_own=0.10). Skill: 0=random, 1=oracle.

## UNSEEN skill

| sigma_obs | precision ON | precision OFF | winner |
|---|---|---|---|
| 0.10 | 0.609 [0.591, 0.625] | 0.674 [0.657, 0.690] | OFF |
| 0.30 | 0.450 [0.427, 0.473] | 0.584 [0.574, 0.595] | OFF |
| 0.60 | 0.281 [0.254, 0.305] | 0.396 [0.366, 0.423] | OFF |
| 1.00 | 0.163 [0.142, 0.183] | 0.196 [0.169, 0.219] | OFF |
| 2.00 | 0.047 [0.028, 0.065] | 0.050 [0.030, 0.070] | OFF |

## ANYTIME skill

| sigma_obs | precision ON | precision OFF | winner |
|---|---|---|---|
| 0.10 | 0.457 [0.448, 0.466] | 0.495 [0.484, 0.505] | OFF |
| 0.30 | 0.390 [0.373, 0.405] | 0.438 [0.425, 0.452] | OFF |
| 0.60 | 0.296 [0.280, 0.311] | 0.336 [0.316, 0.355] | OFF |
| 1.00 | 0.216 [0.194, 0.234] | 0.215 [0.195, 0.235] | ON |
| 2.00 | 0.144 [0.123, 0.164] | 0.120 [0.108, 0.135] | ON |

Takeaway: at LOW/default broadcast noise, UNIFORM weighting is better (it uses the abundant broadcast fully, which powers unseen generalization); precision weighting 1/sigma^2 only pays off once the broadcast is noisy enough that down-weighting it filters more error than coverage it loses. So the recommended default is uniform weights, switching to precision weighting when the broadcast is known to be high-noise.

