# Sensing-grounded observability: does the categorical win survive PHYSICAL masking? (RAS)

Masking + noise are DERIVED from sensing geometry (drone senses a target's engagement iff within radius R_sense; read-off noise grows with distance), not injected. 8 seeds, bootstrap 95%% CI. Effective coverage = mean fraction of (drone,target) engagements sensible.

| R_sense | mean coverage |
|---|---|
| 0.20 | 0.107 |
| 0.35 | 0.287 |
| 0.50 | 0.497 |
| 0.80 | 0.863 |

## Unseen-pair skill (the categorical claim)

| method | R=0.20 | R=0.35 | R=0.50 | R=0.80 |
|---|---|---|---|---|
| **RewardCF** | 0.030 [0.017, 0.044] | 0.147 [0.131, 0.164] | 0.229 [0.208, 0.247] | 0.304 [0.282, 0.329] |
| KNNCF | 0.049 [0.039, 0.059] | 0.170 [0.146, 0.193] | 0.270 [0.245, 0.296] | 0.353 [0.328, 0.378] |
| Tabular | 0.000 [-0.003, 0.004] | -0.002 [-0.010, 0.007] | -0.002 [-0.009, 0.006] | 0.005 [0.001, 0.007] |
| UCBIndep | 0.005 [-0.000, 0.011] | 0.002 [-0.002, 0.006] | 0.006 [0.002, 0.011] | 0.006 [0.000, 0.012] |

## Overall skill

| method | R=0.20 | R=0.35 | R=0.50 | R=0.80 |
|---|---|---|---|---|
| **RewardCF** | 0.445 [0.420, 0.466] | 0.510 [0.481, 0.535] | 0.552 [0.531, 0.572] | 0.599 [0.582, 0.616] |
| KNNCF | 0.169 [0.151, 0.184] | 0.321 [0.302, 0.343] | 0.413 [0.384, 0.448] | 0.438 [0.412, 0.466] |
| Tabular | 0.433 [0.411, 0.450] | 0.439 [0.415, 0.464] | 0.430 [0.404, 0.452] | 0.432 [0.416, 0.449] |
| UCBIndep | 0.593 [0.565, 0.617] | 0.595 [0.568, 0.620] | 0.600 [0.573, 0.626] | 0.604 [0.579, 0.626] |

Read: if RewardCF keeps a high unseen-pair skill above the structure-free floor (~0) even at SMALL sensing radius (sparse, distance-noisy observation), the categorical generalization result is not an artifact of an abstract mask, it survives physically-grounded, geometry-limited sensing, the regime a real drone swarm operates in.

