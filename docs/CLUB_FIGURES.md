# CLUB baseline added to the block-world figure comparisons (data only)

CLUB run through the SAME masked harness, config, and 16 seeds as each experiment, then merged into the data JSON so the figures CAN include it on a later rebuild. Paper and figures are NOT regenerated. Figure 3 (anytime, c16) and Table 3 (c14) already include CLUB. Figures 5-8 use the LatentSwarm package; CLUB is now ported there too (`latentswarm.baselines.CLUB`, registered as `club`), so those figures can include it once their JSONs are regenerated with `club` in the algorithm list (held pending the figure rebuild).


## Figure 2 (masking-robustness crossover)  (c15_crossover_20260524_145135.json)

| rho | CLUB unseen-pair skill |
|---|---|
| 1.0 | 0.436 |
| 0.85 | 0.432 |
| 0.7 | 0.417 |
| 0.55 | 0.383 |
| 0.4 | 0.330 |
| 0.25 | 0.258 |
| 0.15 | 0.168 |
| 0.1 | 0.108 |

## Figure 4a (collaboration value vs rho)  (collab_20260524_145217.json)

| rho | CLUB unseen-pair skill |
|---|---|
| 0.0 | -0.003 |
| 0.1 | 0.108 |
| 0.25 | 0.258 |
| 0.5 | 0.351 |
| 1.0 | 0.436 |

## Figure 4b (positive scaling vs m)  (scale_m_20260524_145325.json)

| m | CLUB unseen-pair skill |
|---|---|
| 5 | 0.075 |
| 10 | 0.149 |
| 20 | 0.290 |
| 40 | 0.429 |
| 80 | 0.491 |
