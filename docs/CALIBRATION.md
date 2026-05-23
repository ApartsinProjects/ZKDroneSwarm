# EMCF predictive-interval calibration (is the confidence real?)

EMCF posterior predictive intervals on UNSEEN pairs, rho=1.0, 8 seeds, bootstrap 95% CI.

## Discrimination: does predicted uncertainty track actual error?

Unseen pairs binned into quintiles by EMCF's predicted sd (Q1=most confident -> Q5=least). A real confidence makes actual RMSE RISE across bins.

| quintile | mean predicted sd | actual RMSE |
|---|---|---|
| Q1 | 0.168 | 0.231 [0.223, 0.239] |
| Q2 | 0.235 | 0.289 [0.281, 0.297] |
| Q3 | 0.317 | 0.361 [0.346, 0.373] |
| Q4 | 0.417 | 0.432 [0.412, 0.447] |
| Q5 | 0.511 | 0.492 [0.471, 0.509] |

Actual RMSE rises +0.261 from the most-confident to least-confident quintile (positive => predicted sd is informative, not decorative).

## Calibration: empirical coverage of nominal intervals

| nominal | EMCF empirical | naive (constant sd) |
|---|---|---|
| 50% | 0.403 [0.383, 0.425] | 0.498 |
| 80% | 0.717 [0.705, 0.732] | 0.786 |
| 90% | 0.854 [0.844, 0.865] | 0.891 |
| 95% | 0.924 [0.917, 0.930] | 0.949 |

Read: a perfectly calibrated model matches nominal exactly; mean-field VI is typically OVER-confident (empirical < nominal). What matters for UCB/shrinkage is DISCRIMINATION (above), which does not require perfect coverage, only that more-uncertain predictions are genuinely worse. The naive constant-sd column has identical 'intervals' for every pair, so it cannot discriminate at all.

