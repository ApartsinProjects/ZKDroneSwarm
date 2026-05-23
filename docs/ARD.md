# ARD: does the method find the rank itself? (true d=5)

Set the guessed rank d_hat absurdly high (20) and check ARD-EMCF keeps skill AND recovers an effective rank ~5, removing the rank hyperparameter. rho=1.0, 8 seeds, bootstrap 95% CI.

| variant | unseen skill | anytime skill | recovered eff. rank |
|---|---|---|---|
| RewardCF (d_hat=8) | 0.443 [0.398, 0.480] | 0.374 [0.355, 0.389] | -- (no ARD) |
| EMCF (d_hat=8) | 0.632 [0.613, 0.649] | 0.419 [0.406, 0.432] | 8.0 |
| EMCF (d_hat=20) | 0.630 [0.613, 0.646] | 0.423 [0.405, 0.441] | 20.0 |
| ARD-EMCF (d_hat=8) | 0.566 [0.549, 0.584] | 0.455 [0.434, 0.476] | 3.3 |
| ARD-EMCF (d_hat=20) | 0.565 [0.543, 0.584] | 0.466 [0.451, 0.482] | 3.2 |

Read: if ARD works, ARD-EMCF (d_hat=20) should (a) match ARD-EMCF/EMCF at the tuned d_hat=8 and the RewardCF baseline on skill (no overfit from the 15 extra dimensions), and (b) report a recovered effective rank near the true d=5, so you never have to know or tune the rank.

