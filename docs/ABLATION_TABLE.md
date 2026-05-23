# Method ablation (12 seeds, bootstrap 95% CI)

Recommended method = **ActiveCFconv** (online weighted-ALS + latent-UCB active exploration; best balanced: top anytime, strong unseen). Each row below toggles ONE design choice; skill: 0 = random, 1 = oracle.

## Design-choice ablation (unseen-pair and anytime skill)

| variant (knob) | unseen rho=1.0 | unseen rho=0.25 | anytime rho=1.0 | anytime rho=0.25 |
|---|---|---|---|---|
| **ActiveCFconv**<br/><sub>online wALS + precision + active-UCB  (RECOMMENDED)</sub> | 0.485 [0.466, 0.506] | 0.360 [0.345, 0.376] | 0.440 [0.424, 0.455] | 0.348 [0.330, 0.367] |
| HybridCFconv<br/><sub>- active; + probe->SVD warm-start (eps-greedy)</sub> | 0.501 [0.478, 0.526] | 0.396 [0.371, 0.421] | 0.353 [0.337, 0.367] | 0.308 [0.295, 0.321] |
| RewardCFconv<br/><sub>- active, - warm-start (plain online wALS, eps-greedy)</sub> | 0.450 [0.427, 0.473] | 0.372 [0.347, 0.395] | 0.390 [0.373, 0.405] | 0.332 [0.321, 0.342] |
| RewardCFconv_noprec<br/><sub>- precision weighting (uniform obs weights)</sub> | 0.584 [0.574, 0.594] | 0.392 [0.371, 0.413] | 0.438 [0.426, 0.452] | 0.339 [0.328, 0.349] |
| PTF<br/><sub>batch SVD + SGD finetune (explore-then-commit)</sub> | 0.505 [0.482, 0.528] | 0.276 [0.248, 0.305] | 0.277 [0.260, 0.295] | 0.231 [0.217, 0.245] |
| ESTR<br/><sub>explore-then-spectral commit</sub> | 0.224 [0.197, 0.251] | 0.047 [0.034, 0.059] | 0.218 [0.196, 0.241] | 0.185 [0.173, 0.197] |

## Rank sensitivity of ActiveCFconv (rho=0.25; true d=5, default d_hat=8)

| guessed rank d_hat | unseen skill | anytime skill |
|---|---|---|
| 2 | 0.222 [0.198, 0.245] | 0.270 [0.247, 0.293] |
| 5 | 0.327 [0.296, 0.356] | 0.351 [0.331, 0.369] |
| 8 | 0.360 [0.345, 0.376] | 0.348 [0.330, 0.368] |
| 12 | 0.359 [0.328, 0.390] | 0.347 [0.329, 0.365] |
| 20 | 0.344 [0.313, 0.369] | 0.340 [0.324, 0.354] |

Takeaways: (i) online weighted-ALS dominates batch / explore-then-commit (PTF, ESTR) under masking and on anytime (anytime rho=1.0: RewardCFconv 0.39 vs PTF 0.28; unseen rho=0.25: 0.37 vs 0.28). (ii) HONEST FINDING: precision weighting 1/sigma^2 does NOT help at the default broadcast noise (sigma_obs=0.3) -- UNIFORM weighting (noprec) is better on both unseen (0.584 vs 0.450 at rho=1.0) and anytime, because it uses the abundant broadcast fully (the broadcast carries the cross-target structure that unseen generalization needs). Precision weighting only pays off once the broadcast is high-noise; see PRECISION_SWEEP.md for the crossover. (iii) the SVD warm-start (HybridCFconv) lifts unseen, most at dense rho, at an anytime cost. (iv) active (latent-UCB) exploration improves anytime and dense-rho unseen over eps-greedy. (v) robust across guessed rank for d_hat >= true d (5..20 all ~0.35 unseen at rho=0.25), degrading only when d_hat=2 < true d=5.

