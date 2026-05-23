# Choice-informativeness EM (joint latent + per-teammate gamma_k)

Does EM-learned per-choice informativeness beat a fixed competence ramp? Sweep the broadcast REWARD noise sigma_obs (rho=1.0, own noise 0.10); 8 seeds, bootstrap 95% CI.

## UNSEEN skill

| method | sigma_obs=0.3 | sigma_obs=0.6 | sigma_obs=1.0 |
|---|---|---|---|
| RewardCF | 0.446 [0.408, 0.477] | 0.295 [0.251, 0.337] | 0.162 [0.128, 0.196] |
| ChoiceCF | 0.093 [0.064, 0.126] | 0.093 [0.063, 0.126] | 0.093 [0.063, 0.125] |
| **ChoiceEM** | 0.012 [-0.011, 0.036] | 0.012 [-0.011, 0.036] | 0.012 [-0.011, 0.036] |

## ANYTIME skill

| method | sigma_obs=0.3 | sigma_obs=0.6 | sigma_obs=1.0 |
|---|---|---|---|
| RewardCF | 0.374 [0.355, 0.389] | 0.290 [0.266, 0.311] | 0.227 [0.199, 0.253] |
| ChoiceCF | 0.219 [0.197, 0.241] | 0.219 [0.197, 0.241] | 0.219 [0.196, 0.241] |
| **ChoiceEM** | 0.163 [0.146, 0.181] | 0.163 [0.146, 0.180] | 0.163 [0.146, 0.180] |

## ChoiceEM - ChoiceCF (mean; + = EM better)

| metric | sigma_obs=0.3 | sigma_obs=0.6 | sigma_obs=1.0 |
|---|---|---|---|
| unseen | -0.081 | -0.081 | -0.081 |
| anytime | -0.056 | -0.056 | -0.056 |

Read: if the choice channel is valuable (high sigma_obs), ChoiceEM should match or beat ChoiceCF by trusting model-based choices and discounting random ones; both choice methods should hold up better than RewardCF as rewards get noisy.

