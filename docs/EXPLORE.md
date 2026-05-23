# Info-directed exploration: sample efficiency (rho=0.25, T=50)

Anytime cumulative-skill at early/mid/final rounds (higher + earlier = more sample-efficient). 8 seeds, bootstrap 95%% CI.

| exploration | round 10 | round 25 | round 50 (final) | rounds to 1/2-final |
|---|---|---|---|---|
| eps-greedy (RewardCF) | 0.064 [0.041, 0.083] | 0.168 [0.155, 0.182] | 0.324 [0.309, 0.339] | 24.5 |
| count-bonus (ActiveCF) | 0.074 [0.065, 0.083] | 0.198 [0.186, 0.210] | 0.333 [0.309, 0.354] | 21.1 |
| **posterior-UCB b=1 (EMCF)** | 0.006 [-0.019, 0.033] | 0.106 [0.087, 0.126] | 0.303 [0.286, 0.322] | 30.1 |
| **collective-UCB b=1 (EMCF)** | 0.010 [-0.012, 0.031] | 0.101 [0.081, 0.121] | 0.305 [0.284, 0.326] | 31.1 |
| **collective b=0.3 (EMCF)** | 0.048 [0.034, 0.062] | 0.173 [0.148, 0.196] | 0.356 [0.330, 0.380] | 26.2 |
| neg-corr (CoordCF) | 0.084 [0.060, 0.108] | 0.197 [0.174, 0.218] | 0.336 [0.315, 0.355] | 21.2 |

Read: if posterior-directed exploration is more sample-efficient, EMCF should show the highest early-round cumulative skill and the fewest rounds-to-half-final (it probes where the factor posterior is most uncertain, the most informative engagements). count-bonus (ActiveCF) is the cheap proxy; eps-greedy is uninformed.

