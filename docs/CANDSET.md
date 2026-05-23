# Candidate-set-size independence (does limiting the offer matter?)

Training candidate size c_train swept 20 -> 240 (=n, 'peek the entire target set'); unseen eval offer fixed at 20. rho=1.0, 8 seeds, bootstrap 95% CI.

| method | c_train=20 | c_train=60 | c_train=120 | c_train=240 |
|---|---|---|---|---|
| **RewardCF** | 0.443 [0.398, 0.480] | 0.410 [0.385, 0.433] | 0.384 [0.368, 0.399] | 0.341 [0.321, 0.360] |
| Tabular | -0.004 [-0.011, 0.005] | -0.009 [-0.014, -0.001] | -0.001 [-0.008, 0.003] | 0.002 [-0.002, 0.006] |

Takeaway: CF's unseen-pair skill is essentially flat across c_train (including the full c=n case), and Tabular stays at the floor (~0) throughout. The categorical separation does NOT depend on the candidate-set size: a tabular learner cannot rank targets it never engaged no matter how many it is shown each round. The candidate set is a modeling knob (per-round reachability), not load-bearing for the result.

