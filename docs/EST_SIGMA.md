# Is the 'noise level known' assumption load-bearing? (est-sigma vs known-sigma vs uniform)

RewardCF with KNOWN sigma (given rvar) vs UNIFORM (ignores sigma) vs EST-sigma (estimates per-source sigma from residuals). rho=1.0; homog (all sigma=1.0) vs hetero (half 0.1, half 1.9). 8 seeds, bootstrap 95%% CI.

## unseen skill

| method | homog | hetero |
|---|---|---|
| known-sigma | 0.161 [0.138, 0.188] | 0.194 [0.164, 0.224] |
| uniform | 0.194 [0.160, 0.225] | 0.345 [0.307, 0.380] |
| est-sigma | 0.125 [0.100, 0.156] | 0.295 [0.273, 0.317] |

## anytime skill

| method | homog | hetero |
|---|---|---|
| known-sigma | 0.279 [0.259, 0.302] | 0.282 [0.258, 0.306] |
| uniform | 0.260 [0.234, 0.286] | 0.302 [0.281, 0.327] |
| est-sigma | 0.234 [0.210, 0.259] | 0.306 [0.287, 0.324] |

est-sigma minus known-sigma (hetero, unseen): +0.101

Read: if EST-sigma MATCHES known-sigma (small gap) and both >= uniform under HETERO noise, the 'noise level known' assumption is NOT load-bearing, sigma can be estimated from residuals; and since uniform is competitive, the categorical headline needs no sigma at all.

