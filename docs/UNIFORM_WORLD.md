# Re-evaluation without the block model: random continuous latent vectors

Does SwarmCF's advantage depend on the block-model world (K=10 discrete robot/target types)? 
Here we regenerate the latent vectors as i.i.d. Gaussian, L2-normalized (uniform on the unit sphere), so the reward is still rank-5 but has NO type clusters. Same masked harness 
(m=30, n=240, d=5, T=50, c=20, sigma_own=0.10, sigma_obs=0.30), 8 seeds. Unseen-pair skill.

Data: uniform_world_20260524_144751.json. NOTE: 8-seed scout, not the 16-seed paper run; paper is unchanged.


## rho = 1.00 (unseen-pair skill)

| method | block world | uniform world | uniform - block |
|---|---|---|---|
| **SwarmCF** | 0.385 | 0.333 | -0.052 |
| SwarmCF-batch | 0.511 | 0.374 | -0.137 |
| CLUB | 0.440 | 0.316 | -0.123 |
| BPMF | 0.239 | 0.130 | -0.109 |
| BiasModel | 0.125 | 0.027 | -0.097 |
| ESTR | 0.229 | 0.185 | -0.044 |
| MFSGD | 0.036 | 0.010 | -0.026 |
| UCBIndep | 0.002 | 0.003 | +0.001 |
| Tabular | 0.001 | 0.001 | -0.001 |
| Random | -0.006 | 0.004 | +0.010 |

SwarmCF - CLUB gap: block -0.055 -> uniform +0.017.

## rho = 0.25 (unseen-pair skill)

| method | block world | uniform world | uniform - block |
|---|---|---|---|
| **SwarmCF** | 0.346 | 0.252 | -0.094 |
| SwarmCF-batch | 0.280 | 0.206 | -0.073 |
| CLUB | 0.257 | 0.153 | -0.104 |
| BPMF | 0.142 | 0.090 | -0.052 |
| BiasModel | 0.084 | 0.015 | -0.069 |
| ESTR | 0.051 | 0.027 | -0.024 |
| MFSGD | 0.007 | 0.000 | -0.006 |
| UCBIndep | 0.005 | 0.002 | -0.003 |
| Tabular | 0.005 | 0.002 | -0.002 |
| Random | -0.001 | 0.007 | +0.007 |

SwarmCF - CLUB gap: block +0.089 -> uniform +0.099.

## Read

At rho=0.25, SwarmCF's lead over the clustering baseline CLUB goes from +0.089 (block) to +0.099 (uniform): the block model FLATTERS clustering, and SwarmCF's advantage widens without it. SwarmCF stays a low-rank method that does not need discrete clusters; CLUB's hard-clustering loses its natural target when the latent vectors are continuous.

