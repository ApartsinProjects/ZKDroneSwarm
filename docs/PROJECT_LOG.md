# Project Log: Confidence-Aware Decentralized Collaborative Filtering (ZK-MRTA)

**Goal**: find a CLEAR, relatively SIMPLE framing/method for decentralized
multi-robot task allocation (via collaborative filtering) that beats all
baselines and is groundbreaking; accept a more complex extension only if no
simple option works. Updated after every cycle. Commit per cycle (revertible).

## Experimental harness (synthetic pilots)
- Low-rank world: m drones, n targets, rank d; reward R = P Uᵀ. Structures:
  one-hot modes, cluster_gauss (random cluster centers), gauss (continuous),
  optional reward sharpening (raises effective rank).
- Sample-starved, CHANGING candidate subsets: each step every drone is offered a
  random subset (size `cand`) and picks one; T pulls/drone < n (camping
  impossible -> generalisation required). Repeated assignment.
- Two-stage noise: own reward observed with sigma_own (clean); teammates'
  rewards observed with sigma_obs (noisy); CHOICES observed cleanly.
- Reward-sharing prob p (1 = rewards broadcast, 0 = decision-only).
- Metric: skill = (greedy_final_model - random)/(oracle - random) in [0,1];
  also online reward/step. greedy = exploit final model over fresh subsets.
- Policies: Tabular (own-row memorisation), RewardCF/MF (pool rewards),
  ChoiceCF (pool choices, BPR/implicit), DualConf (competence-weighted choices),
  DualEM (mixture EM), RewardCFRobust (per-teammate residual trust).
- Files: experiments/pilot_identifiability.py, pilot_choice_only.py,
  pilot_refit.py, pilot_structure.py, pilot_rank.py, pilot_starvation.py,
  pilot_bootstrap.py, pilot_em.py, pilot_confirm.py, pilot_noise.py,
  pilot_trust.py. Python: C:/Python314/python.exe.

## Experiments and results (chronological)

| # | experiment | key result |
|---|---|---|
| 1 | identifiability gate (offline, choices) | U IS recoverable from choices; at low data choices BEAT rewards (greedy 0.448 vs 0.356); choice recovery saturates ~0.45 (≈44% of oracle gap) |
| 2 | online single-pass RLS | conjugate core fixed RECOVERY (Urec 0.06->0.25) but lost to tabular (single-pass underfits bilinear) |
| 3 | warm-started batch refit (ALS) | **GATE PASSED**: RewardMF@p=1 greedy 0.687 vs Tabular 0.485 (+42%), Urec 0.639. Reward-observable CF win is SOLID |
| 4 | decision-only choices online | naive folding + separated + entangled all fail (bootstrap deadlock; corruption) |
| 5 | 2D phase diagram (starvation x sharing) | CF/Tab non-monotonic, peak +51% at ~21% coverage; at p=0 CF/Tab~0.9 flat (reward-sharing NECESSARY) |
| 6 | latent structure sweep | CF wins for ALL geometries (gap 0.10-0.34); more clusters help; reward NONLINEARITY (raises rank) collapses advantage |
| 7 | true-rank sweep | low-rank LOAD-BEARING: gap ~0 at full rank (d=30) AND at d=1 (no personalisation); sweet spot d~3-5 |
| 8 | competence-weighted bootstrap (DualConf) | first p=0 "win" +8% at n=120 (3 seeds) -- later shown to be noise |
| 9 | principled EM (mixture) | EM works at p=1 but FAILS at p=0 (model-agreement responsibility reintroduces deadlock). Lesson: infer competence from BEHAVIOUR not model-agreement |
| 10 | 5-seed confirmation | reward-observable ROCK-SOLID (CF/Tab 1.44-1.89). decision-only "win" EVAPORATES to parity (was a 3-seed fluke) |
| 11 | apples-to-apples channel (clean choices vs noisy rewards) | choice channel FLAT 0.423 vs noise; reward channel 0.728->0.368 (drops BELOW solo at sigma_obs=2). decisions NOISE-IMMUNE; crossover at high noise |
| 12 | per-teammate trust w/ faulty teammates | CHOICE-trust WORKS: Ch_comp robust, gap grows w/ faulty% (+0.159->+0.180). REWARD-trust = BUG (override bypassed). |
| 13 | P1 collaboration-harm threshold | naive pooling drops BELOW solo (reward@50%, choice always); competence-weighted CHOICE pooling robust to 50% faulty, stays ABOVE solo (0.435->0.419). reward-trust fix works (gain grows w/ faulty%) but only parity at 50% |
| 14 | MF approximation audit | CF skill near-flat across als_sweeps (n=240: 0.731@5,3 -> 0.769@40,1); defaults near-converged. CF results are slightly UNDERSTATED (conservative), NOT inflated -> answers "do approximations interfere": no, they handicap CF. Use als_sweeps=10,refit_every=2; prior_prec 1-3 OK, 0.1 hurts |
| 16 | extended faulty breakdown to 80% (greedy+AUC) | competence-weighted CHOICE pooling robust to ~50% faulty (>= solo), BREAKS past 50% (65-80% below solo); ~50% breakdown matches robust-stats theory. CLOSES the parked faulty/Byzantine thread (drift) |
| 17 | CF VARIANT BAKE-OFF (n=120, 3 seeds) | BothCF (fuse reward+choice channels) is the SIMPLE DOMINANT variant: best-or-tied in ALL 6 (structure x observability) cells. RewardCF wins only reward-clean (collapses under noise/decision-only); ChoiceCF only decision-only. ~1% reward-clean penalty from un-gated fusion -> motivates CONFIDENCE-GATED fusion (dual-source confidence) |
| 18 | C8 unseen-pair generalization (FAIR, d_hat=8 guessed != true 5; 5 seeds; data saved) | CATEGORICAL WIN holds fairly: RewardCF UNSEEN-pair skill 0.496 vs Tabular 0.006 (~80x); tabular at floor BY CONSTRUCTION (no estimate for never-observed pairs). Overall 0.717 vs 0.397. Robust to guessed rank -> not an oracle-rank artifact. THE result for the groundbreaking bar (novel-in-setting + categorical). |
| 19 | C12 dynamic TARGET ONBOARDING (block-model core, fair d_hat=8, 5 seeds, data saved) | CLEAN Theta(d) vs Theta(m): CF (ridge fold-in given P) onboards a new target for ALL drones from ~d shared probes (0.93 skill @ 8 probes, ~0.98 saturated); Tabular rises ~linearly, only catching up at probes=m=30. At 3 probes: CF 0.797 vs Tabular 0.243 (3.3x). Operational categorical win = the two-phase idea validated. |
| 20 | C11 unseen-pair win under HETEROGENEOUS MASKING (block-model core, fair d_hat=8, 5 seeds, data saved; stateUniq FIXED) | CATEGORICAL WIN HOLDS in the NATURAL masked regime: CF unseen 0.16-0.41 vs Tabular ~0 (floor) at EVERY rho (1.0->0.10); CF overall 0.50-0.65 vs Tabular 0.42. DECENTRALIZATION REAL: fixed stateUniq (across-drone divergence of learned R_hat=P_i U_i^T) rises monotonically 0.54(rho=1)->0.92(rho=0.10); even rho=1 is 0.54 (per-observer noise). Heterogeneous observability => genuinely unique states. |

### Headline result tables
**Cycle 5 (phase diagram, CF/Tab greedy ratio, 5 seeds):**
n=30:1.16, n=60:1.28, n=120:1.38, n=240:1.51(peak), n=480:1.33 at p=1; ~0.9 flat at p=0.

**Cycle 7 (true-rank, CF-Tab skill gap):**
d=1:0.04, d=2:0.29, d=3:0.33, d=5:0.32, d=8:0.09, d=15:0.14, d=30:0.01.

**Cycle 11 (noise, skill):** Tabular 0.395 flat; RewardCF 0.728/0.591/0.491/0.451/0.368
(sigma_obs 0.1-2.0); ChoiceCF_comp 0.423 flat. Crossover ~sigma_obs 1.5-2.

**Cycle 12 (faulty%, skill):** Ch_naive 0.276/0.251/0.225; Ch_comp 0.435/0.424/0.405
(faulty 0/20/40%). C-gain +0.159/+0.173/+0.180.

**Cycle 13 = P1 (collaboration-harm threshold, 5 seeds, SOLO=0.405-0.414):**
| faulty% | RewardCF | RewRobust | Ch_naive | Ch_comp |
|---|---|---|---|---|
| 0  | 0.636 | 0.653 | 0.276< | 0.435 |
| 20 | 0.527 | 0.545 | 0.251< | 0.424 |
| 35 | 0.451 | 0.490 | 0.230< | 0.423 |
| 50 | 0.335< | 0.404< | 0.265< | 0.419 |
(< = below solo). SANITY: solo flat (ignores teammates) OK; Ch_naive low even at
0% faulty because cold-start exploratory choices corrupt unweighted pooling
(consistent with bootstrap finding); RewRobust now > RewardCF (bug fixed),
gain grows with faulty% (+0.017->+0.069). FINDINGS: (1) naive collaboration is
HARMFUL (below solo); (2) competence-weighted CHOICE pooling robust to 50%
faulty, stays ABOVE solo, best method at 50%; (3) reward-trust limited at 50%
(in-range garbage hard to detect) -> decisions make unreliability more
DETECTABLE than outcomes. TO IMPROVE: RANSAC/consensus to push past 50%;
trust-weighted HYBRID (rewards when reliable, choices when not) to dominate
across the range; clearer-outlier faulty rewards to make reward-trust test fair.

## Key findings (consolidated)
1. **Reward-observable CF beats tabular** (+44-89%, robust) IFF: reward is
   low-rank-but-personalised (1<d<<min(m,n)), regime sample-starved with changing
   availability, AND rewards shared. Non-monotonic sweet spots in starvation and
   rank. Explains all prior negatives (Sets M/N were sample-rich).
2. **Decision-only choices ≈ tabular parity** (capped by choice-recovery ceiling
   ≈ tabular). Not a standalone win.
3. **Competence must be inferred from BEHAVIOUR** (consistency), not
   model-agreement (which deadlocks).
4. **Decisions are NOISE-IMMUNE**: under high observation noise, the clean choice
   channel beats the noisy reward channel; naive reward-pooling can fall below
   solo.
5. **Per-teammate trust (choice side) works**: competence weighting is robust to
   faulty teammates; advantage grows with faulty fraction.

| 21 | C13 unseen-pair skill vs TRUE RANK d (D3 empirical support; reward-observable; fair d_hat=10; 5 seeds; data saved) | EXACTLY as D3 predicts: CF unseen-pair skill DECREASES monotonically with rank (d=2:0.671, 3:0.578, 5:0.381, 8:0.270); Tabular ~0 (floor) at ALL d. CF-Tab gap +0.67->+0.27 (scales with low-rankness). Confirms Prop 2 (O(d) completion) + the corollary. Spine + theory (docs/THEORY.md) now MUTUALLY VALIDATED. |

| 22 | Confidence-gated BothCF capstone (bake-off, 3 seeds; stdout summary) | PARTIAL/instructive: BothGated erases the reward-clean penalty on clustG (0.866 ~ RewardCF 0.869) and wins decision-only clustG (0.603), but HURTS under reward-NOISE (0.672 vs BothCF 0.691): the gate uses reward COUNT not PRECISION (count/sigma^2) -> mis-gates noisy rewards. NOT strictly dominant. CONCLUSION: un-gated BothCF is the recommended simple near-dominant method (the ~1% reward-clean penalty is benign); precision-aware gating = future polish. Core spine unaffected. |
| 23 | C14 METHOD BAKE-OFF: ALL relevant competitors in ONE masked harness (Random, UCBIndep, UCBHomo, Tabular, MFSGD, ESTR, RewardCF, BothCF); block model, fair d_hat=8; 5 seeds; data saved | LADDER on UNSEEN-pair skill (rho=1): RewardCF 0.376 ~ BothCF 0.372 > ESTR 0.232 > UCBHomo 0.167 > MFSGD 0.042 > UCBIndep 0.004 ~ Tabular ~0 ~ Random. Categorical low-rank-vs-no-structure split CONFIRMED across a full method set. KEY POSITIONING: ESTR (centralized explore-then-COMMIT, Kang-Hsieh-Lee'22 style) COLLAPSES under masking (unseen 0.232->0.058 as rho 1->0.25) while ours HOLD (0.376->0.336); gap WIDENS with masking. ESTIMATOR matters within low-rank: weighted-ALS (ours) > batch-SVD explore-then-commit (ESTR) > under-converged online SGD (MFSGD ~floor). UCBHomo captures only rank-1 popularity (partial unseen 0.17->0.07). stateUniq (CF) rises 0.54->0.83 as rho falls. |
| 24 | C14b EXTENDED bake-off: + PTF (probe-then-fit: UCB-probe -> SVD warm-start -> online SGD finetune) and BPMF (Bayesian PMF, conjugate precision + Thompson) for FULL low-rank coverage; 10 methods; 5 seeds; data saved | IMPORTANT/HONEST: PTF is a VERY strong baseline -- it BEATS ours on unseen at rho=1 (PTF 0.516 vs RewardCF 0.376) and ties overall (PTF 0.661 vs 0.650). BUT PTF (and ESTR, BPMF) all rely on a batch SVD of an empirical R_hat and DEGRADE under masking, while our online weighted-ALS stays ~FLAT. CROSSOVER at rho~1: at ANY masking (rho<=0.5) OURS WINS both overall and unseen (rho=0.5: RewardCF unseen 0.411>PTF 0.373, overall 0.654>0.574; rho=0.25: BothCF overall 0.619>PTF 0.538, unseen 0.349>0.280). Reframed claim: PTF wins ONLY at rho=1 (full broadcast = NO real observability limit, the degenerate case the paper excludes); in the LIMITED-observability regime that DEFINES the problem, ours dominates ALL baselines on BOTH metrics. Categorical low-rank vs no-structure split still holds across all 5 low-rank methods. data: results/pilots/c14_compare_20260522_132640.json (supersedes cycle-23 subset). |
| 25 | C15 CROSSOVER: finer rho sweep (8 rho x 7 methods x 8 seeds=384 cells) to pin masking-robustness; figure F5_crossover.png; data saved | Crossover PINNED. UNSEEN skill: PTF leads for rho>=0.7 (0.51->0.46), ours (RewardCF) FLAT ~0.39-0.41 and OVERTAKES near rho~0.55 (RewardCF 0.41>PTF 0.38), ours best through rho=0.4; at rho<=0.15 all converge toward floor (noisy parity). Batch-SVD hybrids DECAY monotonically (PTF 0.51->0.18, ESTR 0.23->0.01, BPMF 0.23->0.07); UCBIndep ~0 (floor) at every rho. OVERALL skill: ours wins/ties at every rho among generalizing methods (RewardCF peaks 0.66 @ rho=0.55), though UCBIndep stays ~0.59 (high overall via own-row exploitation, but ZERO unseen). KEY REALIZATION: this skill metric scores only the FINAL policy and so gives PTF/ESTR a FREE 40%-of-rounds probe phase; an ANYTIME/AUC metric (cumulative reward, "targets killed @K") would charge that cost and is the right next test (cycle 26). [Infra: a transient C:-drive fill during the 6-worker run caused a stdout-flush OSError AFTER data saved to E:; recovered on exit, freed 1.7GB, now route analysis to E: files + light stdout.] |
| 26 | C16 ANYTIME/AUC: cumulative reward EARNED over rounds ("targets destroyed @K"), the operational metric that CHARGES the probe-phase cost; 10 methods x 2 rho x 8 seeds; figure F6_anytime.png; data saved | DEFINITIVE OPERATIONAL WIN. On cumulative-normalized skill OURS (RewardCF/BothCF) wins at EVERY horizon and BOTH rho. rho=0.25 @T/4,@T/2,@final: RewardCF 0.069/0.180/0.341 vs PTF -0.002/0.055/0.230 vs ESTR 0.008/0.064/0.181 vs Tabular 0.071/0.141/0.252 vs UCBIndep -0.002/-0.004/-0.006. rho=1.0 final: RewardCF 0.404 > PTF 0.274 > ESTR 0.216, ours +47%. RESOLVES the cycle-25 caveat: PTF's better FINAL policy is operationally irrelevant -- it earns ~random during its 40% probe phase, so on cumulative reward ours beats it by ~48% at every horizon. EXPOSES UCBIndep: its high final-policy 'overall' (0.59) is a mirage -- on anytime it is STUCK ~0 because n>>T (240 targets, 50 rounds) keeps it perpetually exploring untried arms (never exploits). BPMF (Thompson) also over-explores -> ~floor anytime. Categorical anytime separation: online low-rank CF is the only thing that earns above random in the FIRST quarter (0.07-0.10 vs ~0 for all others). |
| 27 | E9 (Wave 1): PROBE-THEN-ONLINE-ALS HYBRID. New method HybridCF (UCB probe -> SVD warm-start -> our online weighted-ALS) added to the crossover (8 rho x 8 seeds) and anytime (8 seeds) sweeps; figures F5/F6 updated | HYBRID STRENGTHENS THE POSITIONING. Final-policy: HybridCF strictly improves on RewardCF at EVERY rho (UNSEEN 0.41-0.44 in the flat region; masking-robust), and is BEST-or-tied on OVERALL at EVERY rho (0.65-0.66; TIES PTF at rho=1=0.65; dominates at masking, 0.64 vs PTF 0.55 at rho=0.25). On UNSEEN it beats PTF for rho<=0.55 and crosses ~rho=0.6; PTF retains a (significant) UNSEEN lead only at dense broadcast rho>=0.85 (0.51 vs 0.41). ANYTIME: HybridCF pays a probe-phase cost (@T/4 ~0), final 0.357(rho=1)/0.336(rho=0.25), above PTF (0.274/0.230) but BELOW RewardCF (0.404/0.341). NET: a PARETO frontier among OUR methods -- RewardCF/BothCF own ANYTIME (no probe), HybridCF owns FINAL-POLICY (overall everywhere; unseen in the masked regime). Competitors are dominated by one of ours on every metric except PTF's dense-rho unseen (which PTF buys by sacrificing anytime). data: c15_crossover_20260522_145626 + c16_anytime_20260522_145857. |
| 28 | E3 (Wave 1): TWO-CHANNEL GRID rho (action mask) x sigma_obs (reward noise); both channels masked CONSISTENTLY (masked teammate -> neither choice nor reward); methods Tabular/RewardCF/ChoiceCF/BothCF/PTF; 5x3 grid x 8 seeds; figure F7 | CHANNEL TRADEOFF CONFIRMED. ChoiceCF (action channel) is FLAT in sigma_obs (0.48/0.45/0.44 across rho) -- clean choices are noise-invariant. RewardCF (reward channel) DEGRADES with noise (rho=1: 0.65->0.35). ChoiceCF OVERTAKES RewardCF at high noise (Ch-Rew gap rho=1: -0.17@0.3 -> +0.13@2.0; crossover ~sigma_obs=1). BothCF tracks the better channel (robust). PTF collapses at high noise (->0.07; this run used PRE-FIX PTF). CAVEATS: (a) sigma_obs=0 column is a precision-weight artifact (1/sigma^2 -> 1e6 over-weights clean broadcast) -> non-monotonic dip at rho<1; meaningful range is sigma_obs>=0.3. (b) PTF here pre-clip-fix. Both addressed for future runs (PTF clip committed). data: e3_channels_20260522_153527.json. |
| 29 | E12 (user): PERSISTENT vs IID masking, testing Theorem 4. Headline metrics (unseen, anytime, state-uniqueness) under both models across rho, + uniqueness vs horizon T; 6 methods; 8 seeds; figure F8 | THEOREM 4 CONFIRMED EMPIRICALLY. (c) unseen + anytime are ~INVARIANT to the masking model: persistent/iid within ~0.04 at every rho (RewardCF unseen 0.41/0.41 @rho0.5, 0.34/0.34 @rho0.25; anytime 0.38/0.37, 0.34/0.31). Low-rank still >> floor under iid; categorical results robust to the masking choice. (a/b) STATE-UNIQUENESS is DURABLE under persistent (flat in T: 0.86->0.80 over T=25..200 @rho0.25) but TRANSIENT under iid (DECREASING 0.90->0.51) -- exactly the Borel-Cantelli prediction (iid drones converge as T grows; persistent drones keep permanent blind sets). ANSWERS the masking-model question: iid is fine and the main results survive it; persistent is chosen so decentralization is durable not transient. PTF used the clip-fix here (stable: 0.50@rho1 -> 0.12 iid @rho0.1, graceful). data: e12_iid_masking_20260522_154506.json. |
| 30 | E13 (user): CHOICE-ONLY ABLATION + strict-ZK check. Tabular vs ChoiceCF (menu) vs ChoiceZK (global negatives, strict-ZK) vs RewardCF vs BothCF, consistent both-channel masking, sigma_obs=0.30; 8 seeds | TWO CLEAN RESULTS. (1) CHOICE CHANNEL ALONE lifts UNSEEN above the floor: ChoiceCF/ChoiceZK unseen 0.05-0.13 vs Tabular ~0 (Ch-Tab +0.130@rho1 -> +0.048@rho0.1) -- teammates' CHOICES carry recoverable low-rank structure, but it is MUCH weaker than the reward channel (RewardCF unseen 0.39). On ANYTIME the choice channel adds ~0 (Tabular already exploits its own clean reward). (2) STRICT-ZK ROBUST: ChoiceZK (no menu, global negatives) ~= ChoiceCF on both metrics (ZK-menu gap |.| <= 0.03, within noise) -> the choice channel's value is NOT an artifact of observing teammates' offered menus; it holds under strict zero-knowledge. Closes the ZK_COMPLIANCE caveat. NOTE: BothCF < RewardCF on unseen here (0.36 vs 0.39) because at clean-ish sigma_obs=0.30 the weak choice channel dilutes the strong reward channel; the choice channel helps the FUSION only when rewards are NOISY (E3). data: e13_choice_20260522_155023.json. |
| 31 | E2/E4/E6 (Wave 1): SCALING SWEEPS over true rank d, horizon T, targets n, guessed rank d_hat (rho=0.5); 5 methods; 8 seeds; figure F9 | ALL THEORY CONFIRMED across scales. [d] unseen scales with LOW-RANKNESS (RewardCF 0.96@d1 -> 0.10@d20; Tabular floor at every d; estimator-independent: PTF~RewardCF~HybridCF) = Theorem 2/corollary. [T anytime] RewardCF dominates at all T (0.21->0.65@T200); UCBIndep STUCK ~0 EVEN at T=200 because n=240>T keeps an untried arm in nearly every offer (Theorem 3 anytime corollary, dramatic); Tabular(eps-greedy) grows (0.13->0.48) since it is not trapped by an untried-arm bonus. [T unseen] PTF GROWS with T (0.21->0.67, batch SVD needs data) and passes ours for T>=100; ours plateaus (~0.40) -> ours wins in the STARVED regime, PTF with abundant data. [n] both metrics fall with starvation; ours dominates; structure-free worsens. [d_hat] ROBUST across guessed rank 2..20 (true d=5): RewardCF/HybridCF unseen 0.32@dhat2 -> 0.49@dhat20 (over-guessing safe/beneficial); supports fairness/practicality (no need to know true rank). data: e246_scaling_20260522_160250.json. WAVE 1 COMPLETE (E1,E2,E3,E4,E6,E9,E12,E13). |
| 32 | E7 (Wave 2): NEWCOMER COLD-START -- a SECOND categorical result (transpose of C12). A late-joining drone folds in its own factor from k own probes given the swarm's broadcast-learned U (ridge SHRUNK to the population prior p_pop = D7 hierarchical idea); predicts its whole row; 10 seeds; figure F10 | CATEGORICAL cold-start win. With ZERO own history (k=0) the CF newcomer acts at skill 0.275 (from the broadcast-learned U + population prior) while the TABULAR newcomer is at the FLOOR (~0, random) at EVERY probe count. CF rises with probes (0.28@k0 -> 0.41@k3 -> 0.57@k16 -> 0.59@k30), beating the popularity prior (~0.28 flat) once k>=3 (personalization beyond popularity). Theta(d) vs Theta(n): CF personalizes from ~d probes; tabular needs ~n (never, on unseen). [Fix: naive fold-in was unstable for k<d_hat (underdetermined 8-dim from <8 obs); shrinkage to p_pop made it stable + monotonic and is itself the D7 hierarchical-prior result.] data: e7_newcomer_20260522_160904.json. |
| 33 | E10 (Wave 2): FUSION dominance attempts -- precision-gated (BothCFPrec) and validation-stacked (StackCF) fusion, vs the channels across sigma_obs; 8 seeds (BothCFPrec) + smoke (StackCF) | CLEAN RECOMMENDATION (not a partial result): NO dominant fusion is needed because the REWARD CHANNEL already wins in the realistic regime. Crossover at sigma_obs ~= 1 (noise std = half the [-1,1] signal range): for sigma_obs<1 RewardCF/HybridCF dominate ChoiceCF (rho=1: 0.63>0.48 @0.3, 0.55>0.48 @0.6); ChoiceCF wins only at SEVERE noise sigma_obs>=1 (0.48 vs 0.34-0.46). BothCFPrec (gate choice by reward PRECISION) erases the reward-clean penalty to ~0 in the realistic regime (dom-margin ~ -0.01) but does not beat ChoiceCF at severe noise (others' noisy rewards still pollute U). StackCF (per-drone select reward-vs-choice by held-out own-reward fit) UNDERPERFORMS (holdout costs scarce own data: -0.05 to -0.09 vs the better channel) -> stacking is too data-expensive in this starved regime. CONCLUSION: recommend the simple REWARD channel (RewardCF/HybridCF) for sigma_obs<1; the CHOICE channel (ChoiceZK) is documented SEVERE-noise insurance. data: e10_precgate_20260522_162536.json. |
| 34 | CLOSE THE PTF GAP (user push: don't concede). The PTF dense-rho unseen lead was an ARTIFACT of our under-converged default ALS. Converged config HybridCFconv (als_sweeps=20, refit_every=1, eps_decay=0.97); 10-seed paired confirmation with bootstrap 95% CIs | PTF NOW DOMINATED. UNSEEN: HybridCFconv TIES PTF at rho=1 (0.494 vs 0.505; diff CI [-0.033,+0.011] contains 0) and WINS under masking (rho=0.5 +0.120, rho=0.25 +0.119; CIs exclude 0). ANYTIME: HybridCFconv WINS at EVERY rho (+0.078/+0.101/+0.073; CIs exclude 0). So PTF has NO remaining significant advantage on ANY metric/density. The earlier under-convergence (few sweeps, refit_every=3) was understating CF (the MF-audit caveat, now quantified: unseen rho=1 0.38 -> 0.49). Cost: HybridCFconv anytime slightly < default RewardCF (more exploration) -> our methods span the Pareto frontier (RewardCF/BothCF anytime-optimal; HybridCFconv final-policy-optimal), all dominating PTF. Paper 7.4 updated. data: conv_confirm_20260522_172716.json. |
| 35 | E8 (Wave 2, user point 4): ACTIVE EXPLORATION. ActiveCF = latent-space UCB (predicted reward + count-based uncertainty bonus from the BROADCAST -> collective active learning, no comms). Confirm ActiveCFconv (+converged ALS) vs RewardCF/HybridCFconv/PTF; 12 seeds, paired bootstrap CIs | EM/CONFIDENCE REVISIT PAYS OFF (as EXPLORATION, not gating). ActiveCFconv DOMINATES eps-greedy RewardCF: unseen WINS @rho=1 (+0.097) & rho=0.5 (+0.045), tie @0.25; anytime WINS @rho=1 (+0.040), tie @0.5/0.25 (never worse). It has the BEST anytime of ALL methods @rho=1 (0.440 vs RewardCF 0.40, HybridCFconv 0.35, PTF 0.28) AND near-PTF unseen (0.485 vs 0.505) -- it explores WHILE exploiting (no probe-phase cost), and broadcast counts make exploration collective. Pareto frontier among ours: ActiveCFconv = best balanced (best anytime + strong unseen, dominates RewardCF); HybridCFconv = best unseen under masking. Both dominate all competitors. NOTE: plain ActiveCF (no convergence) HURTS under masking (counts unreliable when masked); convergence fixes it. data: e8_active_20260522_182950.json. |
| 36 | E14 (Wave 2, user point 2): GENERALITY over m (drones), K (clusters), within (latent spread). Confirms conclusions are not artifacts of the default world; 5 methods; 8 seeds | GENERAL. Categorical (CF unseen >> Tabular/UCBIndep floor ~0) and anytime dominance hold across ALL axes. [m] unseen scales with population (RewardCF 0.21@m10 -> 0.45@m120; HybridCF 0.24->0.46; floor ~0 throughout); small m=10 weaker but still >>floor. [K] holds for every cluster count INCLUDING K=m=30 (every drone its own type = NO clustering = uniform drone latents): HybridCF unseen 0.47@K2 -> 0.33@K30, all >>floor. [within] holds tight->diffuse (within 0.05->1.0): HybridCF 0.46->0.31. So the categorical advantage is NOT an artifact of the clustered latent SAMPLING -- it holds for uniform-ish latents too. PTF rises with m (dense R_hat) but our anytime dominates throughout. data: e14_robust_20260522_184041.json. WAVE 2 COMPLETE (E7,E8,E10,E14 + converged-config). |
| 37 | E15 (user): BROADER FAIR BASELINES + T5. Added SoftImpute (nuclear-norm convex completion), KNNCF (memory-based model-free CF), BiasModel (additive, no interaction); all per-drone, broadcast-only, guessed rank; 8 seeds + CIs. Figure F12; theorem T5 (additive rank ceiling) | OURS DOMINATE THE BROADER FIELD in the regime that matters. unseen @rho=0.25: HybridCFconv 0.404 > best-new KNNCF 0.308 (+0.096 CI[+0.067,+0.123]); anytime @rho=0.25: ActiveCFconv 0.343 > best-new SoftImpute 0.289 (+0.054 CI[+0.040,+0.067]). HONEST: SoftImpute is the NEW dense-broadcast unseen champion (0.583 @rho=1, tops PTF 0.505 and us) but COLLAPSES under masking (0.305@.25, 0.049@.10) and loses anytime (we win anytime even @rho=1: ActiveCFconv 0.440 > SoftImpute 0.406) -- same dense-only-specialist pattern as PTF. KNNCF generalizes (0.48 unseen @rho=1) but weak anytime. BiasModel ~0.12 unseen = the ADDITIVE CEILING, confirming new Theorem 5 (additive predictor rank<=2 -> popularity-only ranking < CF for d>1). data: e15_morebase_20260522_191327.json. |
| 38 | Wave 3b-2 (user): REAL SIMULATOR VALIDATION. Ported our weighted-ALS (RewardCF) into the tabula_drone PettingZoo env (spatial targets, depleting HP, episodic) as a per-drone IPolicy (weighted_als_policy.py); benchmark (tabula_bench.py) vs env's SGD-MF, UCBIndep, random, oracle; 3 seeds x 16 learning episodes; efficiency metric (reward/step); figure F13 | EXTERNAL VALIDITY CONFIRMED in the real env. Skill (=(policy-random)/(oracle-random)): weighted_als (OURS) 0.806+-0.016 > UCBIndep 0.721+-0.053 > MF(env SGD) 0.251+-0.170 > random 0; approaches oracle. Our method ports cleanly to spatial/depleting/episodic dynamics and DOMINATES the env's own learning policies with LOW variance. NOTE: this env has HP-depletion = a partial CONTENTION setting, so it also previews axis (a). data: results/pilots/tabula_bench_real.json. |
| 39 | Wave 3b-1 (user): ASSUMPTION STRESS (external validity to low-rank). Power nonlinear link (raises eff. rank 5->15) + entrywise noise (approx low-rank, eff. rank 5->29); 5 methods; 8 seeds; figure F14 | GRACEFUL DEGRADATION, ours stay best. [nonlin] methods do slightly BETTER (the power curve sharpens target differences): HybridCFconv unseen 0.49->0.60, ActiveCFconv 0.44->0.52; UCBIndep ~0. [approx] as eff. rank 5->29 (near full-rank), CF unseen degrades SMOOTHLY (HybridCFconv 0.49->0.19, ActiveCFconv 0.44->0.18, RewardCF 0.41->0.15, PTF 0.37->0.14), no cliff, always > floor; anytime similar. So the advantage needs only SOME usable low-rank structure, not exact low-rank -> external validity to the assumption. WAVE 3b COMPLETE (real-sim + assumption-stress). data: stress_assump_20260522_200547.json. |

## STATUS: EXPERIMENT LOOP CONVERGED (2026-05-22)
The groundbreaking spine is empirically validated + theory-backed + paper-outlined.
Recommended method = BothCF (fuse reward+choice; near-dominant). Remaining before
submission is WRITING, not experiments: generate the 5 headline figures from the
saved results/pilots/*.json, write prose, final 5-seed confirmations at paper
settings. See docs/PAPER_OUTLINE.md. Polish backlog (precision-gated fusion, C6
Bayesian, C10 active exploration) is OPTIONAL, not needed for the core claim.

## 10,000-FT REVIEW (after ~20 cycles, 2026-05-22)
THE SPINE (all multi-seed, fair guessed rank, complete data saved):
- C8 static unseen-pair: CF 0.496 vs Tabular 0.006. Categorical.
- C11 natural masked regime: CF unseen 0.16-0.41 vs Tabular ~0 at EVERY rho;
  CF overall 0.50-0.65 vs Tabular 0.42.
- C12 dynamic onboarding: Theta(d) vs Theta(m) (CF 0.93 @ ~d probes; Tabular @ m).
- D3 theory grounded (MC O(d(m+n)) vs mn; block/cluster structure lowers it).
- Reward-observable characterization (cycles 1-17): CF wins iff low-rank-
  personalised + sample-starved + reward-shared; BothCF dominates the grid.

THE CLAIM: in a decentralized swarm with limited/heterogeneous observability and
NO communication, CF over the public broadcast lets agents act well on agent-task
pairs they NEVER observed and onboard NEW targets with O(d) shared probes vs O(m)
-- categorically beating independent/tabular learning (at the floor BY
CONSTRUCTION on unseen pairs), with matrix-completion theory behind it.

WENT WELL: floor-by-construction framing -> DEFINITE (categorical) wins not %;
clean block-model core; fairness (guessed rank) closed; data saved + catalogued.
STRANGE / TO FIX: (a) C11 stateUniq metric flawed (measures p_i, not U
divergence) -> fix to evidence 'unique states'; (b) BothCF ~1% reward-clean
penalty (un-gated fusion) -> confidence-gating (C3); (c) lit sub-agents
API-blocked -> direct web search works.
NEXT (re-ranked): spine empirics DONE. 1) FIX stateUniq + re-run C11 (validity:
decentralization is real). 2) D3 formal theorem. 3) (polish) confidence-gated
BothCF + C6 Bayesian. 4) Begin PAPER assembly (spine + theory ready). Method
refinements (confidence, active exploration) are POLISH, not needed for the core
categorical claim. PARKED (drift): Byzantine/faulty robustness, RANSAC.

## CYCLE 23: METHOD BAKE-OFF vs RELEVANT COMPETITORS (C14, 2026-05-22)
Motivation (user): "scout for relevant other methods to compare against,
including methods with low-rank assumption ... make sure all relevant are
covered." Built experiments/pilot_compare.py: ALL competitors in the SAME C11
masked harness (own clean reward + persistent per-drone broadcast mask rho),
block-model world, FAIR guessed rank d_hat=8, 5 seeds, parallel across CPU cores.
Competitor ports live in experiments/pilot_baselines.py.

METHOD SET (by structural assumption):
  no-structure : Random (floor), UCBIndep (per-(drone,target) UCB1),
                 UCBHomo (single shared arm table = naive pooling),
                 Tabular (eps-greedy own-reward empirical mean).
  low-rank     : MFSGD (online SGD matrix-factorization),
                 ESTR (explore-then-spectral-refit: random explore -> SVD of
                       R_hat -> exploit; centralized, explore-then-COMMIT),
                 RewardCF (OURS: online weighted-ALS, own+others' noisy rewards),
                 BothCF   (OURS: online weighted-ALS fusing rewards + choices).

RESULTS (skill = (greedy-random)/(oracle-random); mean over 5 seeds):
  rho=1.00  overall / UNSEEN
    Random   -0.00 / 0.01   UCBIndep 0.59 / 0.00   UCBHomo 0.24 / 0.17
    Tabular   0.42 /-0.00   MFSGD    0.25 / 0.04   ESTR    0.39 / 0.23
    RewardCF  0.65 / 0.38   BothCF   0.64 / 0.37
  rho=0.25  overall / UNSEEN
    Random    0.00 / 0.00   UCBIndep 0.58 / 0.00   UCBHomo 0.23 / 0.07
    Tabular   0.42 / 0.00   MFSGD    0.20 /-0.02   ESTR    0.33 / 0.06
    RewardCF  0.61 / 0.34   BothCF   0.62 / 0.35

FINDINGS:
1. CATEGORICAL low-rank vs no-structure split holds across a FULL method set:
   every no-structure learner (UCBIndep, Tabular, Random) sits at the UNSEEN
   floor (~0) by construction; low-rank methods lift above it. UCBIndep has the
   STRONGEST overall (0.59, exploits its own row) yet ZERO unseen -- the cleanest
   demonstration of "high in-distribution skill, no generalization."
2. UCBHomo (naive pooling) gets PARTIAL unseen (0.17->0.07): pooling recovers the
   rank-1 target "popularity" main-effect but NOT personalization; the
   CF-minus-UCBHomo gap isolates the value of personalization beyond popularity.
3. AMONG low-rank, the ESTIMATOR is decisive in the sample-starved regime:
   weighted-ALS (ours) > batch-SVD explore-then-commit (ESTR) > under-converged
   online SGD (MFSGD, ~floor). Naive low-rank is NOT enough.
4. POSITIONING vs ESTR (the closest centralized low-rank bandit): ESTR works when
   the broadcast is dense (rho=1, unseen 0.23) but COLLAPSES under masking
   (unseen 0.06 at rho=0.25) because its single batch SVD cannot complete a
   sparse R_hat and it never adapts after committing. OUR online weighted-ALS
   handles missingness natively and HOLDS (0.38->0.34). The advantage WIDENS as
   observation gets sparser -> masking-robust online decentralized CF is the
   novel contribution among low-rank methods (not the unseen win itself, which
   all low-rank methods share over no-structure).
5. stateUniq (per-drone learned R_hat divergence) rises 0.54->0.83 as rho falls
   for RewardCF -> decentralization is genuine.

SANITY / CODE REVIEW: Random ~0 on both metrics (normalization calibrated);
single-cell smoke test before parallel run; ProcessPoolExecutor(6 workers), one
process per (method, rho, seed); world via core.make_world(...)[:3]. CAVEAT to
address next cycle: MFSGD looks weak (vanilla SGD-MF underfits at T=50) and we
have NOT yet included PTF (probe-then-fit, SVD-warm-started MF) or BPMF
(Bayesian PMF), both already in tabula_drone/. Add them for full low-rank
coverage so the comparison cannot be called a strawman. Data:
results/pilots/c14_compare_20260522_131827.json.

## CYCLE 24: FULL LOW-RANK COVERAGE (+PTF +BPMF) -- HONEST CROSSOVER (C14b, 2026-05-22)
Addressed the cycle-23 caveat: ported the two missing low-rank competitors into
the pilot harness (experiments/pilot_baselines.py) and re-ran the 10-method
bake-off (results/pilots/c14_compare_20260522_132640.json):
  PTF  = Probe-Then-Fit: per-(drone,target) UCB probe -> SVD warm-start rank-d
         factors -> ONLINE SGD-MF fine-tune. (the strong hybrid; addresses
         MFSGD's cold-start AND adds online adaptation that ESTR lacks.)
  BPMF = Bayesian PMF (Salakhutdinov-Mnih'08): per-drone conjugate precision
         posterior over factors + Thompson sampling; consumes per-obs noise rvar
         as likelihood variance (principled, like RewardCF). MAP plug-in updates.
Both fully ZK (broadcast-only), fair guessed d_hat=8, per-drone (decentralized).
[Numpy-2.0 fix: batched np.linalg.solve needs b[...,None] then [...,0]; matched
the WeightedMF convention.]

UNSEEN-pair skill (mean/5 seeds), low-rank methods:
  method     rho=1.00  rho=0.50  rho=0.25
  PTF          0.516     0.373     0.280
  RewardCF     0.376     0.411     0.336      (ours)
  BothCF       0.372     0.313     0.349      (ours)
  BPMF         0.233     0.169     0.126
  ESTR         0.232     0.136     0.058
  MFSGD        0.042     0.006    -0.019
OVERALL skill:
  PTF          0.661     0.574     0.538
  RewardCF     0.650     0.654     0.608      (ours)
  BothCF       0.638     0.614     0.619      (ours)

THE HONEST FINDING (do NOT spin): PTF is a VERY strong baseline and BEATS our
methods at rho=1 (unseen 0.516 vs 0.376; overall 0.661 vs 0.650). This does NOT
weaken the paper -- it SHARPENS it:
1. The CATEGORICAL claim (low-rank acts on unseen pairs; no-structure is at the
   floor BY CONSTRUCTION) is now demonstrated across FIVE different low-rank
   estimators (PTF, ESTR, BPMF, RewardCF, BothCF), all lifting above the
   no-structure floor (Random/UCBIndep/Tabular ~0; UCBHomo partial via rank-1
   popularity). The effect is an ESTIMATOR-INDEPENDENT property of low-rank
   structure, not an artifact of our method. This is exactly what a rigorous
   reviewer wants.
2. Our method's specific contribution = MASKING-ROBUSTNESS + true ONLINE
   DECENTRALIZED operation. Every batch-SVD method (PTF, ESTR, BPMF) builds a
   single empirical R_hat and SVDs it; under masking R_hat is sparse+biased
   (unobserved -> imputed 0), so their unseen skill DECAYS with rho (PTF
   0.516->0.280; ESTR 0.232->0.058; BPMF 0.233->0.126). Our online weighted-ALS
   handles missing entries NATIVELY (precision weights), so unseen skill is ~FLAT
   (RewardCF 0.376->0.411->0.336). CROSSOVER is near rho=1: at ANY genuine
   masking (rho<=0.5) OURS WINS BOTH metrics.
3. SCOPING (defensible + true to the premise): PTF wins ONLY at rho=1 = full
   broadcast = every drone sees everything = NO real observation limit and
   "cosmetic" decentralization (the regime the paper explicitly excludes). The
   instant observation is limited (rho<1, the whole point), our method is the
   best on overall AND unseen. So: "in the limited/heterogeneous-observability
   regime that defines decentralized ZK-MRTA, online weighted-ALS dominates all
   baselines; a probe-then-fit hybrid is competitive only when observation is
   complete."

SANITY/CODE REVIEW: single-cell smoke tests for PTF & BPMF before the parallel
run; Random still ~0 (calibration intact); PTF stateUniq rises 0.275->0.808 with
masking (decentralized); MFSGD weakness now EXPLAINED (PTF = MFSGD + UCB-probe +
SVD-warm-start fixes exactly the cold-start/exploration failure -> strong),
internal consistency. NEXT (cycle 25): finer rho sweep around the crossover
(rho in {1.0,0.7,0.5,0.35,0.25,0.15,0.1}) for PTF vs RewardCF vs BothCF with more
seeds to pin the crossover and tighten CIs -> a headline figure
("masking-robustness: ours flat, batch-SVD hybrids decay").

## CYCLE 25: MASKING-ROBUSTNESS CROSSOVER (C15, 2026-05-22)
experiments/pilot_crossover.py reuses pilot_compare._run_cell (identical fair
config) over a finer rho grid {1.0,0.85,0.7,0.55,0.4,0.25,0.15,0.1} x 7 methods
x 8 seeds = 384 cells (all complete). Figure: docs/figures/F5_crossover.png.

UNSEEN-pair skill (mean/8 seeds) vs rho:
  rho       1.00 0.85 0.70 0.55 0.40 0.25 0.15 0.10
  UCBIndep  0.00 0.00 0.00 0.01 0.00 0.00 0.00 0.01   (no-structure floor)
  MFSGD     0.04 0.04 0.04 0.03 0.02 0.00 0.01 0.01   (underfit ~floor)
  ESTR      0.23 0.20 0.20 0.13 0.09 0.05 0.01 0.01   (batch-SVD, steep decay)
  BPMF      0.23 0.22 0.20 0.19 0.15 0.13 0.10 0.07   (Bayesian, milder decay)
  PTF       0.51 0.49 0.46 0.38 0.35 0.29 0.25 0.18   (strongest @hi rho, decays)
  RewardCF  0.39 0.41 0.39 0.41 0.37 0.34 0.23 0.17   (ours, FLAT then declines)
  BothCF    0.36 0.36 0.32 0.32 0.32 0.32 0.24 0.21   (ours, flattest at extreme)
OVERALL skill (mean/8 seeds): RewardCF 0.65 0.65 0.65 0.66 0.63 0.61 0.53 0.51;
  PTF 0.65 0.64 0.62 0.58 0.58 0.55 0.53 0.51; UCBIndep ~0.59 flat (high overall,
  ZERO unseen); BothCF 0.63..0.53.

INTERPRETATION (honest):
- CATEGORICAL spine intact: every low-rank method >> no-structure floor on unseen
  at every rho; UCBIndep/MFSGD pinned at ~0.
- Crossover on UNSEEN is ~rho=0.55: PTF leads for rho>=0.7 (dense broadcast),
  ours leads rho in [0.25,0.55]; at rho<=0.15 all decline to noisy parity.
- MASKING-ROBUSTNESS is the clean differentiator: ours is ~flat for rho>=0.4 while
  PTF/ESTR/BPMF decay monotonically (each SVDs an R_hat whose unobserved entries
  are imputed 0 -> sparse+biased under masking; our weighted-ALS handles missing
  entries natively). On OVERALL skill ours wins/ties at every rho among
  generalizing methods.
- LIMITATION of the metric (-> cycle 26): "skill" scores ONLY the final policy,
  granting explore-then-commit methods (PTF/ESTR) a cost-free 40% probe phase.
  The operationally-relevant metric is ANYTIME cumulative reward / AUC ("targets
  destroyed by round K"), which charges the probe cost. Our online method never
  pauses to probe, so on AUC it should separate more cleanly. RUN NEXT.

INFRA NOTE: during the 384-cell 6-worker run, C: briefly hit 0 free (worker temp
+ harness output spool) -> Python stdout-flush OSError at shutdown, AFTER
save_results wrote the complete JSON to E:. Recovered on process exit; cleaned
1.7GB (C: now ~13GB free). Mitigation adopted: write re-analysis to E: text files
and keep stdout terse; consider max_workers<=4 for big sweeps.

## CYCLE 26: ANYTIME / AUC -- THE OPERATIONAL METRIC (C16, 2026-05-22)
experiments/pilot_anytime.py: instead of scoring the FINAL policy, track the
reward each drone ACTUALLY earns each round (true reward of its pick), normalized
per round against oracle (best-in-offer) and random (mean-in-offer); report the
CUMULATIVE-normalized skill trajectory. This charges the cost of any probe/explore
phase. 10 methods x rho{1.0,0.25} x 8 seeds. Figure: docs/figures/F6_anytime.png.

Anytime cumulative-normalized skill (mean/8 seeds) at K=T/4, T/2, T:
  rho=1.00                @12     @25     @50(final)
    UCBIndep            -0.003  -0.002   0.001    (STUCK ~0: n>>T perpetual explore)
    Tabular              0.061   0.144   0.246    (eps-greedy own-row; best non-LR)
    ESTR                -0.004   0.070   0.216    (flat during probe, jumps @20)
    PTF                 -0.003   0.073   0.274    (probe cost early, strong finish)
    BPMF                 0.014   0.019   0.046    (Thompson over-explores)
    RewardCF             0.098   0.243   0.404    (OURS: best at every horizon)
    BothCF               0.098   0.235   0.400    (OURS)
  rho=0.25
    UCBIndep            -0.002  -0.004  -0.006
    Tabular              0.071   0.141   0.252
    ESTR                 0.008   0.064   0.181
    PTF                 -0.002   0.055   0.230
    BPMF                 0.002  -0.003   0.010
    RewardCF             0.069   0.180   0.341    (OURS)
    BothCF               0.069   0.179   0.342    (OURS)

WHY THIS IS THE HEADLINE (and resolves cycle 25 honestly):
1. On the metric that actually matters operationally -- targets destroyed by round
   K -- OUR online weighted-ALS WINS at EVERY horizon and BOTH rho. At the FINAL
   round ours beats the strongest competitor PTF by ~47% (rho=1: 0.404 vs 0.274)
   to ~48% (rho=0.25: 0.341 vs 0.230). The EARLY-round gap is categorical: in the
   first quarter only online CF earns above random (0.07-0.10) -- every other
   method is ~0.
2. Resolves the cycle-25 nuance: PTF's superior FINAL policy at dense rho is
   operationally irrelevant because PTF earns ~random during its 40% probe phase
   (visible kink at round 20 in F6). The final-policy metric flattered probe-then-
   commit methods by giving them their exploration for free.
3. EXPOSES UCBIndep: its high final-policy 'overall skill' (~0.59 in C14/C15) is a
   mirage. On anytime it is STUCK at ~0 because with n=240 targets and only T=50
   rounds it can't pull each arm once; its offer almost always contains an untried
   target (infinite UCB bonus) so it explores forever and never exploits. The
   sample-starved regime (n>>T) is exactly where structure-free methods fail
   operationally and low-rank generalization pays off.

SYNTHESIS OF THE COMPARISON (cycles 23-26): against the full relevant method set
(no-structure: Random/UCBIndep/UCBHomo/Tabular; low-rank: MFSGD/ESTR/PTF/BPMF):
  - CLAIM 1 (categorical, estimator-independent): low-rank structure lets agents
    act on NEVER-OBSERVED pairs; no-structure is at the floor BY CONSTRUCTION.
    Holds across all 5 low-rank estimators (final-policy unseen skill, C14/C15).
  - CLAIM 2 (our method): online weighted-ALS is (a) MASKING-ROBUST (flat unseen
    skill vs batch-SVD decay, C15) and (b) ANYTIME-OPTIMAL (no probe phase ->
    dominates cumulative reward at every horizon and rho, C16). The only metric a
    competitor (PTF) wins is final-policy quality at dense rho>=0.7, which is
    operationally irrelevant. NEXT: fold C14/C15/C16 into PAPER_OUTLINE related-
    work + results, then write the draft.

## CYCLE 27 ADDENDUM: HybridCF probe-budget tradeoff (E9, 2026-05-22)
Ablation (6 seeds) of HybridCF probe_frac, final-policy unseen vs anytime:
  config        uns@rho1  any@rho1  uns@rho.25  any@rho.25
  Hyb probe0.3    0.423     0.365      0.368       0.334
  Hyb probe0.5    0.449     0.304      0.380       0.268
  RewardCF        0.383     0.405      0.339       0.344
  PTF             0.502     0.273      0.282       0.228
READ: probe budget is a clean knob trading FINAL-POLICY unseen (up) for ANYTIME
(down). probe0.3 is the balanced default (unseen well above RewardCF, anytime
close to it). NOTE: even probe0.5 (more than PTF's 0.4) does NOT reach PTF's
dense-rho unseen (0.449 vs 0.502), so that residual is an ESTIMATOR nuance
(PTF's SGD-from-SVD vs our ridge-ALS-from-SVD), not just probe budget; and PTF
pays for it heavily on anytime (0.273). Confirms: no single config dominates both
metrics; the Pareto frontier among OUR methods (RewardCF/BothCF anytime-optimal,
HybridCF final-policy-optimal) is real and is the honest framing. WAVE-1 E9 DONE.

## CYCLE 40: WAVE-c PAPER TIGHTENING toward AAMAS/JAAMAS (2026-05-22)
(Cycles 28-39 catalogued in DATA_CATALOGUE.md; this resumes detailed logging.)
Executing PAPER_REVIEW.md P1 items.

P1-4 (20-seed headline bootstrap CIs, results/pilots/headline20_*.json -> docs/
HEADLINE_TABLE.md). Confirms the headline at 20 seeds, 95% CI:
  method        UNSEEN rho=1.0 / rho=0.25     ANYTIME rho=1.0 / rho=0.25
  UCBIndep         0.000 / -0.001                0.002 / -0.004    (floor, T1/T3)
  PTF              0.490 /  0.272                0.273 /  0.226
  RewardCF         0.377 /  0.326                0.383 /  0.342
  HybridCFconv     0.488 /  0.379                0.340 /  0.300
  ActiveCFconv     0.476 /  0.339                0.436 /  0.341
READ: UCBIndep at the floor on BOTH metrics (categorical). On UNSEEN, ours TIES PTF
at full broadcast (HybridCFconv 0.488 CI [0.468,0.509] vs PTF 0.490 [0.458,0.517])
and BEATS it under masking (0.379 vs 0.272, non-overlapping). On ANYTIME, ActiveCFconv
wins everywhere (0.436 vs PTF 0.273 at rho=1, non-overlapping). ActiveCFconv = best
balanced; the published "~0.49 unseen / ~0.44 anytime" claims are confirmed.

Also this cycle (deterministic tightening, no new runs):
- P1-5 vector figures: make_figures.py now emits PNG + vector PDF (docs/figures/pdf/)
  for F2-F14; main.tex includes the PDFs.
- P1-7 theorem tightening in main.tex: T2 incoherence + Otilde(d(m+n)) rate + noisy
  fold-in error; T3 starved-regime cT<<n + exact untried-arm mechanism.
- P1-8 contention scope note strengthened (leans on tabula_drone HP-depletion as
  preliminary partial-contention evidence).
- Data-driven tables: make_paper_v2.py + make_tutorial.py render headline/ablation/
  contention tables from the committed .md files (one source of truth) via a small
  markdown->HTML helper. Estimator gained a `precision` on/off toggle; pilot_anytime
  exposes run_anytime_clshp(Cls,hp,...) so ablation variants reuse the same loops.
NEXT (cycles 41-42): pilot_ablation.py (P1-6, RUNNING) -> docs/ABLATION_TABLE.md;
pilot_contention.py (P1-8) -> docs/CONTENTION.md; regenerate HTML; push.

## Cycles 47-49 (2026-05-23): ChoiceEM rescue, adaptive contention, held-out gamma, tutorial
- Cycle 47: ChoiceEM full 8-seed re-run (sigma_obs 0.6/1.0/2.0). Naive EM deadlocks
  (unseen 0.012); rescue (g0=0.1, warm_em=0.3) FIXES the anytime deadlock (0.163->0.217,
  ties the ramp 0.219), confirming the cold-start root cause, but the learned gate gives
  no SKILL edge over the fixed ramp. POSITIVE NICHE: at sigma_obs=2.0 the noise-immune
  ChoiceCF beats RewardCF on BOTH unseen (0.093 vs 0.042) and anytime (0.219 vs 0.179).
- Cycle 48 (H2): ContentionAdaptiveCF = fixed private DIRECTION (T7) + magnitude scaled
  by each drone's own loss rate (convex) + a HARD ZK scarcity gate (engage iff offer <=
  4m). EXTENDS the contention win from pool=15 alone to pool<=60 (beats fixed offset at
  pool=30 0.153 vs 0.134 and pool=60 0.205 vs 0.178; ties at pool=15). HONEST LIMIT: at
  no-contention (pool=240) the offset policies trail plain CF (0.25 vs 0.44) because they
  use pure argmax and DROP eps-exploration; the hard gate confirmed the gap is missing
  COVERAGE, not the offset. FIX (queued): eps-greedy fallback when the gate is off.
- Cycle 49 (ChoiceEM, user-driven): KEY INSIGHT (now Proposition 9). In-sample
  responsibility CANNOT down-weight a uniform-random teammate (E[r]=gamma fixed point) and
  OVERFITS a factor to its choices, INFLATING gamma (random teammate gamma ~0.70 >> prior
  0.1). The diagnosis came from a SANITY experiment (pilot_choicehetero.py: real learners
  + ORACLE choosers vs RANDOM choosers): a working estimator MUST give gamma(oracle) >>
  gamma(random); in-sample FAILS this (0.95 vs 0.70), so the homogeneous null was a real
  limitation, not a bug. FIX: HELD-OUT (predictive) responsibility, score each choice once
  against the model BEFORE the refit incorporates it. Smoke: predictive gamma(oracle) 0.48
  >> gamma(random) 0.11 (sanity PASSES), and good-drone unseen improves. Full 8-seed run in
  progress. NEXT: ChoiceEM-grad (reward-improvement gradient, to also catch consistently-
  WRONG teammates predictive still trusts); precision heterogeneous-noise sanity; eps-greedy
  contention unification; fold confirmed numbers into tutorial 5.6 + catalogue.
- Tutorial: full layman-explanation pass (Notation-at-a-glance table + 23 "In plain words"
  callouts across Sections 1-7 + expanded glossary), making it self-contained for a
  non-expert; verified KaTeX renders in-browser. Also fixed cwd-relative output-path bugs
  in make_tutorial.py.
METHODOLOGY NOTE (adopt going forward): verify every NEGATIVE/weak result with a SANITY
experiment whose answer is obvious (oracle vs random teammates; known-rank ARD; clean-vs-
noisy precision sources; identical-vs-distinct types for contention; d=1 popularity for the
unseen claim). The ChoiceEM sanity already converted a null into a diagnosed-and-fixed result.

## Cycles 50-52 (2026-05-23): sanity-check arc + consolidation + table-regression fix
- Cycle 50 (H11b): precision under HETEROGENEOUS teammate noise. Matched mean noise, only
  heterogeneity varies. BOUNDED precision (relcap) WINS in hetero (unseen 0.483 vs uniform
  0.390; anytime 0.356 vs 0.267, non-overlapping CIs), loses in homog (-0.063); UNBOUNDED
  full loses both (over-concentrates -> starves coverage). Scopes the 'uniform beats
  precision' negative: noise-aware weighting helps iff sources DIFFER in reliability, and
  only ratio-bounded.
- Cycle 51 (H4): EMCF interval calibration. DISCRIMINATION passes (actual RMSE rises
  monotonically across predicted-sd quintiles, Q1 0.231 -> Q5 0.492), so the posterior is
  usable for UCB/shrinkage; CALIBRATION mildly OVER-confident (50%->40%, 95%->92%), typical
  mean-field VI. Honest: use it for RELATIVE uncertainty, not exact coverage.
- Cycle 52: CONSOLIDATION (user ask: simpler, theory-aligned). paper_v2 now leads with one
  core estimator + a "Scoped refinements" MENU table (6 conditional, theorem-backed
  extensions + a diagnosed-nulls line); tutorial gets a top-of-section-5 map box. Turns the
  ~15-method zoo into one core + a short conditional add-on list.
- Cycle 52 BUG FIX (regression): md_tables() used relative docs/*.md paths, so regenerating
  the HTML from experiments/ silently replaced ALL embedded data tables with
  '[not generated yet]' placeholders (had degraded committed tutorial.html since cycle 48).
  md_tables now resolves against ROOT in both generators; make_paper_v2 also got the
  ROOT-anchored OUT/FIG fix. Tables restored (0 placeholders). LESSON: a cwd-relative path in
  a generator can silently drop content; always grep the regenerated HTML for placeholders.
- H11c ARD-known-rank sanity: recovered eff rank NON-monotone in true d (2.00/2.35/2.13/1.73
  for d=2/3/5/8). Honest diagnosis: recovered = identifiable rank <= d (Thm 8); SNR confound
  (unit-norm signal ~1/sqrt(d) at fixed noise/budget). Softened the ARD claim to d_hat-
  invariance (the usable property). Constant-SNR controlled sweep = future work.
- H2 follow-up: eps-greedy fallback for ContentionAdaCF when the scarcity gate is OFF (so it
  reduces to plain CF at no-contention instead of under-exploring); clean 8-seed re-run in
  progress to confirm pool=240 recovers to ~plain-CF while pool<=60 wins hold.
LESSON (compute hygiene): do NOT oversubscribe the CPU by launching multiple ProcessPool
experiments at once; it thrashed the contention run to a stall and it had to be killed and
re-run cleanly. Run heavy experiments sequentially.

(Cycles 53-62 captured in DATA_CATALOGUE rows 53-54 + git log: H6/H6b churn, H3 UnifiedCF
capstone, theory T9/P10/P16, paper/tutorial consolidation + novelty framing.)

- Cycle 63 (STRICT-ZK NEWCOMER + harness audit + path fix): a background agent audited ALL
  46 experiment harnesses for the "no priors, no communication beyond personalized partial
  noisy broadcast" setting. It found exactly ONE cross-learner parameter copy: E7's newcomer
  took U_hat = learners[0].U.copy() and a peer-population p_pop (exact only at rho=1). FIXED:
  the newcomer is now a PASSIVE RewardCF listener that recovers U from its OWN persistent-
  masked broadcast and folds in (prior = mean of the teammate factors IT recovered);
  incumbents masked at the same rho; rho swept {1.0,0.5,0.25}, 10 seeds (catalogue row 55,
  F10 regenerated). The categorical O(d)-vs-O(n) newcomer gap SURVIVES strict ZK at every rho
  (Tabular ~0; CF >> Tabular). At rho=1 fold-in personalizes 0.28->0.57 >> popularity 0.27;
  at rho=0.25 the probe-efficiency SLOPE flattens (CF ~ pop ~ 0.30) because the self-recovered
  U becomes the bottleneck, but both still categorically beat Tabular. HONEST strengthening.
- Cycle 63 (audit fixes, no headline impact): the true-d run_episode family (pilot_noise/
  pilot_refit/pilot_choice_only) handed learners the TRUE rank as factor dim; benign (all
  paper headlines use the d_hat harnesses run_masked/run_anytime_clshp), now documented as an
  ORACLE-RANK DIAGNOSTIC and made overridable via an optional d_hat= arg. Harness-side audit
  written into ZK_COMPLIANCE.md (CLEAN after the E7 fix).
- Cycle 63 (path BUG FIX): save_results() used a cwd-relative "results/pilots", so an
  experiment launched from experiments/ silently wrote its JSON to experiments/results/pilots
  (caught when E7's new file did not appear at the repo root). save_results now ROOT-anchors a
  relative results_dir; misplaced file moved back. LESSON (again): cwd-relative paths in
  generators/savers are a recurring footgun -- anchor to ROOT.
- Cycle 64 (RAS paper TERM/NAME CONSISTENCY pass): single sweep to make terminology and
  method names identical across paper prose, tables, and figures. Standardized: "drone" to
  "robot" (kept the proper env name tabula_drone); "observation density" to the canonical
  "broadcast rate rho"; "no-structure"/"no structure" to "structure-free"; the flagship
  figure label "RewardCF (low-rank CF, ours)" to "SwarmCF (...)" in F16/F18 so every embedded
  figure shows the SwarmCF family name (F5/F6/F13/F17 already did); and one stray "tasks" to
  "targets" inside the all-target mission paragraph (general framework stays "task", concrete
  mission/examples stay "target"). PTF left as-is: the paper deliberately uses the code name
  "PTF" in prose and "SwarmCF-batch (PTF)" in Table 2/3, so figures naming it "PTF" are
  already consistent. method_profiles.py (single source of truth) updated; figures + paper
  regenerated; verified in the rendered HTML (0 stray drone/observation-density/no-structure).
  NOTE: the tutorial keeps its "drone" voice on purpose (drone-domain teaching doc, internally
  self-consistent); it was NOT regenerated so its frozen drone figures do not mix with the
  paper's robot figures.
- Cycle 65 (RAS submission hardening + reviewer cycles to acceptance): acted on direct
  requests and then reviewed the paper line by line as an RAS reviewer over several cycles.
  NAMING: one display name per method (SwarmCF, SwarmCF-batch; external methods keep their
  literature names MF-SGD/ESTR/BPMF/SoftImpute; structure-free Independent-UCB/Tabular/Random;
  ceilings Centralized (clean)/(CTDE)/Oracle); "PTF" dropped from all prose and figures.
  TEMPLATE: removed non-paper text (the "Self-contained manuscript ..." subtitle, "(RAS)" in
  the HTML title); added author block, Declaration of competing interest, Data availability
  (repo link); Highlights trimmed to <=85 chars; abstract 235 words (<=250). TABLE 1: clearer
  headers + streamlined cells. FIGURE 1: redesigned as a clean legend-based schematic.
  REVIEWER FIXES: (1) HTML rendering bug, a literal "<" in math ate scope-condition (i), now
  "&lt;"; (2) SwarmCF-batch reframed as ours, not an external "field"; (3) removed undefined
  "ZK-MRTA"; (4) INTEGRITY: tabula_drone is our own in-repo simulator, so dropped "open-source
  ... we did not design" and reframed as a separate higher-fidelity simulator with different
  dynamics; (5) Table 2/3 notation made consistent (dropped Table 3's redundant profile column)
  and method sets aligned (removed Homogeneous-UCB); (6) Appendix A self-grades -> neutral
  remarks; (7) completed authorless reference [18] (Athira K. A. et al., ACM Comput. Surv.
  2024); (8) condensed a duplicated Section 3 sentence; removed em-dashes. Committed rounds
  16-17, pushed. OPEN (user): fill the placeholder author/affiliation block before submission.
- Cycle 66 (deep reviewer rounds 18-20: theorems, references, citations, related work):
  THEOREMS (round 18): the structure-free floor was near-definitional, so it is now
  Proposition 1; the four substantive results are Theorems 1-4 (row-completion Theta(d) vs
  Theta(n) separation; anytime separation; decentralized masked recovery, the novel one;
  collective speedup). All body/appendix cross-refs renumbered. REFERENCES (round 19): a
  background agent web-verified all 41; only [25] STRATA was wrong (it is a JAAMAS journal
  article, Autonomous Agents and Multi-Agent Systems 34:38, 2020, not the AAMAS conference) --
  fixed. Added verified refs: [42] Ammad-ud-din et al. (federated CF), [43] Ling et al.
  (decentralized matrix completion), [44] McMahan et al. (FedAvg), [45] Auer et al. (UCB1),
  [46] Bernstein et al. (Dec-POMDP), and (round 20) [47] Kuhn (Hungarian method) + [48]
  Sarwar et al. (SVD fold-in). 48 refs total; verified programmatically that EVERY ref is
  cited and no citation is dangling. TABLE 1: every paradigm row now carries citations.
  RELATED WORK: filled gaps (federated CF/learning, decentralized matrix completion,
  Dec-POMDP positioning). CITATION AUDIT: cited weighted ALS [28,31], UCB1 [45], Hungarian
  [47], fold-in [48] where prior-art concepts are used. PROSE: "for example aerial vehicles"
  (was "say"); "Computational cost" (was "On-board cost", colliding with "Onboarding").
  Committed rounds 18-20, pushed.
- Cycle 67 (reviewer rounds 21-23: Table 2, anytime rigor, ceiling attribution): TABLE 2
  (round 21) dropped the redundant profile-badge column (same fix as Table 3), full-word
  headers, Communication reads none/full, ours rows shaded. ANYTIME RIGOR (round 22): Figure 3
  shows eps-greedy Tabular climbing to ~0.25, so "structure-free stuck near random" was false
  -- only per-arm UCB is pinned; corrected prose + caption; fixed the scarcity condition
  ("n>>cT" -> "n>>T" in Section 3/scope/limitations; it contradicted the intro and the headline
  params cT=1000>n=240); qualified Theorem 2's SwarmCF bound as conditional on basis recovery;
  moved the Fig 3 legend off the annotation. CEILING ATTRIBUTION (round 23): Section 6.4's "~81%
  of the ceiling" was the DEFERRED ContentionAdaCF; corrected to the in-paper plain SwarmCF
  (~80%, 0.44 vs 0.55 ceiling, from ctde_20260523_193407.json) and reframed the residual gap as
  what the deferred de-confliction targets; defined Theorem 2's function g; noted skill can be
  negative; fixed two missing-space merges. Verified: cross-references (Figs 1-6, Tables 1-3,
  Prop 1 + Thms 1-4, Appendices A-D) all resolve; 48 refs all cited. Committed rounds 21-23,
  pushed. Reviewer recommendation: accept.
- Cycle 68 (publish: GitHub Pages index + Word export): fixed Table 2's guessed-rank symbol
  (combining circumflex "d&#770;" rendered poorly in the sans-serif table; switched to KaTeX
  "$\hat d$"). make_ras_paper.py now also writes docs/index.html, so the RAS paper is the
  GitHub Pages landing page (Pages serves main:/docs at apartsinprojects.github.io/ZKDroneSwarm/;
  the old "Zero-Knowledge MRTA" index was replaced). Exported a Microsoft Word version via the
  html2doc skill (KaTeX -> MathML -> OMML -> academic styling): docs/ras_paper.docx, 273 native
  editable Word equations, figures embedded, Tables full-width. Added a fixed top-right
  "Download .docx" link in the HTML/index (hidden in print; the docx was generated before the
  link so it stays clean). Committed + pushed.
- Cycle 69 (onboarding semantics fix): the fold-in O(d-hat) onboarding claim was conflating new
  TASKS with new ROBOTS. A new task folds in instantly because the swarm already holds the
  robot-factor basis; a new robot has NO memory and, with no communication, cannot be handed the
  basis, so it must first recover the task factors from the passive broadcast (Theorem 3 coverage
  time) and only then fold in -- bounded by recovery, not O(d-hat). Renamed Section 4 to
  "New-task onboarding (fold-in)", added the honest new-robot caveat, and dropped "and robots"
  from the abstract/contribution fold-in claim (team-growth benefit stays under Theorem 4 /
  positive scaling). Rebuilt docs/ras_paper.docx from a link-stripped copy. Committed + pushed.
- Cycle 70 (Figure 4b consistency: add SwarmCF-batch to the scaling panel): panel (a) included
  SwarmCF-batch but (b) did not. Re-ran pilot_scale_m.py with PTF added (4 methods x 5 swarm
  sizes x 8 seeds, rho=0.5; new data scale_m_20260524_053659.json). Result: SwarmCF-batch also
  scales positively and MORE steeply (unseen 0.098 at m=5 -> 0.576 at m=80), overtaking online
  SwarmCF for large teams (m>=40) as pooled observations sharpen its one-shot refit -- the same
  online/batch crossover seen along rho in Figure 2, now along m. Added PTF to the F18 panel-(b)
  loop, updated the Figure 4 caption and a Section 6.3 sentence. Regenerated figures + paper +
  Word docx. Honest reading: the flagship remains the online variant for its robustness under
  masking (the operational regime); batch wins with dense data (high rho / large m). Pushed.
- Cycle 71 (TabulaDrone naming + Section 6.5 redraft): the simulator is OUR OWN (the in-repo
  tabula_drone package; __init__ declares "TabulaDrone: Reinforcement Learning Environments for
  Drone Target Engagement"), so the paper must not present it as external. Renamed the bare code
  path tabula_drone -> TabulaDrone (proper name) throughout the paper and the F13 figure title;
  retitled Section 6.5 "Robustness and transfer to a higher-fidelity simulator" (was "external
  validity"); redrafted the section to state plainly that TabulaDrone is a PettingZoo/Gymnasium
  RL environment from our own project, built as a general drone-engagement testbed independently
  of the CF method, and to frame the result as transfer/generalization beyond the analytical
  model (not third-party external validation). Also clarified Section 6.4 that the mission
  (Figure 5) is the analytical harness of Section 3 reframed, not a separate simulator (Figure 5
  is one experiment at two broadcast rates; the centralized-ceiling result in the same section is
  a second experiment in the same harness; the only different simulator is TabulaDrone, Figure 6).
  Regenerated figures + paper + Word docx. Pushed.
- Cycle 72 (rename to LatentSwarm + background correctness review + fixes): renamed the simulator
  TabulaDrone -> LatentSwarm (chosen) and rewrote Section 6.5 to state it implements a VARIANT of
  the setting (keeps the low-rank capability x requirement reward + per-observer noise; adds 2-D
  motion, depleting-target HP, rectified damage, episodic). Verified the env actually matches:
  reward = np.dot(drone_latent, target_latent) (low-rank), per-drone reward noise, decentralized.
  Added Figure 4(b) fixed rho=0.5 to caption and x-axis. A background reviewer agent read the
  paper line-by-line; fixed the genuine issues it found: (1) Section 6.4 "beats every other
  low-rank method" was violated by our OWN deferred EMCF (0.360 > RewardCF 0.348) -> scoped to
  "every external low-rank method and our batch variant ... best external alternative"; (2) Table
  3 note "SwarmCF leads" violated by deferred BothCF (0.349 > 0.336) -> "among the methods shown",
  added a note that deferred confidence variants can edge it out; (3) the Table 3 bake-off (c14)
  is 5 seeds not 8 -> corrected Setup/Table 3 note/Appendix D; (4) Theorem 3 tightened to
  pair-identifiability iff (p_i in span{p_k}), full u_j recovery as the stronger case; (5)
  reconciled the noise term with Appendix B via sigma_min(B)=Theta(sqrt|E|); (6) dropped the
  ill-posed "Theta(m)-fold speedup over a lone learner" (a lone learner never recovers); (7)
  LatentSwarm 0.806 +/- 0.016 relabeled as std over 3 seeds (not a bootstrap CI); (8) noted
  Proposition 1's Omega(1) relies on the bounded normalized reward. Background scout: best
  EXTERNAL benchmark to cast the problem onto is Level-Based Foraging (native capability-vs-
  requirement match, partial obs, comms-free); added as a future-work external-validation plan.
  Regenerated figures + paper + Word docx (290 equations). Pushed.
- Cycle 73 (LatentSwarm accuracy: no 2-D motion; Appendix E; follow-up paper): reviewing the env
  (drone_engage_latent_mrta.py) showed there is NO 2-D motion: drones/targets sit at FIXED 2-D
  positions (used only as observation features) and each round a drone SELECTS a target; reward =
  signed cosine of latent traits; target HP depletes by the rectified dot; collisions = capacity
  contention; episodic. So "robots move in 2-D" was an overclaim. Fixed Section 6.5 and the
  Figure 6 caption (fixed 2-D layout, target selection, cosine-trait reward, HP depletion,
  contention), and ADDED Appendix E "The LatentSwarm simulator" with a precise construction +
  Algorithm 3 (episode loop). Section 6.5 sentence rephrased (removed "from our own project",
  tighter style). Dropped "a hardware and physics-based simulation study" from future work (all
  future work is software/simulation). Updated the low-rank external-validation pointer to
  RecoGym / bilinear-bandit (LBF is not low-rank-native). Background: drafted the FOLLOW-UP paper
  (experiments/make_ras_paper2.py -> docs/ras_paper2.html, 510 KB, 10 sections on the deferred
  SwarmCF refinements + a recap; figures F11/F15/F7/F10) and linked it top-right of the main
  paper (under the .docx button). Background scout: best low-rank-native external sims are RecoGym
  (reward = Beta@omega) and a bilinear bandit; we do NOT have LBF code and should not replace
  LatentSwarm with it. Regenerated paper + Word docx (324 equations). Pushed. OPEN: ras_paper2 is
  a first DRAFT (theorem numbering + content want a review pass).
- Cycle 74 (three names locked + suite promoted + Figure 5 trim + 16-point clarity pass): named
  the problem setting Zero-Knowledge MRTA (ZK-MRTA) and promoted it through the abstract,
  Contribution 1, the Section 3 heading, and Table 1 (Ours row now "Ours: ZK-MRTA (this paper)"),
  so the paper carries three crisp names: ZK-MRTA (problem), SwarmCF (method family), LatentSwarm
  (software). Promoted LatentSwarm from "a simulator" to a RELEASED evaluation suite and added it
  as a sixth contribution: an open PettingZoo/Gymnasium suite for ZK-MRTA comprising the analytical
  masked-broadcast harness (headline results) and the spatial environment; also mentioned in
  Setup, Section 6.5, Appendix E, and Data availability. Trimmed the old Figure 5 (the F17
  operational-mission embed): kept the operational categorical result as a sentence plus the
  centralized-ceiling comparison in Section 6.4, and renumbered the LatentSwarm transfer figure
  from Figure 6 to Figure 5 (the paper now has 5 figures). Then applied a background line-by-line
  clarity/completeness review (16 findings, all small, none altering a claim): glossed d/n in the
  abstract; introduced d-hat with a gloss at first use; stated the bounded, normalized reward in
  Section 3 (the one Proposition 1 references); named cT/n as the expected offers per task and g as
  a concave order-statistic function in Theorem 2; glossed sigma_min and the rotation-gauge anchor
  block in Theorem 3; glossed tilde-O in Theorem 4; named sigma_own/sigma_obs in Setup; pointed the
  80% / 0.44-vs-0.55 ceiling numbers to the contention sweep and released data; glossed Hungarian
  assignment (optimal one-to-one matching); stated E=16 episodes (converged skill = last 8) in
  Appendix E; made the signed-cosine-vs-inner-product variant explicit in Section 6.5; relabeled
  the Appendix C 0.30 as a reconstruction error (not a skill). One genuine consistency fix: the
  fold-in cost read O(d) in Contribution 2 and O(d-hat) in the Algorithm 2 comment but O(d-hat^3)
  in the Section 4 body, so standardized on O(d-hat^3) (constant in n, m). Also fixed the scope
  condition (i) math for a clean Word export: it used the HTML entity for less-than (so the browser
  parser would not treat the following letter as a tag), which node-KaTeX could not parse; switched
  to the \lt macro (renders the less-than sign in both the browser and the converter, no entity and
  no literal angle bracket). Regenerated paper + index (750 KB) and rebuilt the Word docx via
  html2doc (336 equations, 0 unconverted $, 5 images, Tables 1-3); both nav links and the KaTeX
  loader scripts are stripped from the Word source, and the docx was verified to contain no link
  text and to render every less-than as native OMML math. Pushed. OPEN: ras_paper2 follow-up still
  wants a theorem-numbering and ZK-MRTA/LatentSwarm naming review pass.
- Cycle 75 (authors + README realign + Table 3 CIs + MF-SGD naming + ras_paper2 reconciliation;
  HTML only, docx deferred by user): added the author list (Alexander Apartsin, corresponding; Yigal
  Meshulam; Yehudit Aperstein; Afeka Tel Aviv Academic College of Engineering) to the main paper and
  the follow-up, and rewrote README.md to the current focus: title = the paper title, the three names
  (ZK-MRTA problem, SwarmCF method, LatentSwarm suite), live links to paper / follow-up / tutorial,
  SwarmCF-batch + MF-SGD + LatentSwarm naming, and the three authors. Added bootstrap 95% confidence
  intervals to the two unseen-skill columns of Table 3 (method_profiles.html_scorecard via a new _ci
  bootstrap helper); the note now explains the brackets and that structure-free rows straddle zero.
  Renamed the LatentSwarm env's SGD matrix-factorization policy to MF-SGD in Section 6.5, the Figure 5
  caption, and Appendix E: it is the same SGD-MF baseline family used in our analytical bake-off (env
  class MatrixFactorizationPolicy = our MF-SGD; env WeightedALSPolicy = SwarmCF). ras_paper2 pass:
  added Zero-Knowledge MRTA (ZK-MRTA) to its abstract, foundation box, and setting recap; reconciled
  the theorem references by giving the follow-up's OWN results an F prefix (Proposition F1, Theorem F1,
  Theorem F2, Proposition F2, Theorems F3-F4, Proposition F3) labeled "this paper", removing the bogus
  "Proposition 6 / Theorem 7-8 / Proposition 9 / Theorems 10-11 / Proposition 12, companion theory"
  numbers that do not exist in the main paper (which has only Proposition 1 and Theorems 1-4); aligned
  the fold-in cost to O(d-hat^3). Background scout (RecoGym): reward is genuinely low-rank (click ~
  Bernoulli(ff(beta.omega + bias)), bilinear logit, inner dim 5) but it is single-agent with no
  inter-agent observation mask and an abandoned 2019 codebase (unpinned gym/tf/numba), so casting
  ZK-MRTA onto it means building the whole multi-agent masking layer ourselves; keep it cited as the
  named external benchmark for the follow-up, do not block the base paper.
- Cycle 77 (new pluggable latentswarm package, ground-up; design decisions locked): per the user's
  directive, created a proper Python package latentswarm/ (pyproject, pip install -e .) with separated,
  name-registered pluggable components: config.py (RunConfig: every knob), registry.py (scenario/
  algorithm/metric/viz registries), scenarios.py (GaussianMixture default + IIDGaussian), env.py
  (ZKMRTAEnv: persistent OR per-round mask, inner-product OR cosine reward, all-targets OR size-c
  offers, capacity-1 contention, per-observer private noise), algorithms.py (per-robot ZK estimators:
  SwarmCF weighted-ALS, MF-SGD, Independent-UCB, Random), metrics.py (earned skill, unseen-pair skill,
  Hungarian capacity-1 oracle, bootstrap CI), run.py (config-driven runner). Locked decisions from the
  user: (a) ALL targets offered by default (offer_size=0; size-c is an option); (b) PERSISTENT mask is
  the headline, per-round/dynamic-line-of-sight is a pluggable option (random/i.i.d. masking reduces to
  standard uniform-sampling completion, so persistent is the novel case); (c) guessed rank d-hat drawn
  at RANDOM per run in [d, 2d] (d-hat>=d is required for exact recovery; under-ranking is a separate
  mis-specification regime, an optional ablation). 16-seed run (m=30 n=240 d=5 d-hat~U[5,10] T=50
  rho=0.5 sigma=0.3, all-targets, capacity-1, mixture traits): SwarmCF earned 0.419 [0.402,0.437],
  unseen-pair 0.583 [0.524,0.639]; MF-SGD earned 0.198, unseen 0.093; Independent-UCB earned -0.169
  (below the random floor under contention), unseen -0.002 (floor); random ~0. The categorical
  separation (structure-free at the unseen floor, SwarmCF far above, SwarmCF >> MF-SGD) is clean and
  tight in the independent package, and stronger than the earlier tabula_bench numbers. Also folded
  Table 2 "ours (hybrid)" into "ours", removed "faithful" from the paper + F13, and applied 8
  scientific-voice fixes from a whole-paper audit. OPEN (next): viz module + regenerate F13 from
  latentswarm_main.json; Figure 1 redesign (unseen target + unseen pair + identifiability); paper
  integration (Section 3 all-targets default + random rank + mask-pluggable note; Section 6.5 /
  Appendix E rewrite to the package + new numbers); unit tests; deprecate tabula_drone; finish the
  win/advantage voice pass.
- Cycle 78 (graceful-degradation rank ablation): added experiments/ranksweep.py (uses the latentswarm
  package) sweeping the guessed rank d-hat from d/2 (=2) through 3d (=15) at true d=5, 8 seeds, with
  SwarmCF and MF-SGD earned + unseen-pair skill (bootstrap CI). SwarmCF unseen-pair skill degrades
  smoothly under-ranking (0.43 at d-hat=2, 0.52 at 4) and is robust at/above the true rank (0.55-0.65,
  d-hat=5..15); earned skill rises 0.24 -> ~0.42 plateau; MF-SGD stays low throughout. Confirms the
  claim that over-guessing is safe (surplus dims are regularized) and under-guessing degrades gracefully
  (mis-specification, not collapse). Saved results/pilots/latentswarm_ranksweep.json and figure
  F21_ranksweep (png+pdf); to be embedded as a robustness figure during the Section 6.5 integration.
- Cycle 79 (paper integration of the latentswarm package): rewired F13 to read
  results/pilots/latentswarm_main.json (new package numbers) with updated labels (swarm_cf/mf_sgd).
  Rewrote Section 6.5 to the latentswarm-package framing with the new results (SwarmCF earned 0.42
  [0.40,0.44], unseen 0.58 [0.52,0.64]; MF-SGD 0.20/0.09; Independent-UCB -0.17/floor), the all-targets
  menu, and a guessed rank drawn at random per run; added a "Robustness to the guessed rank" paragraph +
  Figure 6 (the F21 graceful-degradation sweep). Updated Section 3 (menu = all tasks by default, size-c
  optional; d-hat random per run and robust, Figure 6; persistent mask is the primary case while an
  i.i.d. per-round mask is the easier standard-sampling regime, and the suite supports both), the Setup,
  and Appendix E (the modular pip-installable package; all-targets default; d-hat ~ Uniform{d..2d};
  Algorithm 3 updated). Replaced the "PettingZoo/Gymnasium" claims with "modular Python package" (the new
  package is plain, not PettingZoo). Regenerated F13 (only F13 churned) + HTML (848 KB); docx not rebuilt
  (HTML phase). OPEN: Figure 1 redesign (unseen target + unseen pair + identifiability); unit tests for
  latentswarm; deprecate tabula_drone; finish the win/advantage voice pass.
- Cycle 80 (Figure 1 redesign): redrew F20_setting to deliver the confirmed message. Beyond the focal
  row's unseen pairs ('?'), it now shows (i) a COMPLETELY UNSEEN TARGET (a column the focal robot never
  engaged) that is recoverable because visible teammates engaged it (purple column + purple focal cell =
  the predictable unseen pair), and (ii) a second unseen-target column with NO visible engagers that
  stays non-identifiable (orange dashed), i.e. the Theorem 3 condition. Column callouts moved above the
  matrix to avoid the legend. Updated the Figure 1 caption to match. Regenerated F20 + HTML (870 KB);
  docx not rebuilt. OPEN: unit tests for latentswarm; deprecate tabula_drone; finish the win/advantage
  voice pass.
- Cycle 81 (latentswarm tests + tabula_drone deprecation): added latentswarm/tests/test_latentswarm.py
  (8 smoke + contract tests: registries populated; scenario shapes; env contract incl. all-tasks menu
  and capacity-1; persistent mask + self-visibility; every policy runs and predict_rows has the right
  shape/None; metrics + Hungarian oracle; random rank-guess in range; SwarmCF > random end-to-end) -- all
  pass via `python -m latentswarm.tests.test_latentswarm`. Marked tabula_drone DEPRECATED (docstring +
  DeprecationWarning in its __init__), superseded by latentswarm. Updated README: reproducibility now
  uses `python -m latentswarm.run` and `experiments/ranksweep.py`, describes the modular latentswarm
  package, notes tabula_drone deprecated, and refreshed the LatentSwarm description + validation numbers
  (earned 0.42, unseen 0.58). OPEN: finish the win/advantage scientific-voice pass.
- Cycle 82 (scientific-voice pass, part 2): applied the remaining grounded-voice fixes from the audit:
  "win"/"wins" -> "separation"/"leads" (Section 5 and 6.1 headings, the batch crossover, the
  discussion), "beats every external" -> "exceeds", "what the team and broadcast actually buy" ->
  "contribute", dropped "crucially" and the "We answer yes" rhetorical answer, "the setting in one
  line" -> "in brief", and grounded "the swarm gets smarter as it grows" -> "per-robot competence
  rises with team size" (highlights, abstract, Section 6.3). Regenerated HTML (870 KB); docx still not
  rebuilt (HTML phase). The latentswarm rebuild + paper integration the user requested is now complete
  end to end (package + tests + Section 3/6.5/Appendix E + Figures 1, 5, 6 + README + deprecation).
- Cycle 83 (author/affiliation update, Figure 1 revert, "independent" wording, review fixes): per the
  user: (1) Alexander Apartsin's affiliation -> Holon Institute of Technology (now two affiliations: HIT
  for Apartsin, Afeka for Meshulam + Aperstein), and Yehudit Aperstein is the corresponding author
  (moved the asterisk); applied to ras_paper, ras_paper2, README. (2) Reverted Figure 1 to the previous
  (cleaner) schematic per the user (git checkout of make_figures.py F20 block from 2173f8f, keeping the
  newer F13; reverted the caption). (3) "independent implementation" -> "separate implementation"
  throughout (the user asked what it meant: a second, separately-coded implementation of the same
  setting whose env is written independently of the SwarmCF method, NOT third-party external validation;
  both are ours). A background review agent (correctness/consistency/clarity + structure) flagged real
  issues; fixed: Section 6.4 "best external alternative ~0.29" was actually our SwarmCF-batch, not an
  external method, so removed the unsupported/mislabeled 0.29 + "non-overlapping intervals" and deferred
  to Table 3; standardized the UCB baseline name to "Independent-UCB" (was named 5 ways); named the
  Figure 2 crossover method SwarmCF-batch (was ambiguous "batch spectral completion"). The agent's
  "Figure 6 should be 16 seeds" flag was a false alarm (the rank sweep legitimately uses 8 seeds, the
  main LatentSwarm run uses 16). Regenerated figures + both papers; docx still not rebuilt (HTML phase).
  RECOMMENDED but NOT yet done (needs the user's nod, involves figure renumbering): split the overloaded
  Section 6.5 into 6.5 Robustness (scope + rank, Fig) and 6.6 the LatentSwarm separate implementation
  (Fig). Low-priority remaining: add SoftImpute row to Table 3 or note its omission; gloss ESTR/BPMF at
  first use.
- Cycle 84 (random-d-hat re-run of the main analytical sweeps; Figure 1 fix; abstract; GitHub Pages):
  re-ran the five main sweeps (pilot_compare/c14 -> Table 3, pilot_crossover/c15 -> Fig 2,
  pilot_anytime/c16 -> Fig 3, pilot_collab + pilot_scale_m -> Fig 4) with the guessed rank d-hat drawn
  at RANDOM per run in [d,2d] (shared guessed_rank(seed) in pilot_compare, reused by the others; kept
  c=20 per the chosen scope). Numbers stable vs fixed d-hat=8 (RewardCF unseen 0.357 at rho=0.25),
  confirming rank robustness. Updated Setup + Appendix D to say d-hat drawn at random per run in [d,2d].
  Dropped the explicit "LatentSwarm" name from the abstract (kept the "reproduces in a separate
  implementation with contention" claim; the name stays in contributions / 6.5 / Appendix E / Data
  availability). Fixed Figure 1: the focal robot's own-engagement cells were green OUTLINES merged with
  the blue row border (invisible); now FILLED green + matching legend. Confirmed the RAS paper is the
  GitHub Pages landing (make_ras_paper writes docs/index.html == docs/ras_paper.html). OPEN (next, per
  user): switch the BODY to the unrestricted-c (c=n, all-tasks) variant and move the c=20 offered-subset
  variant to an APPENDIX figure referenced from the body; add a RANDOM (i.i.d. per-round) masking
  variant to the appendix with its figure; do NOT touch the follow-up paper. Per user instruction,
  regenerated HTML + index only and did NOT rebuild the Word docx (HTML-editing phase), so the docx is
  intentionally one round stale. A LatentSwarm implementation review (alignment to the Section 3
  setting) was delivered to the user separately.
- Cycle 76 (LatentSwarm rewritten as a faithful ZK-MRTA instantiation + capacity-1 contention; re-run;
  F13 + Section 6.5 + Appendix E; HTML only): per the user's review decision, made the LatentSwarm
  spatial environment a FAITHFUL instantiation of the Section 3 setting instead of a
  cosine/2-D/HP-depletion/episodic variant. Env (tabula_drone/envs/drone_engage_latent_mrta.py): added
  (a) a persistent per-pair Bernoulli(rho) observation mask with INDEPENDENT per-observer (private)
  reward noise, so no two robots see the same stream (the heart of the setting; previously every robot
  saw the full shared stream); (b) reward_mode="dot" = signed inner product R_ij=<p_i,u_j> (the Section
  3 reward; previously cosine); (c) capacity_one contention (only the first robot to pick a task each
  round succeeds; previously collisions were merely counted); (d) per-round per-robot offered size-c
  subsets (forcing broad coverage so unseen-pair recovery is testable). 2-D positions are now inert
  (dropped; t-SNE of the latent traits is for visualization only). All four are opt-in params with
  legacy-preserving defaults. Bench (experiments/tabula_bench.py): signed Gaussian-MIXTURE (block)
  traits matching the analytical harness, m=30 n=240 d=5 d-hat=8 c=20 T=50 rho=0.5 sigma=0.3, the SAME
  guessed rank and exploration schedule for MF-SGD and SwarmCF (fair), 16 seeds, an analytic capacity-1
  Hungarian oracle as the de-conflicted ceiling, and two metrics (earned anytime skill, self-normalized
  unseen-pair skill). Result (bootstrap 95% CI): SwarmCF earned 0.313 [0.293,0.330] >> MF-SGD 0.130
  [0.104,0.157] >> Independent-UCB -0.083 (below the random floor: under contention its persistent
  exploration collides without coordinating); SwarmCF is the ONLY method with unseen-pair skill
  significantly above the floor (0.104 [0.043,0.167]; MF-SGD/UCB/random all straddle 0), the Proposition
  1 categorical separation reproduced in independent code. The gap to the Hungarian ceiling is
  un-de-conflicted contention (the deferred de-confliction refinement), not estimation. The old 0.806
  was an artifact of the unfaithful easy setting (full visibility, cosine, sample-rich, no contention);
  the faithful setting is much harder, so the absolute number is lower but the categorical result is
  cleaner and honest. Rewrote F13 (two bars: earned + unseen skill with bootstrap CIs), Section 6.5
  (independent-implementation-with-contention framing + new numbers), the Figure 5 caption, Appendix E
  (faithful env + Algorithm 3 mission loop), and the abstract / contributions / Setup / Data-availability
  mentions (dropped "higher-fidelity spatial / different dynamics"). Regenerated F13 (only F13 churned)
  and the HTML + index; per user instruction the Word docx was NOT rebuilt (HTML phase). Background
  scout (RecoGym): genuinely low-rank reward but single-agent, no inter-agent mask, abandoned 2019
  codebase, so keep it as a follow-up external benchmark, not for the base paper.

## Cycle 85 (c=20 body / c=n appendix; LatentSwarm = the only codebase; Section 6.5 = contention; de-confliction; Figure 1 blue circles)
Re-ran the five main sweeps under the new defaults (random d-hat per run in [d,2d]; offer size restored to
c=20 as the body default) plus pilot_iid (persistent vs i.i.d. masking, now random d-hat) and a NEW
LatentSwarm contention experiment (experiments/latentswarm_contention.py: earned skill + collision rate
per policy at offer sizes {all-tasks, c=20}, capacity-1, 16 seeds). Fixed run_masked's eval coupling
(ev=min(cand,20)) so the unseen eval is comparable across menu sizes and the all-tasks (c=n) run works.
Headline decision (user): keep c=20 in the BODY (where online SwarmCF leads the batch methods on
unseen-pair skill) and move the unrestricted all-tasks menu (c=n) to a new Appendix F. Appendix F
honestly documents that under an unrestricted menu the greedy online estimator under-explores (collective
engagement coverage narrows) so its lead over batch completion narrows, while the categorical
low-rank-vs-structure-free separation is unchanged; it also adds the persistent-vs-i.i.d. masking-model
robustness figure. The c=n bake-off at rho=0.25: RewardCF unseen 0.064 vs PTF 0.137 / BPMF 0.168 (lead
gone), vs c=20 RewardCF 0.357 (lead intact), confirming the offer-size contingency.

Restructured Section 6 to a SINGLE-codebase narrative (user: "no reason to mention any codebase other
than LatentSwarm"): dropped every "separate/independent implementation", "reproduced in independent
code", "not an artifact of one implementation", and "analytical harness" two-codebase phrase from the
abstract, contributions, Setup, Section 6, and Appendix E. New 6.5 "Capacity-1 contention and
communication-free de-confliction" carries the contention story (its main story IS contention): (a) the
separation survives contention (SwarmCF earned 0.42 of the capacity-1 Hungarian ceiling, unseen 0.58;
Independent-UCB earns -0.17, below random, because it collides); (b) the new collision finding (Figure 6):
under the all-tasks menu Independent-UCB collides on ~0.97 of engagements while SwarmCF collides only
~0.12, because its HETEROGENEOUS learned models implicitly de-conflict with no message passing;
restricting to a size-c menu cuts UCB collisions 0.97 to 0.25 (one reason the body uses c=20). Section 6.4
now establishes that coordination (not estimation) is the binding constraint and points to 6.5, removing
the contention/ceiling double-coverage that previously sat in both 6.4 and the LatentSwarm section. Section
6.6 "Robustness across configurations" absorbs the rank-robustness sweep + offer-size/masking pointers to
Appendix F. Added three one-line forward-pointers to the follow-up (de-confliction private offset ~2x
earned reward at severe contention; confidence-directed exploration restores coverage under the
unrestricted menu; ARD removes the guessed rank). Figures renumbered 1-9 (bake-off 5, collision 6, rank 7,
offer-size 8, masking 9), all captions and cross-references checked.

Figure 1: own clean engagements are now bold BLUE CIRCLES (were green cell fills, camouflaged on the
red-yellow-green heatmap); teammate engagements relabeled "sensed, noisy". Figure-caption math fix: a raw
"<" in "$\hat d<d$" was parsed as an HTML tag and swallowed part of the rank caption; replaced with the
KaTeX-safe \lt / \gt macros (scanned the whole paper, no other raw "<" in math). Re-ran opmetrics so Table
3 regret/ttc are on the c=20 anytime data. Regenerated all figures (F5/F6/F18 byte-identical to the
committed c=20, confirming the deterministic re-run; F8/F19/F20 changed; new F22/F23) and rebuilt
docs/ras_paper.html + docs/index.html (1325 KB). Per standing instruction the Word .docx was NOT rebuilt
(HTML phase) and the follow-up paper (ras_paper2) was untouched. NOTE for a later code cleanup: the
masked-broadcast experiments (core/pilot_*) and the contention package (latentswarm/) are still distinct
modules presented under the single name LatentSwarm; porting the body drivers into the package for literal
single-command reproducibility is deferred.

## Cycle 86 (top-tier RAS reviewer pass; edit-bucket E1-E10)
Acted as a top-tier RAS reviewer and read the RENDERED docs/ras_paper.html line by line (not the
generator), producing a feedback list partitioned into (a) edits and (b) more-simulation experiments.
The edit bucket E1-E10 was applied (E11, an Elsevier/LaTeX reformat, is deferred to the production stage):
honesty and scoping fixes for the headline claims (scope the "floor" language to the unseen columns,
note the tabular learner is competitive on the operational metrics, soften the within-CI batch margin),
cross-reference and define-before-use tightening, and consistency of method names (SwarmCF-batch, not
PTF, in the rendered tables). Committed as e5b8d79. The experiment bucket (X1 tight 16-seed CIs, X2
robotics-grounded instance, X3 Theorem-2 strict-regime anytime, X4 probe-restores-the-online-lead at the
all-tasks menu, X5 approximate-low-rank, the last not yet started) was queued for the following cycles.

## Cycle 87 (X2 robotics-grounded instance; LatentSwarm in a separate folder; referee infra)
Built the robotics-grounded ZK-MRTA instance in LatentSwarm: a sensing_coalition scenario (non-negative
capability/requirement profiles over d sensing modalities: electro-optical, infrared, acoustic, LiDAR,
range-endurance, so the reward is a modality match, rank-d by construction) and a line_of_sight mask
(visibility from a range-limited disk graph induced by 2-D patrol positions, with per-observer noise that
grows with distance, sigma^2 proportional to 1+(r/R_s)^2). Added the X2 driver
(experiments/latentswarm_grounded.py) and the X2-precursor collision study
(experiments/latentswarm_contention.py). Moved LatentSwarm into its own folder with a detailed README and
a user guide and a developer guide. Ran X2 (16 seeds, capacity-1, random rank guess). Committed as f195126.

## Cycle 88 (LatentSwarm unification: configurable, pluggable, parity-checked)
Unified the suite so the package can reproduce the paper from one codebase without re-running here. Made
every parameter configurable through RunConfig (no hard-coded constants): geometry knobs, world/reward
knobs, and the per-policy hyperparameters. Added a block_cosine parity world that ports core.make_world
and is bit-identical to the analytical harness at jitter=0.15 (default jitter is 0.2; exact parity needs
0.15, documented). Made the world and the reward pluggable (block_cosine / gaussian_mixture /
iid_gaussian / sensing_coalition worlds; inner_product / cosine rewards; normalized vs unnormalized).
Ported the baselines (tabular, ucb_homo, estr, swarmcf_batch = PTF, bpmf) and metrics (anytime_trajectory,
cumulative_regret, time_to_competence, state_uniqueness, unseen_pair_skill_heldout) as registered
drop-ins, and added config-driven sweep drivers (python -m latentswarm.sweeps --which
crossover|anytime|collab|scale_m|ranksweep|offersize|iid_vs_persistent). Verified, NOT re-ran the paper:
13 passing tests, a 3-seed parity check (all metrics within overlapping ranges), and an independent
bit-identity check of the parity world. Tagged pre-unification as a revert anchor. Refreshed the package
README/user/developer guides and the root README to link LatentSwarm and describe the single codebase
(e3b35c8, README 35a1fae).

## Cycle 89 (integrate X1 + X2 + X3 + X4 into the base paper; rebuild HTML)
Re-ran the headline sweeps at 16 seeds (X1: crossover, anytime, and the operational metrics at the c=20
body default) and integrated the tighter CIs into Table 3 and the operational-metrics page; the ordering
is unchanged (SwarmCF leads the operational columns, SwarmCF-batch wins only full broadcast, structure-free
at the floor on the unseen columns), and SwarmCF's masked-unseen margin over the batch variant stays
within the 16-seed interval (kept the honest "within the interval" wording rather than claiming a clean
win). Wrote a new Section 6.7 "A robotics-grounded instance" from the X2 run (sensing-modality traits +
line-of-sight mask + distance noise; SwarmCF reaches unseen-pair skill 0.19 [0.15,0.24] and earns 0.32 of
the centralized ceiling while structure-free learners sit at the floor and Independent-UCB earns below
random by colliding; Figure 8). Added two Appendix-F robustness figures: Figure 11 (X3) re-runs the
anytime comparison in Theorem 2's strict scarce-offer regime (c=3, cT < n) where the structure-free
collapse appears as predicted, and Figure 12 (X4) shows a short UCB probe restores SwarmCF's lead over
batch completion under the all-tasks menu, isolating the contingency as the exploration schedule, not the
estimator. Renumbered Appendix F (former Figures 8/9 are now 9/10) and updated the abstract and
contribution 5 with the robotics-grounded clause; updated all seed-count text to "16 random seeds (8 for
the scaling sweeps of Figure 4)". Fixed a figure-builder crash in make_figures (np.eye(mm, bool) read the
dtype as the columns argument; now np.eye(mm, dtype=bool)) so F18/F26 generate. Rebuilt
docs/ras_paper.html + docs/index.html (1682 KB). Per standing instruction the Word .docx was NOT rebuilt
(HTML phase) and the follow-up paper (ras_paper2) was untouched. The scaling sweeps (Figure 4) were not
re-run, so they remain at 8 seeds, as stated in the Setup and caption.

## Cycle 90 (second RAS-reviewer pass: address review points M1-M4; HTML only)
A line-by-line top-tier-reviewer read of the rendered HTML surfaced four substantive points; addressed all
four in the generator (make_ras_paper.py) and rebuilt (no figures or data changed). M1 (reconcile the
unseen-pair-skill numbers that appear as 0.316 in Table 3, 0.58 in Section 6.5, and 0.19 in Section 6.7):
added a non-comparability note in 6.5 (the contention study uses rho=0.5, the all-tasks menu, and the
stricter capacity-1 Hungarian normalization, so its values run higher and are not directly comparable to
Table 3's rho=0.25 size-c numbers; only the categorical above-floor-vs-at-floor pattern carries across) and
a matching note in 6.7 (geometry-limited line-of-sight is a harsher channel, hence lower absolute skill, a
separate grounded instance). M2 (Theorem 2's g was promised "made precise in Appendix A" but never
defined): defined g explicitly in Appendix A as the expected order statistic
g(y)=E[(max_{j in A} R_ij - mu_i)/(max_{j in S} R_ij - mu_i)] (A = engaged-and-offered subset of expected
size y), increasing and concave with g(0)=0, with the Jensen step across rounds and the honest loose-constant
remark; body wording changed "made precise" -> "defined in Appendix A". M3 (SoftImpute appeared in Table 2
but had no row in the Table 3 scorecard, since the c14 bake-off data contains no SoftImpute): removed
SoftImpute from the Table 2 method list so the two tables' non-ceiling method sets match. M4 (the "about 80%
of the ceiling" claim was asserted in prose only): sourced it from the released results/pilots/ctde JSON,
now stated with the actual matching-normalized anytime earned-skill numbers at n=240 (SwarmCF 0.44 vs the
full-communication ceiling 0.52 and the noiseless matcher 0.55, ~84%), and updated the dependent "drop from
~80% to 42%" in 6.5 to ~84%. Verified in the rebuilt HTML: SoftImpute count 0, no stale "about 80%" or "made
precise", g-definition LaTeX well-formed (balanced), both notes present. Word .docx not rebuilt; ras_paper2
untouched.

## Cycle 91 (external PNG figures + callout-box processing + minor review points; HTML only)
Three groups of changes, all in the generator. (1) Figures: switched the make_ras_paper.py img() helper from
inline base64 data URIs to external references (src="figures/NAME.png", loading="lazy"); the build-time
existence check is retained and the base64 import dropped. docs/ras_paper.html and docs/index.html fell from
~1683 KB to 83 KB (~20x), and prose diffs are now readable instead of being buried under megabyte image
blobs. GitHub Pages serves docs/figures/ alongside the HTML, so the relative refs resolve; all 12 referenced
PNGs are git-tracked. (2) Callout boxes, per the reviewer recommendation: cut box #1 ("The setting in brief",
redundant with the abstract and highlights), de-boxed box #2 ("The observation channel") into normal body
text (it is a core definition, not a sidebar), and kept box #3 (the Section 6.7 scope box) once while removing
its verbatim duplicate from Section 7 (now a pointer to the box). The Highlights, theorem, and algorithm
environments are standard and were left as-is; only the scope callout remains as a box. (3) Minor / line-level
review points: defined sigma_own in the Section 3 model (own-observation noise sigma_own < sigma, with the
Section 6 values), removed the "default" offer-size ambiguity (the model permits all n; the body uses c=20,
Appendix F studies c=n), added seed counts to the Figure 2 / 3 / 4 captions (16 / 16 / 8), softened "provably
useless" -> "provably uninformative" (x2), and changed Table 1's "the open cell (hardest)" -> "(most
constrained)". Not done: demoting Theorem 1 to a Lemma (a "consider" item; it would cascade a renumber, was
already weighed in the Cycle-80 theorem audit, and the theorem already carries a "standard given U" remark),
and trimming the abstract (it is ~230 words, within the RAS 250-word limit). Word .docx not rebuilt
(HTML phase); follow-up paper ras_paper2 untouched.

## Cycle 92 (third RAS-reviewer pass: scope subsection, CRediT, submission-readiness; HTML only)
Round-3 reviewer read of the rendered HTML. Tackled the substantive items and folded in the cheap
correctness/compliance fixes. (1) Scope callout: the "Scope: when does SwarmCF beat structure-free
learning?" box was the only colored box left in the body and was topically orphaned at the end of 6.7;
de-boxed it and promoted it to a proper subsection 6.8 "Scope of the advantage", and repointed both
cross-references (6.6 "stated below" -> "of Section 6.8"; Section 7 "the scope box of Section 6.7" ->
"Section 6.8"). There are now zero colored callout boxes in the body (Highlights/abstract/theorem/algorithm
environments are standard and remain). (2) CRediT: added the Elsevier-required "CRediT authorship
contribution statement" section before the competing-interest declaration, with a per-author role draft
(Apartsin: conceptualization/methodology/software/formal analysis/investigation/visualization/writing-
original; Meshulam: methodology/validation/writing-review; Aperstein: conceptualization/supervision/writing-
review). NOTE: the role assignments are a reasonable draft and must be verified/adjusted by the authors. (3)
Theory clarity: Theorem 4's rate Otilde(d(1+n/m)) now states it is at constant broadcast (rho=Theta(1)) and
that the general rate carries the 1/rho of Theorem 3, removing the apparent rho-dependence mismatch with
Theorem 3. (4) Cheap fixes: dropped the dangling "convex completion" from the Section 6 method prose (the
convex/nuclear-norm baseline SoftImpute was removed in Cycle 90, so the body now reads "spectral and
Bayesian completion"); trimmed the keyword list from 7 to 6 (removed "multi-agent bandits") for the Elsevier
6-keyword cap; and noted in Algorithm 1 that the displayed noise weight w=1/sigma^2 is run with uniform w=1
in our headline runs (matching Section 4 / Appendix D). Verified in the rebuild: zero box divs, Section 6
headings 6.1-6.8 in order, CRediT present with all three authors, cross-references resolve to 6.8, keyword
count 6, no stale "convex completion". HTML ~84 KB (external PNGs). Deferred (cosmetic/optional): one-term
standardization of the "privately-*" adjective, and adding an earned-skill/percent-of-ceiling column to
Table 3. Word .docx not rebuilt; ras_paper2 untouched.

## Cycle 93 (close the last two review points + CRediT role correction; HTML only)
Cleared the two deferred round-3 items and applied an author-supplied CRediT correction. (4) Sourced the
ceiling claim in a table: added <b>Table 4</b> in Section 6.4 (centralized full-communication Hungarian
ceiling 0.52, SwarmCF 0.44 = 0.84 of the ceiling, Independent-UCB 0.01; matching-normalized anytime earned
skill, rho=0.25, n=240, 8 seeds, from results/pilots/ctde) and referenced it from the 6.4 prose, so the 84%
no longer lives only in prose. (7) Terminology: standardized the overall private-observation property to the
dominant "privately-noisy", collapsing the abstract's "partial, noisy, privately-perceived" to "partial and
privately-noisy" and the intro's "partial (range-limited), noisy, and privately-perceived" to "partial
(range-limited) and privately-noisy" (removes the noisy/privately redundancy); "privately-perceived" now
appears 0 times, and the one remaining "per-observer-private observation mask" is mask-specific and correct,
so kept. CRediT: per the corresponding author, Yigal Meshulam's contributions are Software, Validation,
Investigation (rephrased from "Implementation, Validation, Experiment running" into CRediT terms), and the
overlapping Software/Investigation were removed from A. Apartsin (now Conceptualization, Methodology, Formal
analysis, Visualization, Writing - original draft); Aperstein unchanged (Conceptualization, Supervision,
Writing - review & editing). Verified in the rebuild: Table labels 1-4 present, 6.4 references Table 4,
Yigal = Software/Validation/Investigation, zero "privately-perceived". With this, round-3 review points 1-7
are all addressed (1,2,3,5,6 in Cycle 92; 4,7 here) and the scope subsection 6.8 is in. Word .docx not
rebuilt; ras_paper2 untouched.

## Cycle 94 (fresh-review minor fixes + clean reference audit; HTML only)
A fresh first-time-reviewer read of the base paper surfaced four minor items; fixed all four in the
generator. (M-a) Numbered the two display equations: the reward (Eq. 1) and the skill score (Eq. 2) now
carry KaTeX \tag{} numbers. (M-b) Appendix A intro relabeled honestly: "proofs ... (full for Proposition 1
and Theorems 1 and 3; order-argument sketches for the anytime and collective-speedup bounds, Theorems 2 and
4)" instead of "self-contained proofs". (M-c) Quantified the abstract claim "recovering most of a centralized
full-communication ceiling" -> "recovering most (about 84%) of ...". (M-d) Added the Elsevier-expected
Funding section (standard "no specific grant" sentence; AUTHORS MUST replace if grants exist). Background
reference audit (agent, web): all 48 references verified real with correct authors/title/venue/year (48 OK,
0 issues; only cosmetic notes such as #11 "MCMC" abbreviation), so the bibliography is submission-ready.
Word .docx not rebuilt; ras_paper2 handled separately (Cycle 95).

## Cycle 95 (follow-up paper polish + alignment to the accepted base; HTML only)
Per user instruction (the earlier ras_paper2 freeze is lifted), polished the follow-up paper assuming the
base is accepted. Done via agent on make_ras_paper2.py only (base paper untouched). (1) External images:
img() now emits src="figures/NAME.png" loading="lazy" (base64 dropped); ras_paper2.html fell ~499 KB ->
~43 KB; all 4 referenced figures (F11_pareto, F15_deconfliction, F7_channels, F10_newcomer) exist. (2)
Terminology aligned to base: abstract "privately-perceived" -> "privately-noisy"; method names already route
through method_profiles.disp() (SwarmCF-* scheme), so no stale PTF/RewardCF/ContentionCF/ChoiceEM/TabulaDrone
remain; matched "categorical separation", "structure-free", "unseen-pair", "anytime", "guessed rank",
"LatentSwarm". (3) Framing as companion to the accepted base: subtitle + scope box + one-paragraph background
recap now treat the foundation paper as published prior work [1] (with a References [1] entry linking
ras_paper.html and a reciprocal back-link), without re-deriving it. (4) Fixed two broken figure
cross-references (in-text Figure 1->2 and 2->3 to match captions). Verified: 0 base64, 4 external imgs, 0
"privately-perceived", 0 em/en/double dashes, 0 stale names. Not yet acted on (need author judgment): explicit
[1, Thm/Sec] pointers into the base once its final numbering is fixed; confirm the follow-up's headline
numbers against the latest logs; optionally promote a few prose-only claims (ARD, churn) to figures; ensure
ras_paper2.docx exists before the .docx link is used. Word .docx not rebuilt; base paper untouched.
Hotfix: the "Preliminary / scope" .prelim box before Section 5 was missing its </div>, so its background
spilled into Section 5; added the closing tag (rendered ras_paper2.html div balance now 14/14).

## Cycle 96 (fresh-review quick wins: Table 4 CIs, scarcity-regime framing, synthetic-evaluation reframing; HTML only)
Knocked out the three quick wins from the fresh review. (Moderate 2) Table 4 now carries bootstrap 95% CIs
(from the per-seed ctde data, 3-decimal to match Table 3): ceiling 0.520 [0.483, 0.559], SwarmCF 0.439
[0.418, 0.462], Independent-UCB 0.005 [-0.004, 0.014]; caption updated to "means with bootstrap 95% CIs over
8 seeds", so no table reports bare point estimates anymore. (Moderate 3) Theory-vs-headline scarcity gap
framed: the Section 6 Setup now states the headline is a moderate scarcity (each task offered about
cT/n approx 4 times) and that Theorem 2's strict scarce-offer regime cT=o(n) is shown separately in Appendix
F (Figure 11), and Section 6.2 adds the same forward-pointer. (Major 1, cheap honest-framing tier) Section 7
Limitations now states plainly that all evidence is in simulation on a reward that is low-rank by
construction, that we do not validate on physical robots or an external benchmark, and that the low-rank
premise is an assumption rather than a measured deployment property, with external/higher-fidelity validation
flagged as future work. Verified in the rebuild (Table 4 CIs present, Section 6/6.2 pointers present, Section
7 reframing present). Still open from the fresh review: the substantive Major-1 approximate-low-rank
robustness experiment, and (optional) re-running the ceiling at 16 seeds and the bake-off at more seeds for
Moderate 4. Word .docx not rebuilt; ras_paper2 untouched.

## Cycle 97 (Major-1 substantive: approximate-low-rank robustness experiment + Figure 13)
New experiment answering the fresh review's Major-1 ("the categorical result may depend on EXACTLY low-rank").
New driver experiments/pilot_approxrank.py perturbs the rank-d block reward with a full-rank Gaussian term,
R_eps=(R+eps*s*G)/sqrt(1+eps^2), s=std(R)/std(G) (entry-wise scale held fixed so the observation SNR is
constant; low-rank energy fraction 1/(1+eps^2); effective rank rises from d toward min(m,n) as eps grows),
and sweeps eps at the masked headline rho=0.25, reusing pilot_compare.REGISTRY/guessed_rank and
pilot_c11_masking.run_masked (run_masked derives reward/oracle/unseen-skill entirely from R, so perturbing R
is sufficient and faithful). 16 seeds. Result (bootstrap 95% CI): SwarmCF unseen-pair skill degrades
GRACEFULLY 0.316 [0.287,0.346] at eps=0 (eff rank 5; reproduces the Table 3 headline exactly) -> 0.285
(eps=0.2, eff rank ~19) -> 0.203 (eps=0.5, eff rank ~28, ~80% low-rank energy) -> 0.079 [0.067,0.092]
(eps=1.0, eff rank ~29, ~50% low-rank energy), staying ABOVE the structure-free floor (Independent-UCB ~0.00,
intervals straddling zero) at every eps; SwarmCF-batch tracks it. So the advantage is a property of
EXPLOITABLE low-rank structure, not of exact low-rankness. Added F27 builder to make_figures.py (bootstrap
CIs + a top axis showing effective rank) -> docs/figures/F27_approxrank.png/pdf; integrated into the base
paper as Figure 13 + an "Approximate low-rank" paragraph in Appendix F (heading renamed), with pointers from
Section 6.6 and Section 7. PDF timestamp-only churn from the figure rebuild was reverted (PNGs are
byte-stable). HTML 87 KB; .docx not rebuilt.

## Cycle 98 (follow-up polish round 2 + external-benchmark scouting; HTML only, base untouched)
Two background agents. (a) Follow-up polish: added 23 explicit base-paper pointers ([1, Prop. 1], [1, Thm 3],
[1, Sec. 6.5], [1, Table 4], etc.), and notably caught and fixed a FABRICATED number (the follow-up Sec 4
claimed SwarmCF-D+ "recovers about 81%"; the base has no such figure and SwarmCF-D+ is a follow-up method ->
rewrote to cite the base's actual plain-SwarmCF ~84% of the ceiling, [1, Sec. 6.4, Table 4]) and an overclaim
(the fold-in bound and collective-speedup law were billed as the follow-up's Theorems F3-F4; both are proven
in the base, relabeled as imported [1, App. B] / [1, Thm 4], leaving only the genuinely-new Proposition F3).
Verified div balance 14/14, zero dashes/base64, figures 1-4 in order; NOT committed by the agent (committed
here after review). (b) External-benchmark scouting (Major-1 gold-standard option): RecoGym (Criteo) is the
most tractable single external benchmark because its reward is a latent-factor inner product (low-rank by
construction), estimated ~4-6 person-days (its blockers: single-agent, so the multi-robot masked broadcast is
a wrapper to build, plus legacy gym==0.14.0/Py3.6 install friction); Level-Based Foraging is worst (reward is
combinatorial, not low-rank, so hosting ZK-MRTA guts its semantics); no public code exists for the
bilinear-bandit papers. Recommendation recorded: attempt RecoGym if we want the gold-standard external
benchmark, else the approximate-low-rank experiment (Cycle 97) + the honest §7 framing already substantially
answer Major-1. Word .docx not rebuilt; base paper untouched.

## Cycle 99 (drop external-benchmark promise; reframe as "no environment exists, so we built LatentSwarm")
Per user decision after the scouting report: dropped the future-work promise to validate on an external
environment. Level-based foraging is a poor fit (its reward is combinatorial, not low-rank, and its
observations are spatial, so hosting ZK-MRTA would override reward+observation+dynamics, leaving the
benchmark cosmetic); RecoGym is not a robot domain; and a bilinear bandit is essentially the single-agent
core of our own model (hidden latents), so it is not a genuinely external benchmark. Removed the
"level-based foraging / RecoGym / bilinear-bandit" external-validation clause from Section 7 future work.
Instead stated the motivation where LatentSwarm is introduced (Section 6 Setup): no existing benchmark
instantiates the ZK-MRTA regime (hidden low-rank robot-task reward seen only through a persistent,
per-observer-private, masked-and-noisy broadcast under communication-free, task-scarce decentralized
choice), so we built and openly released LatentSwarm and run every experiment on it. Reframed the Section 7
limitation accordingly (simulation-only on our own simulator because none exists; low-rank remains an
assumption; a higher-fidelity or physical instantiation is itself an open problem) and dropped the named
external benchmarks. The Related-Work citation of bilinear bandits as prior art (Section 2) is retained
(correct). HTML 88 KB; .docx not rebuilt; follow-up paper has no such promise (checked).
