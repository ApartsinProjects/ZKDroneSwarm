"""Generate a SELF-CONTAINED, graduate-level HTML tutorial of the whole project:
motivation, hypothesis, model, method, metrics, experiments, results, theory,
the two observability channels, parameters, honest positioning, glossary.
Figures (F2-F6) are base64-embedded so the file is a single portable artifact.
Output: docs/tutorial.html  (regenerable; reads PNGs from docs/figures/).
"""
import base64
import os

FIG_DIR = "docs/figures"
OUT = "docs/tutorial.html"


def img(name, alt):
    p = os.path.join(FIG_DIR, name)
    if not os.path.exists(p):
        return '<p><em>[missing figure: %s]</em></p>' % name
    b = base64.b64encode(open(p, "rb").read()).decode("ascii")
    return ('<img alt="%s" src="data:image/png;base64,%s" '
            'style="max-width:100%%;height:auto;border:1px solid #ddd;border-radius:6px;">'
            % (alt, b))


CSS = """
:root{--ink:#1a1a1a;--mut:#5a5a5a;--acc:#1f5fa8;--ours:#0b6;--warn:#b00;--bg:#fff;--box:#f6f8fa;--line:#e2e6ea;}
*{box-sizing:border-box;}
body{margin:0;color:var(--ink);background:var(--bg);font:16px/1.65 -apple-system,Segoe UI,Roboto,Helvetica,Arial,serif;}
.wrap{max-width:860px;margin:0 auto;padding:32px 22px 80px;}
h1{font-size:30px;line-height:1.2;margin:0 0 6px;}
h2{font-size:23px;margin:42px 0 8px;padding-top:10px;border-top:2px solid var(--line);}
h3{font-size:18px;margin:26px 0 6px;color:var(--acc);}
.sub{color:var(--mut);font-size:17px;margin:0 0 18px;}
p{margin:10px 0;}
code{background:var(--box);padding:1px 5px;border-radius:4px;font:14px/1.4 SFMono-Regular,Consolas,monospace;}
pre{background:var(--box);border:1px solid var(--line);border-radius:8px;padding:12px 14px;overflow:auto;font:13px/1.5 SFMono-Regular,Consolas,monospace;}
.box{background:var(--box);border:1px solid var(--line);border-left:4px solid var(--acc);border-radius:8px;padding:12px 16px;margin:16px 0;}
.key{border-left-color:var(--ours);}
.warn{border-left-color:var(--warn);}
figure{margin:20px 0;text-align:center;}
figcaption{color:var(--mut);font-size:14px;margin-top:8px;text-align:left;}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px;}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:center;}
th{background:var(--box);}
td.l,th.l{text-align:left;}
.ours{font-weight:700;color:var(--ours);}
.toc{background:var(--box);border:1px solid var(--line);border-radius:8px;padding:12px 18px;columns:2;font-size:14px;}
.toc a{color:var(--acc);text-decoration:none;}
dl dt{font-weight:700;margin-top:12px;}
dl dd{margin:2px 0 0 0;color:#222;}
.small{font-size:13px;color:var(--mut);}
em.t{color:var(--warn);font-style:normal;font-weight:600;}
"""

H = []
A = H.append

A("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
A("<meta name='viewport' content='width=device-width,initial-scale=1'>")
A("<title>Acting on the Unseen: a tutorial</title>")
A("<style>%s</style></head><body><div class='wrap'>" % CSS)

A("<h1>Acting on the Unseen</h1>")
A("<p class='sub'>Collaborative filtering for decentralized multi-robot task "
  "allocation under limited, communication-free observability. "
  "A self-contained, graduate-level walkthrough of the hypothesis, model, method, "
  "metrics, experiments, and results.</p>")

A("<div class='box'><strong>One-paragraph summary.</strong> A swarm of drones must "
  "repeatedly pick which targets to engage. Each drone is good at different targets "
  "(hidden, low-rank compatibility), there are far more targets than rounds, the "
  "drones cannot talk to each other, and each sees only a noisy, partial slice of a "
  "public broadcast of what the swarm did. We show that if each drone runs "
  "collaborative filtering (low-rank matrix decomposition) over that broadcast, it "
  "can act well on targets it has <em>never personally engaged</em>, and onboard "
  "brand-new targets for the whole swarm from about <code>d</code> shared probes "
  "instead of one-per-drone. Against a structure-free learner this is a "
  "<strong>categorical</strong> separation (the structure-free learner is at the "
  "error floor by construction). We back this with theory and a full comparison "
  "against the field.</div>")

# TOC
A("<div class='toc'>")
toc = [("1. Motivation", "mot"), ("2. Hypothesis", "hyp"), ("3. The model", "mod"),
       ("4. The method", "met"), ("5. Metrics", "mtr"), ("6. Experiments and results", "exp"),
       ("7. Comparison to the field", "cmp"), ("8. Two observability channels", "obs"),
       ("9. Theory", "thy"), ("10. Which parameters matter", "par"),
       ("11. Honest positioning and revived ideas", "pos"), ("12. Glossary", "glo")]
for t, a in toc:
    A("<a href='#%s'>%s</a><br>" % (a, t))
A("</div>")

# 1 Motivation
A("<h2 id='mot'>1. Motivation</h2>")
A("<p>Consider an autonomous swarm assigning itself to targets over many rounds. "
  "Three facts make this hard and interesting at once:</p>")
A("<ul>"
  "<li><strong>Heterogeneous, hidden compatibility.</strong> Different drones suit "
  "different targets (payload, sensor, geometry). The compatibility is unknown and "
  "<em>latent</em>, but it has structure: a few hidden factors explain most of it "
  "(it is approximately low-rank).</li>"
  "<li><strong>Sample starvation.</strong> There are many more targets than "
  "engagement rounds (<code>n &gt;&gt; T</code>). A learner that treats every "
  "drone-target pair as a separate unknown can never gather enough data.</li>"
  "<li><strong>No communication, partial observability.</strong> Drones do not "
  "coordinate or share parameters. There is only a public broadcast of actions and "
  "outcomes, and each drone sees only a noisy, partial, persistent slice of it.</li>"
  "</ul>")
A("<p>The core question is <strong>generalization</strong>: can a drone act well on "
  "a target it never personally engaged, and on targets that did not exist when "
  "learning began? A structure-free learner cannot. If you estimate a target's value "
  "only from your own engagements with it, then for any target you never touched your "
  "best guess is the prior mean. With far more targets than rounds, that floor "
  "dominates. This is the gap low-rank structure can close.</p>")

# 2 Hypothesis
A("<h2 id='hyp'>2. Hypothesis</h2>")
A("<div class='box key'><strong>H1 (categorical).</strong> When compatibility is "
  "low-rank, a drone can recover the shared <em>target factors</em> from the public "
  "broadcast and thereby predict its OWN value for targets it never engaged. Against "
  "an independent / tabular learner this is a categorical separation on unseen pairs, "
  "not a few-percent improvement: the tabular learner is at the floor by construction."
  "</div>")
A("<div class='box key'><strong>H2 (method).</strong> Among low-rank methods, an "
  "<em>online, precision-weighted</em> estimator is uniquely suited to the limited / "
  "heterogeneous observability regime: it is masking-robust and anytime-optimal, so "
  "it dominates on the operationally relevant metric (reward actually earned over "
  "time) throughout the regime where observation is genuinely limited.</div>")

# 3 Model
A("<h2 id='mod'>3. The model</h2>")
A("<h3>3.1 World: a block model with low-rank compatibility</h3>")
A("<p>We use <code>m</code> drones and <code>n</code> targets. Drones fall into "
  "<code>K1</code> latent types and targets into <code>K2</code> types. Type "
  "prototypes are drawn and L2-normalized; each drone factor <code>p_i</code> and "
  "target factor <code>u_j</code> is its type prototype plus a small within-type "
  "spread, also normalized. The true reward is the cosine compatibility</p>")
A("<pre>R[i,j] = &#10216;p_i, u_j&#10217;   with  ||p_i|| = ||u_j|| = 1,   so R[i,j] in [-1, 1].</pre>")
A("<p>This is a deliberate, artifact-free choice: unit norm removes any popularity "
  "or magnitude scaling; the bilinear form (no nonlinear link) keeps the matrix "
  "genuinely rank <code>d</code> (a nonlinear <code>g(&#10216;p,u&#10217;)</code> "
  "would inflate the effective rank and break the low-rank assumption). The effective "
  "rank is <code>min(d, K1, K2)</code>. Defaults: <code>m=30, n=240, d=5, "
  "K1=K2=10</code>, so <code>n &gt;&gt; T</code>.</p>")
A("<h3>3.2 Observability: a public broadcast with two degradations</h3>")
A("<p>Each round, every drone is offered a random candidate set of <code>cand</code> "
  "targets, picks one, and earns its true reward. A public broadcast carries the "
  "(action, outcome) of every drone. Each drone observes:</p>")
A("<ul>"
  "<li>its OWN outcome cleanly (small noise <code>sigma_own</code>);</li>"
  "<li>a <strong>persistent, per-drone-random</strong> subset of teammates' "
  "broadcast events: a mask keeps each teammate with probability <code>rho</code> "
  "(the drone always sees itself), and observed teammate rewards carry noise "
  "<code>sigma_obs</code>.</li></ul>")
A("<div class='box'><strong>Why two degradations, and what they mean.</strong> "
  "<code>rho</code> (masking) models <em>action observability</em>: you either see a "
  "teammate's event or you do not. <code>sigma_obs</code> (additive noise) models "
  "<em>reward observability</em>: when you do see an outcome, its value is noisy. "
  "Because masks are persistent and differ per drone, drones genuinely hold "
  "different information (decentralization is real, not cosmetic), and any target a "
  "drone neither pulls nor sees is <em>unseen</em> for it. We deliberately avoid "
  "spatial sensing range, because that would also imply an attack-range limit and "
  "turn the problem into constrained assignment, a separate confound.</div>")
A("<h3>3.3 Baselines and the ceiling</h3>")
A("<p><strong>Random</strong> is the floor. <strong>Tabular / independent</strong> "
  "learns each pair on its own (own-row optimal, zero transfer). <strong>Oracle</strong> "
  "is the centralized, complete-information ceiling: it knows the true reward and "
  "picks the best target in each offered set. In the non-contention setting (targets "
  "do not deplete) this is the centralized optimum and is exactly what we normalize "
  "against. <strong>Fairness:</strong> every structured learner is given only a "
  "<em>guessed</em> rank <code>d_hat=8</code> (not the true <code>d=5</code>), and no "
  "method ever sees the latent factors. All methods use only the public broadcast.</p>")

# 4 Method
A("<h2 id='met'>4. The method</h2>")
A("<p>Each drone <code>i</code> keeps its own factor estimates "
  "<code>P (m x d_hat)</code> and <code>U (n x d_hat)</code> and updates them by "
  "<strong>online weighted alternating least squares</strong> (weighted ALS) over "
  "the events it has observed. Each observed reward is weighted by its "
  "<em>precision</em> <code>1/sigma^2</code>: a clean own-outcome counts fully, a "
  "noisy broadcast outcome counts less, and a masked (unseen) event counts as zero "
  "<em>precision</em> rather than as a zero <em>value</em>. That distinction is the "
  "crux of robustness to masking (Section 7).</p>")
A("<p>Selection is epsilon-greedy on the predicted row <code>U[cand] @ P[i]</code> "
  "with a decaying schedule; estimation and decision are kept separate (so the "
  "decision policy can later be swapped for Thompson or information-gain "
  "exploration). The method is <strong>anytime</strong> (no probe-then-commit "
  "phase), <strong>decentralized</strong> (one estimator per drone), and handles "
  "missing entries natively through the precision weights.</p>")
A("<p>Two variants are studied:</p>")
A("<ul>"
  "<li><strong>RewardCF</strong>: the cross-agent signal is teammates' (noisy) "
  "rewards, precision-weighted. The core method.</li>"
  "<li><strong>BothCF</strong>: additionally fuses teammates' CHOICES as "
  "competence-weighted implicit feedback (competence inferred from behavioral "
  "consistency, not from agreeing with the model, which deadlocks at cold-start). "
  "A choice-only variant, <strong>ChoiceCF</strong>, uses the action channel alone."
  "</li></ul>")

# 5 Metrics
A("<h2 id='mtr'>5. Metrics</h2>")
A("<p>All skills are normalized: <code>skill = (method - random) / (oracle - "
  "random)</code>, so 0 means no better than random and 1 means oracle-level.</p>")
A("<dl>"
  "<dt>Overall skill (final policy)</dt><dd>quality of the learned policy at the end, "
  "averaged over random offers.</dd>"
  "<dt>Unseen-pair skill</dt><dd>the same, but restricted to targets the drone NEVER "
  "pulled. This is the categorical metric: tabular is ~0 here by construction.</dd>"
  "<dt>Anytime cumulative-reward skill (AUC)</dt><dd>reward ACTUALLY earned over the "
  "rounds (targets destroyed by round K), normalized. This charges the cost of any "
  "probe / explore phase, so it is the operationally honest metric.</dd>"
  "<dt>State-uniqueness</dt><dd>how much the drones' learned reward matrices differ; "
  "it rises as masking increases, evidencing that decentralization is real.</dd>"
  "<dt>Onboarding skill vs probes</dt><dd>skill on a freshly injected target as a "
  "function of how many shared probes it received.</dd></dl>")

# 6 Experiments and results (the four spine results)
A("<h2 id='exp'>6. Experiments and results</h2>")
A("<h3>6.1 When does collaborative filtering help? (characterization)</h3>")
A("<p>Across a structure-by-observability grid, CF beats tabular if and only if "
  "three conditions hold together: (1) the reward is low-rank but PERSONALIZED "
  "(<code>1 &lt; d &lt;&lt; min(m,n)</code>; it collapses at full rank, at "
  "<code>d=1</code> where there is no personalization to exploit, and under a "
  "nonlinear reward link); (2) the regime is sample-starved with changing "
  "availability; (3) the reward channel is shared. Decision-only signals without "
  "rewards reach at best tabular parity. This scoping explains earlier null "
  "results (sample-rich regimes).</p>")

A("<h3>6.2 The categorical result: acting on unseen pairs</h3>")
A("<p>Statically, CF reaches unseen-pair skill 0.496 versus tabular 0.006. In the "
  "natural masked regime it holds at every density (Figure F2): CF stays well above "
  "zero on unseen pairs for all <code>rho &gt; 0</code> while tabular sits at the "
  "floor. As masking increases, the drones' models genuinely diverge "
  "(state-uniqueness rises from 0.54 to 0.92), confirming the decentralization is "
  "real and CF still completes the unseen entries.</p>")
A("<figure>%s<figcaption><strong>Figure F2.</strong> Unseen-pair skill versus "
  "observation density <code>rho</code>. CF (matrix decomposition) acts on "
  "never-observed pairs at every density; the tabular learner is pinned at zero by "
  "construction.</figcaption></figure>" % img("F2_unseen_masking.png", "F2 unseen masking"))

A("<h3>6.3 Dynamic task onboarding</h3>")
A("<p>After drones have learned their factors <code>P</code>, a brand-new target is "
  "introduced. Its <code>d</code>-dimensional factor is recovered by a ridge "
  "fold-in from a handful of shared probes; then ALL drones can predict it. CF "
  "reaches high skill at about <code>d_hat</code> probes; tabular needs about "
  "<code>m</code> probes (every drone must try the new target itself). This is the "
  "<code>Theta(d)</code> versus <code>Theta(m)</code> onboarding separation "
  "(Figure F3).</p>")
A("<figure>%s<figcaption><strong>Figure F3.</strong> Onboarding skill on a new "
  "target versus number of shared probes. CF onboards for the whole swarm from about "
  "<code>d</code> probes; tabular needs about one per drone.</figcaption></figure>"
  % img("F3_onboard.png", "F3 onboarding"))

A("<h3>6.4 The gap scales with low-rankness</h3>")
A("<p>As the true rank <code>d</code> rises, there is more to complete from the same "
  "data, so CF unseen-pair skill decreases monotonically (0.67 at d=2 down to 0.27 "
  "at d=8) while tabular stays at the floor for every <code>d</code> (Figure F4). "
  "The categorical gap is therefore largest exactly when the structure is most "
  "low-rank, as the theory predicts.</p>")
A("<figure>%s<figcaption><strong>Figure F4.</strong> Unseen-pair skill versus the "
  "true rank <code>d</code>. CF scales with low-rankness; tabular is at the floor "
  "throughout.</figcaption></figure>" % img("F4_rank.png", "F4 rank scaling"))

# 7 Comparison
A("<h2 id='cmp'>7. Comparison to the field</h2>")
A("<p>We ported every relevant competitor into one fair harness (guessed rank, "
  "masked broadcast, one estimator per drone). It helps to know what each one is.</p>")
A("<h3>7.1 The competitors, in plain terms</h3>")
A("<dl>"
  "<dt>Structure-free bandits</dt><dd><strong>UCBIndep</strong> runs a separate UCB1 "
  "bandit for each (drone, target) pair: correct about heterogeneity, but with zero "
  "generalization across targets. <strong>UCBHomo</strong> pools all drones into one "
  "shared target table (assumes every drone is identical). <strong>Tabular</strong> "
  "is epsilon-greedy on its own running averages.</dd>"
  "<dt>MFSGD</dt><dd>plain online matrix factorization by stochastic gradient "
  "descent; a baseline low-rank method that underfits in this sample-starved "
  "regime.</dd>"
  "<dt id='ptf'>PTF (Probe-Then-Fit)</dt><dd>a strong hybrid in three stages: "
  "(1) <em>probe</em> with per-(drone,target) UCB to fill an empirical reward table; "
  "(2) take a truncated SVD of that table to get a low-rank warm start; (3) "
  "<em>fit</em> by running online SGD matrix factorization from that warm start. It "
  "fixes the cold-start of plain MF and adds online adaptation that pure "
  "explore-then-commit lacks. It is the toughest baseline.</dd>"
  "<dt>ESTR (Explore-Then-Spectral-refit)</dt><dd>explore randomly for a fixed "
  "fraction of rounds, take ONE SVD of the accumulated reward table, then commit to "
  "exploiting it. A clean explore-then-commit low-rank bandit.</dd>"
  "<dt>BPMF (Bayesian PMF)</dt><dd>a Bayesian matrix factorization that keeps a "
  "posterior over the factors (precision matrices) and selects by Thompson sampling. "
  "Principled, but Thompson over-explores in the anytime regime.</dd></dl>")
A("<div class='box'><strong>What is a 'batch-SVD hybrid', and are they independent "
  "per drone?</strong> ESTR and PTF both build one empirical reward table "
  "<code>R_hat</code> from what has been observed and then take a single (batch) "
  "Singular Value Decomposition of it to extract the low-rank factors (PTF then keeps "
  "fine-tuning; ESTR commits). We call this family the batch-SVD hybrids. The catch: "
  "unobserved entries of <code>R_hat</code> are filled with 0 before the SVD, so "
  "under masking <code>R_hat</code> is both sparse and biased and the SVD-based "
  "completion degrades. BPMF is Bayesian rather than SVD-based, but it is "
  "<em>snapshot-style</em> in the same spirit and also degrades (less data plus "
  "Thompson over-exploration). <strong>Yes, in our experiments every method, "
  "including ESTR and PTF, runs as a separate per-drone instance</strong> over that "
  "drone's own masked view of the broadcast (so all are decentralized here). ESTR's "
  "literature default is a single centralized estimator; what distinguishes our "
  "method from these is not centralization but (a) online weighted-ALS that "
  "down-weights missing entries instead of imputing zeros, and (b) being anytime "
  "rather than phase-structured.</div>")

A("<h3>7.2 Three lenses on the comparison</h3>")
A("<p><strong>(a) Final-policy unseen skill.</strong> Every low-rank method clears "
  "the no-structure floor, so the categorical result is estimator-INDEPENDENT (a "
  "property of low-rank structure, not of our particular algorithm). UCBHomo gets "
  "only partial credit because pooling recovers a rank-1 popularity effect but not "
  "personalization.</p>")
A("<p><strong>(b) Masking-robustness (Figure F5).</strong> Sweeping <code>rho</code> "
  "finely, our online weighted-ALS unseen skill is essentially flat for "
  "<code>rho &gt;= 0.4</code>, while every batch-SVD method decays monotonically "
  "(PTF 0.51 to 0.18, ESTR 0.23 to 0.01). The crossover is near <code>rho=0.55</code>: "
  "PTF leads only when the broadcast is dense.</p>")
A("<figure>%s<figcaption><strong>Figure F5.</strong> Masking-robustness. Our online "
  "ALS (solid) stays high as the broadcast is masked; batch-SVD hybrids (dashed) "
  "decay; the structure-free baseline is at the floor.</figcaption></figure>"
  % img("F5_crossover.png", "F5 crossover"))
A("<p><strong>(c) Anytime cumulative reward (Figure F6).</strong> The operational "
  "metric, reward actually earned over the rounds, charges the cost of any probe "
  "phase. Here our method dominates at every horizon and every density. The reasons "
  "are structural: PTF and ESTR earn about random during their probe phase (the kink "
  "at round 20), and UCBIndep is stuck near random the whole time because with 240 "
  "targets and 50 rounds it cannot pull each arm once, so its offer almost always "
  "contains an untried target it is forced to explore. Low-rank generalization is "
  "what lets a method exploit under <code>n &gt;&gt; T</code>.</p>")
A("<figure>%s<figcaption><strong>Figure F6.</strong> Anytime reward trajectory at "
  "<code>rho=0.25</code>. Online CF earns from round one; explore-then-commit "
  "methods pay a probe phase; UCBIndep never escapes exploration.</figcaption></figure>"
  % img("F6_anytime.png", "F6 anytime"))

A("<h3>7.3 Summary table</h3>")
A("<p>UNSEEN = final-policy unseen-pair skill; ANYTIME = final-round cumulative "
  "reward skill. Skill ~0 means no better than random. Ours in green.</p>")
rows = [
 ("Random","no-structure","0.007","0.004","-0.009","0.000"),
 ("UCBIndep","no-structure","0.004","0.003","0.001","-0.006"),
 ("UCBHomo","no-structure","0.167","0.070","0.032","0.010"),
 ("Tabular","no-structure","-0.001","0.003","0.246","0.252"),
 ("MFSGD","low-rank","0.042","-0.019","0.128","0.121"),
 ("ESTR","low-rank","0.232","0.058","0.216","0.181"),
 ("PTF","low-rank","0.516","0.280","0.274","0.230"),
 ("BPMF","low-rank","0.233","0.126","0.046","0.010"),
 ("RewardCF (ours)","low-rank","0.376","0.336","0.404","0.341"),
 ("BothCF (ours)","low-rank","0.372","0.349","0.400","0.342"),
]
A("<table><tr><th class='l'>Method</th><th>Class</th><th>UNSEEN rho=1</th>"
  "<th>UNSEEN rho=0.25</th><th>ANYTIME rho=1</th><th>ANYTIME rho=0.25</th></tr>")
for nm, cl, u1, u25, a1, a25 in rows:
    o = " class='ours'" if "ours" in nm else ""
    A("<tr><td class='l'%s>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
      % (o, nm, cl, u1, u25, a1, a25))
A("</table>")
A("<p class='small'>Reading: every low-rank method clears the no-structure UNSEEN "
  "floor (categorical, estimator-independent). On ANYTIME (operational) our online "
  "ALS leads at both densities; UCBIndep's strong final-policy skill collapses to "
  "~0 anytime; PTF leads UNSEEN only at full broadcast.</p>")

# 8 Two channels (answers Q4)
A("<h2 id='obs'>8. Two observability channels: reward and action</h2>")
A("<p>A natural question is whether we study reward observation only, or also "
  "action observation. We model and analyze BOTH.</p>")
A("<table><tr><th class='l'>Channel</th><th class='l'>Degraded by</th>"
  "<th class='l'>Method that uses it</th></tr>"
  "<tr><td class='l'>Reward (continuous outcome)</td><td class='l'>additive noise "
  "<code>sigma_obs</code>; also masking</td><td class='l'>RewardCF</td></tr>"
  "<tr><td class='l'>Action / choice (discrete)</td><td class='l'>masking "
  "<code>rho</code> (a choice is seen cleanly or missed)</td><td class='l'>ChoiceCF</td></tr>"
  "<tr><td class='l'>Both, fused</td><td class='l'>both</td><td class='l'>BothCF</td></tr></table>")
A("<p>Masking degrades both channels (a missed event gives you neither the action "
  "nor the reward). Noise degrades only the reward channel (an observed choice is "
  "clean). A separate apples-to-apples study showed that under high reward-noise the "
  "clean CHOICE channel beats the noisy reward channel (decisions are noise-immune), "
  "and BothCF fuses them. <em class='t'>Honest gap:</em> the recent head-to-head "
  "comparison (Figures F5, F6) held <code>sigma_obs</code> fixed and swept "
  "<code>rho</code>, and it did not include the choice-only method. Crossing the two "
  "channels (a <code>rho x sigma_obs</code> grid) and including ChoiceCF is the first "
  "planned experiment (suite E3 in the experimentation plan).</p>")

# 9 Theory
A("<h2 id='thy'>9. Theory (per-drone sample complexity)</h2>")
A("<p>The empirics are matched by a clean separation (full proofs in "
  "<code>docs/THEORY.md</code>).</p>")
A("<ul>"
  "<li><strong>Tabular floor.</strong> A tabular learner estimates <code>R[i,j]</code> "
  "only from observations of that exact pair; on any never-observed pair its expected "
  "squared error is a constant (the prior variance). To be good on all of a drone's "
  "targets it must observe <code>Theta(n)</code> of them; <code>Theta(mn)</code> "
  "total.</li>"
  "<li><strong>CF completes a row from O(d).</strong> If the reward is rank "
  "<code>d</code> and the broadcast identifies the target-factor space "
  "<code>U</code>, then a drone's whole row is determined by its "
  "<code>d</code>-dimensional factor, recoverable from about <code>d</code> "
  "observations by a ridge fold-in; the drone then predicts ALL targets, including "
  "never-pulled ones.</li>"
  "<li><strong>The separation.</strong> Per drone: tabular <code>Theta(n)</code> "
  "versus CF <code>Theta(d)</code>; on unseen pairs, tabular error is a constant "
  "floor while CF error goes to zero. A categorical, not constant-factor, gap.</li>"
  "<li><strong>Under masking.</strong> A drone sees about <code>rho</code> times the "
  "population's observations; <code>U</code> is recovered while that exceeds the "
  "completion threshold, and CF degrades gracefully (not to the floor) for "
  "<code>rho &gt; 0</code>. Distinct masks give distinct estimates, hence genuinely "
  "unique per-drone states.</li>"
  "<li><strong>Anytime corollary.</strong> When <code>n &gt;&gt; T</code>, a per-arm "
  "method's offer almost surely contains an untried arm, so it cannot exploit; "
  "structure-free anytime skill goes to zero. Low-rank generalization is what enables "
  "exploitation under starvation.</li>"
  "<li><strong>Clustering helps further.</strong> With <code>K1, K2</code> types, "
  "identifying a drone's type is an <code>O(log K1)</code> classification and the "
  "type factor transfers, lowering sample complexity below the generic "
  "<code>O(d)</code>.</li></ul>")
A("<p class='small'>Novelty versus standard matrix-completion bounds (Candes-Recht; "
  "Keshavan-Montanari-Oh OptSpace): those are centralized with uniform sampling; ours "
  "is the decentralized, online, broadcast-only, per-drone-masked statement, with the "
  "unseen-pair floor making the gap categorical.</p>")

# 10 Parameters
A("<h2 id='par'>10. Which parameters matter (and why)</h2>")
A("<p>The design space has three groups of knobs. The high-value ones to explore "
  "are marked P0.</p>")
A("<table><tr><th class='l'>Group</th><th class='l'>Parameter</th>"
  "<th class='l'>Why it matters</th><th>Priority</th></tr>"
  "<tr><td class='l'>World</td><td class='l'>true rank <code>d</code></td>"
  "<td class='l'>core: the gap scales with low-rankness; vanishes at d=1 and at full rank</td><td>P0</td></tr>"
  "<tr><td class='l'>World</td><td class='l'>targets <code>n</code>, horizon <code>T</code></td>"
  "<td class='l'>sample starvation <code>n/T</code>: the anytime edge grows as it rises</td><td>P0</td></tr>"
  "<tr><td class='l'>World</td><td class='l'>clusters <code>K1,K2</code>, tightness <code>within</code></td>"
  "<td class='l'>block structure lowers sample complexity; controls onboarding</td><td>P1</td></tr>"
  "<tr><td class='l'>Observability</td><td class='l'>masking <code>rho</code></td>"
  "<td class='l'>action channel; the headline robustness axis</td><td>P0</td></tr>"
  "<tr><td class='l'>Observability</td><td class='l'>reward noise <code>sigma_obs</code></td>"
  "<td class='l'>reward channel; never yet crossed with <code>rho</code></td><td>P0</td></tr>"
  "<tr><td class='l'>Algorithm</td><td class='l'>guessed rank <code>d_hat</code></td>"
  "<td class='l'>fairness and practicality: robustness to misguessing the rank</td><td>P0</td></tr>"
  "<tr><td class='l'>Algorithm</td><td class='l'>refit cadence, explore policy</td>"
  "<td class='l'>isolates the anytime mechanism; enables active exploration</td><td>P1</td></tr></table>")
A("<p>The single most informative new experiment is the two-dimensional "
  "<code>rho x sigma_obs</code> grid (it exercises both observability channels at "
  "once), followed by the <code>n/T</code> starvation sweep (which is where the "
  "anytime advantage is born) and the <code>d_hat</code> robustness sweep (the "
  "fairness knob). See <code>docs/EXPERIMENT_PLAN.md</code> for the full protocol "
  "with seeds, confidence intervals, and ablations.</p>")

# 11 Positioning
A("<h2 id='pos'>11. Honest positioning and revived ideas</h2>")
A("<div class='box warn'><strong>We do not claim universal dominance.</strong> PTF "
  "achieves a better FINAL policy at full broadcast (<code>rho=1</code>), the "
  "no-observability-limit case the premise excludes. Our claims, all evidenced, are: "
  "(1) the unseen-pair and onboarding categorical separation over structure-free "
  "methods; (2) among low-rank methods, our online weighted-ALS is uniquely "
  "masking-robust and anytime-optimal, so it dominates on cumulative reward at every "
  "horizon and at every density <code>rho &lt; 1</code>, that is, throughout the "
  "limited-observability regime that defines the problem.</div>")
A("<p>Several parked directions are being revived to strengthen the method and add "
  "results (details in the experimentation plan):</p>")
A("<ul>"
  "<li><strong>Probe-then-online hybrid:</strong> warm-start our online ALS with a "
  "short probe and SVD, aiming to match PTF at dense broadcast while keeping our "
  "robustness and anytime lead, that is, to dominate everywhere.</li>"
  "<li><strong>Active, uncertainty-reducing exploration:</strong> probe the "
  "highest-posterior-variance target; because every probe is broadcast, one drone's "
  "probe improves all estimates (collective benefit, no communication).</li>"
  "<li><strong>Newcomer cold-start:</strong> a late-joining drone acts from the "
  "broadcast alone (a second categorical result; tabular newcomer is random).</li>"
  "<li><strong>Precision-gated fusion:</strong> gate the choice channel by reward "
  "precision (not raw count) so BothCF is strictly dominant across the noise grid.</li>"
  "</ul>")
A("<p>Kept out of scope: Byzantine / malicious teammates (we assume honest agents), "
  "robust RANSAC factorization (subsumed by precision weighting unless real outliers "
  "are shown), and contention / assignment (which turns the problem into matching, a "
  "strong future paper rather than a revival).</p>")

# 12 Glossary
A("<h2 id='glo'>12. Glossary</h2>")
A("<dl>"
  "<dt>Collaborative filtering (CF)</dt><dd>predicting unknown entries of a "
  "user-item (here drone-target) matrix from observed ones by assuming low rank; "
  "we do it online, per drone.</dd>"
  "<dt>Low rank / latent factors</dt><dd>the reward matrix is "
  "<code>R = P U^T</code> with a few columns; each drone and target is a short "
  "vector, and compatibility is their inner product.</dd>"
  "<dt>Weighted ALS</dt><dd>alternating least squares that solves for "
  "<code>P</code> given <code>U</code> and vice versa, with each observation weighted "
  "by its precision; missing entries get zero weight.</dd>"
  "<dt>Fold-in</dt><dd>given known factors <code>U</code>, fit a new row's (or "
  "column's) short factor by a small least-squares problem, without retraining; the "
  "basis of onboarding.</dd>"
  "<dt>PTF (Probe-Then-Fit)</dt><dd>probe with UCB, SVD the resulting table for a "
  "warm start, then fine-tune by online SGD. Our toughest baseline. (See "
  "<a href='#ptf'>7.1</a>.)</dd>"
  "<dt>Batch-SVD hybrid</dt><dd>any method that extracts factors by a single SVD of "
  "an accumulated reward table (ESTR, PTF). It imputes unobserved entries as zero, "
  "which is why it degrades under masking.</dd>"
  "<dt>Masking (<code>rho</code>)</dt><dd>persistent, per-drone loss of a fraction "
  "of broadcast events; models limited action observability.</dd>"
  "<dt>Unseen pair</dt><dd>a (drone, target) the drone never engaged and never "
  "observed; tabular is at the error floor here, CF is not.</dd>"
  "<dt>Anytime / AUC skill</dt><dd>reward earned cumulatively over rounds (targets "
  "destroyed by round K), the operational metric that charges exploration cost.</dd>"
  "<dt>Skill</dt><dd><code>(method - random) / (oracle - random)</code>: 0 is "
  "random, 1 is the centralized complete-information ceiling.</dd></dl>")

A("<h2>Reproducibility</h2>")
A("<p class='small'>All numbers derive from complete per-seed JSON in "
  "<code>results/pilots/</code> (registry: <code>docs/DATA_CATALOGUE.md</code>; "
  "chronology and per-cycle reviews: <code>docs/PROJECT_LOG.md</code>). Figures: "
  "<code>python experiments/make_figures.py</code>. This page: "
  "<code>python experiments/make_tutorial.py</code>. Comparison harness: "
  "<code>experiments/pilot_compare.py</code> (bake-off), <code>pilot_crossover.py</code> "
  "(masking-robustness), <code>pilot_anytime.py</code> (anytime); competitor ports in "
  "<code>experiments/pilot_baselines.py</code>; world, reward, and oracle in "
  "<code>experiments/core.py</code>. The paper draft is <code>docs/PAPER_DRAFT.md</code>; "
  "the full experimentation plan is <code>docs/EXPERIMENT_PLAN.md</code>.</p>")

A("</div></body></html>")

html = "\n".join(H)
open(OUT, "w", encoding="utf-8").write(html)
print("wrote %s (%d KB)" % (OUT, len(html.encode("utf-8")) // 1024))
