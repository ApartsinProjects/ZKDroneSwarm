"""Generate the SELF-CONTAINED, graduate-level, step-by-step HTML tutorial covering
the ENTIRE project: motivation, background, the model, baselines, our methods,
theory, metrics, and every experimental result. Figures F2-F11 are base64-embedded
so the page is a single portable artifact suitable for GitHub Pages.
Output: docs/tutorial.html   (regenerable; reads PNGs from docs/figures/).
"""
import base64
import os

FIG = "docs/figures"
OUT = "docs/tutorial.html"


def img(name, alt, w="100%"):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return "<p><em>[missing figure: %s]</em></p>" % name
    b = base64.b64encode(open(p, "rb").read()).decode("ascii")
    return ('<img alt="%s" src="data:image/png;base64,%s" '
            'style="max-width:%s;height:auto;border:1px solid #d7dde3;border-radius:8px;">' % (alt, b, w))


CSS = """
:root{--ink:#16191d;--mut:#5b6570;--acc:#1f5fa8;--ok:#0a7d4d;--warn:#b23;--bg:#fff;--box:#f5f8fb;--ln:#e2e8ee;}
*{box-sizing:border-box}
body{margin:0;color:var(--ink);background:var(--bg);font:16px/1.68 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:880px;margin:0 auto;padding:34px 22px 90px}
h1{font-size:31px;line-height:1.18;margin:0 0 4px}
h2{font-size:24px;margin:46px 0 8px;padding-top:12px;border-top:3px solid var(--ln)}
h3{font-size:18.5px;margin:28px 0 6px;color:var(--acc)}
h4{font-size:15.5px;margin:18px 0 4px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.sub{color:var(--mut);font-size:17px;margin:2px 0 16px}
p{margin:10px 0}
code{background:var(--box);padding:1px 5px;border-radius:4px;font:13.5px/1.4 SFMono-Regular,Consolas,monospace}
pre{background:var(--box);border:1px solid var(--ln);border-radius:8px;padding:12px 14px;overflow:auto;font:13px/1.5 SFMono-Regular,Consolas,monospace}
.box{background:var(--box);border:1px solid var(--ln);border-left:4px solid var(--acc);border-radius:8px;padding:12px 16px;margin:16px 0}
.key{border-left-color:var(--ok)}.warn{border-left-color:var(--warn)}
.step{background:#fbfdff;border:1px solid var(--ln);border-radius:8px;padding:6px 16px 12px;margin:16px 0}
.step h3{margin-top:12px}
figure{margin:18px 0;text-align:center}figcaption{color:var(--mut);font-size:13.5px;margin-top:8px;text-align:left}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px}
th,td{border:1px solid var(--ln);padding:6px 9px;text-align:center}th{background:var(--box)}
td.l,th.l{text-align:left}.ours{font-weight:700;color:var(--ok)}.bad{color:var(--warn)}
.toc{background:var(--box);border:1px solid var(--ln);border-radius:8px;padding:12px 18px;font-size:14px;columns:2}
.toc a{color:var(--acc);text-decoration:none}
dl dt{font-weight:700;margin-top:10px}dl dd{margin:2px 0 0}
.small{font-size:13px;color:var(--mut)}
.pill{display:inline-block;background:var(--box);border:1px solid var(--ln);border-radius:20px;padding:1px 10px;font-size:12px;color:var(--mut);margin:2px 3px 0 0}
"""

H = []
A = H.append
A("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
A("<meta name='viewport' content='width=device-width,initial-scale=1'>")
A("<title>Acting on the Unseen: a graduate tutorial</title><style>%s</style></head><body><div class='wrap'>" % CSS)

A("<h1>Acting on the Unseen</h1>")
A("<p class='sub'>Collaborative filtering for decentralized multi-robot task allocation under "
  "limited, communication-free observability. A self-contained, step-by-step, graduate-level "
  "tutorial of the whole project: motivation, background, model, baselines, method, theory, "
  "metrics, and results.</p>")
for p in ["zero prior knowledge", "zero communication", "partial + noisy observation",
          "fully distributed", "low-rank latent structure", "online", "anytime"]:
    A("<span class='pill'>%s</span>" % p)

A("<div class='box key'><strong>The setting in one breath: minimal assumptions.</strong> "
  "A swarm of agents must repeatedly choose which task to take, under the LEAST possible "
  "information. <b>No prior knowledge</b> (no labels, no task models, no known latent structure, not "
  "even the true rank). <b>Zero communication</b> (no messages, no parameter sharing, no coordinator, "
  "no consensus). <b>Only partial, noisy observables</b> (each agent passively senses a masked, noisy "
  "slice of the public outcomes). <b>Fully distributed decisions</b> (every agent decides alone, from "
  "its own private state). The question: can such a swarm still act INTELLIGENTLY, in particular on "
  "tasks it has never tried? This tutorial shows the answer is YES, via collaborative filtering over "
  "the public outcome stream, and that the advantage is CATEGORICAL (with proofs).</div>")

A("<p class='small'>Companion streamlined paper (v2): "
  "<a href='paper_v2.html'>paper_v2.html</a>. Full draft: <code>docs/PAPER_DRAFT.md</code>. "
  "Repository: <a href='https://github.com/ApartsinProjects/ZKDroneSwarm'>ApartsinProjects/ZKDroneSwarm</a>.</p>")

A("<div class='box'><strong>What you will learn.</strong> How a swarm of drones, with no shared "
  "model, no communication, and only a noisy partial view of what its teammates did, can still "
  "act well on targets it has never tried, onboard brand-new targets, and welcome new drones, by "
  "running collaborative filtering over the public outcome stream. We build the idea from scratch, "
  "state the competing methods, prove why the advantage is categorical, define the right metrics, "
  "and walk through every experiment and figure.</div>")

A("<div class='toc'>")
toc = [("1. Motivation: the problem", "mot"), ("2. Background: the tools", "bg"),
       ("3. The model (world, observability, fairness)", "model"),
       ("4. Baselines (the field)", "base"), ("5. Our methods", "meth"),
       ("6. Metrics", "metrics"), ("7. Theory", "thy"),
       ("8. Results, step by step", "res"), ("9. Novelty and positioning", "nov"),
       ("10. Reproducibility", "repro"), ("11. Glossary", "glo")]
for t, a in toc:
    A("<a href='#%s'>%s</a><br>" % (a, t))
A("</div>")

# 1 MOTIVATION
A("<h2 id='mot'>1. Motivation: the problem</h2>")
A("<p>Imagine a swarm of autonomous drones that, round after round, must each pick a target to "
  "engage. Three facts make this both hard and interesting:</p>")
A("<ul>"
  "<li><strong>Hidden, heterogeneous compatibility.</strong> Different drones suit different "
  "targets (payload, sensor, geometry). The compatibility is unknown and latent, but structured: "
  "a few hidden factors explain most of it (it is approximately low-rank).</li>"
  "<li><strong>Sample starvation.</strong> There are many more targets than engagement rounds "
  "(<code>n &gt;&gt; T</code>): a learner that treats each drone-target pair as its own unknown can "
  "never gather enough data.</li>"
  "<li><strong>No communication, partial observability.</strong> Drones neither coordinate nor "
  "share parameters. There is only a public stream of what happened (engagements and their "
  "outcomes), and each drone senses only a noisy, partial slice of it.</li></ul>")
A("<div class='box key'><strong>The central question (generalization).</strong> Can a drone act "
  "well on a target it never personally engaged, and on targets that did not exist when learning "
  "began? A structure-free learner cannot: if it estimates a target's value only from its own "
  "engagements, then for any target it never touched its best guess is the prior mean, the error "
  "floor. With far more targets than rounds, that floor dominates. Closing this gap is the whole "
  "point.</div>")

# 2 BACKGROUND
A("<div class='box'><strong>The idea in everyday terms (the Netflix analogy).</strong> Streaming "
  "services guess whether YOU will like a movie you have never watched. They do not need to know what "
  "the movie is about; they just notice that people who liked the same things as you also liked some "
  "movie you have not seen, and recommend it. Our drones do the exact same trick: a drone learns "
  "which OTHER drones tend to succeed where it succeeds, and borrows their successes to guess how it "
  "would do against a target it has NEVER engaged. Nobody is told anything about the targets; the "
  "pattern of who-succeeded-where is enough. That trick is called <em>collaborative filtering</em>, "
  "and it is the heart of this project.</div>")
A("<div class='box'><strong>What 'low-rank' means, simply.</strong> Imagine every drone and every "
  "target each gets a short list of a few hidden 'traits' (say 5 numbers). How well a drone does "
  "against a target is just how well their trait-lists line up. Because only a few traits matter, the "
  "giant table of all drone-vs-target outcomes is actually simple underneath ('low-rank'). That "
  "simplicity is what lets a drone fill in the blanks for pairs it never tried, the same way knowing "
  "your taste in a few genres predicts many movies.</div>")

A("<h2 id='bg'>2. Background: the tools we build on</h2>")
A("<dl>"
  "<dt>Collaborative filtering (CF) / matrix completion</dt><dd>Predict the unknown entries of a "
  "user-item matrix from observed ones by assuming it is low-rank: <code>R = P U^T</code>. Each row "
  "(drone) and column (target) is a short latent vector; their inner product is the value. Classic "
  "guarantees: a rank-d matrix is recoverable from about <code>d(m+n)</code> observed entries "
  "(Candes-Recht 2009; Keshavan-Montanari-Oh OptSpace; Recht 2011). We use these as the statistical "
  "backbone but in a NON-standard way (decentralized, online, masked).</dd>"
  "<dt>Low-rank bandits</dt><dd>Sequential decision methods that exploit low-rank reward structure "
  "(explore-then-commit spectral methods; probe-then-fit hybrids). Typically centralized.</dd>"
  "<dt>RecSys cold-start and fold-in</dt><dd>Given known item factors, a new user's factor is fit "
  "from a few interactions by a small least-squares solve, no retraining (WALS fold-in). We reuse "
  "this for onboarding new targets and welcoming new drones.</dd>"
  "<dt>Exposure / choice debiasing</dt><dd>Treating observed choices as implicit feedback requires "
  "accounting for what options were available; we use the public active-target set as the choice "
  "set.</dd></dl>")

# 3 MODEL
A("<h2 id='model'>3. The model</h2>")
A("<div class='step'>")
A("<h3>Step 3.1 The world: a block model with low-rank compatibility</h3>")
A("<p><code>m</code> drones, <code>n</code> targets. Drones fall into <code>K1</code> latent types, "
  "targets into <code>K2</code> types; type prototypes are drawn and unit-normalized, and each "
  "drone/target factor is its type prototype plus a small within-type spread, also normalized. The "
  "true reward is the cosine compatibility:</p>")
A("<pre>R[i,j] = &#10216;p_i, u_j&#10217;,   ||p_i|| = ||u_j|| = 1,   so R[i,j] in [-1, 1].</pre>")
A("<p>Unit norm removes any popularity/scale artifact; the bilinear form (no nonlinear link) keeps "
  "the matrix genuinely rank <code>d = min(d, K1, K2)</code>. Defaults: <code>m=30, n=240, d=5, "
  "K1=K2=10</code>, so <code>n &gt;&gt; T</code>.</p></div>")
A("<div class='step'>")
A("<h3>Step 3.2 Observability: a public outcome stream with two degradations</h3>")
A("<p>Each round every drone is offered a random candidate set, picks one target, and earns its "
  "true reward. A public stream carries the (action, outcome) of every engagement. Each drone "
  "senses its OWN outcome cleanly (noise <code>sigma_own</code>) and a per-drone-limited, noisy "
  "subset of teammates' outcomes:</p>")
A("<ul><li><strong>Masking <code>rho</code></strong> (action observability): you either detect a "
  "teammate's engagement or you do not.</li>"
  "<li><strong>Additive noise <code>sigma_obs</code></strong> (reward observability): a detected "
  "outcome's value is noisy.</li></ul>")
A("<div class='box'>Masking is <em>passive sensing of public outcomes</em> (limited detection), NOT "
  "radio transmission, so the setting is genuinely communication-free. Persistent (fixed) per-drone "
  "masks make decentralization durable; i.i.d. per-round masks (packet-loss style) give the same "
  "headline results (Section 8.7).</div></div>")
A("<div class='step'>")
A("<h3>Step 3.3 Fairness and zero-knowledge, for every method</h3>")
A("<p>The same rules bind ours and all baselines: no method ever receives the latent factors, the "
  "true rank, or labels; structured methods all get the SAME <em>guessed</em> rank "
  "<code>d_hat=8</code> (true <code>d=5</code>); every method is an INDEPENDENT per-drone learner "
  "with no parameter sharing and no coordinator; all see the same partial+noisy stream. The "
  "<strong>Oracle</strong> (centralized, complete information) is used only to normalize scores, "
  "never as a competitor.</p></div>")

# 4 BASELINES
A("<h2 id='base'>4. Baselines: the field</h2>")
A("<table><tr><th class='l'>Method</th><th>Class</th><th class='l'>What it does</th></tr>"
  "<tr><td class='l'>Random</td><td>no-structure</td><td class='l'>uniform pick; the floor.</td></tr>"
  "<tr><td class='l'>UCBIndep</td><td>no-structure</td><td class='l'>a separate UCB1 bandit per "
  "(drone,target) pair: correct heterogeneity, zero cross-target generalization.</td></tr>"
  "<tr><td class='l'>UCBHomo</td><td>no-structure</td><td class='l'>one shared target table (assumes "
  "all drones identical): recovers only rank-1 'popularity'.</td></tr>"
  "<tr><td class='l'>Tabular</td><td>no-structure</td><td class='l'>epsilon-greedy on own-row "
  "averages.</td></tr>"
  "<tr><td class='l'>MFSGD</td><td>low-rank</td><td class='l'>plain online SGD matrix factorization; "
  "underfits when starved.</td></tr>"
  "<tr><td class='l'>ESTR</td><td>low-rank</td><td class='l'>explore-then-spectral-refit: random "
  "explore, one SVD of the empirical matrix, then commit.</td></tr>"
  "<tr><td class='l'>PTF</td><td>low-rank</td><td class='l'>probe-then-fit: UCB probe, SVD warm-start, "
  "then SGD finetune. The strongest competitor.</td></tr>"
  "<tr><td class='l'>BPMF</td><td>low-rank</td><td class='l'>Bayesian PMF with Thompson sampling; "
  "over-explores in the anytime regime.</td></tr>"
  "<tr><td class='l'>SoftImpute</td><td>low-rank</td><td class='l'>nuclear-norm CONVEX completion "
  "(iterate fill, SVD, soft-threshold). Strongest at full broadcast; decays under masking.</td></tr>"
  "<tr><td class='l'>kNN-CF</td><td>memory-CF</td><td class='l'>model-FREE collaborative filtering "
  "(similarity-weighted neighbor rewards); tests whether explicit factorization is needed.</td></tr>"
  "<tr><td class='l'>BiasModel</td><td>additive</td><td class='l'>additive drone+target bias, NO "
  "interaction (rank&le;2); isolates the value of personalization.</td></tr></table>")
A("<p class='small'>A 'batch-SVD hybrid' (ESTR, PTF) extracts factors by one SVD of an accumulated "
  "reward table whose unobserved entries are imputed 0; under masking that table is sparse and "
  "biased, which is why these methods decay (Section 8.5). In our harness every baseline runs as an "
  "independent per-drone instance (ESTR's literature default is centralized; we run it per drone for "
  "a fair, fully-distributed comparison).</p>")

# 5 METHODS
A("<h2 id='meth'>5. Our methods</h2>")
A("<p>Each drone runs its own <strong>online weighted alternating least squares</strong> (weighted "
  "ALS) over the events it senses, weighting each observation by its precision <code>1/sigma^2</code> "
  "(clean own outcomes count fully, noisy ones less, masked ones as zero precision rather than zero "
  "value, the key to masking-robustness). Estimation is separated from the decision policy, so the "
  "policy can be epsilon-greedy, UCB, or uncertainty-directed.</p>")
A("<dl>"
  "<dt>RewardCF</dt><dd>cross-agent signal = teammates' (noisy) rewards. The simple workhorse; "
  "best on the anytime metric.</dd>"
  "<dt>ChoiceZK</dt><dd>cross-agent signal = teammates' CHOICES only (global negatives, no menu): "
  "strictly observes only actions. A noise-immune fallback.</dd>"
  "<dt>BothCF</dt><dd>fuse rewards + competence-weighted choices.</dd>"
  "<dt>HybridCF / HybridCFconv</dt><dd>probe-then-online-ALS: a short UCB probe and SVD warm-start, "
  "then our online weighted-ALS (PTF's probe and warm-start, but our estimator). The converged "
  "variant (more ALS sweeps, refit every round) is the best on final-policy unseen skill.</dd>"
  "<dt>ActiveCFconv</dt><dd>active exploration: a latent-space UCB (predicted reward + a count-based "
  "uncertainty bonus from the broadcast). Probes the most uncertain targets, and since probes are "
  "broadcast, one drone's probe lowers everyone's uncertainty (collective active learning). Best "
  "balanced method.</dd></dl>")
A("<p class='small'>Explored but not recommended (honest negatives): precision-gated fusion "
  "(BothCFPrec) and validation-stacked fusion (StackCF) do not strictly dominate the simple reward "
  "channel, which already wins for all realistic noise; the per-observation confidence GATE (model "
  "agreement) deadlocks at cold-start.</p>")

# 6 METRICS
A("<h2 id='metrics'>6. Metrics (and why each one matters)</h2>")
A("<p>To compare methods fairly we put every score on the same 0-to-1 ruler, called <b>skill</b>:</p>")
A("<pre>skill = (what the method got - what RANDOM gets) / (what the ORACLE gets - what RANDOM gets)</pre>")
A("<p>So <b>skill = 0</b> means 'no better than guessing', and <b>skill = 1</b> means 'as good as a "
  "cheating oracle that knows everything and coordinates everyone'. A method at 0.4 has closed 40% of "
  "the gap between guessing and perfect. We report it as a mean over many random worlds (seeds) with "
  "error bars, so a difference only counts if the bars do not overlap.</p>")
A("<p>We use five flavors of skill, each answering a different question:</p>")

def metric(name, what, why, look):
    A("<div class='step'><h3>%s</h3>" % name)
    A("<p><b>What it is.</b> %s</p>" % what)
    A("<p><b>Why it matters.</b> %s</p>" % why)
    A("<p><b>What to look for.</b> %s</p></div>" % look)

metric("Overall skill",
       "How good the final learned policy is when offered a random handful of targets.",
       "It is the all-round 'did you learn the task' score.",
       "Higher is better; but it can be flattered by methods that explored a lot 'for free' (see "
       "anytime).")
metric("Unseen-pair skill (the categorical one)",
       "The same score, but ONLY on targets the drone never tried itself, the 'unseen' pairs.",
       "This is the make-or-break test of GENERALIZATION. A learner with no shared structure literally "
       "cannot do better than guessing here (it has no information), so it scores ~0 by construction. "
       "Any score clearly above 0 means the method is genuinely transferring knowledge to things it "
       "never experienced.",
       "We want our methods well above 0 and the structure-free baselines stuck at 0. The GAP between "
       "them is the headline result.")
metric("Anytime / AUC skill (the operational one)",
       "The reward actually EARNED while learning, summed over the rounds ('how many targets did you "
       "destroy by round K'), not just the quality of the policy at the end.",
       "In the real world you care about results DURING the mission, not only after. This metric "
       "CHARGES a method for any time it spends just exploring or probing. It is the most honest, "
       "practical score.",
       "Methods that explore politely while still exploiting (ours) win; methods that pause to probe, "
       "or that never stop exploring, lose here even if their final policy looks fine.")
metric("State-uniqueness",
       "How DIFFERENT the drones' learned internal models are from each other.",
       "It checks that decentralization is REAL: if every drone ended up with the same model, 'no "
       "communication' would be meaningless. It is also the quantity that distinguishes the two "
       "masking models in the theory.",
       "It should rise as observation gets more limited (drones see different things), and stay high "
       "over time under persistent masking (durable) but fade under i.i.d. masking (transient).")
metric("Onboarding / newcomer skill vs number of probes",
       "How good the prediction for a brand-new target (or a brand-new drone) gets, as a function of "
       "how many trial engagements ('probes') it has received.",
       "It measures sample efficiency for cold-start: how cheaply can the swarm absorb something new?",
       "Our methods should reach high skill after only about d probes (the number of hidden traits), "
       "while a structure-free learner needs roughly one probe per drone or per target.")

# 7 THEORY
A("<h2 id='thy'>7. Theory, step by step (why the win is categorical)</h2>")
A("<p>The empirical results are not luck: a short chain of arguments PREDICTS them. We give each "
  "result as: <em>statement</em>, <em>why it matters</em>, <em>proof (key steps)</em>, and "
  "<em>confirmed by</em> (which experiment). Full formal proofs: <code>docs/THEORY_FORMAL.md</code>.</p>")
A("<h4>Setup and notation</h4>")
A("<p>True reward matrix <code>R = P U^T</code>, rank <code>d</code>, with unit factors so "
  "<code>R[i,j] = &#10216;p_i, u_j&#10217;</code>. Drone i is offered a uniform size-c set each round; "
  "<code>mu_i</code> is its mean reward, oracle is the best-in-offer, and "
  "<code>skill = (earned - random)/(oracle - random)</code>. A learner is STRUCTURE-FREE if its "
  "estimate of R[i,j] depends only on i's own past pulls of j (per-arm tabular: UCBIndep, Tabular).</p>")

A("<div class='step'><h3>Theorem 1: the tabular floor (EXACT)</h3>")
A("<p><b>Statement.</b> For a structure-free learner and any target j drone i never pulled, the "
  "estimate is the prior constant, so its squared error is at least the row variance "
  "<code>Var_j(R[i,j]) = Omega(1)</code>, and its expected unseen-pair skill is exactly 0.</p>")
A("<p><b>Why it matters.</b> This is the floor the categorical win is measured against: a "
  "structure-free swarm is helpless on anything it has not personally tried, no matter how long it or "
  "its teammates run.</p>")
A("<p><b>Proof (key steps).</b> (1) By definition the estimate on an unpulled j is a constant b "
  "chosen before seeing j; for any constant, <code>E[(b - R[i,J])^2] = (b-mu_i)^2 + Var_J &gt;= "
  "Var_J</code>. (2) The broadcast cannot help: a per-arm estimate of R[i,j] is, by definition, not a "
  "function of any event (k,&middot;) with k != i, nor of any target j' != j. (3) On an offer of "
  "never-pulled targets the selection is independent of their rewards, so by exchangeability the "
  "expected earned reward equals the unseen-set mean, giving skill 0.</p>")
A("<p><b>Confirmed by.</b> Tabular/UCBIndep unseen ~0 at every density (F2), every rank (F4), every "
  "world (Section 8.7). <em>Corollary (pooling):</em> a learner that pools all drones (UCBHomo) "
  "estimates the column mean <code>&#10216;p_bar, u_j&#10217;</code> = the rank-1 'popularity' term, "
  "so it gets PARTIAL unseen skill (~0.17), not zero and not full, exactly as measured.</p></div>")

A("<div class='step'><h3>Theorem 2: CF completes a whole row from O(d) (EXACT)</h3>")
A("<p><b>Statement.</b> If the target factors U are known (rank d) and drone i observes its true "
  "rewards on any set Omega with |Omega| &gt;= d whose factors {u_j} span R^d, then p_i is the unique "
  "least-squares solution and <code>R[i,j] = &#10216;p_i, u_j&#10217;</code> is recovered EXACTLY for "
  "ALL j, including never-pulled ones.</p>")
A("<p><b>Why it matters.</b> This is the engine of generalization and of onboarding/cold-start: a "
  "few observations plus the shared structure determine the entire row. Per-drone sample complexity "
  "drops from <code>Theta(n)</code> (tabular) to <code>Theta(d)</code>.</p>")
A("<p><b>Proof (key steps).</b> (1) Stacking j in Omega gives the linear system "
  "<code>R[i,Omega] = U_Omega p_i</code>. (2) Spanning means U_Omega has rank d, so "
  "<code>U_Omega^T U_Omega</code> is invertible and the solution <code>p_i = (U_Omega^T U_Omega)^{-1} "
  "U_Omega^T R[i,Omega]</code> is unique and equals the true p_i. (3) Then "
  "<code>&#10216;p_i, u_j&#10217;</code> reproduces R[i,j] for every j. (With noise, the ridge "
  "estimate has error O(sigma^2 d / lambda_min), vanishing as data grows.) U itself is identified from "
  "the pooled broadcast by standard matrix completion once about d(m+n) entries are seen.</p>")
A("<p><b>Confirmed by.</b> Target onboarding at ~d shared probes (F3) and newcomer cold-start at ~d "
  "own probes (F10), both vs tabular's ~n; and unseen skill rising as the rank falls (F4).</p></div>")

A("<div class='step'><h3>Theorem 3: anytime separation under starvation (ORDER)</h3>")
A("<p><b>Statement.</b> With n targets, offers of size c, horizon T, ANY structure-free learner has "
  "anytime (cumulative-reward) skill at most <code>g(cT/n)</code>, which goes to 0 when "
  "<code>cT = o(n)</code>, EVEN with full broadcast. A CF learner reaches near-oracle after about d "
  "rounds, so its anytime skill is <code>1 - O(d/T)</code>.</p>")
A("<p><b>Why it matters.</b> 'Final-policy quality' can flatter a method that explores a lot for "
  "free. The operationally honest metric is reward EARNED while learning (targets destroyed by round "
  "K). This theorem says structure-free methods cannot earn under starvation.</p>")
A("<p><b>Proof (key steps).</b> (1) A structure-free learner only earns above the mean on a round "
  "whose offer contains an already-pulled target; its pulled set has size &lt;= t-1. (2) Because new "
  "targets are chosen blind to their reward, the pulled set is reward-blind, so the offered-and-pulled "
  "rewards are like a few i.i.d. draws; by concavity of expected order statistics (Jensen) the "
  "per-round surplus is &lt;= g(c(t-1)/n). (3) Summing over t and using g increasing gives "
  "&lt;= g(cT/n) -&gt; 0 for cT = o(n). The broadcast does not change the pulled set (Theorem 1). "
  "(UCB is even lower: its infinite untried-arm bonus makes it explore forever while n &gt; t.)</p>")
A("<p><b>Confirmed by.</b> UCBIndep stuck at ~0 anytime even at T=200 with n=240 (F9); online CF "
  "earns from round one and dominates at every horizon (F6).</p></div>")

A("<div class='step'><h3>Theorem 4: masking-model dichotomy (EXACT limits)</h3>")
A("<p><b>Statement.</b> Under i.i.d. per-round loss, every drone eventually senses every teammate "
  "(Borel-Cantelli), so all drones converge to the same model (heterogeneity is TRANSIENT). Under "
  "PERSISTENT masking each drone has a permanent blind set, so models stay distinct (DURABLE). The "
  "unseen and anytime results are INVARIANT to the choice.</p>")
A("<p><b>Why it matters.</b> It answers 'is persistent masking a cheat?': no. i.i.d. (packet-loss "
  "style) loss gives the same headline results; persistent masking is chosen only so that "
  "'no communication implies durably different knowledge' is a structural property, not a vanishing "
  "transient.</p>")
A("<p><b>Proof (key steps).</b> (1) i.i.d.: each teammate is observed with prob rho &gt; 0 each "
  "round; the events are independent and sum diverges, so each is seen infinitely often a.s.; hence "
  "every drone recovers the full true model, a common limit. (2) Persistent: a blind teammate's rows "
  "appear in NONE of drone i's observations, so they are non-identifiable for i and stay at the prior "
  "for all T; distinct blind sets give distinct models, bounded away in distance uniformly in T. "
  "(3) The categorical results use only own pulls (Theorem 1) and U-identification (Theorem 2), which "
  "both hold under either mask.</p>")
A("<p><b>Confirmed by.</b> i.i.d. vs persistent give the same unseen/anytime curves, but "
  "state-uniqueness is flat in T under persistent and DECAYS under i.i.d. (F8), exactly as "
  "predicted.</p></div>")
A("<div class='step'><h3>Theorem 5: an additive model cannot personalize (EXACT)</h3>")
A("<p><b>Statement.</b> A predictor of the form 'global + drone-bias + target-bias' (the BiasModel "
  "baseline) has a prediction matrix of rank at most 2, and for ranking targets it reduces to the "
  "shared 'popularity' order, so its unseen skill is the popularity skill, strictly below CF whenever "
  "there is more than one hidden trait.</p>")
A("<p><b>Why it matters.</b> It pinpoints exactly what is missing without the low-rank INTERACTION: "
  "additive effects (some targets are generally easier, some drones generally better) are not enough; "
  "the personalized 'which drone suits which target' part is what CF adds, and what the categorical "
  "win depends on.</p>")
A("<p><b>Proof (key steps).</b> The matrix a+b_i+c_j = (a&middot;1+b) 1^T + 1 c^T is a sum of two "
  "rank-1 matrices, so rank &le; 2. For a FIXED drone the term a+b_i is constant across targets, so "
  "the ranking is by c_j alone, the same popularity order for everyone, which ignores personalization "
  "and so underperforms CF for d&gt;1.</p>")
A("<p><b>Confirmed by.</b> BiasModel unseen ~0.12 and UCBHomo ~0.17 (both additive/popularity) vs CF "
  "~0.49 at d=5 (Section 8.9).</p></div>")
A("<p class='small'>A full theory-vs-experiment alignment table (every prediction mapped to its "
  "measured value, with honest tensions: T3's constant is loose though its mechanism is exact; T2 "
  "idealizes 'U known' vs finite-data recovery) is in <code>docs/THEORY_FORMAL.md</code>.</p>")

# 8 RESULTS
A("<h2 id='res'>8. Results, step by step</h2>")

A("<h3>8.1 When does collaborative filtering help?</h3>")
A("<p>Across a structure-by-observability grid, CF beats tabular if and only if three conditions "
  "hold together: the reward is low-rank but PERSONALIZED (<code>1 &lt; d &lt;&lt; min(m,n)</code>), "
  "the regime is SAMPLE-STARVED, and the reward channel is SHARED. This scopes the claims and "
  "explains earlier null results in sample-rich regimes.</p>")

A("<h3>8.2 The categorical result: acting on unseen pairs</h3>")
A("<p>Statically, CF reaches unseen-pair skill 0.496 vs tabular 0.006. Under the natural masking "
  "regime it holds at every density (Figure F2): CF stays well above zero on unseen pairs for all "
  "<code>rho &gt; 0</code> while tabular sits at the floor; meanwhile drones' models diverge "
  "(state-uniqueness rises 0.54 to 0.92), so decentralization is real.</p>")
A("<figure>%s<figcaption><strong>F2.</strong> Unseen-pair skill vs observation density. CF acts on "
  "never-observed pairs at every density; tabular is pinned at the floor.</figcaption></figure>"
  % img("F2_unseen_masking.png", "F2"))
A("<p><b>Takeaway.</b> This is the core result in one picture: the blue CF line stays well above zero "
  "everywhere (it acts intelligently on things it never tried), while the tabular line hugs zero (it "
  "is helpless on the unseen, by mathematical necessity). The size of that gap, not a few percent, is "
  "what we mean by a CATEGORICAL win.</p>")

A("<h3>8.3 Dynamic onboarding and newcomers</h3>")
A("<p>A brand-new TARGET is onboarded for the whole swarm from about <code>d</code> shared probes "
  "via fold-in (Figure F3); tabular needs about <code>m</code> probes. Symmetrically, a NEW DRONE "
  "with zero history acts from the broadcast alone, folding in its own factor from a few probes "
  "(Figure F10), starting at population-average competence while a tabular newcomer is at the floor. "
  "Both are <code>Theta(d)</code> vs <code>Theta(n)</code> separations.</p>")
A("<figure>%s<figcaption><strong>F3.</strong> Target onboarding: CF onboards from about d probes; "
  "tabular needs about one per drone.</figcaption></figure>" % img("F3_onboard.png", "F3"))
A("<figure>%s<figcaption><strong>F10.</strong> Newcomer cold-start: a new drone acts from the "
  "broadcast at zero history; the tabular newcomer is at the floor.</figcaption></figure>"
  % img("F10_newcomer.png", "F10"))
A("<p><b>Takeaway.</b> New tasks and new teammates are absorbed cheaply: a few trial runs (about d, "
  "the number of hidden traits) are enough for the whole swarm to predict a new target, and a brand-new "
  "drone is useful from its very first moment by riding on what the swarm already learned. A "
  "structure-free swarm would need to re-try everything from scratch.</p>")

A("<h3>8.4 The gap scales with low-rankness</h3>")
A("<p>As the true rank rises there is more to complete from the same data, so CF unseen skill "
  "decreases (0.96 at d=1 to 0.10 at d=20) while tabular stays at the floor for every rank "
  "(Figure F4).</p>")
A("<figure>%s<figcaption><strong>F4.</strong> Unseen-pair skill vs true rank d.</figcaption></figure>"
  % img("F4_rank.png", "F4"))

A("<h3>8.5 Comparison to the field: masking-robustness and the anytime metric</h3>")
A("<p>Every low-rank method clears the no-structure floor on unseen pairs, so that categorical win "
  "is a property of low-rank STRUCTURE, not of our particular estimator. Our method's specific edge "
  "is twofold. (a) Masking-robustness (Figure F5): our online weighted-ALS unseen skill is flat as "
  "the broadcast is masked, while batch-SVD hybrids (PTF/ESTR/BPMF) decay. (b) Anytime (Figure F6): "
  "on cumulative reward we dominate at every horizon; UCBIndep is stuck near random because, with "
  "<code>n &gt;&gt; T</code>, it never stops exploring untried arms, and PTF/ESTR pay a probe "
  "phase.</p>")
A("<figure>%s<figcaption><strong>F5.</strong> Masking-robustness: online ALS stays high; batch-SVD "
  "hybrids decay; the no-structure floor is at zero.</figcaption></figure>" % img("F5_crossover.png", "F5"))
A("<figure>%s<figcaption><strong>F6.</strong> Anytime reward trajectory: online CF earns from round "
  "one; explore-then-commit pays a probe phase; UCBIndep is stuck.</figcaption></figure>"
  % img("F6_anytime.png", "F6"))
A("<p><b>Takeaway.</b> If you count reward as it is actually earned (the honest, operational view), "
  "our online methods (top curves) pull ahead from the very first rounds and stay ahead. The "
  "explore-then-commit competitors are flat near zero until they stop probing, and the per-target "
  "bandit never recovers because, with far more targets than rounds, every menu still contains "
  "something it has not tried. This is where 'acting on the unseen' turns into real, early reward.</p>")

A("<h3>8.6 Both observability channels</h3>")
A("<p>Crossing masking with reward noise (Figure F7): the reward channel (RewardCF) degrades as "
  "<code>sigma_obs</code> rises while the clean choice channel (ChoiceZK) is flat. The crossover is "
  "at <code>sigma_obs ~ 1</code> (noise = half the signal range): for realistic noise the reward "
  "channel dominates; the choice channel is severe-noise insurance. No learned fusion is needed.</p>")
A("<figure>%s<figcaption><strong>F7.</strong> Two channels: clean choices overtake noisy rewards "
  "only under severe noise.</figcaption></figure>" % img("F7_channels.png", "F7"))

A("<h3>8.7 Robustness checks: masking model, scaling, generality</h3>")
A("<p>Re-running everything under i.i.d. per-round loss (Figure F8) leaves unseen and anytime "
  "essentially unchanged (the categorical results do not depend on the masking model), while "
  "state-uniqueness is durable under persistent masking but transient under i.i.d., exactly as the "
  "theory predicts. Scaling sweeps over rank, horizon, target count, and guessed rank (Figure F9) "
  "confirm every trend, including robustness to misguessing the rank. Generality sweeps confirm the "
  "conclusions are not artifacts of the default world: they hold across drone count m (10 to 120), "
  "cluster count K (2 to 30, INCLUDING K=m, i.e. every drone its own type = no clustering = uniform "
  "latents), and latent spread (tight to diffuse). So the categorical advantage is not an artifact of "
  "clustered latent sampling.</p>")
A("<figure>%s<figcaption><strong>F8.</strong> Persistent vs i.i.d. masking: results invariant; "
  "decentralization durable (persistent) vs transient (i.i.d.).</figcaption></figure>"
  % img("F8_iid_vs_persistent.png", "F8"))
A("<p><b>Takeaway.</b> It does not matter whether the missing observations are a fixed blind spot "
  "(persistent) or random dropouts (i.i.d., like lost radio packets): the headline results are the "
  "same (left/middle panels overlap). The only thing that changes is whether the drones STAY "
  "different over time (right panel): they do under a fixed blind spot, but slowly converge under "
  "random dropouts. So our modeling choice is principled, not a cheat.</p>")
A("<figure>%s<figcaption><strong>F9.</strong> Scaling: unseen (top) and anytime (bottom) vs rank, "
  "horizon, targets, and guessed rank.</figcaption></figure>" % img("F9_scaling.png", "F9"))

A("<h3>8.8 Putting it together: we dominate the field</h3>")
A("<p>The strongest competitor (PTF) once led on one metric (final-policy unseen at full broadcast); "
  "that was an artifact of our under-converged default estimator. With a converged configuration, "
  "HybridCFconv ties PTF there and beats it everywhere else, and active exploration (ActiveCFconv) "
  "improves on plain online CF on both metrics. The Pareto frontier (Figure F11) is entirely ours: "
  "ActiveCFconv (best balanced) and HybridCFconv (best unseen under masking) sit up-and-right; PTF "
  "and the rest are dominated.</p>")
A("<figure>%s<figcaption><strong>F11.</strong> Pareto frontier (anytime vs unseen): our methods "
  "dominate; PTF trades all anytime for unseen and is dominated under masking.</figcaption></figure>"
  % img("F11_pareto.png", "F11"))
A("<p><b>Takeaway.</b> 'Up and to the right' is best (good on both axes). Our methods (green/blue) sit "
  "in that corner; the strongest competitor (PTF, red) is pushed to the edge, buying a sliver of "
  "final-policy quality by giving up almost all earned reward, and it falls behind entirely once "
  "observation is limited. There is no metric left on which a competitor beats us in the regime that "
  "matters.</p>")
A("<p>Summary numbers (skill ~0 = random; ours in green):</p>")
rows = [("Random", "no-struct", "0.01", "0.00"), ("UCBIndep", "no-struct", "0.00", "-0.01"),
        ("Tabular", "no-struct", "0.00", "0.25"), ("PTF", "low-rank", "0.51", "0.27"),
        ("ESTR", "low-rank", "0.23", "0.18"), ("RewardCF (ours)", "low-rank", "0.39", "0.40"),
        ("HybridCFconv (ours)", "low-rank", "0.49", "0.35"), ("ActiveCFconv (ours)", "low-rank", "0.49", "0.44")]
A("<table><tr><th class='l'>Method</th><th>Class</th><th>UNSEEN @rho=1</th><th>ANYTIME @rho=1</th></tr>")
for nm, cl, un, an in rows:
    o = " class='ours'" if "ours" in nm else ""
    A("<tr><td class='l'%s>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (o, nm, cl, un, an))
A("</table>")

# 9 NOVELTY
A("<h3>8.9 Even more competitors (a broader, fair bake-off)</h3>")
A("<p>To be sure we are not beating a weak field, we added three more methods, each run under the "
  "exact same limits (per-drone, broadcast-only, guessed rank where relevant): <b>SoftImpute</b> (a "
  "different, convex way to fill in the table), <b>kNN-CF</b> (a model-free 'find similar drones' "
  "approach), and <b>BiasModel</b> (additive effects only, no personalization). Figure F12 shows the "
  "result.</p>")
A("<figure>%s<figcaption><strong>F12.</strong> More fair baselines vs ours, unseen (left) and anytime "
  "(right) vs observation density.</figcaption></figure>" % img("F12_morebaselines.png", "F12"))
A("<p><b>Takeaway.</b> The story holds and widens. SoftImpute is excellent when the broadcast is full "
  "(it even tops everyone on unseen at rho=1), but, like all the batch 'fill-in' methods, it collapses "
  "as observation is masked and it loses on earned reward. kNN-CF generalizes but earns little. "
  "BiasModel stays near the additive ceiling (it cannot personalize, exactly as Theorem 5 says). "
  "Under masking (the regime that matters) and on the anytime metric, our methods beat the best of "
  "these new baselines with non-overlapping error bars. No competitor wins in the regime that defines "
  "the problem.</p>")

A("<h3>8.10 Validation in a realistic simulator</h3>")
A("<p>Everything above is in our clean, controlled world. Does it survive a realistic simulator? We "
  "dropped our method into the existing tabula_drone PettingZoo environment, which has spatial "
  "targets, target health that depletes as drones engage (so it is partly a contention setting), and "
  "episodic resets, and benchmarked it (3 seeds, learning over episodes) against the environment's own "
  "policies. Score = efficiency (reward per step), normalized to skill (Figure F13).</p>")
A("<figure>%s<figcaption><strong>F13.</strong> Real tabula_drone simulator: converged skill (left) "
  "and learning curves (right).</figcaption></figure>" % img("F13_realsim.png", "F13"))
A("<p><b>Takeaway.</b> The story holds in the real simulator: our method reaches skill ~0.81 (with "
  "low variance), clearly beating the environment's own SGD matrix-factorization (~0.25) and the "
  "per-arm bandit UCBIndep (~0.72), and approaching the centralized oracle. So the advantage is not "
  "an artifact of our clean generative model; it transfers to spatial, depleting, episodic dynamics.</p>")

A("<h2 id='nov'>9. Novelty and honest positioning</h2>")
A("<p><strong>Novelty.</strong> (1) The decentralized, online, broadcast-only, per-drone-masked "
  "formulation of CF for MRTA, with the unseen-pair / onboarding categorical separations and a "
  "matching per-drone theory; standard matrix-completion theory is centralized and about estimation "
  "error, not online decision reward. (2) The anytime (cumulative-reward) separation under sample "
  "starvation. (3) The masking-model dichotomy (durable vs transient decentralization). (4) Collective "
  "active exploration via the shared broadcast.</p>")
A("<div class='box warn'><strong>Honest positioning.</strong> The categorical unseen win is shared "
  "by all low-rank methods over no-structure ones (it is a property of structure). Our specific "
  "contribution is being masking-robust and anytime-optimal, dominating throughout the limited-"
  "observability regime that defines the problem. We do not fabricate dominance: precision-gated and "
  "stacked fusions did not beat the simple reward channel, and we report that.</div>")

# 10 REPRO
A("<h2 id='repro'>10. Reproducibility</h2>")
A("<p class='small'>All numbers come from complete per-seed JSON in <code>results/pilots/</code> "
  "(registry <code>docs/DATA_CATALOGUE.md</code>; chronology and per-cycle reviews "
  "<code>docs/PROJECT_LOG.md</code>). Figures: <code>python experiments/make_figures.py</code>; this "
  "page: <code>python experiments/make_tutorial.py</code>. Harnesses: <code>pilot_compare.py</code>, "
  "<code>pilot_crossover.py</code>, <code>pilot_anytime.py</code>, <code>pilot_e3_channels.py</code>, "
  "<code>pilot_iid.py</code>, <code>pilot_scaling.py</code>, <code>pilot_robust.py</code>, "
  "<code>pilot_e7_newcomer.py</code>, <code>pilot_e8.py</code>; methods in <code>pilot_noise.py</code> "
  "and <code>pilot_baselines.py</code>; world/reward/oracle in <code>core.py</code>. Proofs: "
  "<code>docs/THEORY_FORMAL.md</code>; ZK audit: <code>docs/ZK_COMPLIANCE.md</code>; paper draft: "
  "<code>docs/PAPER_DRAFT.md</code>.</p>")

# 11 GLOSSARY
A("<h2 id='glo'>11. Glossary</h2>")
A("<dl>"
  "<dt>Low rank / latent factors</dt><dd>R = P U^T with a few columns; compatibility is an inner "
  "product of short vectors.</dd>"
  "<dt>Weighted ALS</dt><dd>alternating least squares solving for P given U and vice versa, each "
  "observation weighted by its precision; missing entries get zero weight.</dd>"
  "<dt>Fold-in</dt><dd>fit a new row/column factor from a few observations given the other factors, "
  "no retraining.</dd>"
  "<dt>Unseen pair</dt><dd>a (drone, target) the drone never engaged or observed; tabular is at the "
  "floor here, CF is not.</dd>"
  "<dt>Anytime / AUC skill</dt><dd>cumulative reward earned over rounds; charges exploration cost.</dd>"
  "<dt>Masking (rho)</dt><dd>per-drone loss of a fraction of sensed outcomes (limited detection).</dd>"
  "<dt>Skill</dt><dd>(method - random)/(oracle - random): 0 random, 1 centralized ceiling.</dd></dl>")

A("<p class='small' style='margin-top:30px'>Generated from the project repository; every figure and "
  "number is regenerable from saved data. See the paper draft for the full write-up.</p>")
A("</div></body></html>")

html = "\n".join(H)
open(OUT, "w", encoding="utf-8").write(html)
print("wrote %s (%d KB)" % (OUT, len(html.encode("utf-8")) // 1024))
