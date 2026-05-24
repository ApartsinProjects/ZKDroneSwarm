"""Generate the FOLLOW-UP journal paper (companion to docs/ras_paper.html) as a
self-contained HTML with KaTeX math and callout boxes. The main paper establishes the
core categorical communication-free collaborative-filtering result (SwarmCF) and DEFERS
the refinements; this generator drafts that follow-up: confidence-directed exploration,
communication-free de-confliction under contention, rank self-determination, the choice
channel, non-stationarity / churn, and a unified method.

Output: docs/ras_paper2.html. Run from REPO ROOT (reads docs/figures/*.png, imports
method_profiles). DRAFT: every empirical number is taken from the project's logged data
(docs/*.md, results/pilots/*.json); no fabricated figures.
"""
import base64, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import method_profiles as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures")
OUT = os.path.join(ROOT, "docs", "ras_paper2.html")


def img(name, alt, w="90%"):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return "<p><em>[missing %s]</em></p>" % name
    b = base64.b64encode(open(p, "rb").read()).decode("ascii")
    return ('<img alt="%s" src="data:image/png;base64,%s" style="max-width:%s;height:auto;'
            'border:1px solid #d7dde3;border-radius:6px">' % (alt, b, w))


CSS = """
body{margin:0;color:#16191d;background:#fff;font:16px/1.66 Georgia,'Times New Roman',serif;text-align:justify;hyphens:auto;-webkit-hyphens:auto}
h1,h2,h3,.sub{text-align:left;hyphens:manual}
.wrap{max-width:860px;margin:0 auto;padding:34px 22px 90px}
h1{font-size:28px;line-height:1.22;margin:0 0 8px;font-family:-apple-system,Segoe UI,Roboto,sans-serif}
h2{font-size:21px;margin:32px 0 8px;font-family:-apple-system,Segoe UI,Roboto,sans-serif;border-bottom:2px solid #e2e8ee;padding-bottom:3px}
h3{font-size:17px;margin:20px 0 6px;color:#1f5fa8;font-family:-apple-system,Segoe UI,Roboto,sans-serif}
.sub{color:#5b6570;font-size:14.5px;margin:0 0 14px;font-family:-apple-system,Segoe UI,Roboto,sans-serif}
p{margin:9px 0}code{background:#f5f8fb;padding:1px 4px;border-radius:4px;font:13px/1.4 Consolas,monospace}
.abs{background:#f5f8fb;border:1px solid #e2e8ee;border-radius:8px;padding:14px 20px;margin:16px 0;font-size:15px}
.hl{background:#fffdf3;border:1px solid #ece3c0;border-radius:8px;padding:8px 18px 8px 20px;margin:14px 0;font-size:14px}
.box{background:#f7fbf8;border-left:4px solid #0a7d4d;border:1px solid #e2e8ee;border-radius:8px;padding:10px 16px;margin:14px 0}
.thm{background:#f6f8fc;border-left:4px solid #1f5fa8;border:1px solid #e2e8ee;border-radius:8px;padding:8px 16px;margin:12px 0;font-size:15px}
.prelim{background:#fdf6f6;border-left:4px solid #b3541e;border:1px solid #e8d7d7;border-radius:8px;padding:8px 16px;margin:12px 0;font-size:14px}
.algo{background:#fbfbfd;border:1px solid #d7dde3;border-radius:8px;padding:8px 14px;margin:14px 0;font:13px/1.5 Consolas,monospace;white-space:pre-wrap;text-align:left;hyphens:manual}
.algo .cap{font-family:-apple-system,Segoe UI,sans-serif;font-weight:700;font-size:13px;color:#16191d;border-bottom:1px solid #e2e8ee;padding-bottom:4px;margin-bottom:6px;white-space:normal}
figure{margin:18px 0;text-align:center}figcaption{color:#5b6570;font-size:13px;margin-top:6px;text-align:justify}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:12.5px;font-family:-apple-system,Segoe UI,sans-serif}
th,td{border:1px solid #e2e8ee;padding:5px 8px;text-align:center}td.l,th.l{text-align:left}
.small{font-size:13px;color:#5b6570}a{color:#1f5fa8}
ol.contrib>li{margin:4px 0}
.docxlink{position:fixed;top:10px;right:12px;z-index:50;background:#1f5fa8;color:#fff;text-decoration:none;font:600 12px/1.2 -apple-system,Segoe UI,Roboto,sans-serif;padding:7px 11px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.18)}
.docxlink:hover{background:#17487f}
@media print{.docxlink{display:none}}
"""

KATEX = ("<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css'>"
         "<script defer src='https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js'></script>"
         "<script defer src='https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js' "
         "onload=\"renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},"
         "{left:'$',right:'$',display:false}]})\"></script>")

H = []
A = H.append
A("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
  "<meta name='viewport' content='width=device-width,initial-scale=1'>"
  "<title>Refinements of Communication-Free Collaborative Filtering for Decentralized MRTA</title>"
  "%s<style>%s</style></head><body><div class='wrap'>" % (KATEX, CSS))

# ---------------- title / meta ----------------
A("<a class='docxlink' href='ras_paper2.docx'>Download .docx</a>")
A("<h1>Refinements of Communication-Free Collaborative Filtering for Decentralized "
  "Multi-Robot Task Allocation: confidence, contention, rank, and beyond</h1>")
A("<p class='sub'>Follow-up to &ldquo;Acting on the Unseen&rdquo;.<br>"
  "Author One<sup>a,&lowast;</sup>, Author Two<sup>a</sup><br>"
  "<span class='small'><sup>a</sup>Affiliation, City, Country.&ensp;"
  "<sup>&lowast;</sup>Corresponding author.</span></p>")

A("<div class='hl'><b>This is a draft follow-up paper.</b> It builds on the companion paper, which "
  "establishes the core result; here we study the refinements that foundation enables. Several results "
  "below are reported as <b>preliminary</b> and are marked as such; they are grounded in logged "
  "experiments but have not received the full statistical treatment of the companion paper's headline "
  "claims.</div>")

# ---------------- abstract ----------------
A("<div class='abs'><b>Abstract.</b> A companion paper establishes that a robot team with no prior "
  "knowledge, no communication, and only a partial, noisy, privately-perceived view of teammates' "
  "outcomes can still act well on tasks it never attempted, by running decentralized online low-rank "
  "collaborative filtering over the passive broadcast (<b>SwarmCF</b>); the advantage over any "
  "structure-free learner is categorical, not a constant factor. That paper deliberately keeps to a "
  "single core estimator and defers the refinements. This follow-up studies six of them, each a member "
  "of the SwarmCF family and each evaluated on the same communication-free harness: "
  "<b>(i) confidence-directed exploration</b> through a Bayesian posterior over the shared factors "
  "(<b>SwarmCF-B</b>), which raises both final and anytime quality and lets the swarm probe where the "
  "shared structure is least certain; <b>(ii) communication-free de-confliction</b> under capacity-1 "
  "contention via a fixed, private offset (<b>SwarmCF-D+</b>), which roughly doubles earned reward at "
  "severe contention with no messages; <b>(iii) rank self-determination</b> that removes the guessed "
  "rank (<b>SwarmCF-B-ARD</b>), recovering a stable effective rank independent of the guess; "
  "<b>(iv) the action / choice channel</b> as a noise-immune alternative to cardinal rewards "
  "(<b>SwarmCF-Ch</b>), which overtakes the reward channel once observation noise is large; "
  "<b>(v) non-stationarity and team churn</b>, handled only when low-rank fold-in is united with "
  "confidence-directed probing of newcomers; and <b>(vi) a unified method</b> (<b>SwarmCF-U</b>) whose "
  "refinements activate only on their triggering condition, giving one policy that is best-or-tied "
  "across regimes. We close with what carries over from the foundation and an honest account of the "
  "limitations, marking preliminary results throughout.</div>")
A("<p class='small'><b>Keywords:</b> multi-robot task allocation; decentralized learning; collaborative "
  "filtering; low-rank matrix completion; communication-free coordination; Bayesian factorization; "
  "automatic relevance determination; swarm robotics.</p>")

# ---------------- 1. the foundation ----------------
A("<h2>1. Introduction and the foundation</h2>")
A("<div class='box'><b>The companion paper, in one paragraph.</b> The foundation paper formalizes "
  "multi-robot task allocation (MRTA) in its most restrictive but practically common form: a team of "
  "$m$ robots faces $n$ tasks whose hidden robot$\\times$task reward $R=PU^\\top$ is low-rank (a few "
  "latent capability and requirement traits govern fit), with <b>no prior knowledge</b> (not even the "
  "rank), <b>no communication</b>, and a <b>partial, per-observer-noisy</b> view of a public outcome "
  "stream, in the task-scarce regime $n\\gg T$ where most pairs are never attempted. Its method, "
  "<b>SwarmCF</b>, is a decentralized online low-rank collaborative filter that each robot runs over "
  "the passive broadcast, with an $O(\\hat d)$ fold-in that lets it act on never-attempted tasks. Its "
  "central result is a <b>categorical separation</b>: a structure-free learner is provably pinned at "
  "the prior-mean error floor on unseen pairs (Proposition 1), while SwarmCF attains per-robot sample "
  "complexity $\\Theta(d)$ versus $\\Theta(n)$ with a matching anytime separation (Theorems 1-2), a "
  "deterministic condition under which decentralized recovery from the privately-masked broadcast is "
  "exact (Theorem 3), and a collective-speedup law by which the swarm gets smarter and faster as it "
  "grows (Theorem 4); the advantage is validated on a robotics-grounded mission and transfers to the "
  "higher-fidelity <b>LatentSwarm</b> spatial simulator. This paper studies the refinements that "
  "foundation enables.</div>")
A("<p>The foundation makes a single, simple estimator do one thing well: generalize to the unseen with "
  "no communication. That deliberate minimalism leaves a family of natural questions open, and the "
  "companion paper lists them as future work. Each question is a refinement of the same estimator rather "
  "than a new paradigm, so each can be studied on the identical communication-free harness and read "
  "against the same categorical baseline. The contributions of this follow-up are exactly those "
  "refinements.</p>")
A("<p><b>Contributions of this follow-up.</b></p><ol class='contrib'>"
  "<li><b>Confidence-directed exploration</b> via a Bayesian posterior over the shared factors "
  "(%s), and the active / coordinated variants that probe where the structure is least certain "
  "(Section 3).</li>"
  "<li><b>Communication-free de-confliction</b> under capacity-1 contention through a fixed, private "
  "offset, and a self-tuning, scarcity-gated form (%s) that matches plain SwarmCF when targets are "
  "plentiful and roughly doubles earned reward at severe contention (Section 4).</li>"
  "<li><b>Rank self-determination</b> that removes the guessed rank $\\hat d$ via automatic relevance "
  "determination (%s), recovering a stable effective rank invariant to the guess (Section 5).</li>"
  "<li><b>The action / choice channel</b> (%s) as a noise-immune alternative to cardinal rewards, "
  "with the crossover at which it overtakes the reward channel (Section 6).</li>"
  "<li><b>Non-stationarity and team churn</b>: plain exploitative filtering is shown to be insufficient, "
  "and the fix is to unite low-rank fold-in with confidence-directed newcomer probing (Section 7).</li>"
  "<li><b>A unified method</b> (%s) whose refinements activate only on their condition, giving one "
  "communication-free policy that is best-or-tied across the standard, churn, and contention regimes "
  "(Section 8).</li></ol>"
  % (mp.disp("EMCF"), mp.disp("ContentionAdaCF"), mp.disp("ARD-EMCF"),
     mp.disp("ChoiceCF"), mp.disp("UnifiedCF")))

A("<p>All refinements are members of one family. Table 1 lays out how each member differs from the "
  "core estimator along the axes that matter (signal channel, exploration rule, confidence handling, "
  "contention handling, rank, and coordination); the body of the paper then takes the axes one at a "
  "time. Throughout, we cite each member by its display name and reserve numbered formal results "
  "(Propositions and Theorems, proved in the companion paper's theory appendix) for the claims they "
  "support.</p>")
A("<p class='small'><b>Table 1.</b> The SwarmCF family by mechanism: each refinement changes one or two "
  "axes of the same decentralized, communication-free online estimator. The core method (%s) is the "
  "top row of the foundation; the remaining rows are the refinements studied here.</p>"
  % mp.disp("RewardCF"))
A(mp.html_mechanisms())

# ---------------- 2. setting recap ----------------
A("<h2>2. Setting and conventions (recap)</h2>")
A("<p>We keep the foundation's setting verbatim, so this paper is self-contained. A team of $m$ robots "
  "faces $n$ tasks; robot $i$ has a hidden capability vector $p_i\\in\\mathbb{R}^d$ and task $j$ a "
  "hidden requirement vector $u_j\\in\\mathbb{R}^d$, and the expected reward of engaging is the inner "
  "product $R_{ij}=\\langle p_i,u_j\\rangle$, so $R=PU^\\top$ has rank $d\\ll\\min(m,n)$. The team knows "
  "neither $P$, $U$, nor $d$; it uses a guessed rank $\\hat d$. Each round every robot is offered a "
  "size-$c$ subset of tasks, selects one, engages it, and earns its reward; the regime is task-scarce "
  "($n\\gg T$). There is no communication: each robot passively senses a public stream of engagement "
  "outcomes, but only for teammates it can see (a persistent per-pair mask of rate $\\rho$) and read "
  "with private per-observer noise $\\sigma$, so no two robots ever see the same stream.</p>")
A("<p>We report the same normalized <b>skill</b> as the foundation, "
  "$\\mathrm{skill}=(\\text{earned}-\\text{random})/(\\text{oracle}-\\text{random})$, where $0$ is the "
  "no-information floor and $1$ is omniscient. <b>Unseen-pair skill</b> is restricted to tasks the robot "
  "never engaged (the categorical generalization test that the companion paper turns into a theorem); "
  "<b>anytime skill</b> is the cumulative-reward (operational) measure. Under contention we additionally "
  "report earned reward normalized by the per-round matching optimum (Hungarian), and the collision "
  "rate. Unless stated otherwise the configuration is the foundation's: $m=30$ robots, $n=240$ tasks, "
  "true rank $d=5$, guessed rank $\\hat d=8$, partial broadcast $\\rho$ swept, own-observation noise "
  "$0.1$ and broadcast-observation noise $0.3$, with bootstrap 95% confidence intervals over 8 seeds "
  "(6 for the confidence bake-off).</p>")
A("<p class='small'>The methods compared here all run on the same harness as the foundation: one "
  "estimator per robot, decentralized and communication-free, reading only the passive broadcast. "
  "Communication-based methods (auctions, consensus, centralized training, federated or gossip "
  "exchange) remain inadmissible by the problem definition and appear only as reference ceilings; the "
  "admissible communication-free comparison is the structure-free paradigm (independent UCB, tabular). "
  "Table 2 of the companion paper fixes those operating profiles.</p>")

# ---------------- 3. confidence ----------------
A("<h2>3. Confidence-directed exploration via a Bayesian posterior over the factors</h2>")
A("<p><b>The idea.</b> The core estimator explores with a simple $\\varepsilon$-greedy schedule, which "
  "is uninformed: it probes uniformly at random rather than where its knowledge is weakest. The "
  "refinement replaces the point estimate of the factors with a <b>Bayesian posterior</b>, so the swarm "
  "can act optimistically under calibrated uncertainty and direct its scarce engagements toward the "
  "tasks whose shared requirement vectors are least pinned down. The mechanism is "
  "<b>%s</b> (variational Bayesian probabilistic matrix factorization with a predictive-interval "
  "rule), with active and coordinated exploration variants <b>%s</b> and <b>%s</b>."
  % (mp.disp("EMCF"), mp.disp("ActiveCF"), mp.disp("CoordCF")))
A("<p><b>The mechanism.</b> Each observation enters the variational posterior with its likelihood "
  "precision inside the model while the prior fixes the data-versus-regularizer scale, giving a valid "
  "predictive variance "
  "$\\operatorname{Var}(\\hat R_{ij})=p_i^\\top\\Sigma_{u_j}p_i+u_j^\\top\\Sigma_{p_i}u_j+"
  "\\operatorname{tr}(\\Sigma_{p_i}\\Sigma_{u_j})$ on which an upper-confidence rule is optimism under "
  "uncertainty. A subtlety the foundation's theory pins down (Proposition 6) is that naive "
  "inverse-variance weighting of the <i>fit</i> is the wrong object: for a task the robot never engaged "
  "the requirement vector $u_j$ is identified only from teammates' broadcast, so over-weighting the "
  "robot's own (zero-information) rewards inflates the unseen error, and uniform, coverage-preserving "
  "weighting weakly dominates. The Bayesian posterior is the right way to use confidence because it "
  "keeps the broadcast at full weight in the fit while remaining noise- and coverage-aware. For "
  "<i>exploration</i>, only the collective term $p_i^\\top\\Sigma_{u_j}p_i$ (the shared-factor "
  "uncertainty) is used: it is task-specific and anneals as the swarm pins down the structure, whereas "
  "the own-factor term over-explores early.</p>")
A("<p><b>The evidence (confidence preserves generalization and adds skill).</b> On the canonical masked "
  "harness, %s is at least as good as uniform weighting in every observation condition and strictly "
  "better in at least one, the only confidence mechanism in our bake-off to dominate. Its advantage is "
  "largest when the broadcast is noisy: at full broadcast with high observation noise its unseen skill "
  "is $0.326$ versus $0.196$ for uniform weighting, and a shrinkage variant reaches $0.689$ versus "
  "$0.594$ at low noise. Crucially the broadcast stays at full weight, so the categorical "
  "generalization is preserved rather than traded away. Naive inverse-variance ('full precision') "
  "weighting instead falls below uniform on unseen skill ($0.449$ versus $0.594$ at full broadcast, low "
  "noise), exactly as Proposition 6 predicts; ratio-bounded precision only helps when the teammate "
  "sources genuinely differ in reliability." % mp.disp("EMCF"))
A("<p><b>The posterior is informative, not decorative.</b> Binning unseen pairs by %s's predicted "
  "standard deviation, the actual root-mean-square error rises monotonically from the most-confident to "
  "the least-confident quintile (RMSE $0.231\\to 0.492$, a $+0.261$ spread), so the uncertainty "
  "genuinely tracks error and is usable for an upper-confidence rule or for shrinkage. As is typical of "
  "mean-field variational inference the nominal intervals are mildly over-confident (a $90\\%%$ interval "
  "covers about $85\\%%$ empirically), a caveat we state plainly; what an exploration rule needs is "
  "discrimination, which holds, not exact coverage. A constant-standard-deviation baseline has no "
  "discrimination at all." % mp.disp("EMCF"))
A("<p><b>Information-directed exploration buys early sample efficiency.</b> Sweeping exploration rules "
  "at partial broadcast, a strongly optimistic posterior-UCB over-explores early (it spends rounds on "
  "uniformly uncertain tasks), but a tempered collective-UCB attains the best final anytime skill "
  "($0.356$ versus $0.324$ for $\\varepsilon$-greedy and $0.333$ for a cheap count-bonus). A "
  "coordinated variant, %s, down-weights tasks the swarm has already probed through the broadcast (an "
  "explicit division of labor with no communication) and is the fastest early (highest round-10 "
  "cumulative skill), though the exploitation-biased collective rule still wins the final value. The "
  "honest summary is that confidence-directed and coordinated exploration improve early coverage and "
  "final anytime quality, not that any single rule dominates at every horizon." % mp.disp("CoordCF"))
A("<figure>%s<figcaption><b>Figure 1.</b> The anytime-versus-unseen frontier (up and to the right is "
  "better). The confidence- and exploration-directed members of the family (including the active "
  "variant %s) match or dominate the batch variant %s on both axes; the batch variant trades all "
  "anytime quality for unseen skill and is dominated under masking ($\\rho=0.25$). Confidence-directed "
  "exploration thus adds operational value without giving up the categorical unseen "
  "capability.</figcaption></figure>"
  % (img("F11_pareto.png", "anytime vs unseen Pareto frontier"),
     mp.disp("ActiveCF"), mp.disp("PTF")))
A("<div class='thm'><b>Supporting result (Proposition 6, companion theory).</b> When a never-engaged "
  "task's requirement vector is identified only from the broadcast and the broadcast sources are "
  "homogeneously noisy, uniform (coverage-preserving) weighting weakly dominates inverse-variance "
  "fit-weighting for unseen prediction; the variational-Bayes posterior is the consistent "
  "full-information estimator and its collective predictive variance gives a valid optimism rule.</div>")

# ---------------- 4. contention ----------------
A("<h2>4. Communication-free de-confliction under capacity-1 contention</h2>")
A("<p><b>The idea.</b> When tasks deplete or have unit capacity, several robots that have learned "
  "accurate, similar preferences will converge on the same best task and collide; the foundation's "
  "mission section identifies this within-round coordination as the binding constraint and defers its "
  "solution. The refinement breaks the symmetry <b>without any communication</b> by giving each robot a "
  "fixed, private offset over tasks, so otherwise-identical robots spread across distinct good tasks. "
  "The mechanism is <b>%s</b> (a fixed private offset) and its self-tuning form <b>%s</b> "
  "(a scarcity-gated offset that scales with the robot's own recent loss rate)."
  % (mp.disp("ContentionCF"), mp.disp("ContentionAdaCF")))
A("<p><b>The mechanism.</b> Each robot draws a fixed offset $h_i\\in\\mathbb{R}^n$ with independent "
  "continuous entries and selects $\\arg\\max_{j\\in S}(\\hat R_{ij}+\\varepsilon h_i[j])$. Within a "
  "group of similar robots the perturbed argmaxes are almost surely distinct, so same-type collisions "
  "vanish once the group's top targets are offered, while any task with a reward margin above "
  "$2\\varepsilon\\lVert h\\rVert_\\infty$ is unchanged, preserving value up to $O(\\varepsilon)$. Two "
  "design choices are essential and are theorems, not heuristics (Theorem 7): the offset must be "
  "<b>fixed</b> (a re-randomized per-round offset gives the same expected collision probability and no "
  "stable assignment) and <b>private</b> (a shared-signal offset such as a popularity or collective "
  "count shifts every robot identically and re-synchronizes them). A constant offset, however, hurts "
  "preference quality when there is no contention, so the deployed method %s scales the offset toward "
  "zero when the robot is winning its engagements (recovering greedy, value-preserving behavior) and "
  "toward the fixed offset under saturation, gated by a strict scarcity test on the offer size."
  % mp.disp("ContentionAdaCF"))
A("<p><b>The evidence (a roughly 2x earned-reward win at severe contention, no messages).</b> Sweeping "
  "the shared offer-pool size from plentiful ($240$) to severely contended ($15$) at full broadcast, "
  "%s is best-or-tied on matching-normalized earned reward at every pool: it matches plain SwarmCF when "
  "targets are plentiful (pool $240$: $0.448$ versus $0.439$, with the lowest collision rate $0.126$ "
  "among reward-seekers and unseen skill recovering to $0.320$) and roughly doubles earned reward at "
  "the most contended pool (pool $15$: $0.100$ versus $\\le 0.06$ for the no-offset methods, with "
  "non-overlapping intervals). The fixed-offset form %s wins similarly at severe contention "
  "($0.105$ at pool $15$). Figure 1 shows the de-confliction sweep." % (mp.disp("ContentionAdaCF"), mp.disp("ContentionCF")))
A("<figure>%s<figcaption><b>Figure 2.</b> Communication-free de-confliction under capacity-1 "
  "contention. As the shared task pool shrinks (more contention, left to right), the private-offset "
  "methods (%s, %s) sustain the most earned reward among reward-seeking policies, roughly doubling the "
  "no-offset methods at the most contended pool, while structure-free dispatch earns near zero. The "
  "categorical unseen-pair quality is contention-invariant because it is a property of the learned "
  "model.</figcaption></figure>"
  % (img("F15_deconfliction.png", "de-confliction under contention"),
     mp.disp("ContentionAdaCF"), mp.disp("ContentionCF")))
A("<p><b>Why proactive private offsets beat the field primitives.</b> Against communication-free "
  "reductions of recognized MRTA de-confliction baselines given the <i>same</i> SwarmCF utility, a "
  "consensus-auction-with-backoff (CBBA with the consensus step removed) and a musical-chairs "
  "re-seating, the proactive private offset earns more at severe contention ($0.100$ versus $0.064$ for "
  "the auction and $0.028$ for re-seating at pool $15$). A reactive shared backoff makes all colliders "
  "flee the same task together and re-synchronizes them; randomized re-seating adds collisions. A "
  "static private offset instead spreads the robots once and for all. The categorical unseen quality is "
  "untouched by contention (it is learned contention-free), so the contention story is purely "
  "operational: the offset reduces collisions among robots that have <i>accurate</i> preferences.</p>")
A("<p>For operational context, the foundation's mission section reports that %s recovers about "
  "$81\\%%$ of a centralized full-communication ceiling when targets are plentiful and that the residual "
  "gap is precisely this within-round coordination cost; the two centralized ceilings differ little, so "
  "coordination, not estimation, is the binding constraint that this refinement targets." % mp.disp("ContentionAdaCF"))
A("<div class='thm'><b>Supporting result (Theorem 7, companion theory).</b> Under a shared offer pool "
  "with capacity-1 matching and no communication, a deterministic argmax over similar robots collides "
  "on $\\Theta(m-K)$ engagements; a fixed, private continuous offset makes same-type argmaxes almost "
  "surely distinct while preserving value up to $O(\\varepsilon)$. The offset must be both fixed and "
  "private: re-randomized or shared offsets do not yield a stable de-confliction.</div>")
A("<div class='prelim'><b>Preliminary / scope.</b> Theorem 7 proves the fixed-offset case; the "
  "self-tuning, scarcity-gated %s that we deploy is supported empirically (best-or-tied earned reward "
  "at every pool, with the categorical unseen metric recovered) but is not yet covered by a matching "
  "theorem. An adaptive-offset envelope that interpolates from greedy at no contention to the fixed "
  "offset at saturation is the natural object to prove and is left to future work." % mp.disp("ContentionAdaCF"))

# ---------------- 5. rank self-determination ----------------
A("<h2>5. Rank self-determination: removing the guessed rank</h2>")
A("<p><b>The idea.</b> The core method needs a guessed rank $\\hat d$. Although the foundation shows "
  "the result is robust to over-guessing, carrying any rank hyperparameter at all is unsatisfying for a "
  "prior-free method. The refinement makes the swarm <b>learn the rank itself</b> through automatic "
  "relevance determination (ARD), pruning latent directions that the observed, masked design does not "
  "excite. The mechanism is <b>%s</b> (the Bayesian variant with per-column ARD priors)."
  % mp.disp("ARD-EMCF"))
A("<p><b>The mechanism.</b> Each latent column $r$ carries a prior precision $\\alpha_r$ updated by the "
  "variational rule; a column is retained when the observed (masked) design excites that direction with "
  "second-moment energy above the prior and noise floor, and is pruned ($\\alpha_r\\to\\infty$) "
  "otherwise. The recovered effective rank therefore equals the number of latent directions that are "
  "<b>identifiable</b> from what the swarm actually observes, which is at most the generative rank. "
  "Critically the retained set does not depend on the guess $\\hat d$: extra columns are simply pruned, "
  "so the rank knob is removed.</p>")
A("<p><b>The evidence (a stable effective rank, invariant to the guess).</b> Setting the guessed rank "
  "deliberately high, %s recovers an effective rank of about $3.2$ whether the guess is $\\hat d=8$ or "
  "$\\hat d=20$ (the underlying Bayesian variant without ARD instead reports the full guessed rank, "
  "$8.0$ and $20.0$), with no accuracy penalty from the surplus dimensions and an improved anytime "
  "skill ($0.466$ at $\\hat d=20$ versus $0.374$ for the core method). The guess no longer matters: the "
  "method self-determines its working dimension." % mp.disp("ARD-EMCF"))
A("<div class='thm'><b>Supporting result (Theorem 8, companion theory).</b> Under variational PMF with "
  "ARD, a latent direction is retained iff the observed masked design excites it above the prior/noise "
  "floor, so the recovered effective rank equals the identifiable rank ($\\le d$) and, crucially, does "
  "not depend on the guessed $\\hat d$.</div>")
A("<div class='prelim'><b>Preliminary / honest reading.</b> The recovered effective rank is the "
  "<i>identifiable</i> rank, not the raw generative $d$: in a controlled sweep it was about $3.2$ at "
  "$d=5$ and non-monotone across $d\\in\\{2,3,5,8\\}$ because, at a fixed observation budget and noise, "
  "the per-direction signal scales roughly as $1/\\sqrt{d}$, so the weakest directions fall below the "
  "identifiability floor. The solidly supported, usable claim is the invariance to $\\hat d$ (it "
  "removes the knob); a constant-signal-to-noise controlled study of rank recovery remains future "
  "work.</div>")

# ---------------- 6. choice channel ----------------
A("<h2>6. The action / choice channel as a noise-immune alternative</h2>")
A("<p><b>The idea.</b> The core method reads a cardinal reward off the broadcast, which is exactly the "
  "quantity corrupted by per-observer sensing noise. But a robot can often perceive <i>which</i> task a "
  "teammate chose far more reliably than <i>how well</i> it turned out. The refinement learns from the "
  "<b>choice channel</b> alone, who engaged what, treating each observed choice as a noisy-rational "
  "preference signal. The mechanism is <b>%s</b> (collaborative filtering on the choice channel), with "
  "a fusion variant <b>%s</b> that combines reward and competence-weighted choice."
  % (mp.disp("ChoiceCF"), mp.disp("BothCF")))
A("<p><b>The mechanism.</b> An observed choice $a$ in an offered set $S$ is modeled as a mixture: with "
  "some informativeness a Boltzmann-rational pick on the teammate's predicted scores, otherwise "
  "uniform. Because the channel is categorical, it is immune to the cardinal read-off noise $\\sigma$ "
  "entirely: a teammate's choice carries the same information whether the reward is read cleanly or "
  "noisily. The two-channel grid (broadcast rate $\\rho$ against observation noise $\\sigma$) is the "
  "natural place to see when each channel wins; the reward channel is best at low noise and the choice "
  "channel takes over as noise grows.</p>")
A("<p><b>The evidence (a clean noise-immune niche).</b> Sweeping the broadcast reward noise at full "
  "broadcast, the reward channel degrades with noise while the choice channel is flat by construction. "
  "They cross over: at high observation noise ($\\sigma=2.0$) %s beats the reward channel on both "
  "unseen skill ($0.093$ versus $0.042$) and anytime skill ($0.219$ versus $0.179$) with "
  "non-overlapping intervals, whereas at low noise the reward channel leads. Figure 2 shows the "
  "two-channel grid that locates this crossover." % mp.disp("ChoiceCF"))
A("<figure>%s<figcaption><b>Figure 3.</b> The two channels across the observation grid (broadcast rate "
  "$\\rho$ and per-observer reward noise $\\sigma$). The reward channel is strongest when observation "
  "is clean; the action / choice channel, being categorical, is immune to the cardinal read-off noise "
  "and overtakes the reward channel as $\\sigma$ grows. This locates the regime in which a robot should "
  "trust what teammates <i>chose</i> rather than how well it <i>turned out</i>.</figcaption></figure>"
  % img("F7_channels.png", "two-channel grid"))
A("<div class='thm'><b>Supporting result (Proposition 9, companion theory).</b> Per-teammate choice "
  "informativeness is identifiable only <b>held-out</b>: scoring a choice against a model fit to that "
  "same choice spuriously inflates a random teammate's informativeness, whereas scoring it before the "
  "refit incorporates it lets genuinely informative and uninformative teammates separate.</div>")
A("<div class='prelim'><b>Preliminary / honest negative.</b> Learning per-teammate informativeness "
  "jointly by expectation-maximization gave no edge over a fixed competence ramp on skill: a naive "
  "scheme deadlocks (unseen $0.012$) and a rescued, held-out variant fixes the deadlock and ties the "
  "fixed ramp on anytime skill but does not beat it on unseen skill ($0.031$ versus $0.093$). The "
  "robust, supported claim is narrower and is the one we make: the <i>fixed-ramp</i> choice channel is "
  "a genuine noise-immune alternative in the high-noise niche; the learned-gate elaboration is not yet "
  "a win and is reported as a negative.</div>")

# ---------------- 7. non-stationarity / churn ----------------
A("<h2>7. Non-stationarity and team churn</h2>")
A("<p><b>The idea.</b> The foundation assumes a stationary task set. In the field, tasks appear and "
  "expire and robots join and leave. The refinement asks whether the same family stays adapted under "
  "<b>continuous turnover</b>, and the answer is a useful arc: plain exploitative filtering does "
  "<i>not</i> suffice, and the fix is to unite low-rank fold-in with confidence-directed probing of the "
  "uncertain (fresh) entities, the mechanisms of Section 3 (<b>%s</b> and <b>%s</b>)."
  % (mp.disp("EMCF"), mp.disp("ActiveCF")))
A("<p><b>The mechanism and the diagnosis.</b> Under fast churn (a fixed active set with a steady stream "
  "of departures and fresh arrivals), plain %s ties a structure-free learner on the active set "
  "($0.632$ versus $0.619$) and actually trails it on fresh arrivals ($0.074$ versus $0.132$): "
  "collective fold-in needs on the order of $\\hat d$ probes to pin a newcomer, and a purely "
  "exploitative policy never spends them. The diagnosis is therefore not that low-rank structure fails "
  "but that exploitation starves newcomers of the few probes fold-in needs." % mp.disp("RewardCF"))
A("<p><b>The evidence (churn is handled, but only by the united method).</b> Adding confidence-directed "
  "probing reverses the result: %s and %s dominate on both metrics, on the active set %s reaches "
  "$0.842$ versus $0.619$ for the structure-free learner and $0.632$ for plain filtering, and on fresh "
  "arrivals %s and %s reach $0.363$ and $0.371$ versus $0.132$, all with non-overlapping intervals. "
  "Read as an operational readiness metric, a freshly added robot becomes effective on tasks it never "
  "tried after a handful of engagements (order of the latent rank), while a structure-free newcomer "
  "never does. Non-stationarity is thus handled categorically, but the win requires combining low-rank "
  "fold-in with directed newcomer probing; neither structure-free optimism nor exploitative filtering "
  "alone is enough." % (mp.disp("ActiveCF"), mp.disp("EMCF"), mp.disp("EMCF"),
                          mp.disp("ActiveCF"), mp.disp("EMCF")))
A("<figure>%s<figcaption><b>Figure 4.</b> Newcomer cold-start readiness (strict communication-free "
  "setting): a freshly added robot's skill on tasks it never engaged, versus the number of its own "
  "probes, at several broadcast rates. The robot recovers the shared structure from its own masked "
  "broadcast and folds in, reaching population-average competence after a handful of probes (order of "
  "the latent rank), while a structure-free newcomer stays at the floor regardless of probes; the slope "
  "flattens as the broadcast is masked more heavily. This is the readiness that confidence-directed "
  "probing restores under churn.</figcaption></figure>"
  % img("F10_newcomer.png", "newcomer cold-start readiness"))
A("<div class='thm'><b>Supporting results (Theorems 10-11 and Proposition 12, companion theory).</b> "
  "The fold-in perturbation bound splits a newcomer's cold-start error into basis-recovery, own-probe "
  "noise, and ridge bias, exact once it has $\\ge\\hat d$ probes; the collective-speedup law makes "
  "recovery faster as the team grows; and the churn fold-in latency balances probe cost against "
  "turnover rate, explaining why directed probing is needed under fast churn.</div>")
A("<div class='prelim'><b>Preliminary.</b> The churn study uses a single turnover schedule (a fixed "
  "active set with periodic departures and arrivals) at full broadcast; the finite-time coverage rate "
  "under a strongly exploiting, adaptive policy, the same residual the foundation flags for its "
  "recovery theorem, is the binding open question here too.</div>")

# ---------------- 8. unified method ----------------
A("<h2>8. A unified communication-free method</h2>")
A("<p><b>The idea.</b> The refinements above are specialists: confidence-directed exploration for "
  "sample efficiency and churn, a private offset for contention, ARD for the rank. A practical swarm "
  "should not be re-tuned per regime. The capstone refinement folds them into <b>one</b> method whose "
  "components <b>activate only on their triggering condition</b>. The mechanism is <b>%s</b> (the "
  "Bayesian variant with a loss-self-gated de-confliction offset and a loss-gated exploration anneal, "
  "plus an abundance gate that damps exploration when tasks are plentiful)." % mp.disp("UnifiedCF"))
A("<p><b>The mechanism.</b> %s runs the confidence-directed estimator of Section 3; its de-confliction "
  "offset (Section 4) is scaled by the robot's own recent loss rate, so it is dormant when the robot is "
  "winning its engagements and engages only under contention; its exploration anneal is gated by the "
  "same loss signal; and an abundance gate damps the upper-confidence exploration when the offered set "
  "is large (no scarcity), where exploration costs earned reward under capacity-1. Every gate is driven "
  "by quantities the robot already observes, so the method remains communication-free." % mp.disp("UnifiedCF"))
A("<p><b>The evidence (one policy, best-or-tied across regimes).</b> Against the per-regime specialists "
  "on the same seeds, %s ties or wins in all of the standard, churn, and contention regimes: standard "
  "anytime skill $0.437$ (specialist %s $0.433$); churn active-set $0.851$ and recent-arrival $0.347$ "
  "(specialist %s $0.842$ and $0.371$); contention earned reward $0.104$ at the most contended pool "
  "(specialist %s $0.100$, roughly double the greedy $0.059$). The abundance gate closes the one "
  "residual, no-contention earned reward, lifting pool-240 earned reward from $0.344$ to $0.425$ "
  "(greedy $0.439$) while leaving the small-offer regimes byte-identical because the gate fires only "
  "when the offer is large. One communication-free policy, with no per-regime tuning, is therefore "
  "best-or-statistically-tied everywhere we tested."
  % (mp.disp("UnifiedCF"), mp.disp("EMCF"), mp.disp("EMCF"), mp.disp("ContentionAdaCF")))
A("<div class='prelim'><b>Preliminary.</b> The unified method is validated against specialists on the "
  "standard, churn, and contention regimes at full broadcast; it has not been swept jointly over "
  "masking, noise, team size, and rank, and the gating thresholds (loss gate, abundance gate) are "
  "fixed rather than learned. A theoretical envelope for the gated policy is sketched in the companion "
  "theory (the loss-and-abundance-gated envelope) but not fully proved.</div>")

# ---------------- 9. discussion ----------------
A("<h2>9. Discussion: what carries over from the foundation</h2>")
A("<p><b>The categorical baseline is the through-line.</b> Every refinement is measured against the "
  "same structure-free floor the foundation proves, and in every case the categorical unseen-pair "
  "advantage is preserved: confidence keeps the broadcast at full weight rather than trading "
  "generalization for noise-awareness; the contention offset is value-preserving up to $O(\\varepsilon)$ "
  "and leaves the learned preferences (hence the unseen metric) intact; ARD prunes only unidentifiable "
  "directions; the choice channel is a different but still structure-exploiting signal; and the churn "
  "fix restores generalization to newcomers precisely by feeding the shared structure. The refinements "
  "add operational competence (sample efficiency, de-confliction, hyperparameter removal, noise "
  "immunity, adaptivity) <i>on top of</i> the categorical capability, not in place of it.</p>")
A("<p><b>The communication-free constraint carries over too.</b> No refinement reintroduces "
  "communication: the Bayesian posterior is local to each robot, the de-confliction offset is private "
  "and fixed, ARD is a per-robot prior, the choice channel is read from the same passive stream, and "
  "the churn and unified methods gate on locally observed quantities. The reference ceilings remain the "
  "only place where full communication appears. The same collective-speedup intuition also recurs: "
  "directed exploration and newcomer probing work because the swarm's pooled observations feed one "
  "shared structure, the mechanism the foundation's Theorem 4 makes precise.</p>")
A("<p><b>A single deployable method.</b> Taken together, the refinements collapse into %s: one "
  "communication-free policy that generalizes to the unseen, explores by confidence, de-conflicts under "
  "contention, self-determines its rank, can fall back to the noise-immune choice channel, and stays "
  "adapted under churn, with each capability dormant until its condition fires." % mp.disp("UnifiedCF"))

# ---------------- 10. limitations ----------------
A("<h2>10. Limitations and the status of these results</h2>")
A("<p><b>Status.</b> This is a draft follow-up. The companion paper's headline claims carry full "
  "theory and multi-seed validation with bootstrap confidence intervals; the results here are grounded "
  "in logged experiments (typically 6-12 seeds with bootstrap intervals) but several are explicitly "
  "<b>preliminary</b>, as flagged in the relevant sections: the self-tuning contention offset lacks a "
  "matching theorem (only the fixed offset is proved); ARD recovers the identifiable, not the raw, "
  "rank and wants a constant-signal-to-noise study; the learned choice-informativeness gate is an "
  "honest negative and only the fixed-ramp choice channel is a confirmed win; the churn and unified "
  "studies each use a single schedule at full broadcast and have not been swept jointly over the full "
  "operating grid.</p>")
A("<p><b>Inherited assumptions.</b> All refinements inherit the foundation's premises: the reward is "
  "(approximately) low-rank and the advantage requires structure beyond mere popularity ($d>1$), task "
  "scarcity ($n\\gg T$), and a shared channel ($\\rho>0$); rewards are real-valued and bilinear in "
  "latent traits; and the recovery rate is established for non-adaptive exploration, with the "
  "finite-time rate under a strongly exploiting policy still open. The refinements relax none of these; "
  "they add capabilities within the same regime.</p>")
A("<p><b>What remains.</b> The clearest next steps are an adaptive-offset envelope theorem covering the "
  "deployed contention method, a constant-signal-to-noise study of ARD rank recovery, a reward-value "
  "(not merely predictability) signal for the choice channel to handle consistently-wrong teammates, a "
  "joint sweep of the unified method over masking, noise, team size, and rank, and the finite-time "
  "adaptive coverage rate that both papers flag. A hardware and physics-based study, and external "
  "validation on an independent third-party benchmark, would test the refinements as the foundation "
  "was tested on the LatentSwarm simulator.</p>")

# ---------------- declarations ----------------
A("<h2>Declaration of competing interest</h2>")
A("<p class='small'>The authors declare that they have no known competing financial interests or "
  "personal relationships that could have appeared to influence the work reported in this paper.</p>")
A("<h2>Data availability</h2>")
A("<p class='small'>The source code, the simulation harness, and the per-seed data required to "
  "reproduce every figure and result in this paper are openly available at "
  "<a href='https://github.com/ApartsinProjects/ZKDroneSwarm'>github.com/ApartsinProjects/ZKDroneSwarm</a>. "
  "This follow-up is a companion to &ldquo;Acting on the Unseen: Communication-Free Collaborative "
  "Filtering for Decentralized Multi-Robot Task Allocation&rdquo;, which establishes the core result "
  "and the formal theory cited here.</p>")

A("</div></body></html>")
html_str = "\n".join(H)
open(OUT, "w", encoding="utf-8").write(html_str)
print("wrote", OUT, "(%d KB)" % (len(html_str.encode("utf-8")) // 1024))
