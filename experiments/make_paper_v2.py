"""Generate the STREAMLINED v2 paper as a self-contained HTML for GitHub Pages:
a clean linear narrative (minimal-assumptions framing) from motivation through
theory and experiments. Key figures base64-embedded. Output: docs/paper_v2.html.
"""
import base64, os
FIG = "docs/figures"
OUT = "docs/paper_v2.html"


def img(name, alt):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return "<p><em>[missing %s]</em></p>" % name
    b = base64.b64encode(open(p, "rb").read()).decode("ascii")
    return ('<img alt="%s" src="data:image/png;base64,%s" '
            'style="max-width:88%%;height:auto;border:1px solid #d7dde3;border-radius:6px">' % (alt, b))


def _bold(s):
    while "**" in s:
        s = s.replace("**", "<b>", 1).replace("**", "</b>", 1)
    return s.strip()


def md_tables(path, drop_h1=True):
    """Render the SUBSET of markdown our experiments emit (## headers, | tables |,
    **bold**, inline <br/>/<sub> kept) as HTML, so the paper draws headline/ablation/
    contention tables from the SAME committed .md files (one source of truth)."""
    if not path or not os.path.exists(path):
        return "<p><em>[%s not generated yet]</em></p>" % os.path.basename(str(path))
    out = []; rows = []

    def flush():
        if not rows:
            return
        cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
        h = "".join("<th>%s</th>" % _bold(c) for c in cells[0])
        body = "".join("<tr>%s</tr>" % "".join(
            "<td class='l'>%s</td>" % _bold(c) if k == 0 else "<td>%s</td>" % _bold(c)
            for k, c in enumerate(r)) for r in cells[1:])
        out.append("<table><tr>%s</tr>%s</table>" % (h, body)); rows.clear()

    for ln in open(path, encoding="utf-8").read().splitlines():
        s = ln.rstrip()
        if s.startswith("|") and set(s.replace("|", "").replace("-", "").strip()) == set():
            continue                       # separator row |---|
        if s.startswith("|"):
            rows.append(s); continue
        flush()
        if s.startswith("## "):
            out.append("<h3>%s</h3>" % _bold(s[3:]))
        elif s.startswith("# "):
            if not drop_h1:
                out.append("<h3>%s</h3>" % _bold(s[2:]))
        elif s.strip():
            out.append("<p class='small'>%s</p>" % _bold(s.strip()))
    flush()
    return "\n".join(out)


CSS = """
body{margin:0;color:#16191d;background:#fff;font:16px/1.66 Georgia,'Times New Roman',serif}
.wrap{max-width:820px;margin:0 auto;padding:34px 22px 80px}
h1{font-size:27px;line-height:1.2;margin:0 0 6px;font-family:-apple-system,Segoe UI,Roboto,sans-serif}
h2{font-size:21px;margin:34px 0 8px;font-family:-apple-system,Segoe UI,Roboto,sans-serif;border-bottom:2px solid #e2e8ee;padding-bottom:3px}
h3{font-size:17px;margin:20px 0 6px;color:#1f5fa8;font-family:-apple-system,Segoe UI,Roboto,sans-serif}
.sub{color:#5b6570;font-size:15px;margin:0 0 16px;font-family:-apple-system,Segoe UI,Roboto,sans-serif}
p{margin:9px 0}code{background:#f5f8fb;padding:1px 4px;border-radius:4px;font:13px/1.4 Consolas,monospace}
.abs{background:#f5f8fb;border:1px solid #e2e8ee;border-radius:8px;padding:12px 18px;margin:16px 0;font-size:15px}
.box{background:#f7fbf8;border-left:4px solid #0a7d4d;border:1px solid #e2e8ee;border-radius:8px;padding:10px 16px;margin:14px 0}
figure{margin:16px 0;text-align:center}figcaption{color:#5b6570;font-size:13px;margin-top:6px;text-align:left}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px;font-family:-apple-system,Segoe UI,sans-serif}
th,td{border:1px solid #e2e8ee;padding:5px 8px;text-align:center}th{background:#f5f8fb}td.l,th.l{text-align:left}
.ours{font-weight:700;color:#0a7d4d}.small{font-size:13px;color:#5b6570}
a{color:#1f5fa8}
"""
H = []
A = H.append
A("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
  "content='width=device-width,initial-scale=1'><title>Acting on the Unseen (v2)</title>"
  "<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css'>"
  "<style>%s</style></head><body><div class='wrap'>" % CSS)
A("<h1>Acting on the Unseen: Collaborative Filtering for Decentralized Multi-Robot Task Allocation "
  "under Minimal Assumptions</h1>")
A("<p class='sub'>Streamlined draft (v2). Companion tutorial: <a href='tutorial.html'>tutorial.html</a>. "
  "Repo: <a href='https://github.com/ApartsinProjects/ZKDroneSwarm'>ApartsinProjects/ZKDroneSwarm</a>. "
  "All numbers regenerable from saved per-seed data.</p>")

A("<div class='abs'><b>Abstract.</b> We study multi-robot task allocation under the LEAST possible "
  "information: a swarm with NO prior knowledge (no labels, no task models, no known latent structure, "
  "not even the problem's rank), ZERO communication (no messages, no parameter sharing, no "
  "coordinator), and only PARTIAL, NOISY observation of a public outcome stream, with every decision "
  "made independently and distributedly. We show that, even so, each agent can act well on tasks it "
  "has NEVER tried, onboard brand-new tasks, and welcome new agents, by running online collaborative "
  "filtering over the public broadcast. Against any structure-free learner this is a CATEGORICAL "
  "separation, not a constant factor (the structure-free learner is at the error floor on unseen "
  "pairs by construction), and we prove a matching per-agent sample-complexity theory "
  "($\\Theta(d)$ vs $\\Theta(n)$) plus an anytime (cumulative-reward) separation. "
  "Against the field of low-rank methods our online, masking-robust, anytime-optimal estimators "
  "dominate throughout the limited-observability regime that defines the problem.</div>")

A("<h2>1. Introduction</h2>")
A("<div class='box'><b>Minimal assumptions.</b> No prior knowledge. Zero communication. Only partial, "
  "noisy observables. Fully distributed decisions. Can a swarm still act intelligently, in particular "
  "on the unseen? We answer yes, and prove the advantage is categorical.</div>")
A("<p>A swarm repeatedly chooses which target to engage. Compatibility between agents and targets is "
  "hidden but low-rank; there are far more targets than rounds (<code>n &gt;&gt; T</code>); agents "
  "neither coordinate nor share parameters and each senses only a noisy, partial slice of the public "
  "outcomes. The crux is GENERALIZATION: a structure-free learner that estimates a target only from "
  "its own engagements is, on any target it never touched, at the prior-mean error floor, which "
  "dominates when targets outnumber rounds.</p>")
A("<p><b>Contributions.</b> (1) A minimal-assumption, decentralized, online, broadcast-only "
  "formulation of CF for MRTA. (2) The categorical unseen-pair and onboarding/cold-start results "
  "(<code>Theta(d)</code> vs <code>Theta(n)</code>). (3) An anytime separation under sample "
  "starvation. (4) A masking-model dichotomy (durable vs transient decentralization). (5) A method "
  "family that is masking-robust and anytime-optimal, dominating the prior-art low-rank competitors; "
  "and collective active exploration via the shared broadcast.</p>")

A("<h2>2. Setting</h2>")
A("<p><b>World.</b> $m$ drones, $n$ targets, with $K_1,K_2$ latent types; the reward is the cosine of "
  "unit latent factors $p_i,u_j\\in\\mathbb{R}^d$,</p>")
A("<p style='text-align:center'>$R_{ij} = \\langle p_i, u_j\\rangle \\in [-1,1], \\qquad R = PU^\\top, "
  "\\qquad \\operatorname{rank}(R)=\\min(d,K_1,K_2)$ (no nonlinear link).</p>")
A("<p><b>Observability.</b> A public stream of (action, outcome); each agent senses its own outcome "
  "cleanly (noise $\\sigma_{\\mathrm{own}}$) and a per-agent masked (rate $\\rho$), noisy (noise "
  "$\\sigma_{\\mathrm{obs}}$) slice of teammates' outcomes (passive sensing, not transmission, so "
  "communication-free). The observation precision is $\\beta=1/\\sigma^2$, with masked events absent "
  "(weight $0$). "
  "<b>Baselines.</b> Random, UCBIndep, UCBHomo, Tabular (no-structure); MFSGD, ESTR, PTF, BPMF "
  "(low-rank). <b>Fairness/ZK.</b> Every method gets a guessed rank (or none), an independent "
  "per-agent instance with no parameter sharing, and the same noisy partial stream; the Oracle "
  "(centralized, complete information) only normalizes scores. <b>Metrics.</b> normalized skill, "
  "unseen-pair skill, anytime cumulative-reward skill, state-uniqueness.</p>")

A("<h2>3. Method</h2>")
A("<p>Each agent runs its own ONLINE weighted alternating least squares over the events it senses. "
  "An unobserved/masked event gets zero WEIGHT, not an imputed zero VALUE; this is what makes the "
  "estimator masking-robust (batch fill-in methods impute zeros for missing entries and decay under "
  "masking). Among OBSERVED events, inverse-variance (<code>1/sigma^2</code>) weighting is an optional "
  "noise-adaptive refinement that pays off when the broadcast is noisy; at low noise uniform weighting "
  "generalizes slightly better to unseen pairs (we ablate this honestly). Estimation is separated from "
  "the decision policy. The "
  "family: <b>RewardCF</b> (teammates' rewards; anytime-optimal), <b>ChoiceZK</b> (teammates' actions "
  "only; noise-immune fallback), <b>HybridCFconv</b> (probe, SVD warm-start, then converged online "
  "ALS; best final-policy unseen), and <b>ActiveCFconv</b> (latent-space UCB with a broadcast "
  "count-based uncertainty bonus, so one agent's probe lowers everyone's uncertainty; best "
  "balanced).</p>")

A("<h2>4. Theory</h2>")
A("<p>(Full proofs in <code>THEORY_FORMAL.md</code>; step-by-step in the tutorial.)</p>")
A("<p><b>T1 (tabular floor).</b> A structure-free learner has $\\Omega(1)$ error and $0$ skill on any "
  "unseen pair, and the broadcast is provably useless to it; pooling recovers only the rank-1 "
  "popularity term. <b>T2 (CF row completion).</b> Given $U$, an agent's whole row is recovered exactly "
  "by an $O(d)$ least-squares fold-in; per-agent $\\Theta(d)$ vs tabular $\\Theta(n)$, "
  "categorical on unseen pairs. <b>T3 (anytime).</b> Under $n \\gg T$ any structure-free learner's "
  "anytime skill $\\le g(cT/n) \\to 0$ (even with full broadcast); CF reaches near-oracle in $O(d)$ "
  "rounds. <b>T4 (masking).</b> i.i.d. loss makes decentralization transient; persistent masking makes "
  "it durable; the categorical results are invariant to the choice. <b>T5 (additive ceiling).</b> An "
  "additive predictor $a+b_i+c_j$ has rank $\\le 2$ and reduces to the popularity order, so it cannot "
  "personalize and is strictly below CF for $d > 1$.</p>")

A("<h2>5. Experiments</h2>")
A("<p><b>Categorical (acting on the unseen).</b> CF acts well on never-observed pairs at every "
  "observation density while structure-free learners sit at the floor (Fig. 1); new targets onboard "
  "for the whole swarm from ~d shared probes and new agents from ~d own probes (vs ~n for tabular).</p>")
A("<figure>%s<figcaption><b>Fig. 1.</b> Unseen-pair skill vs observation density: CF acts on the "
  "unseen at every density; structure-free is pinned at the floor.</figcaption></figure>"
  % img("F2_unseen_masking.png", "F2"))
A("<p><b>Operational (anytime).</b> On reward actually earned over time, our online methods dominate "
  "at every horizon; per-arm bandits are stuck near random because, with n &gt;&gt; T, they never "
  "stop exploring untried arms (Fig. 2).</p>")
A("<figure>%s<figcaption><b>Fig. 2.</b> Anytime cumulative reward: online CF earns from round one; "
  "explore-then-commit pays a probe phase; UCBIndep is stuck.</figcaption></figure>"
  % img("F6_anytime.png", "F6"))
A("<p><b>Robustness.</b> Results are invariant to the masking model (i.i.d. vs persistent), while "
  "decentralization is durable only under persistent masking (Fig. 3); and they hold across rank, "
  "horizon, target/agent counts, guessed rank, cluster structure, and latent spread.</p>")
A("<figure>%s<figcaption><b>Fig. 3.</b> Persistent vs i.i.d. masking: unseen/anytime invariant; "
  "state-uniqueness durable (persistent) vs transient (i.i.d.).</figcaption></figure>"
  % img("F8_iid_vs_persistent.png", "F8"))
A("<p><b>Real-simulator validation.</b> Beyond our generative model, we drop the method into the "
  "tabula_drone PettingZoo environment (spatial targets, depleting target health, episodic resets) "
  "and benchmark against its own policies (Fig. 3b). On the efficiency metric (reward per step), our "
  "method reaches skill ~0.81 with low variance, beating the environment's SGD matrix-factorization "
  "(~0.25) and UCBIndep (~0.72) and approaching the oracle. We also stress the assumptions directly "
  "(approximate low-rank via entrywise noise; a nonlinear link that raises the effective rank) and "
  "observe graceful degradation with the ranking preserved.</p>")
A("<figure>%s<figcaption><b>Fig. 3b.</b> Real tabula_drone simulator: converged skill and learning "
  "curves; ours beats the env's SGD-MF and UCBIndep, approaching the oracle.</figcaption></figure>"
  % img("F13_realsim.png", "F13"))
A("<p><b>Versus the field.</b> The categorical unseen win is shared by all low-rank methods over "
  "no-structure ones (a property of structure). Our specific edge is masking-robustness and anytime "
  "optimality; with a converged estimator our methods dominate the strongest competitor (PTF) on "
  "every metric and density (Fig. 4). We further compare against a broader, fair set, run under the "
  "same limits: SoftImpute (nuclear-norm convex completion), kNN-CF (memory-based, model-free), and "
  "BiasModel (additive, no interaction). SoftImpute is the strongest at full broadcast (it tops unseen "
  "at rho=1) but, like all batch fill-in methods, decays under masking and loses on anytime; kNN-CF "
  "generalizes but earns little; BiasModel sits at the additive ceiling (Theorem 5). Under masking and "
  "on anytime, our methods beat the best of these with non-overlapping CIs.</p>")
A("<figure>%s<figcaption><b>Fig. 4.</b> Pareto frontier (anytime vs unseen): our methods dominate; "
  "PTF trades all anytime for unseen and is dominated under masking.</figcaption></figure>"
  % img("F11_pareto.png", "F11"))
A("<p><b>Headline numbers (20 seeds, bootstrap 95%% CI).</b> The unseen-pair and anytime "
  "skill of the no-structure floor (UCBIndep), the strongest competitor (PTF), and our "
  "methods, at full broadcast (rho=1) and heavy masking (rho=0.25):</p>")
A(md_tables("docs/HEADLINE_TABLE.md"))
A("<h3>Method ablation</h3>")
A("<p>We foreground ONE recommended method (ActiveCFconv) and present the rest as "
  "ablations rather than a method zoo. The table below isolates each design choice "
  "(online vs batch, precision weighting, SVD warm-start, active vs eps-greedy exploration) "
  "and the robustness to a guessed rank, under one identical config (12 seeds, 95%% CI):</p>")
A(md_tables("docs/ABLATION_TABLE.md"))

A("<div class='box'><b>Scope: when does CF beat structure-free?</b> The advantage is not "
  "universal. Across a structure-by-observability grid it holds precisely when THREE "
  "conditions co-occur. <b>(1) Low-rank but PERSONALIZED</b> (1 &lt; d &lt;&lt; min(m,n)): "
  "the rank-1 part of the reward is shared 'popularity' that a pooling/bias baseline already "
  "captures, so at d=1 there is nothing personal to transfer (Theorem 5); at the other "
  "extreme, d near full means no structure to exploit (graceful collapse, Fig. on stress). "
  "Personalization is the interaction BEYOND popularity that only a rank&gt;1 model can "
  "represent. <b>(2) SAMPLE-STARVED</b> (n &gt;&gt; cT): if instead sample-rich, a tabular "
  "learner eventually measures every entry directly and the unseen advantage vanishes. "
  "<b>(3) SHARED reward channel</b> (rho &gt; 0): without a broadcast each agent has only its "
  "own outcomes and CF degenerates to tabular. This scopes the contribution and explains why "
  "structure-free methods are competitive in sample-rich, low-personalization, or unshared "
  "settings, our earlier null results.</div>")

A("<h2>6. Related work</h2>")
A("<p>Matrix completion (Candes-Recht; Keshavan-Montanari-Oh; Recht) gives centralized estimation "
  "bounds; we make them decentralized, online, and broadcast-only, with the unseen-pair floor making "
  "the gap categorical and an anytime (decision-reward) separation the completion theory does not "
  "address. Low-rank bandits (explore-then-commit spectral; probe-then-fit) are centralized and/or "
  "phase-structured; we are anytime and distributed. Bayesian PMF over-explores in the anytime "
  "regime. Federated/gossip CF shares factors; we share nothing but a passively-sensed public "
  "stream.</p>")

A("<h2>7. Limitations and conclusion</h2>")
A("<p><b>Contention.</b> Our formal results assume non-contention (a target may be engaged "
  "by more than one drone). Hard contention turns allocation into matching, where explicit "
  "coordination may add value beyond preference quality. We test this directly with a "
  "capacity-1 matching variant (each engaged target is awarded to one drone; collisions earn "
  "nothing; earned reward is normalized by the per-round Hungarian optimum). Two things hold: "
  "the CATEGORICAL preference quality (unseen-pair skill, evaluated contention-free) is "
  "contention-invariant because it is a property of the learned model; and on EARNED reward "
  "our methods still lead, because diverse, accurate preferences spread drones across targets "
  "and reduce collisions, though the operational gap narrows as collisions, not preferences, "
  "become the bottleneck.</p>")
A(md_tables("docs/CONTENTION.md"))
A("<p>Under minimal assumptions, collaborative filtering over the public broadcast lets a "
  "swarm act on the unseen, onboard new tasks and agents, and earn from the first round, with "
  "a categorical, theory-backed advantage over structure-free learning and dominance over "
  "prior-art low-rank methods in the limited-observability regime.</p>")
A("<p class='small'>Reproducibility: complete per-seed data in <code>results/pilots/</code>; figures "
  "via <code>experiments/make_figures.py</code>; proofs in <code>docs/THEORY_FORMAL.md</code>; ZK "
  "audit in <code>docs/ZK_COMPLIANCE.md</code>; full v1 draft in <code>docs/PAPER_DRAFT.md</code>.</p>")
A("</div>")
A("<script defer src='https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js'></script>")
A("<script defer src='https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js' "
  "onload='renderMathInElement(document.body,{delimiters:[{left:\"$$\",right:\"$$\",display:true},"
  "{left:\"$\",right:\"$\",display:false},{left:\"\\\\(\",right:\"\\\\)\",display:false},"
  "{left:\"\\\\[\",right:\"\\\\]\",display:true}],throwOnError:false})'></script>")
A("</body></html>")
open(OUT, "w", encoding="utf-8").write("\n".join(H))
print("wrote %s (%d KB)" % (OUT, len("\n".join(H).encode("utf-8")) // 1024))
