"""Generate the STREAMLINED v2 paper as a self-contained HTML for GitHub Pages:
a clean linear narrative (minimal-assumptions framing) from motivation through
theory and experiments. Key figures base64-embedded. Output: docs/paper_v2.html.
"""
import base64, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures")
OUT = os.path.join(ROOT, "docs", "paper_v2.html")


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
    if path and not os.path.isabs(path):
        path = os.path.join(ROOT, path)          # resolve docs/*.md against repo root (cwd-independent)
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
  "(<code>Theta(d)</code> vs <code>Theta(n)</code>); the cold-start holds even for a strictly "
  "zero-knowledge newcomer that recovers the target basis from its OWN masked broadcast rather than "
  "receiving any peer's model, with the separation degrading gracefully as its visibility falls. "
  "(3) An anytime separation under sample "
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
A("<p><b>One core, scoped extensions (read this if nothing else).</b> The hero is a SINGLE estimator, "
  "online noise-weighted ridge ALS over the public stream, with an $\\epsilon$-greedy policy. "
  "Everything else, Bayesian confidence (EMCF), directed exploration, rank self-determination (ARD), "
  "contention de-confliction, and the choice channel, is a SCOPED EXTENSION of that one estimator: each "
  "is applied only when a stated condition holds and otherwise reduces to the core. We present them as a "
  "single theorem-backed menu (Section 5) rather than a method zoo, so the whole design space collapses "
  "to one method plus a short list of conditional add-ons.</p>")

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
A("<p><b>Recent results and honest scope.</b> <b>T9 (precise scope iff):</b> the CATEGORICAL "
  "unseen gap vs a structure-free learner needs only a shared channel ($\\rho>0$) and some structure; "
  "$d>1$ is what additionally beats a POPULARITY baseline (T5); sample-starvation is what makes the "
  "unseen advantage OPERATIONALLY dominant (T3), three conditions with three distinct roles, not one "
  "AND. <b>Confidence:</b> precision fit-weighting is suboptimal for unseen prediction (it over-weights "
  "the drone's own, here irrelevant, data); the Bayesian posterior is the right object (P6), and "
  "<b>P10</b> sharpens it, bounded ratio-capped precision dominates uniform exactly when teammate "
  "sources DIFFER in reliability, while unbounded $1/\\sigma^2$ starves coverage. <b>Exploration:</b> "
  "EMCF's predictive-variance UCB is LinUCB on the recovered latent features, giving $\\tilde O(d\\sqrt "
  "T)$ per-drone regret given $U$ (P16). <b>Contention</b> = fixed private offset (T7); <b>rank</b> = "
  "ARD recovers the identifiable rank, removing the hyperparameter (T8); <b>choice channel</b> = "
  "held-out informativeness is identifiable where in-sample is not (P9). <b>Honest keystone:</b> the "
  "per-drone recovery of $U$ from the STRUCTURED (persistent-masked) broadcast is currently invoked "
  "from centralized completion theory, not yet proven for non-uniform masking, the main open problem "
  "(full statements + proofs and a critical self-audit in <code>THEORY_FORMAL.md</code>).</p>")

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
A("<p><b>Out-of-harness transfer (independent simulator).</b> Beyond our generative code, we drop the "
  "method as a drop-in policy into the separately-built tabula_drone PettingZoo environment, which has "
  "dynamics our model lacks: spatial geometry, target health that DEPLETES as engaged (a soft "
  "contention/matching pressure), episodic resets, and a Hungarian optimal-assignment oracle. We "
  "benchmark against the environment's own policies (Fig. 3b). On the efficiency metric (reward per "
  "step), our method reaches skill ~0.81 with low variance, beating the environment's SGD "
  "matrix-factorization (~0.25) and UCBIndep (~0.72) and approaching the matching oracle. This tests "
  "TRANSFER and partial-contention robustness, not real-world realism (it is still a simulator, and "
  "itself latent-based, so it is not an independent test of low-rankness). For the low-rank assumption "
  "itself we stress it directly (approximate low-rank via entrywise noise; a nonlinear link that "
  "raises the effective rank) and observe graceful degradation with the ranking preserved.</p>")
A("<figure>%s<figcaption><b>Fig. 3b.</b> Independent tabula_drone simulator (spatial, depleting HP, "
  "episodic; matching oracle): converged skill and learning curves; ours beats the env's SGD-MF and "
  "UCBIndep, approaching the oracle.</figcaption></figure>"
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

A("<h3>Scoped refinements: one core, a menu of conditional extensions</h3>")
A("<p>To avoid a method zoo, every refinement beyond the core estimator is presented as a single "
  "MENU of scoped, theorem-backed extensions. Each helps ONLY under a stated condition and reduces "
  "to the core otherwise, so the practitioner adds an extension exactly when its condition holds. "
  "This is the entire design space in one view.</p>")
A("<table><tr><th class='l'>Refinement</th><th class='l'>Helps when</th><th class='l'>Result (8-20 seeds, CIs)</th><th>Theory</th></tr>"
  "<tr><td class='l'>Bayesian confidence (EMCF)</td><td class='l'>directed exploration; coverage-scarce or noisy broadcast</td>"
  "<td class='l'>predictive uncertainty is DISCRIMINATIVE (actual RMSE rises monotonically Q1$\\to$Q5), and beats uniform when the broadcast is noisy</td><td>P6</td></tr>"
  "<tr><td class='l'>Bounded precision weighting</td><td class='l'>observation sources DIFFER in noise</td>"
  "<td class='l'>ratio-capped precision wins under heterogeneous noise ($+0.09$ unseen, non-overlapping CIs); UNBOUNDED $1/\\sigma^2$ over-concentrates and loses everywhere</td><td>P6</td></tr>"
  "<tr><td class='l'>Collective info-directed exploration</td><td class='l'>anytime sample efficiency</td>"
  "<td class='l'>one agent's broadcast probe lowers everyone's uncertainty; best FINAL anytime skill</td><td>--</td></tr>"
  "<tr><td class='l'>ARD rank adaptation</td><td class='l'>rank unknown</td>"
  "<td class='l'>self-determines the IDENTIFIABLE rank ($\\le d$) and is INVARIANT to the guessed $\\hat d$, which is what removes the rank hyperparameter (the recovered value reflects SNR/coverage, not the raw true rank)</td><td>T8</td></tr>"
  "<tr><td class='l'>Fixed private offset ($+$ adaptive)</td><td class='l'>capacity-1 contention</td>"
  "<td class='l'>$\\approx 2\\times$ earned reward at severe contention; the adaptive, scarcity-gated version extends the win across the whole contested range ($|S|\\le m$)</td><td>T7</td></tr>"
  "<tr><td class='l'>Choice channel $+$ held-out $\\gamma$</td><td class='l'>extreme reward-noise; mixed-reliability teammates</td>"
  "<td class='l'>noise-immune fallback (overtakes the reward channel at $\\sigma_{\\mathrm{obs}}{=}2$); held-out $\\gamma_k$ correctly ranks teammates (oracle $0.49\\gg$ random $0.10$)</td><td>P9</td></tr>"
  "</table>")
A("<p class='small'><b>What does NOT help (honest, diagnosed nulls):</b> UNBOUNDED precision-ALS at "
  "homogeneous noise (coverage, not noise, binds); a LEARNED per-teammate choice gate vs a simple "
  "competence schedule when teammates are homogeneous (it ties, and in-sample it over-trusts random "
  "teammates, the bug held-out $\\gamma$ fixes); confidence-GATED reward$+$choice fusion. Each is a "
  "documented null with a stated cause, several surfaced by deliberate SANITY experiments whose "
  "answer is obvious (oracle-vs-random teammates; homogeneous-vs-heterogeneous noise).</p>")

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
  "address. Low-rank / bilinear bandits (explore-then-commit spectral ESTR; probe-then-fit; rank-1 "
  "and generalized low-rank matrix bandits) are centralized and/or phase-structured; we are anytime "
  "and distributed. Bayesian PMF over-explores in the anytime regime. Federated/gossip CF shares "
  "factors; we share nothing but a passively-sensed public stream. The CLOSEST neighbor is multi-user "
  "RL with low-rank rewards (Nagaraj/Agarwal 2022), which shares the low-rank-across-agents structure "
  "but CENTRALIZES trajectory aggregation, whereas we keep parameters private. CF-bandits / "
  "clustering-of-bandits (CLUB, COFIBA, BanditMF) cluster users centrally and optimize regret on "
  "PULLED arms, not unseen-pair generalization. For contention our communication-free "
  "symmetry-breaking parallels no-communication multiplayer bandits but acts on a LEARNED low-rank "
  "preference. MRTA (auction / consensus CBBA / DCOP) and cooperative MARL (CTDE: MAPPO/QMIX/VDN) "
  "assume communication, known utilities, or centralized training; the admissible no-comms case "
  "reduces to our per-agent bandit baselines.</p>")
A("<p class='small'><b>Honest positioning.</b> Every individual ingredient (low-rank bandits, online "
  "MF, observe-neighbors bandits, no-comms multiplayer matching, Dawid-Skene reliability) exists in "
  "prior work; the contribution is COMPOSITIONAL, this exact combination (decentralized + private "
  "parameters + broadcast-only + online + sample-starved, with an unseen-pair categorical separation) "
  "is the open niche, and the genuinely new mechanisms are the masking-robust zero-weight (not "
  "zero-value) estimator, the anytime separation under starvation, the comms-free de-confliction on a "
  "learned preference, and held-out choice-informativeness. We do not claim a new estimator.</p>")

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
  "become the bottleneck. And the operational loss is recoverable WITHOUT communication: keeping the "
  "CF estimate but swapping greedy argmax for a FIXED PRIVATE per-target offset (Theorem 7) "
  "de-conflicts look-alike drones and roughly DOUBLES earned reward at severe contention; an adaptive, "
  "scarcity-gated version of that offset extends the win across the contested range and reduces to "
  "plain CF when targets are plentiful.</p>")
A(md_tables("docs/CONTENTION.md"))
A("<p><b>Non-stationarity (churn).</b> When targets continuously turn over (departures plus fresh "
  "arrivals), PLAIN exploitative CF does not suffice: fold-in's order-$d$ probe latency is outpaced by "
  "churn on the freshest arrivals, where exploitative CF even trails a structure-free optimist "
  "($0.074$ vs $0.132$). The fix is the SAME confidence-directed exploration from the refinements "
  "menu: a CF estimator that PROBES uncertain (fresh) targets and folds them in (ActiveCF count-bonus, "
  "or EMCF predictive-variance UCB) DOMINATES under churn on both the active set and the freshest "
  "arrivals (EMCF active $0.842$ vs $0.619$ structure-free; fresh arrivals $0.371$ vs $0.132$, "
  "non-overlapping CIs). The swarm stays adapted under continuous change, provided it unites low-rank "
  "fold-in with directed newcomer-probing, neither alone suffices.</p>")
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
