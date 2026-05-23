"""Generate the FOCUSED first journal paper (target: Robotics and Autonomous Systems) as a
self-contained HTML with KaTeX math and pseudocode callout boxes. This is the GOLD subset:
the categorical communication-free collaborative-filtering result, the necessary theory, and
the strongest empirical evidence; advanced machinery is deferred to "future work".

Output: docs/ras_paper.html. Run from REPO ROOT (reads docs/figures/*.png, imports method_profiles).
"""
import base64, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import method_profiles as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures")
OUT = os.path.join(ROOT, "docs", "ras_paper.html")


def img(name, alt, w="90%"):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return "<p><em>[missing %s]</em></p>" % name
    b = base64.b64encode(open(p, "rb").read()).decode("ascii")
    return ('<img alt="%s" src="data:image/png;base64,%s" style="max-width:%s;height:auto;'
            'border:1px solid #d7dde3;border-radius:6px">' % (alt, b, w))


CSS = """
body{margin:0;color:#16191d;background:#fff;font:16px/1.66 Georgia,'Times New Roman',serif}
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
.algo{background:#fbfbfd;border:1px solid #d7dde3;border-radius:8px;padding:8px 14px;margin:14px 0;font:13px/1.5 Consolas,monospace;white-space:pre-wrap}
.algo .cap{font-family:-apple-system,Segoe UI,sans-serif;font-weight:700;font-size:13px;color:#16191d;border-bottom:1px solid #e2e8ee;padding-bottom:4px;margin-bottom:6px;white-space:normal}
figure{margin:18px 0;text-align:center}figcaption{color:#5b6570;font-size:13px;margin-top:6px;text-align:left}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:12.5px;font-family:-apple-system,Segoe UI,sans-serif}
th,td{border:1px solid #e2e8ee;padding:5px 8px;text-align:center}th{background:#f5f8fb}td.l,th.l{text-align:left}
.small{font-size:13px;color:#5b6570}a{color:#1f5fa8}
ol.contrib>li{margin:4px 0}
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
  "<title>Acting on the Unseen (RAS)</title>%s<style>%s</style></head><body><div class='wrap'>" % (KATEX, CSS))

# ---------------- title / meta ----------------
A("<h1>Acting on the Unseen: Communication-Free Collaborative Filtering for Decentralized "
  "Multi-Robot Task Allocation</h1>")
A("<p class='sub'>A first paper targeting <i>Robotics and Autonomous Systems</i>. Companion materials "
  "(tutorial, full theory, per-seed data) at <a href='https://github.com/ApartsinProjects/ZKDroneSwarm'>"
  "ApartsinProjects/ZKDroneSwarm</a>. All reported numbers are regenerable from saved per-seed results.</p>")

A("<div class='hl'><b>Highlights</b><ul>"
  "<li>A prior-free, communication-free MRTA regime with partial, privately-noisy observation.</li>"
  "<li>Decentralized online collaborative filtering acts well on never-attempted tasks.</li>"
  "<li>A categorical separation over structure-free learning, proven and empirically validated.</li>"
  "<li>The broadcast's value and a positive scaling law: the swarm gets smarter as it grows.</li>"
  "<li>Robust under masking; recovers most of a centralized full-communication ceiling.</li></ul></div>")

# ---------------- abstract ----------------
A("<div class='abs'><b>Abstract.</b> Multi-robot task allocation usually assumes some combination of "
  "communication, known task models, or a coordinator. We study the opposite extreme, a regime that is "
  "common in practice yet largely overlooked in theory: a robot team with <b>no prior knowledge</b> (no "
  "task models, no labels, not even the rank of the latent structure), <b>no communication</b> (no "
  "messages, no parameter sharing, no coordinator), and only a <b>partial, noisy, privately-perceived</b> "
  "view of a public stream of teammates' outcomes, with every robot deciding independently. Compatibility "
  "between a robot's capabilities and a task's requirements is governed by a hidden low-rank structure, "
  "and there are far more tasks than rounds, so most (robot, task) pairs are never attempted. We show that "
  "each robot can nonetheless act well on tasks it has never attempted, onboard new tasks, and absorb new "
  "robots, by running online low-rank collaborative filtering over the public broadcast (we call the "
  "method <b>SwarmCF</b>). The advantage over any structure-free learner is <b>categorical</b>, not a "
  "constant factor: a structure-free learner is provably at the prior-mean error floor on unseen pairs. We "
  "prove a matching per-robot sample complexity ($\\Theta(d)$ vs $\\Theta(n)$), an anytime "
  "(cumulative-reward) separation under task scarcity, and a deterministic condition under which "
  "decentralized recovery from the masked broadcast is exact (validated empirically). Experiments quantify "
  "the value of the broadcast, a positive scaling law (the swarm gets smarter as it grows), and dominance "
  "over the low-rank field under limited observability, recovering most of a centralized "
  "full-communication ceiling.</div>")
A("<p class='small'><b>Keywords:</b> multi-robot task allocation; decentralized learning; collaborative "
  "filtering; low-rank matrix completion; communication-free coordination; multi-agent bandits; swarm "
  "robotics.</p>")

# ---------------- 1. introduction ----------------
A("<h2>1. Introduction</h2>")
A("<div class='box'><b>The setting in one line.</b> No prior knowledge, no communication, and only a "
  "partial, noisy, privately-sensed view of teammates' outcomes, with every decision made independently. "
  "Can a swarm still act intelligently, in particular on tasks it has never attempted? We answer yes, and "
  "show the advantage is categorical.</div>")
A("<p>Consider a team of autonomous robots, say aerial vehicles, that must repeatedly decide which task "
  "to engage: which area to inspect, which target to service, which sensor reading to pursue. Whether a "
  "given robot does well on a given task depends on a match between the robot's <b>capabilities</b> "
  "(sensor modalities, payload, effector type, endurance, remaining consumables) and the task's "
  "<b>requirements</b>. This capability-requirement view is the basis of trait-based task allocation "
  "[5], and the match is typically governed by only a few underlying factors, so the full "
  "robot $\\times$ task reward is <b>low-rank</b>.</p>")
A("<p>Most multi-robot task allocation (MRTA) methods obtain coordination from at least one of three "
  "resources: explicit <b>communication</b> (auctions, consensus, message-passing), <b>known task "
  "models or utilities</b>, or a <b>central coordinator / centralized training</b>. In many real "
  "deployments none of these is available: communication is jammed, bandwidth-limited, or deliberately "
  "withheld for stealth; the task structure is unknown a priori; and there is no coordinator. What a "
  "robot can often still do is <b>passively sense</b> some of what its teammates are doing and how it "
  "turned out, imperfectly, at a distance, and differently from every other robot. This is the regime we "
  "formalize and solve: a robot observes a <b>partial</b> (range-limited), <b>noisy</b>, and crucially "
  "<b>privately-perceived</b> slice of a public outcome stream, never the same slice as a teammate.</p>")
A("<p>The technical crux is <b>generalization to the unseen</b>. There are far more tasks than rounds "
  "($n \\gg T$), so each robot personally attempts only a vanishing fraction of tasks. A learner that "
  "estimates each task only from its own attempts (a <b>structure-free</b> learner: independent "
  "per-task bandits, tabular value tables) has, on any task it never attempted, nothing better than the "
  "prior mean, the error floor, which dominates exactly because tasks outnumber rounds. The opportunity "
  "is that the shared low-rank structure links tasks: outcomes a robot <i>observes</i> teammates obtain "
  "(even noisily and partially) constrain that structure, and a few observations then determine the "
  "robot's reward on tasks it has never touched. We show a single, simple estimator turns this "
  "opportunity into a categorical capability.</p>")
A("<p><b>The gap.</b> To our knowledge no prior method targets this cell. Every established paradigm "
  "relaxes at least one of its defining constraints, no prior knowledge, no communication, decentralized "
  "decisions, and partial and privately-noisy observation, by assuming communication, known "
  "utilities/traits, a coordinator or centralized training, or clean/shared observation (Section 2, "
  "Table 1). The regime is not exotic: it is the default when communication is jammed, bandwidth-limited, "
  "or withheld for stealth, and when task structure must be discovered in the field. We close it.</p>")
A("<p><b>Contributions.</b></p><ol class='contrib'>"
  "<li>We formalize <b>communication-free MRTA under partial, privately-noisy observation and zero "
  "prior knowledge</b>, a most-restrictive but practically common regime that prior MRTA and "
  "decentralized-learning work does not address (Section 2, Table 1).</li>"
  "<li>We propose <b>SwarmCF</b>, a decentralized, online, low-rank collaborative filter that each robot "
  "runs over the passive broadcast, with an $O(d)$ fold-in that lets it act on unseen tasks and onboard "
  "new tasks/robots (Section 4).</li>"
  "<li>We prove the advantage is <b>categorical</b>: a structure-free learner is at the error floor on "
  "unseen pairs and the broadcast is provably useless to it, whereas SwarmCF attains a per-robot sample "
  "complexity of $\\Theta(d)$ versus $\\Theta(n)$, with a matching <b>anytime</b> separation under task "
  "scarcity (Section 5).</li>"
  "<li>We give a <b>deterministic condition</b> under which decentralized recovery of the shared "
  "structure from the privately-masked broadcast is exact, with a coverage-time bound that improves as "
  "the team grows, and validate the condition empirically (Section 5, Appendix C).</li>"
  "<li>We quantify, on a robotics-grounded task-servicing mission, the <b>value of the broadcast</b>, a "
  "<b>positive scaling law</b> (per-robot competence rises with team size), and <b>dominance over the "
  "low-rank field</b> under limited observability, recovering most of a centralized full-communication "
  "ceiling (Section 6).</li></ol>")

# ---------------- 2. related work ----------------
A("<h2>2. Related work</h2>")
A("<p><b>Multi-robot task allocation.</b> The taxonomies of Gerkey and Matari&cacute; [1] and Korsah et "
  "al. [2] organize MRTA by single/multi-task robots, single/multi-robot tasks, and "
  "instantaneous/time-extended assignment with interrelated utilities. Classical solvers, market and "
  "auction mechanisms [3] and consensus-based bundle algorithms (CBBA) [4], and distributed constraint "
  "optimization, achieve coordination through <b>communication</b> and assume <b>known</b> task "
  "utilities or costs. Trait-based MRTA [5] matches robot capability vectors to task requirement "
  "vectors, but takes the traits as <b>given</b>. Our setting keeps the trait/low-rank view but makes "
  "the traits <b>unknown and learned online</b>, with neither communication nor known utilities.</p>")
A("<p><b>Decentralized and learning-based coordination.</b> Communication-free multiplayer bandits "
  "(musical chairs [6], SIC-MMAB [7]) break symmetry without messages but are <b>structure-free</b> "
  "(per-arm), so they cannot generalize to unseen arms. Cooperative multi-agent RL (CTDE methods such as "
  "MAPPO [14]) and learned-communication methods (e.g. CommNet [15]) rely on centralized training or "
  "message passing. Federated and gossip collaborative filtering share model parameters; we share "
  "nothing but a passively-sensed outcome stream.</p>")
A("<p><b>Low-rank estimation and bandits.</b> Matrix completion gives centralized recovery guarantees "
  "under (near-)uniform sampling [8,9], with practical factorization estimators (matrix factorization "
  "[13], Bayesian PMF [11], soft-impute [12]); low-rank and bilinear bandits (e.g. explore-then-spectral "
  "[10], and clustering-of-bandits [16]) are centralized and/or phase-structured. We make estimation "
  "decentralized, online, broadcast-only, and robust to a structured (non-uniform) per-robot observation "
  "mask, with the unseen-pair error floor turning the gap into a categorical, rather than "
  "constant-factor, separation.</p>")
A("<p>Table 1 places the major paradigms on the four axes that define our problem; each established "
  "family relaxes at least one axis we hold fixed, and our cell, low-rank with only a guessed rank, no "
  "communication, decentralized, masked and noisy, is the one left open.</p>")
A("<p class='small'><b>Table 1.</b> Paradigms on the axes prior knowledge / communication / distribution "
  "/ observability.</p>")
A(mp.html_paradigms())

# ---------------- 3. problem setting ----------------
A("<h2>3. Problem setting</h2>")
A("<p><b>Reward.</b> A team of $m$ robots faces $n$ tasks. Robot $i$ has a hidden capability vector "
  "$p_i\\in\\mathbb{R}^d$ and task $j$ a hidden requirement vector $u_j\\in\\mathbb{R}^d$; the expected "
  "reward of robot $i$ engaging task $j$ is their inner product"
  "$$ R_{ij} \\;=\\; \\langle p_i, u_j\\rangle, \\qquad R = P U^\\top \\in \\mathbb{R}^{m\\times n},\\ "
  "\\operatorname{rank}(R)=d \\ll \\min(m,n). $$"
  "The low rank $d$ encodes that only a few traits govern fit. The team does <b>not</b> know $P$, $U$, or "
  "even $d$ (it uses a guessed rank $\\hat d$).</p>")
A("<p><b>Interaction.</b> Each round $t=1,\\dots,T$ every robot $i$ is offered a uniform random size-$c$ "
  "subset $S_{it}\\subseteq[n]$ of tasks, selects one $a_{it}\\in S_{it}$, engages it, and earns "
  "$R_{i,a_{it}}$. The operating regime is <b>task-scarce</b>: $n \\gg cT$, so each robot personally "
  "engages only $O(T)$ of the $n$ tasks.</p>")
A("<div class='box'><b>The observation channel (the heart of the setting).</b> There is no communication. "
  "Each robot instead <b>passively senses</b> a public stream of engagement outcomes, but only partially "
  "and noisily, and <b>privately</b>: <b>(persistent partial visibility)</b> robot $i$ observes "
  "teammate $k$'s engagements only if $M_{ik}=1$, where $M_{ik}\\sim\\mathrm{Bernoulli}(\\rho)$ is fixed "
  "for the whole mission ($M_{ii}=1$); <b>(private per-observer noise)</b> when $i$ observes the outcome "
  "of action $a_{kt}$ it reads $R_{k,a_{kt}}+\\eta_{ikt}$ with $\\eta_{ikt}\\sim\\mathcal N(0,\\sigma^2)$ "
  "drawn independently per observer, so the same action is read differently by different robots. No robot "
  "ever sees the clean stream, and no two robots see the same stream.</div>")
A("<p>This channel is the formal counterpart of physical sensing: a robot perceives a teammate's action "
  "and its effect only when the teammate is within range (partial, persistent) and with a fidelity that "
  "degrades with distance and the robot's own sensor (noisy, private). It is strictly weaker than the "
  "shared, clean broadcast usually assumed, and it makes decentralization <b>real</b>: persistent blind "
  "spots give every robot a permanently different view, so the robots cannot converge to a common model "
  "by symmetry. Figure 1 illustrates the setting.</p>")
A("<figure>%s<figcaption><b>Figure 1.</b> The setting. The robot $\\times$ task reward is hidden and "
  "low-rank, $R=PU^\\top$ (capability traits $p_i$, requirement traits $u_j$). A focal robot (blue row) "
  "must act on its <i>whole</i> row, including the many pairs it never engaged ('?'), using only its own "
  "clean engagements (green), a partial and per-observer-noisy view of the teammates it can sense (dots; "
  "greyed rows are persistently invisible to it), and no communication.</figcaption></figure>"
  % img("F20_setting.png", "setting schematic"))
A("<p><b>Objective and metric.</b> We measure decision quality by the normalized <b>skill</b> "
  "(a Murphy skill score [17] / normalized return),"
  "$$ \\mathrm{skill} \\;=\\; \\frac{\\text{earned} - \\text{random}}{\\text{oracle} - \\text{random}}, $$"
  "where for an offered set the <i>oracle</i> picks $\\arg\\max_{j\\in S}R_{ij}$ and <i>random</i> picks "
  "uniformly; skill $=0$ is the no-information floor and $1$ is omniscient. We report <b>unseen-pair "
  "skill</b> (restricted to tasks the robot never engaged, the generalization test), and the "
  "<b>anytime</b> (cumulative-reward) skill over the mission.</p>")

# ---------------- 4. method ----------------
A("<h2>4. Method: SwarmCF</h2>")
A("<p>Each robot independently maintains low-rank factor estimates $\\hat P\\in\\mathbb{R}^{m\\times\\hat "
  "d}$, $\\hat U\\in\\mathbb{R}^{n\\times\\hat d}$ and updates them online from whatever it senses, using "
  "<b>noise-weighted alternating least squares</b> (weighted ridge ALS, the workhorse of low-rank "
  "completion) on the observed entries. The key design choice for masking-robustness: an unobserved "
  "entry receives zero <i>weight</i>, never an imputed zero <i>value</i>. The robot then acts greedily "
  "(with a small $\\varepsilon$ for exploration) on its <b>completed</b> reward row, which is defined for "
  "every task, including ones it never engaged.</p>")
A("<div class='algo'><div class='cap'>Algorithm 1 &mdash; SwarmCF (run independently by each robot $i$)</div>"
  "init $\\hat P,\\hat U$ small-random; observed set $\\Omega_i\\leftarrow\\emptyset$\n"
  "for round $t=1,\\dots,T$:\n"
  "    offered $S_{it}$; act $a_{it}\\leftarrow \\arg\\max_{j\\in S_{it}}\\langle\\hat p_i,\\hat u_j\\rangle$"
  " (w.p. $1-\\varepsilon$, else random); engage, earn $R_{i,a_{it}}$\n"
  "    sense broadcast: for each visible teammate $k$ (i.e. $M_{ik}=1$) record\n"
  "        $(k,\\,a_{kt},\\,\\tilde r=R_{k,a_{kt}}+\\eta_{ikt})$ with weight $w=1/\\sigma^2$ into $\\Omega_i$\n"
  "        (own outcome recorded with its own, lower, noise)\n"
  "    every $\\tau$ rounds: refit by weighted ridge ALS sweeps over $\\Omega_i$:\n"
  "        $\\hat u_j \\leftarrow (\\sum_{(k,j)\\in\\Omega_i} w\\,\\hat p_k\\hat p_k^\\top+\\lambda I)^{-1}"
  "\\sum w\\,\\tilde r\\,\\hat p_k$;   symmetric update for $\\hat p_i$\n"
  "predict full row $\\hat R_{i\\cdot}=\\hat U\\hat p_i$ (defined on EVERY task, seen or not)</div>")
A("<p><b>Onboarding (fold-in).</b> A new task $j^\\star$ (or a new robot) is absorbed without retraining: "
  "given the current factor basis, its hidden vector is the ridge least-squares solution of its few "
  "observed engagements against the corresponding known factors, an $O(\\hat d)$ computation. This is the "
  "same operation that lets an existing robot predict an unseen task once the basis is recovered "
  "(Algorithm 2).</p>")
A("<div class='algo'><div class='cap'>Algorithm 2 &mdash; Fold-in (onboard a new task / robot / unseen pair)</div>"
  "given basis $B$ (the $\\hat d$-dim factors of the entities the newcomer has been observed against)\n"
  "and observations $y$ (the few rewards seen for the newcomer), with weights $W$:\n"
  "    $\\hat x \\leftarrow (B^\\top W B+\\lambda I)^{-1} B^\\top W y$    // $O(\\hat d)$ ridge solve\n"
  "predict reward on any other entity with factor $b$ as $\\langle b,\\hat x\\rangle$</div>")
A("<p><b>What SwarmCF does and does not assume.</b> It is fully decentralized (one estimator per robot, "
  "no shared state), communication-free (it only reads the passive stream), and prior-free beyond a "
  "guessed rank $\\hat d$ (which it does not need to be exact, and which can itself be removed by a "
  "rank-adaptive variant we defer to follow-up work). It does not assume the noise level is known: "
  "uniform weighting suffices and is what we use for the headline results.</p>")

# ---------------- 5. theory ----------------
A("<h2>5. Theory: why the advantage is categorical</h2>")
A("<p>We formalize the separation between structure-free learning and SwarmCF, give the per-robot sample "
  "complexity, and state the conditions under which decentralized recovery from the masked broadcast "
  "succeeds. Proofs are in Appendix A; here we give the statements and the intuition. A learner is "
  "<b>structure-free</b> if its estimate of $R_{ij}$ depends only on robot $i$'s own past engagements of "
  "task $j$ and equals a fixed prior on any task it never engaged (the per-arm class: independent UCB, "
  "tabular).</p>")
A("<div class='thm'><b>Theorem 1 (structure-free floor).</b> For a structure-free learner and any task "
  "$j$ that robot $i$ never engaged, the estimate is the prior constant, so its expected unseen-pair "
  "skill is exactly $0$ and its squared error is at least the row variance $\\Omega(1)$. Moreover the "
  "broadcast is provably useless to it: its per-task estimate is by definition not a function of any "
  "other task or robot.</div>")
A("<div class='thm'><b>Theorem 2 (CF row completion, $\\Theta(d)$ vs $\\Theta(n)$).</b> If the task "
  "factors $U$ are known (rank $d$) and robot $i$ observes its true rewards on any set $\\Omega$ with "
  "$|\\Omega|\\ge d$ whose factors span $\\mathbb{R}^d$, then $p_i$ is the unique least-squares solution "
  "and $R_{ij}=\\langle p_i,u_j\\rangle$ is recovered <b>exactly for all</b> $j$. Per-robot sample "
  "complexity is therefore $\\Theta(d)$, versus $\\Theta(n)$ for any structure-free learner.</div>")
A("<div class='thm'><b>Theorem 3 (anytime separation under task scarcity).</b> With $n$ tasks, offers of "
  "size $c$ and horizon $T$, any structure-free learner has anytime (cumulative-reward) skill at most "
  "$g(cT/n)\\to 0$ when $cT=o(n)$, even with a full broadcast; SwarmCF reaches near-oracle in $O(d)$ "
  "rounds, so its anytime skill is $1-O(d/T)$.</div>")
A("<p>Theorems 1-3 make the separation categorical (zero vs nonzero) and operational (it shows up in "
  "reward earned while learning), but Theorem 2 assumes $U$ is known. The remaining question, the crux "
  "of the decentralized setting, is whether each robot can <b>recover</b> the shared structure from its "
  "own privately-masked, noisy stream. Because the mask is over robot pairs, robot $i$'s observations form "
  "a structured (non-uniform) sub-sample, exactly where off-the-shelf uniform-sampling completion does "
  "not apply. We give a deterministic condition instead.</p>")
A("<div class='thm'><b>Theorem 4 (decentralized masked recovery).</b> Let $E_i(j)$ be the set of robot "
  "$i$'s visible teammates that engaged task $j$. If $i$'s observation graph contains a $\\hat d\\times "
  "\\hat d$ fully-observed invertible anchor block (fixing the factor frame) and, for task $j$, the "
  "factors $\\{p_k : k\\in E_i(j)\\}$ span $\\mathbb{R}^d$, then robot $i$ recovers $u_j$ exactly "
  "(noiseless) and predicts $(i,j)$ exactly after folding in its own $\\ge d$ engagements; under "
  "per-observation noise the error is bounded by the fold-in bound with the coverage degree $|E_i(j)|$ "
  "controlling the noise term. Conversely, if $\\{p_k:k\\in E_i(j)\\}$ does not span the direction of "
  "$p_i$, the pair $(i,j)$ is non-identifiable and the learner is at the prior floor. Under non-adaptive "
  "exploration the spanning condition is met for all tasks, with high probability, after "
  "$T=O\\!\\big(\\tfrac{nd}{\\rho m}\\log n\\big)$ rounds, a rate that improves as the team grows.</div>")
A("<p>Theorem 4 turns the previously-cited completion step into a self-contained, checkable condition: "
  "recovery is exact when a robot has seen a task engaged by a spanning set of visible teammates, "
  "impossible without it, and is reached the faster the larger the team. Appendix C validates it "
  "directly: on the actual observation patterns the swarm produces, reconstruction error collapses from "
  "the prior floor to (numerically) zero exactly at the spanning threshold.</p>")
A("<div class='thm'><b>Theorem 5 (collective speedup, why a swarm).</b> An isolated robot ($\\rho=0$) "
  "sees only its own row and cannot identify a rank-$d>1$ column space, so its unseen skill stays at the "
  "floor: sharing is <b>necessary</b>. With the broadcast, the swarm's pooled observations cross the "
  "recovery threshold after $\\tilde O(d(1+n/m))$ rounds, an $\\Theta(m)$-fold speedup over a lone "
  "learner, achieved with no communication.</div>")
A("<p>Theorems 4-5 are, to our knowledge, the first results that pin decentralized low-rank recovery from "
  "a persistent, private, per-robot mask to an explicit condition and tie its rate to team size; they are "
  "what make the categorical claim self-contained rather than imported from centralized theory. We assess "
  "the theory's correctness, novelty, and utility in Appendix A.</p>")

# ---------------- 6. experiments ----------------
A("<h2>6. Experiments</h2>")
A("<p><b>Setup.</b> Unless noted, $m=30$ robots, $n=240$ tasks, true rank $d=5$, guessed rank "
  "$\\hat d=8$ (so no method is given the rank), horizon $T=50$, offers of size $c=20$, partial "
  "broadcast $\\rho$ swept, private noise on own ($0.1$) and observed ($0.3$) outcomes, 8 seeds, "
  "bootstrap 95% confidence intervals. We compare on one canonical masked harness.</p>")
A("<p><b>How to read the comparison.</b> The setting itself is new, so this is a controlled sweep across "
  "the low-rank design space against the genuinely external structure-free paradigm and "
  "full-information reference ceilings, not a contest of rival systems. SwarmCF is our method; structure-"
  "free learners (independent UCB, tabular) are the external paradigm; standard low-rank estimators "
  "(online SGD-MF, batch spectral/Bayesian/convex completion) are adapted to the setting for the "
  "low-rank comparison; the Oracle and a centralized full-communication matcher are upper bounds, not "
  "competitors. In our harness <b>every</b> method runs decentralized and communication-free (one "
  "estimator per robot); the low-rank methods differ only in the update rule. Table 2 fixes each "
  "method's operating profile.</p>")
A("<p class='small'><b>Table 2.</b> Operating profiles. Notation [distribution | communication | "
  "observability | prior | computation].</p>")
A(mp.html_profiles())

A("<h3>6.1 The categorical win and masking robustness</h3>")
A("<p>Figure 2 sweeps the broadcast rate $\\rho$ and reports unseen-pair skill. SwarmCF acts well on "
  "tasks it never engaged at every observation density, while the structure-free learners sit at the "
  "floor ($\\approx 0$) by construction, the categorical separation of Theorems 1-2. Among low-rank "
  "methods, SwarmCF's online updates stay robust as the broadcast is masked, whereas batch spectral "
  "completion (which imputes unobserved entries) decays; the two cross over near $\\rho\\approx 0.6$ and "
  "batch wins only at full broadcast, where its one-shot factorization on a near-complete matrix is the "
  "best case for completion. The operationally relevant regime, partial observation, is exactly where "
  "the online estimator leads.</p>")
A("<figure>%s<figcaption><b>Figure 2.</b> Unseen-pair skill versus broadcast rate $\\rho$. Structure-"
  "free learners are pinned at the floor at every density; SwarmCF acts on the "
  "unseen throughout and is robust under masking, while batch spectral completion decays as observation "
  "becomes partial. (SwarmCF-RC and SwarmCF-H are deferred SwarmCF-family variants, Section 7; the "
  "headline method is the core SwarmCF.)</figcaption></figure>" % img("F5_crossover.png", "categorical + masking"))

A("<h3>6.2 The operational (anytime) separation</h3>")
A("<p>Final-policy quality can flatter a method that explores cheaply. The operationally honest measure "
  "is reward <i>earned while learning</i>. Figure 3 shows cumulative-reward skill over the mission: "
  "SwarmCF earns from the first rounds, while per-arm bandits remain near random because, with "
  "$n\\gg T$, they never stop exploring untried tasks (Theorem 3). Phase-structured low-rank methods pay "
  "an explore-then-commit penalty early.</p>")
A("<figure>%s<figcaption><b>Figure 3.</b> Anytime cumulative-reward skill. SwarmCF earns from round one; "
  "explore-then-commit pays a probe phase; structure-free learners are stuck near random. (SwarmCF-RC is "
  "a deferred family variant; PTF/ESTR are batch low-rank baselines.)</figcaption></figure>"
  % img("F6_anytime.png", "anytime"))

A("<h3>6.3 Why a swarm: the value of the broadcast and a positive scaling law</h3>")
A("<p>Two experiments isolate what the team and the broadcast actually buy (Figure 4). "
  "<i>(a) Value of the broadcast.</i> Sweeping from $\\rho=0$ (each robot isolated, sees only its own "
  "outcomes) to $\\rho=1$ (full passive sensing): a lone robot cannot recover the shared structure from "
  "its single matrix row, so isolated unseen skill is $\\approx 0$; the broadcast lifts SwarmCF by "
  "$+0.39$ unseen skill but a structure-free learner by $\\approx 0$, which has no model linking tasks "
  "and so cannot use it (Theorem 5). <i>(b) Positive scaling.</i> Holding $n$, horizon and $\\rho$ "
  "fixed and growing the team from $m=5$ to $80$, SwarmCF's unseen skill rises monotonically "
  "($0.08\\to 0.43$): more robots feed more observations into the one shared structure, so each robot's "
  "competence on tasks it never engaged grows with the team. Structure-free learning is flat. A swarm "
  "that gets smarter as it grows, the opposite of the usual interference penalty, is a direct consequence "
  "of sharing structure.</p>")
A("<figure>%s<figcaption><b>Figure 4.</b> (a) Value of the broadcast: unseen skill versus $\\rho$ "
  "(left edge $=$ isolated). (b) Positive scaling: unseen skill versus team size $m$. Both rise for "
  "low-rank CF and are flat for structure-free. (PTF is a batch-refit low-rank method shown for "
  "comparison.)</figcaption></figure>" % img("F18_collab_scaling.png", "why a swarm"))

A("<h3>6.4 An operational mission and a centralized ceiling</h3>")
A("<p>Framed as a target-servicing / dispatch mission, latent factors are robot capability traits and "
  "task requirement traits, each robot repeatedly services an offered target under range-limited, "
  "distance-noisy sensing, SwarmCF on the same servicing-skill metric beats the entire low-rank field "
  "under limited observability ($\\rho=0.25$): $\\approx 0.36$ versus the best of the field "
  "$\\approx 0.29$ with non-overlapping intervals, while structure-free learners sit at the random-"
  "dispatch floor (Figure 5). To bound the cost of our constraints we add two reference ceilings (not "
  "competitors): a centralized full-communication matcher with Hungarian assignment, and the same with "
  "noiseless, unmasked observation. SwarmCF's communication-free de-confliction recovers about 81% of "
  "the full-communication ceiling when tasks are plentiful; the residual gap is the genuine price of "
  "within-round coordination under contention. Both ceilings sit below the omniscient oracle, and the "
  "price of observation noise is small, indicating that coordination, not estimation, is the binding "
  "constraint.</p>")
A("<figure>%s<figcaption><b>Figure 5.</b> Operational target-servicing mission: SwarmCF (blue) versus "
  "the low-rank field (orange) versus structure-free (gray). The separation opens under limited "
  "observability ($\\rho=0.25$).</figcaption></figure>" % img("F17_mission.png", "mission"))
A("<p>Table 3 consolidates the comparison on the canonical masked harness.</p>")
A("<p class='small'><b>Table 3.</b> Performance scorecard on one canonical masked harness.</p>")
A(mp.html_scorecard(ROOT))

# ---------------- 7. discussion ----------------
A("<h2>7. Discussion, limitations, and future work</h2>")
A("<p><b>What the results say.</b> Under the least information, no prior, no communication, partial and "
  "privately-noisy observation, a single simple estimator gives a swarm a capability that structure-free "
  "learning provably cannot have: acting well on the unseen, getting more competent as the team grows, "
  "and recovering most of what a centralized, communicating system could achieve. The win is structural "
  "(it is a property of exploiting the shared low-rank trait structure) and operational (it shows up in "
  "reward earned while learning, and in a robotics-grounded mission).</p>")
A("<p><b>Limitations.</b> The reward is (approximately) low-rank and stationary; rewards are real-valued "
  "and bilinear in latent traits; and the recovery rate of Theorem 4 is established for non-adaptive "
  "exploration, the finite-time rate under a strongly exploiting policy (which can starve low-reward "
  "tasks of coverage) remains open. We report the regime boundaries honestly: the advantage requires "
  "structure beyond mere popularity ($d>1$), task scarcity ($n\\gg cT$), and a shared channel "
  "($\\rho>0$); outside these, structure-free methods are competitive.</p>")
A("<p><b>Future work (a planned follow-up).</b> The present paper deliberately keeps to a single core "
  "estimator. In a follow-up we plan to study the refinements that this foundation enables, each of "
  "which we have prototyped: <i>(i)</i> confidence-directed exploration via a Bayesian posterior over the "
  "factors (collective, information-directed probing through the shared broadcast); <i>(ii)</i> "
  "communication-free de-confliction under capacity-1 contention via a fixed private offset, against "
  "no-communication auction and musical-chairs primitives; <i>(iii)</i> rank self-determination "
  "(removing the guessed $\\hat d$); <i>(iv)</i> the action/choice channel as a noise-immune alternative "
  "to cardinal rewards; and <i>(v)</i> non-stationarity and team churn. We also plan a hardware / "
  "high-fidelity-simulation study and a tightening of the adaptive-policy coverage rate.</p>")

# ---------------- 8. conclusion ----------------
A("<h2>8. Conclusion</h2>")
A("<p>We formalized multi-robot task allocation in its most restrictive but practically common form, no "
  "prior knowledge, no communication, partial and privately-noisy observation, and showed that "
  "decentralized online collaborative filtering over the passive broadcast lets each robot act well on "
  "tasks it has never attempted. The advantage over structure-free learning is categorical and proven; "
  "the broadcast is what makes it possible and the team is what makes it fast; and the method recovers "
  "most of a centralized full-communication ceiling while assuming far less. We hope the setting, a swarm "
  "that must learn to coordinate from nothing but what it can quietly see, becomes a useful baseline "
  "regime for autonomous multi-robot systems.</p>")

# ---------------- appendices ----------------
A("<h2>References</h2>")
A("<ol class='small' style='line-height:1.5'>"
  "<li>B. P. Gerkey, M. J. Matari&cacute;. A formal analysis and taxonomy of task allocation in "
  "multi-robot systems. <i>Int. J. Robotics Research</i>, 23(9):939&ndash;954, 2004.</li>"
  "<li>G. A. Korsah, A. Stentz, M. B. Dias. A comprehensive taxonomy for multi-robot task allocation. "
  "<i>Int. J. Robotics Research</i>, 32(12):1495&ndash;1512, 2013.</li>"
  "<li>M. B. Dias, R. Zlot, N. Kalra, A. Stentz. Market-based multirobot coordination: a survey and "
  "analysis. <i>Proc. IEEE</i>, 94(7):1257&ndash;1270, 2006.</li>"
  "<li>H.-L. Choi, L. Brunet, J. P. How. Consensus-based decentralized auctions for robust task "
  "allocation. <i>IEEE Trans. Robotics</i>, 25(4):912&ndash;926, 2009.</li>"
  "<li>A. Prorok, M. A. Hsieh, V. Kumar. The impact of diversity on optimal control policies for "
  "heterogeneous robot swarms. <i>IEEE Trans. Robotics</i>, 33(2):346&ndash;358, 2017.</li>"
  "<li>J. Rosenski, O. Shamir, L. Szlak. Multi-player bandits: a musical chairs approach. <i>ICML</i>, "
  "2016.</li>"
  "<li>E. Boursier, V. Perchet. SIC-MMAB: synchronisation involves communication in multiplayer "
  "multi-armed bandits. <i>NeurIPS</i>, 2019.</li>"
  "<li>E. J. Cand&egrave;s, B. Recht. Exact matrix completion via convex optimization. <i>Found. Comput. "
  "Math.</i>, 9(6):717&ndash;772, 2009.</li>"
  "<li>R. H. Keshavan, A. Montanari, S. Oh. Matrix completion from a few entries. <i>IEEE Trans. Inf. "
  "Theory</i>, 56(6):2980&ndash;2998, 2010.</li>"
  "<li>Y. Kang, C.-J. Hsieh, T. C. M. Lee. Efficient frameworks for generalized low-rank matrix bandit "
  "problems. <i>NeurIPS</i>, 2022.</li>"
  "<li>R. Salakhutdinov, A. Mnih. Bayesian probabilistic matrix factorization using MCMC. <i>ICML</i>, "
  "2008.</li>"
  "<li>R. Mazumder, T. Hastie, R. Tibshirani. Spectral regularization algorithms for learning large "
  "incomplete matrices. <i>JMLR</i>, 11:2287&ndash;2322, 2010.</li>"
  "<li>Y. Koren, R. Bell, C. Volinsky. Matrix factorization techniques for recommender systems. "
  "<i>IEEE Computer</i>, 42(8):30&ndash;37, 2009.</li>"
  "<li>C. Yu, A. Velu, E. Vinitsky, et al. The surprising effectiveness of PPO in cooperative "
  "multi-agent games. <i>NeurIPS</i>, 2022.</li>"
  "<li>S. Sukhbaatar, A. Szlam, R. Fergus. Learning multiagent communication with backpropagation. "
  "<i>NeurIPS</i>, 2016.</li>"
  "<li>C. Gentile, S. Li, G. Zappella. Online clustering of bandits. <i>ICML</i>, 2014.</li>"
  "<li>A. H. Murphy. Skill scores based on the mean square error and their relationships to the "
  "correlation coefficient. <i>Monthly Weather Review</i>, 116:2417&ndash;2424, 1988.</li>"
  "</ol>")

A("<h2>Appendix A. Theory: proofs and an audit (correctness, novelty, utility)</h2>")
A("<p class='small'>Full statements and proofs are maintained in the companion "
  "<code>docs/THEORY_FORMAL.md</code>; we summarize the proof ideas and an honest audit here.</p>")
A("<p class='small'><b>Theorem 1 (floor).</b> For an unobserved $j$ the estimate is a pre-chosen "
  "constant $b$; $\\mathbb{E}[(b-R_{ij})^2]=(b-\\mu_i)^2+\\mathrm{Var}_j\\ge\\mathrm{Var}_j=\\Omega(1)$, "
  "and on an offer of never-engaged tasks selection is independent of their rewards, giving skill $0$. "
  "The broadcast cannot help a per-task estimate by definition. <i>Correct</i> (elementary); "
  "<i>novel</i> as a clean impossibility for this setting; <i>necessary</i> (defines the floor).</p>")
A("<p class='small'><b>Theorem 2 (row completion).</b> Stacking the observed entries gives "
  "$R_{i,\\Omega}=U_\\Omega p_i$; spanning makes $U_\\Omega$ full column rank, so "
  "$p_i=(U_\\Omega^\\top U_\\Omega)^{-1}U_\\Omega^\\top R_{i,\\Omega}$ is unique and exact, hence all "
  "$R_{ij}$. <i>Correct</i> (linear algebra); standard given $U$, but the $\\Theta(d)$-vs-$\\Theta(n)$ "
  "framing against the floor is the contribution; <i>necessary</i> (the mechanism).</p>")
A("<p class='small'><b>Theorem 3 (anytime).</b> A structure-free learner earns above the mean only on an "
  "offer containing an already-engaged task; the engaged set has size $\\le t-1$ and is reward-blind, so "
  "by concavity of expected order statistics the per-round surplus is $\\le g(c(t-1)/n)$; summing gives "
  "$\\le g(cT/n)\\to 0$. <i>Correct</i> (order; constant loose, mechanism exact); <i>high utility</i> "
  "(justifies the anytime metric).</p>")
A("<p class='small'><b>Theorem 4 (recovery).</b> A fully-observed invertible $\\hat d\\times\\hat d$ "
  "block pins the factor frame; per task, $R_{E_i(j),j}=P_{E_i(j)}u_j$ has a unique solution iff "
  "$P_{E_i(j)}$ has full column rank, giving exact $u_j$ (noiseless) and, with noise, error "
  "$O(\\sigma\\sqrt{d/|E_i(j)|})$; non-identifiability below the spanning threshold is the per-task "
  "analogue of Theorem 1. The coverage time is a coupon-collector bound on the bipartite observation "
  "graph. <i>Correct</i> (proved + validated, Appendix C); <i>novel</i> (deterministic condition for "
  "persistent per-robot masking, where uniform-sampling completion fails); <i>high utility</i> (makes "
  "the categorical claim self-contained). Residual: the finite-time rate under an adaptive policy.</p>")
A("<p class='small'><b>Theorem 5 (collective speedup).</b> A single row leaves a rank-$d>1$ column space "
  "unconstrained (floor); the pooled support reaches the completion threshold in $\\tilde O(d(1+n/m))$ "
  "rounds. <i>Correct</i> (order, given Theorem 4); <i>high utility</i> (the why-a-swarm theory, matching "
  "Section 6.3). The fold-in perturbation bound used above (cold-start error $=$ basis-recovery $+$ "
  "own-probe noise $+$ ridge bias, exact at $k\\ge d$) is in <code>THEORY_FORMAL.md</code>.</p>")

A("<h2>Appendix B. The fold-in error bound</h2>")
A("<p class='small'>For a newcomer factor $x_\\star$ probed against an estimated basis "
  "$\\hat B=B+\\Delta$ ($\\lVert\\Delta\\rVert\\le\\varepsilon$) with $k\\ge d$ observations of noise "
  "$\\sigma$ and ridge $\\lambda$, the ridge fold-in prediction error splits cleanly into three sources, "
  "$\\mathbb{E}|\\hat r-r|\\le C_1\\varepsilon\\lVert x_\\star\\rVert(1+\\lVert b\\rVert/s) + "
  "C_2\\lVert b\\rVert\\sigma\\sqrt{d}/s + C_3\\lambda\\lVert x_\\star\\rVert\\lVert b\\rVert/s^2$ with "
  "$s=\\sigma_{\\min}(B)$, and is exact ($\\hat r=r$) when $\\varepsilon=\\sigma=0,\\lambda\\to0,k\\ge d$. "
  "It quantitatively explains the graceful degradation of cold-start skill as sensing becomes "
  "sparser.</p>")
A("<h2>Appendix C. Empirical validation of the recovery condition (Theorem 4)</h2>")
A("<p class='small'>On the swarm's actual coverage patterns ($m=30,n=240,d=5,\\rho=0.5$, noiseless to "
  "isolate identifiability), reconstructing each unseen pair $(i,j)$ from the observed entries by least "
  "squares gives error $0.000$ exactly when robot $i$'s factor lies in the span of its visible engagers "
  "of $j$ (the condition of Theorem 4), versus the prior floor ($\\approx0.30$) otherwise, with graceful "
  "partial recovery as the spanning rank rises to $d$. The identifiability threshold is exactly the "
  "spanning condition. Full table in <code>docs/P15_VALIDATION.md</code>.</p>")
A("<h2>Appendix D. Reproducibility</h2>")
A("<p class='small'>All numbers come from saved per-seed JSON under <code>results/pilots/</code> "
  "(registry <code>docs/DATA_CATALOGUE.md</code>); figures via <code>experiments/make_figures.py</code>; "
  "this paper via <code>experiments/make_ras_paper.py</code>; methods in "
  "<code>experiments/pilot_noise.py</code> and <code>pilot_baselines.py</code>; proofs in "
  "<code>docs/THEORY_FORMAL.md</code>. Block-model world, signed-cosine reward, 8 seeds, bootstrap "
  "95% CIs throughout.</p>")

A("</div></body></html>")
open(OUT, "w", encoding="utf-8").write("\n".join(H))
print("wrote", OUT, "(%d KB)" % (len(("\n".join(H)).encode("utf-8")) // 1024))
