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
.algo{background:#fbfbfd;border:1px solid #d7dde3;border-radius:8px;padding:8px 14px;margin:14px 0;font:13px/1.5 Consolas,monospace;white-space:pre-wrap;text-align:left;hyphens:manual}
.algo .cap{font-family:-apple-system,Segoe UI,sans-serif;font-weight:700;font-size:13px;color:#16191d;border-bottom:1px solid #e2e8ee;padding-bottom:4px;margin-bottom:6px;white-space:normal}
figure{margin:18px 0;text-align:center}figcaption{color:#5b6570;font-size:13px;margin-top:6px;text-align:justify}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:12.5px;font-family:-apple-system,Segoe UI,sans-serif}
th,td{border:1px solid #e2e8ee;padding:5px 8px;text-align:center}th{background:#f5f8fb}td.l,th.l{text-align:left}
.small{font-size:13px;color:#5b6570}a{color:#1f5fa8}
ol.contrib>li{margin:4px 0}
.docxlink{position:fixed;top:10px;right:12px;z-index:50;background:#1f5fa8;color:#fff;text-decoration:none;font:600 12px/1.2 -apple-system,Segoe UI,Roboto,sans-serif;padding:7px 11px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.18)}
.docxlink:hover{background:#17487f}
.followlink{position:fixed;top:46px;right:12px;z-index:50;background:#0a7d4d;color:#fff;text-decoration:none;font:600 12px/1.2 -apple-system,Segoe UI,Roboto,sans-serif;padding:7px 11px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.18)}
.followlink:hover{background:#085f3a}
@media print{.docxlink,.followlink{display:none}}
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
  "<title>Acting on the Unseen</title>%s<style>%s</style></head><body><div class='wrap'>" % (KATEX, CSS))

# ---------------- title / meta ----------------
A("<a class='docxlink' href='ras_paper.docx'>Download .docx</a>")
A("<a class='followlink' href='ras_paper2.html'>Follow-up paper &#8599;</a>")
A("<h1>Acting on the Unseen: Communication-Free Collaborative Filtering for Decentralized "
  "Multi-Robot Task Allocation</h1>")
A("<p class='sub'>Author One<sup>a,&lowast;</sup>, Author Two<sup>a</sup><br>"
  "<span class='small'><sup>a</sup>Affiliation, City, Country.&ensp;"
  "<sup>&lowast;</sup>Corresponding author.</span></p>")

A("<div class='hl'><b>Highlights</b><ul>"
  "<li>Prior-free, communication-free MRTA under partial, privately-noisy observation.</li>"
  "<li>Decentralized online collaborative filtering acts on never-attempted tasks.</li>"
  "<li>Categorical separation over structure-free learning: proven and validated.</li>"
  "<li>A positive scaling law: the swarm gets smarter as it grows.</li>"
  "<li>Robust to masking; recovers most of a centralized full-communication ceiling.</li></ul></div>")

# ---------------- abstract ----------------
A("<div class='abs'><b>Abstract.</b> Multi-robot task allocation usually assumes some combination of "
  "communication, known task models, or a coordinator. We study the opposite extreme, a regime that is "
  "common in practice yet largely overlooked in theory, which we name <b>Zero-Knowledge MRTA</b> "
  "(ZK-MRTA): a robot team with <b>no prior knowledge</b> (no "
  "task models, not even the latent rank), <b>no communication</b> (no messages, no parameter sharing, "
  "no coordinator), and only a <b>partial, noisy, privately-perceived</b> view of a public stream of "
  "teammates' outcomes. A hidden low-rank structure governs which robot suits which task, and there are "
  "far more tasks than rounds, so most (robot, task) pairs are never attempted. Yet each robot can act "
  "well on tasks it never attempted, and onboard new tasks, by running online low-rank "
  "collaborative filtering over the broadcast (<b>SwarmCF</b>). The advantage over any structure-free learner is <b>categorical</b>, not a "
  "constant factor: a structure-free learner is provably at the prior-mean error floor on unseen pairs. We "
  "prove a matching per-robot sample complexity ($\\Theta(d)$ versus $\\Theta(n)$, in the rank $d$ and "
  "the task count $n$), an anytime "
  "(cumulative-reward) separation under task scarcity, and a deterministic condition under which "
  "decentralized recovery from the masked broadcast is exact (validated empirically). Experiments quantify "
  "the value of the broadcast, a positive scaling law (the swarm gets smarter as it grows), and dominance "
  "over the other low-rank methods under limited observability, recovering most of a centralized "
  "full-communication ceiling; the advantage also transfers to a separate, higher-fidelity spatial "
  "robot simulator with different dynamics.</div>")
A("<p class='small'><b>Keywords:</b> multi-robot task allocation; decentralized learning; collaborative "
  "filtering; low-rank matrix completion; communication-free coordination; multi-agent bandits; swarm "
  "robotics.</p>")

# ---------------- 1. introduction ----------------
A("<h2>1. Introduction</h2>")
A("<div class='box'><b>The setting in one line.</b> No prior knowledge, no communication, and only a "
  "partial, noisy, privately-sensed view of teammates' outcomes, with every decision made independently. "
  "Can a swarm still act intelligently, in particular on tasks it has never attempted? We answer yes, and "
  "show the advantage is categorical.</div>")
A("<p>Consider a team of autonomous robots, for example aerial vehicles, that must repeatedly decide which task "
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
A("<p><b>Relation to collaborative filtering.</b> Low-rank collaborative filtering is itself classical; "
  "our contribution is not the estimator but the demonstration, with theory, that it works <i>at all</i> "
  "in this regime, fully decentralized, communication-free, and under a persistent, per-observer-private "
  "observation mask where standard uniform-sampling completion guarantees do not apply, together with the "
  "recovery condition that says exactly when it works and the collective-speedup law that says why a "
  "swarm helps. The categorical floor against structure-free learning, the decentralized recovery "
  "condition under a structured private mask, and the value-of-broadcast / positive-scaling results are, "
  "to our knowledge, new.</p>")
A("<p><b>Contributions.</b></p><ol class='contrib'>"
  "<li>We formalize and name <b>Zero-Knowledge MRTA</b> (ZK-MRTA): communication-free MRTA under "
  "partial, privately-noisy observation and zero prior knowledge, a most-restrictive but practically "
  "common regime that prior MRTA and decentralized-learning work does not address (Section 2, Table 1).</li>"
  "<li>We propose <b>SwarmCF</b>, a decentralized, online, low-rank collaborative filter that each robot "
  "runs over the passive broadcast, with a constant-time <b>fold-in</b> (an $O(\\hat d^3)$ ridge solve in "
  "the guessed rank $\\hat d$, independent of the task count $n$; Section 4) that lets it act on unseen "
  "tasks and onboard new tasks.</li>"
  "<li>We prove the advantage is <b>categorical</b>: a structure-free learner is at the error floor on "
  "unseen pairs and the broadcast is provably useless to it, whereas SwarmCF attains a per-robot sample "
  "complexity of $\\Theta(d)$ versus $\\Theta(n)$, with a matching <b>anytime</b> separation under task "
  "scarcity (Section 5).</li>"
  "<li>We give a <b>deterministic condition</b> under which decentralized recovery of the shared "
  "structure from the privately-masked broadcast is exact, with a coverage-time bound that improves as "
  "the team grows, and validate the condition empirically (Section 5, Appendix C).</li>"
  "<li>We quantify, on a robotics-grounded task-servicing mission, the <b>value of the broadcast</b>, a "
  "<b>positive scaling law</b> (per-robot competence rises with team size), and <b>dominance over the "
  "other low-rank methods</b> under limited observability, recovering most of a centralized full-communication "
  "ceiling; and we show the advantage <b>transfers to a separate, higher-fidelity spatial simulator</b> "
  "with different dynamics (Section 6).</li>"
  "<li>We release <b>LatentSwarm</b>, an open PettingZoo/Gymnasium evaluation suite for ZK-MRTA, "
  "comprising the controlled analytical masked-broadcast harness used for our headline results and a "
  "spatial environment (Section 6.5, Appendix E), so communication-free low-rank MRTA methods can be "
  "compared on a common footing.</li></ol>")

# ---------------- 2. related work ----------------
A("<h2>2. Related work</h2>")
A("<p>Swarm robotics seeks collective competence from simple local rules [38,39], and coverage and "
  "patrolling control coordinate where robots <i>move</i> [40]; we instead address which <i>task</i> each "
  "robot should engage when the capability-to-task match is unknown and there is no communication.</p>")
A("<p><b>Multi-robot task allocation.</b> The taxonomies of Gerkey and Matari&cacute; [1], Korsah et al. "
  "[2], and Nunes et al. [20] organize MRTA by single/multi-task robots, single/multi-robot tasks, and "
  "instantaneous/time-extended assignment with interrelated utilities and temporal constraints. Classical "
  "solvers, market and auction mechanisms [3], consensus-based bundle algorithms (CBBA) [4], and "
  "distributed constraint optimization achieve coordination through <b>communication</b> and assume "
  "<b>known</b> task utilities or costs (surveys: [21]); even work that explicitly limits communication "
  "still relies on auctions and messages [22], and a recent review underscores how central communication "
  "remains to multi-robot systems [23]. Trait-based and heterogeneity-aware MRTA [5,24,25] matches robot "
  "capability vectors to task requirement vectors, but takes the traits as <b>given</b>. Our setting "
  "keeps the trait/low-rank view but makes the traits <b>unknown and learned online</b>, with neither "
  "communication nor known utilities; recent surveys [18] document the rapid growth of learning-based "
  "MRTA, but the prior-free, communication-free regime we study remains unaddressed.</p>")
A("<p><b>Decentralized and learning-based coordination.</b> Communication-free multiplayer bandits "
  "(musical chairs [6], SIC-MMAB [7], game-of-thrones [26]) break symmetry without messages but are "
  "<b>structure-free</b> (per-arm), so they cannot generalize to unseen arms. Cooperative multi-agent RL "
  "(CTDE: MAPPO [14], QMIX [33], VDN [34], MADDPG [35]) and learned-communication methods (CommNet [15], "
  "TarMAC [36], DIAL [37]) rely on centralized training or message passing; recent learning-based "
  "decentralized assignment (graph neural networks for goal assignment [19] and scheduling [41]) likewise "
  "presumes communication or centralized training. Federated and gossip collaborative filtering [42], and "
  "federated learning more broadly [44], coordinate by exchanging model parameters or gradients; we "
  "exchange nothing but a passively-sensed outcome stream. Decentralized partially-observable control "
  "(Dec-POMDPs [46]) addresses long-horizon coordination under shared latent-state dynamics; our rounds "
  "are one-shot offered-set choices with no shared state, so the challenge is cross-task generalization "
  "from a privately-masked stream rather than long-horizon credit assignment.</p>")
A("<p><b>Low-rank estimation and bandits.</b> Matrix completion gives centralized recovery guarantees "
  "under (near-)uniform sampling [8,9,27,28], with practical factorization estimators (matrix "
  "factorization [13], Bayesian PMF [11], soft-impute [12], implicit-feedback ALS [31], ranking CF [32]); "
  "low-rank and bilinear bandits (explore-then-spectral [10], bilinear [29], generalized-linear [30], "
  "clustering-of-bandits [16]) are centralized and/or phase-structured. Decentralized matrix completion "
  "[43] distributes the factorization across nodes but still exchanges factors or residuals over a "
  "connected communication graph; we forbid all such exchange and recover from passive observation alone. "
  "We make estimation decentralized, "
  "online, broadcast-only, and robust to a structured (non-uniform) per-robot observation mask, with the "
  "unseen-pair error floor turning the gap into a categorical, rather than constant-factor, separation.</p>")
A("<p>Table 1 places the major paradigms on the four axes that define our problem; each established "
  "family relaxes at least one axis we hold fixed, and our cell, low-rank with only a guessed rank, no "
  "communication, decentralized, masked and noisy, is the one left open.</p>")
A("<p class='small'><b>Table 1.</b> Established paradigms versus our regime across prior knowledge, "
  "communication, decision locus, and observation; the final column names the constraint each relaxes "
  "(ours relaxes none).</p>")
A(mp.html_paradigms())

# ---------------- 3. problem setting ----------------
A("<h2>3. Problem setting: Zero-Knowledge MRTA (ZK-MRTA)</h2>")
A("<p><b>Reward.</b> A team of $m$ robots faces $n$ tasks. Robot $i$ has a hidden capability vector "
  "$p_i\\in\\mathbb{R}^d$ and task $j$ a hidden requirement vector $u_j\\in\\mathbb{R}^d$; the expected "
  "reward of robot $i$ engaging task $j$ is their inner product"
  "$$ R_{ij} \\;=\\; \\langle p_i, u_j\\rangle, \\qquad R = P U^\\top \\in \\mathbb{R}^{m\\times n},\\ "
  "\\operatorname{rank}(R)=d \\ll \\min(m,n). $$"
  "The low rank $d$ encodes that only a few traits govern fit. The team does <b>not</b> know $P$, $U$, or "
  "even $d$ (it uses a guessed rank $\\hat d$). We take the reward in normalized form, scaling the latent "
  "traits so that $R_{ij}$ is bounded and zero-mean across tasks; this is the bounded, normalized reward "
  "referenced by Proposition 1.</p>")
A("<p><b>Interaction.</b> Each round $t=1,\\dots,T$ every robot $i$ is offered a uniform random size-$c$ "
  "subset $S_{it}\\subseteq[n]$ of tasks, selects one $a_{it}\\in S_{it}$, engages it, and earns "
  "$R_{i,a_{it}}$. The operating regime is <b>task-scarce</b>: $n \\gg T$, so each robot personally "
  "engages only $O(T)$ of the $n$ tasks.</p>")
A("<div class='box'><b>The observation channel (the heart of the setting).</b> There is no communication. "
  "Each robot instead <b>passively senses</b> a public stream of engagement outcomes, but only partially "
  "and noisily, and <b>privately</b>: <b>(persistent partial visibility)</b> robot $i$ observes "
  "teammate $k$'s engagements only if $M_{ik}=1$, where $M_{ik}\\sim\\mathrm{Bernoulli}(\\rho)$ is fixed "
  "for the whole mission ($M_{ii}=1$); <b>(private per-observer noise)</b> when $i$ observes the outcome "
  "of action $a_{kt}$ it reads $R_{k,a_{kt}}+\\eta_{ikt}$ with $\\eta_{ikt}\\sim\\mathcal N(0,\\sigma^2)$ "
  "drawn independently per observer, so the same action is read differently by different robots. No robot "
  "ever sees the clean stream, and no two robots see the same stream.</div>")
A("<p>This channel is the formal counterpart of physical sensing: a robot perceives a teammate's "
  "engagement and its outcome only when the teammate is within range (the persistent partial mask) and "
  "reads it with a fidelity that degrades with distance and with its own sensor (the per-observer noise), "
  "so it is physically realizable rather than a convenient abstraction. It is strictly weaker than the "
  "shared, clean broadcast usually assumed, and it makes decentralization <b>real</b>: persistent blind "
  "spots give every robot a permanently different view, and the private per-observer noise means even "
  "commonly-visible outcomes are read differently by each robot, so there is no shared, clean signal to "
  "average toward agreement and the robots cannot converge to a common model by symmetry. Figure 1 "
  "illustrates the setting.</p>")
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
  "uniformly; skill $=0$ is the no-information floor and $1$ is omniscient (a policy worse than random "
  "scores below $0$). We report <b>unseen-pair "
  "skill</b> (restricted to tasks the robot never engaged, the generalization test), and the "
  "<b>anytime</b> (cumulative-reward) skill over the mission.</p>")

# ---------------- 4. method ----------------
A("<h2>4. Method: SwarmCF</h2>")
A("<p>Each robot independently maintains low-rank factor estimates $\\hat P\\in\\mathbb{R}^{m\\times\\hat "
  "d}$, $\\hat U\\in\\mathbb{R}^{n\\times\\hat d}$ and updates them online from whatever it senses, using "
  "<b>noise-weighted alternating least squares</b> (weighted ridge ALS [28,31], the workhorse of low-rank "
  "completion) on the observed entries. The key design choice for masking-robustness: an unobserved "
  "entry receives zero <i>weight</i>, never an imputed zero <i>value</i>. The robot then acts greedily "
  "(with a small $\\varepsilon$ for exploration) on its <b>completed</b> reward row, which is defined for "
  "every task, including ones it never engaged.</p>")
A("<div class='algo'><div class='cap'>Algorithm 1: SwarmCF (run independently by each robot $i$)</div>"
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
A("<p><b>New-task onboarding (fold-in [48]).</b> Because the swarm already holds the robot-factor basis, "
  "a new task $j^\\star$ is absorbed without retraining: its hidden vector is the ridge least-squares "
  "solution of its few observed engagements against the corresponding known robot factors, an "
  "$O(\\hat d^3)$ computation, after which every robot can score $j^\\star$. The same fold-in lets a robot "
  "predict any unseen pair once it has recovered the basis (Algorithm 2). <b>A new robot is different:</b> "
  "it arrives with no memory, and with no communication it cannot be handed the basis, so it must first "
  "recover the task factors from the passive broadcast (the coverage time of Theorem 3) and only then fold "
  "in its own $\\ge\\hat d$ engagements; its onboarding is bounded by recovery, not by the $O(\\hat d)$ "
  "fold-in.</p>")
A("<div class='algo'><div class='cap'>Algorithm 2: Fold-in (solve a new entity's $\\hat d$-vector from a few observations against a known basis $B$)</div>"
  "given basis $B$ (the $\\hat d$-dim factors of the entities the newcomer has been observed against)\n"
  "and observations $y$ (the few rewards seen for the newcomer), with weights $W$:\n"
  "    $\\hat x \\leftarrow (B^\\top W B+\\lambda I)^{-1} B^\\top W y$    // $O(\\hat d^3)$ ridge solve\n"
  "predict reward on any other entity with factor $b$ as $\\langle b,\\hat x\\rangle$</div>")
A("<p><b>What SwarmCF does and does not assume.</b> It is fully decentralized (one estimator per robot, "
  "no shared state), communication-free (it only reads the passive stream), and prior-free beyond a "
  "guessed rank $\\hat d$ (which it does not need to be exact, and which can itself be removed by a "
  "rank-adaptive variant we defer to follow-up work). It does not assume the noise level is known: "
  "uniform weighting suffices and is what we use for the headline results.</p>")
A("<p><b>Computational cost.</b> SwarmCF is light enough to run on each robot. Acting is $O(c\\hat d)$ per "
  "round (score the offered set). A periodic refit is a few alternating ridge sweeps, each solving "
  "$O(m+n)$ linear systems of size $\\hat d\\times\\hat d$, i.e. $O\\big(\\text{sweeps}\\cdot(m+n)\\hat "
  "d^3 + |\\Omega_i|\\hat d^2\\big)$ time, with $O((m+n)\\hat d + |\\Omega_i|)$ memory; since $\\hat d$ is "
  "a small guessed rank this is negligible for swarm-scale $m,n$. Folding a new task (or an unseen pair) "
  "into the recovered basis is a single $O(\\hat d^3)$ ridge solve. There is no inter-robot computation: each robot updates only its "
  "own factors and reads the passive stream.</p>")

# ---------------- 5. theory ----------------
A("<h2>5. Theory: why the advantage is categorical</h2>")
A("<p>We formalize the separation between structure-free learning and SwarmCF, give the per-robot sample "
  "complexity, and state the conditions under which decentralized recovery from the masked broadcast "
  "succeeds. Proofs are in Appendix A; here we give the statements and the intuition. A learner is "
  "<b>structure-free</b> if its estimate of $R_{ij}$ depends only on robot $i$'s own past engagements of "
  "task $j$ and equals a fixed prior on any task it never engaged (the per-arm class: independent UCB, "
  "tabular).</p>")
A("<div class='thm'><b>Proposition 1 (structure-free floor).</b> For a structure-free learner and any task "
  "$j$ that robot $i$ never engaged, the estimate is the prior constant, so its expected unseen-pair "
  "skill is exactly $0$ and its squared error is at least the row variance $\\Omega(1)$ (under the bounded, "
  "normalized reward of Section 3). Moreover the "
  "broadcast is provably useless to it: its per-task estimate is by definition not a function of any "
  "other task or robot.</div>")
A("<div class='thm'><b>Theorem 1 (CF row completion, $\\Theta(d)$ versus $\\Theta(n)$).</b> If the task "
  "factors $U$ are known (rank $d$) and robot $i$ observes its true rewards on any set $\\Omega$ with "
  "$|\\Omega|\\ge d$ whose factors span $\\mathbb{R}^d$, then $p_i$ is the unique least-squares solution "
  "and $R_{ij}=\\langle p_i,u_j\\rangle$ is recovered <b>exactly for all</b> $j$. Per-robot sample "
  "complexity is therefore $\\Theta(d)$, versus $\\Theta(n)$ for any structure-free learner.</div>")
A("<div class='thm'><b>Theorem 2 (anytime separation under task scarcity).</b> With $n$ tasks, offers of "
  "size $c$ and horizon $T$, any structure-free learner has anytime (cumulative-reward) skill at most "
  "$g(cT/n)$, where $cT/n$ is the expected number of times each task is offered over the mission and $g$ "
  "is a concave, increasing order-statistic function with $g(0)=0$ (made precise in Appendix A), so the "
  "bound vanishes when $cT=o(n)$, even with a full broadcast; once the shared basis is available "
  "(Theorem 1, recovered per Theorem 3) SwarmCF reaches near-oracle in $O(d)$ further rounds and earns "
  "$\\Theta(1)$ anytime skill, a categorical gap.</div>")
A("<p>Proposition 1 and Theorems 1-2 make the separation categorical (zero versus nonzero) and "
  "operational (it shows up in reward earned while learning), but Theorem 1 assumes $U$ is known. The "
  "remaining question, the crux "
  "of the decentralized setting, is whether each robot can <b>recover</b> the shared structure from its "
  "own privately-masked, noisy stream. Because the mask is over robot pairs, robot $i$'s observations form "
  "a structured (non-uniform) sub-sample, exactly where off-the-shelf uniform-sampling completion does "
  "not apply. We give a deterministic condition instead.</p>")
A("<div class='thm'><b>Theorem 3 (decentralized masked recovery).</b> Let $E_i(j)$ be the set of robot "
  "$i$'s visible teammates that engaged task $j$, with factor matrix $B=P_{E_i(j)}$, and suppose $i$'s "
  "observation graph contains a $\\hat d\\times\\hat d$ fully-observed invertible anchor block (fixing the "
  "global factor frame, i.e. the rotation gauge). Then robot $i$ predicts the pair $(i,j)$ exactly (noiseless) <b>iff</b> "
  "$p_i\\in\\mathrm{span}\\{p_k:k\\in E_i(j)\\}$; the full task vector $u_j$ is recovered when those "
  "factors span $\\mathbb{R}^d$. Under per-observation noise the prediction error is "
  "$O(\\sigma\\sqrt{d}/\\sigma_{\\min}(B))$, where $\\sigma_{\\min}(B)$ is the smallest singular value of "
  "the engager factor matrix $B$; since $\\sigma_{\\min}(B)$ grows with the number of "
  "engagers (generically $\\Theta(\\sqrt{|E_i(j)|})$) the error decreases as $O(\\sigma\\sqrt{d/|E_i(j)|})$. "
  "Conversely, if $p_i\\notin\\mathrm{span}\\{p_k:k\\in E_i(j)\\}$ the pair is non-identifiable and the "
  "learner is at the prior floor. Under non-adaptive exploration the condition holds for all tasks, with "
  "high probability, after $T=O\\!\\big(\\tfrac{nd}{\\rho m}\\log n\\big)$ rounds, a rate that improves as "
  "the team grows.</div>")
A("<p>Theorem 3 turns the previously-cited completion step into a self-contained, checkable condition: a "
  "pair $(i,j)$ is predicted exactly when robot $i$'s own factor lies in the span of the visible "
  "teammates that engaged $j$, and not otherwise, with the threshold reached faster the larger the team. "
  "Appendix C validates it directly: on the actual observation patterns the swarm produces, "
  "reconstruction error collapses from the prior floor to (numerically) zero exactly at the spanning "
  "threshold.</p>")
A("<div class='thm'><b>Theorem 4 (collective speedup, why a swarm).</b> An isolated robot ($\\rho=0$) "
  "sees only its own row and cannot identify a rank-$d>1$ column space, so its unseen skill stays at the "
  "floor: sharing is <b>necessary</b>. With the broadcast, the swarm's pooled observations cross the "
  "recovery threshold after $\\tilde O(d(1+n/m))$ rounds ($\\tilde O$ suppresses logarithmic factors), a "
  "time that shrinks as $1/m$ as the team grows "
  "while a lone learner never crosses it: the broadcast makes recovery possible and a larger team makes "
  "it fast, with no communication.</div>")
A("<p>Theorems 3-4 are, to our knowledge, the first results that pin decentralized low-rank recovery from "
  "a persistent, private, per-robot mask to an explicit condition and tie its rate to team size; they are "
  "what make the categorical claim self-contained rather than imported from centralized theory. Full "
  "proofs are given in Appendix A.</p>")

# ---------------- 6. experiments ----------------
A("<h2>6. Experiments</h2>")
A("<p><b>Setup.</b> Unless noted, $m=30$ robots, $n=240$ tasks, true rank $d=5$, guessed rank "
  "$\\hat d=8$ (so no method is given the rank), horizon $T=50$, offers of size $c=20$, partial "
  "broadcast $\\rho$ swept, private noise on own ($\\sigma_{\\mathrm{own}}=0.1$) and observed "
  "($\\sigma_{\\mathrm{obs}}=0.3$) outcomes (Appendix D), 8 random seeds "
  "(the consolidated bake-off of Table 3 uses 5), bootstrap 95% confidence intervals. We compare on one "
  "canonical masked harness, the analytical mode of our released <b>LatentSwarm</b> suite (Section 6.5 "
  "evaluates its spatial environment).</p>")
A("<p><b>How to read the comparison.</b> The setting itself is new, so this is a controlled sweep across "
  "the low-rank design space against the genuinely external structure-free paradigm and "
  "full-information reference ceilings, not a contest of rival systems. SwarmCF is our method; structure-"
  "free learners (independent UCB1 [45], tabular) are the external paradigm; standard low-rank estimators "
  "(online SGD-MF, batch spectral/Bayesian/convex completion) are adapted to the setting for the "
  "low-rank comparison; the Oracle and a centralized full-communication matcher are upper bounds, not "
  "competitors. We emphasize that communication-based methods (auctions/CBBA, CTDE training, "
  "federated/gossip CF) are <b>inadmissible by the problem definition</b>, not omitted: they require "
  "messages or a coordinator, which our setting forbids, so they can appear only as the centralized "
  "ceilings. The admissible communication-free competitors are exactly the structure-free learners "
  "(independent UCB / tabular), which are the per-arm reductions of no-communication multiplayer-bandit "
  "methods. In our harness <b>every</b> method runs decentralized and communication-free (one estimator "
  "per robot); the low-rank methods differ only in the update rule. Table 2 fixes each method's operating "
  "profile.</p>")
A("<p class='small'><b>Table 2.</b> Operating profiles of the methods compared; SwarmCF-family "
  "refinements are deferred to future work (column abbreviations are defined in the key).</p>")
A(mp.html_profiles(subset=["RewardCF", "MFSGD", "ESTR", "PTF", "BPMF", "SoftImpute",
                           "UCBIndep", "Tabular", "Random", "CentralClean-ceiling", "CTDE-ceiling", "Oracle"]))

A("<h3>6.1 The categorical win and masking robustness</h3>")
A("<p>Figure 2 sweeps the broadcast rate $\\rho$ and reports unseen-pair skill. SwarmCF acts well on "
  "tasks it never engaged at every broadcast rate, while the structure-free learners sit at the "
  "floor ($\\approx 0$) by construction, the categorical separation of Proposition 1 and Theorem 1. Among low-rank "
  "methods, SwarmCF's online updates stay robust as the broadcast is masked, whereas batch spectral "
  "completion (which imputes unobserved entries) decays; the two cross over near $\\rho\\approx 0.6$ and "
  "batch wins only at full broadcast, where its one-shot factorization on a near-complete matrix is the "
  "best case for completion. The operationally relevant regime, partial observation, is exactly where "
  "the online estimator leads.</p>")
A("<figure>%s<figcaption><b>Figure 2.</b> Unseen-pair skill versus broadcast rate $\\rho$. Structure-"
  "free learners are pinned at the floor at every broadcast rate; SwarmCF acts on the "
  "unseen throughout and is robust under masking, while batch spectral completion decays as observation "
  "becomes partial.</figcaption></figure>" % img("F5_crossover.png", "categorical + masking"))

A("<h3>6.2 The operational (anytime) separation</h3>")
A("<p>Final-policy quality can flatter a method that explores cheaply. The operationally honest measure "
  "is reward <i>earned while learning</i>. Figure 3 shows cumulative-reward skill over the mission: "
  "SwarmCF earns from the first rounds. Per-arm UCB stays near the random floor: with $n\\gg T$ arms its "
  "optimism keeps it exploring untried tasks. Phase-structured low-rank methods pay an explore-then-commit "
  "penalty early. An $\\varepsilon$-greedy tabular learner does earn some reward by re-exploiting tasks it "
  "has already engaged (each task recurs in offers about $cT/n\\approx 4$ times here), but it stays well "
  "below SwarmCF and at the floor on unseen pairs (Table 3). The clean anytime collapse of Theorem 2 holds "
  "in the strict regime $cT=o(n)$; SwarmCF's early-earning advantage is broader, as seen here.</p>")
A("<figure>%s<figcaption><b>Figure 3.</b> Anytime cumulative-reward skill ($\\rho=0.25$). SwarmCF earns "
  "from round one; explore-then-commit pays a probe phase; per-arm UCB stays near the random floor, while "
  "$\\varepsilon$-greedy tabular earns only by re-exploiting tasks it has already "
  "engaged.</figcaption></figure>" % img("F6_anytime.png", "anytime"))

A("<h3>6.3 Why a swarm: the value of the broadcast and a positive scaling law</h3>")
A("<p>Two experiments isolate what the team and the broadcast actually buy (Figure 4). "
  "<i>(a) Value of the broadcast.</i> Sweeping from $\\rho=0$ (each robot isolated, sees only its own "
  "outcomes) to $\\rho=1$ (full passive sensing): a lone robot cannot recover the shared structure from "
  "its single matrix row, so isolated unseen skill is $\\approx 0$; the broadcast lifts SwarmCF by "
  "$+0.39$ unseen skill but a structure-free learner by $\\approx 0$, which has no model linking tasks "
  "and so cannot use it (Theorem 4). <i>(b) Positive scaling.</i> Holding $n$, horizon and $\\rho$ "
  "fixed and growing the team from $m=5$ to $80$, SwarmCF's unseen skill rises monotonically "
  "($0.08\\to 0.43$): more robots feed more observations into the one shared structure, so each robot's "
  "competence on tasks it never engaged grows with the team. The batch variant SwarmCF-batch rises even "
  "more steeply (Figure 4b), overtaking the online variant for large teams as the pooled observations "
  "sharpen its one-shot refit (mirroring the broadcast-rate crossover of Figure 2), so the positive "
  "scaling is a structural property of the shared low-rank reward, not specific to the online update. "
  "Structure-free learning is flat. A swarm "
  "that gets smarter as it grows, the opposite of the usual interference penalty, is a direct consequence "
  "of sharing structure.</p>")
A("<figure>%s<figcaption><b>Figure 4.</b> (a) Value of the broadcast: unseen skill versus $\\rho$ "
  "(left edge $=$ isolated). (b) Positive scaling: unseen skill versus team size $m$ at fixed broadcast "
  "rate $\\rho=0.5$ (and fixed $n$, horizon $T$). In both panels our online SwarmCF and its batch variant "
  "SwarmCF-batch rise (the gain is structural), while structure-free learners stay "
  "flat.</figcaption></figure>" % img("F18_collab_scaling.png", "why a swarm"))

A("<h3>6.4 An operational mission and a centralized ceiling</h3>")
A("<p>Framed as a target-servicing / dispatch mission (still the analytical harness of Section 3, reframed), latent factors are robot capability traits and "
  "task requirement traits, each robot repeatedly services an offered target under range-limited, "
  "distance-noisy sensing, SwarmCF on the same servicing-skill metric beats every external low-rank "
  "method and our batch variant SwarmCF-batch under limited observability ($\\rho=0.25$): "
  "$\\approx 0.35$ versus the best external alternative "
  "$\\approx 0.29$ with non-overlapping intervals, while structure-free learners sit at the random-"
  "dispatch floor (the same categorical separation as Figures 2-3, in operational form). To bound the "
  "cost of our constraints we add two reference ceilings (not "
  "competitors): a centralized full-communication matcher with Hungarian assignment [47] (optimal "
  "one-to-one robot-task matching), and the same "
  "with noiseless, unmasked observation. When targets are plentiful (low contention), SwarmCF recovers "
  "about 80% of the full-communication ceiling (servicing skill $\\approx 0.44$ versus $\\approx 0.55$ in "
  "the low-contention regime of the controlled contention sweep; per-seed values are in the released "
  "data); the residual gap is the price of within-round coordination under contention, which a communication-free "
  "de-confliction mechanism (deferred to future work) is designed to close. The two ceilings differ "
  "little (both $\\approx 0.5$), so the price of observation noise is small: coordination, not estimation, "
  "is the binding constraint.</p>")
A("<p>Table 3 consolidates the comparison on the canonical masked harness.</p>")
A("<p class='small'><b>Table 3.</b> Performance scorecard on one canonical masked harness.</p>")
A(mp.html_scorecard(ROOT))

A("<h3>6.5 Robustness and transfer to a higher-fidelity simulator</h3>")
A("<p><b>Beyond one configuration.</b> The separation is structural rather than tuned: it follows from "
  "the three scope conditions stated below (an exploitable low-rank-but-personalized reward, task "
  "scarcity, and a shared channel), not from any particular team size or task count, and we observe the "
  "same pattern across $m$, $n$, $d$, and heterogeneity in additional sweeps. The most demanding "
  "test, transfer to our higher-fidelity <i>LatentSwarm</i> simulator, is reported next.</p>")
A("<p><b>Transfer to LatentSwarm, a higher-fidelity simulator.</b> The results so far use the clean "
  "analytical harness that instantiates the model of Section 3 exactly. To assess whether the method "
  "generalizes beyond that model, we re-run the comparison in the spatial environment of our "
  "<b>LatentSwarm</b> suite, a multi-robot reinforcement-learning environment (PettingZoo/Gymnasium). It realizes a "
  "<i>variant</i> of the setting rather than the model itself: each round every robot selects a target "
  "and earns the <i>signed-cosine</i> alignment of their latent capability and requirement traits, a "
  "normalized variant of the Section 3 inner-product reward (with the same low-rank trait structure and "
  "per-observer noise), while the environment adds a fixed 2-D spatial "
  "layout, depleting-target health (reduced by the rectified trait match), capacity contention "
  "(collisions when robots select the same target), and episodic resets; its dynamics were specified "
  "independently of the collaborative-filtering method (Appendix E gives the exact construction and "
  "pseudocode). Dropped in as one policy alongside the simulator's own built-in "
  "policies, SwarmCF attains the best converged skill ($0.806$, std $0.016$ over 3 seeds), approaching the oracle and "
  "beating both an independent-UCB learner ($0.721$) and the simulator's own SGD matrix-factorization "
  "policy ($0.251$) (Figure 5). In this smaller, less task-scarce environment a structure-free learner is "
  "no longer pinned at the floor, exactly as our scope predicts, yet SwarmCF still wins; that the "
  "advantage persists under these different dynamics is evidence it reflects exploiting the shared "
  "low-rank structure, not an artifact of the analytical harness.</p>")
A("<figure>%s<figcaption><b>Figure 5.</b> Transfer to <b>LatentSwarm</b>, our higher-fidelity spatial "
  "simulator and a variant of the setting (fixed 2-D layout, depleting-target health, episodic dynamics). Left: "
  "converged skill, SwarmCF best and near-oracle, above independent-UCB and the simulator's own SGD-MF "
  "policy. Right: learning curves.</figcaption></figure>" % img("F13_realsim.png", "LatentSwarm simulator transfer"))

A("<div class='box'><b>Scope: when does SwarmCF beat structure-free learning?</b> The advantage is not "
  "universal, and we state its boundary precisely. It holds when three conditions co-occur: "
  "<b>(i) low-rank but personalized</b> structure ($1\\lt d\\ll\\min(m,n)$): at $d=1$ the reward reduces to a "
  "shared popularity order that a bias/pooling baseline already captures, so there is nothing personal to "
  "transfer; <b>(ii) task scarcity</b> ($n\\gg T$): if instead sample-rich, a tabular learner eventually "
  "measures every entry and the unseen advantage vanishes; <b>(iii) a shared channel</b> ($\\rho>0$): "
  "with no broadcast each robot has only its own row and collaborative filtering degenerates to tabular. "
  "These are exactly the conditions of the regime we target, and they delimit honestly where "
  "structure-free methods remain competitive.</div>")

# ---------------- 7. discussion ----------------
A("<h2>7. Discussion, limitations, and future work</h2>")
A("<p><b>What the results say.</b> Under the least information, no prior, no communication, partial and "
  "privately-noisy observation, a single simple estimator gives a swarm a capability that structure-free "
  "learning provably cannot have: acting well on the unseen, getting more competent as the team grows, "
  "and recovering most of what a centralized, communicating system could achieve. The win is structural "
  "(it is a property of exploiting the shared low-rank trait structure) and operational (it shows up in "
  "reward earned while learning, and in a robotics-grounded mission).</p>")
A("<p><b>Limitations.</b> The reward is assumed (approximately) low-rank and stationary, the standard "
  "trait-based premise; the categorical advantage degrades gracefully as the structure becomes only "
  "approximately low-rank or the rank grows toward full, vanishing only "
  "when there is no exploitable structure. Rewards are real-valued and bilinear in latent traits; and the "
  "recovery rate of Theorem 3 is established for non-adaptive "
  "exploration, the finite-time rate under a strongly exploiting policy (which can starve low-reward "
  "tasks of coverage) remains open. We report the regime boundaries honestly: the advantage requires "
  "structure beyond mere popularity ($d>1$), task scarcity ($n\\gg T$), and a shared channel "
  "($\\rho>0$); outside these, structure-free methods are competitive.</p>")
A("<p><b>Future work (a planned follow-up).</b> The present paper deliberately keeps to a single core "
  "estimator. In a follow-up we plan to study the refinements that this foundation enables, each of "
  "which we have prototyped: <i>(i)</i> confidence-directed exploration via a Bayesian posterior over the "
  "factors (collective, information-directed probing through the shared broadcast); <i>(ii)</i> "
  "communication-free de-confliction under capacity-1 contention via a fixed private offset, against "
  "no-communication auction and musical-chairs primitives; <i>(iii)</i> rank self-determination "
  "(removing the guessed $\\hat d$); <i>(iv)</i> the action/choice channel as a noise-immune alternative "
  "to cardinal rewards; and <i>(v)</i> non-stationarity and team churn. We also plan external "
  "validation on independent benchmarks (an embodied "
  "multi-agent environment such as level-based foraging, and a low-rank-native recommender or "
  "bilinear-bandit testbed such as RecoGym, where the low-rank reward holds by construction), and a "
  "tightening of the adaptive-policy coverage rate.</p>")

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

# ---------------- declarations ----------------
A("<h2>Declaration of competing interest</h2>")
A("<p class='small'>The authors declare that they have no known competing financial interests or personal "
  "relationships that could have appeared to influence the work reported in this paper.</p>")
A("<h2>Data availability</h2>")
A("<p class='small'>The source code, the per-seed data required to reproduce every figure and table, and "
  "the <b>LatentSwarm</b> evaluation suite (the analytical masked-broadcast harness and the spatial "
  "environment) are openly available at "
  "<a href='https://github.com/ApartsinProjects/ZKDroneSwarm'>github.com/ApartsinProjects/ZKDroneSwarm</a>.</p>")

# ---------------- references + appendices ----------------
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
  "<li>Athira K. A., J. Divya Udayan, U. Subramaniam. A systematic literature review on multi-robot "
  "task allocation. <i>ACM Computing Surveys</i>, 2024.</li>"
  "<li>P. Goarin, G. Loianno. Graph neural network for decentralized multi-robot goal assignment. "
  "<i>IEEE Robotics and Automation Letters</i>, 2024.</li>"
  "<li>E. Nunes, M. Manner, H. Mitiche, M. Gini. A taxonomy for task allocation problems with temporal "
  "and ordering constraints. <i>Robotics and Autonomous Systems</i>, 90:55&ndash;70, 2017.</li>"
  "<li>A. Khamis, A. Hussein, A. Elmogy. Multi-robot task allocation: a review of the state-of-the-art. "
  "In <i>Cooperative Robots and Sensor Networks</i>, Springer, 2015.</li>"
  "<li>M. Otte, M. J. Kuhlman, D. Sofge. Auctions for multi-robot task allocation in communication "
  "limited environments. <i>Autonomous Robots</i>, 44:547&ndash;584, 2020.</li>"
  "<li>J. Gielis, A. Shankar, A. Prorok. A critical review of communications in multi-robot systems. "
  "<i>Current Robotics Reports</i>, 3:213&ndash;225, 2022.</li>"
  "<li>G. Notomista, S. Mayya, S. Hutchinson, M. Egerstedt. An optimal task allocation strategy for "
  "heterogeneous multi-robot systems. <i>European Control Conference</i>, 2019.</li>"
  "<li>H. Ravichandar, K. Shaw, S. Chernova. STRATA: a unified framework for task assignments in large "
  "teams of heterogeneous agents. <i>Autonomous Agents and Multi-Agent Systems</i>, 34:38, 2020.</li>"
  "<li>I. Bistritz, A. Leshem. Distributed multi-player bandits: a game of thrones approach. "
  "<i>NeurIPS</i>, 2018.</li>"
  "<li>B. Recht. A simpler approach to matrix completion. <i>JMLR</i>, 12:3413&ndash;3430, 2011.</li>"
  "<li>P. Jain, P. Netrapalli, S. Sanghavi. Low-rank matrix completion using alternating minimization. "
  "<i>STOC</i>, 2013.</li>"
  "<li>K.-S. Jun, R. Willett, S. Wright, R. Nowak. Bilinear bandits with low-rank structure. "
  "<i>ICML</i>, 2019.</li>"
  "<li>Y. Lu, A. Meisami, A. Tewari. Low-rank generalized linear bandit problems. <i>AISTATS</i>, "
  "2021.</li>"
  "<li>Y. Hu, Y. Koren, C. Volinsky. Collaborative filtering for implicit feedback datasets. "
  "<i>ICDM</i>, 2008.</li>"
  "<li>S. Rendle, C. Freudenthaler, Z. Gantner, L. Schmidt-Thieme. BPR: Bayesian personalized ranking "
  "from implicit feedback. <i>UAI</i>, 2009.</li>"
  "<li>T. Rashid, M. Samvelyan, et al. QMIX: monotonic value function factorisation for deep multi-agent "
  "reinforcement learning. <i>ICML</i>, 2018.</li>"
  "<li>P. Sunehag, G. Lever, et al. Value-decomposition networks for cooperative multi-agent learning. "
  "<i>AAMAS</i>, 2018.</li>"
  "<li>R. Lowe, Y. Wu, et al. Multi-agent actor-critic for mixed cooperative-competitive environments. "
  "<i>NeurIPS</i>, 2017.</li>"
  "<li>A. Das, T. Gervet, et al. TarMAC: targeted multi-agent communication. <i>ICML</i>, 2019.</li>"
  "<li>J. Foerster, Y. Assael, N. de Freitas, S. Whiteson. Learning to communicate with deep multi-agent "
  "reinforcement learning. <i>NeurIPS</i>, 2016.</li>"
  "<li>M. Brambilla, E. Ferrante, M. Birattari, M. Dorigo. Swarm robotics: a review from the swarm "
  "engineering perspective. <i>Swarm Intelligence</i>, 7:1&ndash;41, 2013.</li>"
  "<li>E. &#350;ahin. Swarm robotics: from sources of inspiration to domains of application. "
  "<i>LNCS 3342</i>, Springer, 2005.</li>"
  "<li>J. Cort&eacute;s, S. Mart&iacute;nez, T. Karatas, F. Bullo. Coverage control for mobile sensing "
  "networks. <i>IEEE Trans. Robotics and Automation</i>, 20(2):243&ndash;255, 2004.</li>"
  "<li>Z. Wang, M. Gombolay. Learning scheduling policies for multi-robot coordination with graph "
  "attention networks. <i>IEEE Robotics and Automation Letters</i>, 5(3):4509&ndash;4516, 2020.</li>"
  "<li>M. Ammad-ud-din, E. Ivannikova, S. A. Khan, W. Oyomno, Q. Fu, K. E. Tan, A. Flanagan. Federated "
  "collaborative filtering for privacy-preserving personalized recommendation system. "
  "<i>arXiv:1901.09888</i>, 2019.</li>"
  "<li>Q. Ling, Y. Xu, W. Yin, Z. Wen. Decentralized low-rank matrix completion. <i>IEEE ICASSP</i>, "
  "2925&ndash;2928, 2012.</li>"
  "<li>B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Ag&uuml;era y Arcas. Communication-efficient "
  "learning of deep networks from decentralized data. <i>AISTATS</i>, 1273&ndash;1282, 2017.</li>"
  "<li>P. Auer, N. Cesa-Bianchi, P. Fischer. Finite-time analysis of the multiarmed bandit problem. "
  "<i>Machine Learning</i>, 47(2&ndash;3):235&ndash;256, 2002.</li>"
  "<li>D. S. Bernstein, R. Givan, N. Immerman, S. Zilberstein. The complexity of decentralized control "
  "of Markov decision processes. <i>Mathematics of Operations Research</i>, 27(4):819&ndash;840, 2002.</li>"
  "<li>H. W. Kuhn. The Hungarian method for the assignment problem. <i>Naval Research Logistics "
  "Quarterly</i>, 2(1&ndash;2):83&ndash;97, 1955.</li>"
  "<li>B. Sarwar, G. Karypis, J. Konstan, J. Riedl. Incremental singular value decomposition algorithms "
  "for highly scalable recommender systems. <i>Proc. 5th Int. Conf. Computer and Information "
  "Technology</i>, 2002.</li>"
  "</ol>")

A("<h2>Appendix A. Proofs of the main results</h2>")
A("<p class='small'>We give self-contained proofs of the main results below, each closed by a short "
  "remark on the proof technique and its relation to existing results. Throughout, factors are assumed "
  "in general position (generic $P,U$), the persistent mask is $M_{ik}\\sim\\mathrm{Bernoulli}(\\rho)$ "
  "over robot pairs, and $\\hat d\\ge d$.</p>")
A("<p class='small'><b>Proposition 1 (floor).</b> For an unobserved $j$ the estimate is a pre-chosen "
  "constant $b$; $\\mathbb{E}[(b-R_{ij})^2]=(b-\\mu_i)^2+\\mathrm{Var}_j\\ge\\mathrm{Var}_j=\\Omega(1)$, "
  "and on an offer of never-engaged tasks selection is independent of their rewards, giving skill $0$. "
  "The broadcast cannot help a per-task estimate by definition. <i>Remark:</i> elementary; it fixes the "
  "floor against which the categorical claim is measured.</p>")
A("<p class='small'><b>Theorem 1 (row completion).</b> Stacking the observed entries gives "
  "$R_{i,\\Omega}=U_\\Omega p_i$; spanning makes $U_\\Omega$ full column rank, so "
  "$p_i=(U_\\Omega^\\top U_\\Omega)^{-1}U_\\Omega^\\top R_{i,\\Omega}$ is unique and exact, hence all "
  "$R_{ij}$. <i>Remark:</i> the linear algebra is standard given $U$; the contribution is the "
  "$\\Theta(d)$-versus-$\\Theta(n)$ contrast against the floor of Proposition 1.</p>")
A("<p class='small'><b>Theorem 2 (anytime).</b> A structure-free learner earns above the mean only on an "
  "offer containing an already-engaged task; the engaged set has size $\\le t-1$ and is reward-blind, so "
  "by concavity of expected order statistics the per-round surplus is $\\le g(c(t-1)/n)$; summing gives "
  "$\\le g(cT/n)\\to 0$. <i>Remark:</i> an order argument (the mechanism is exact, the constant loose); "
  "it justifies reporting the anytime metric alongside final-policy skill.</p>")
A("<p class='small'><b>Theorem 3 (recovery).</b> A fully-observed invertible $\\hat d\\times\\hat d$ "
  "block pins the factor frame; per task, $R_{E_i(j),j}=B\\,u_j$ with $B=P_{E_i(j)}$ determines $u_j$ "
  "uniquely iff $B$ has full column rank, and determines the pair $\\langle p_i,u_j\\rangle$ iff "
  "$p_i\\in\\mathrm{span}\\{p_k:k\\in E_i(j)\\}$ (the per-task analogue of Proposition 1). With noise the "
  "error is $O(\\sigma\\sqrt{d}/\\sigma_{\\min}(B))=O(\\sigma\\sqrt{d/|E_i(j)|})$ since generically "
  "$\\sigma_{\\min}(B)=\\Theta(\\sqrt{|E_i(j)|})$. Coverage time (non-adaptive policy): each visible teammate engages each task "
  "with probability $1/n$ per round, so it has engaged task $j$ at least once after $T$ rounds with "
  "probability $1-(1-1/n)^T\\approx 1-e^{-T/n}$; the number of distinct visible engagers of $j$ is "
  "$\\mathrm{Binomial}(|N_i|,1-e^{-T/n})$ with $|N_i|\\approx\\rho m$, and requiring $\\ge d$ spanning "
  "engagers for all $n$ tasks with a union bound gives $T=O\\!\\big(\\tfrac{nd}{\\rho m}\\log n\\big)$, "
  "which decreases in the team size $\\rho m$ and requires $\\rho m>d$. <i>Remark:</i> the condition is "
  "deterministic and is validated directly in Appendix C. We are not aware of a comparable recovery "
  "condition for a persistent, per-observer mask, the regime where uniform-sampling matrix-completion "
  "guarantees do not apply; the finite-time rate under a strongly-exploiting (adaptive) policy is left "
  "open.</p>")
A("<p class='small'><b>Theorem 4 (collective speedup).</b> A single row leaves a rank-$d>1$ column space "
  "unconstrained (floor); the pooled support reaches the completion threshold in $\\tilde O(d(1+n/m))$ "
  "rounds. <i>Remark:</i> an order argument given Theorem 3; it is the formal counterpart of the "
  "value-of-broadcast and positive-scaling results in Section 6.3. The fold-in perturbation bound used "
  "above (cold-start error $=$ basis-recovery $+$ own-probe noise $+$ ridge bias, exact at $k\\ge d$) is "
  "stated and proved in Appendix B.</p>")

A("<h2>Appendix B. The fold-in error bound</h2>")
A("<p class='small'>For a newcomer factor $x_\\star$ probed against an estimated basis "
  "$\\hat B=B+\\Delta$ ($\\lVert\\Delta\\rVert\\le\\varepsilon$) with $k\\ge d$ observations of noise "
  "$\\sigma$ and ridge $\\lambda$, the ridge fold-in prediction error splits cleanly into three sources, "
  "$\\mathbb{E}|\\hat r-r|\\le C_1\\varepsilon\\lVert x_\\star\\rVert(1+\\lVert b\\rVert/s) + "
  "C_2\\lVert b\\rVert\\sigma\\sqrt{d}/s + C_3\\lambda\\lVert x_\\star\\rVert\\lVert b\\rVert/s^2$ with "
  "$s=\\sigma_{\\min}(B)$, and is exact ($\\hat r=r$) when $\\varepsilon=\\sigma=0,\\lambda\\to0,k\\ge d$. "
  "It quantitatively explains the graceful degradation of cold-start skill as sensing becomes "
  "sparser.</p>")
A("<h2>Appendix C. Empirical validation of the recovery condition (Theorem 3)</h2>")
A("<p class='small'>On the swarm's actual coverage patterns ($m=30,n=240,d=5,\\rho=0.5$, noiseless to "
  "isolate identifiability), reconstructing each unseen pair $(i,j)$ from the observed entries by least "
  "squares gives error $0.000$ exactly when robot $i$'s factor lies in the span of its visible engagers "
  "of $j$ (the condition of Theorem 3), versus a prior-floor reconstruction error of $\\approx0.30$ (an "
  "error, not a skill) otherwise, with graceful "
  "partial recovery as the spanning rank rises to $d$. The identifiability threshold is therefore "
  "exactly the spanning condition of Theorem 3.</p>")
A("<h2>Appendix D. Reproducibility</h2>")
A("<p class='small'>All experiments use a block-model world with signed-cosine reward and bootstrap 95% "
  "confidence intervals; the sweeps use 8 random seeds and the consolidated bake-off (Table 3) uses 5; "
  "each reported number is averaged over the per-seed "
  "runs. The code and per-seed data needed to regenerate every figure and table are openly available "
  "(see Data availability).</p>")
A("<p class='small'><b>Hyperparameters.</b> Headline configuration: $m=30$ robots, $n=240$ tasks, true "
  "rank $d=5$, guessed rank $\\hat d=8$, horizon $T=50$, offer size $c=20$, own-observation noise "
  "$\\sigma_{\\mathrm{own}}=0.1$, broadcast-observation noise $\\sigma_{\\mathrm{obs}}=0.3$, persistent "
  "mask rate $\\rho$ swept. SwarmCF: $\\varepsilon$-greedy with $\\varepsilon_0=0.5$ decaying by $0.93$ "
  "per round to $\\varepsilon_{\\min}=0.05$; ridge $\\lambda=10^{-2}$; 8 alternating-least-squares sweeps "
  "per refit; refit every 3 rounds; uniform observation weights (the noise level is not used). Identical "
  "exploration schedules and the same guessed rank $\\hat d$ are given to every structured baseline for "
  "fairness; the choices are conservative for our claims (generous to baselines).</p>")

A("<h2>Appendix E. The LatentSwarm spatial environment</h2>")
A("<p class='small'>The spatial environment of the LatentSwarm suite is the PettingZoo/Gymnasium "
  "environment used for the transfer test of Section 6.5. It realizes a variant of the "
  "Section 3 setting with episodic, health-depletion dynamics; the dynamics were specified independently "
  "of the collaborative-filtering method, and SwarmCF enters only as one drop-in policy among the "
  "simulator's own (random, oracle, SGD matrix factorization, independent UCB).</p>")
A("<p class='small'><b>Traits and reward.</b> Each robot $i$ and target $j$ is assigned a hidden "
  "$d$-dimensional trait vector ($p_i$, $u_j$) drawn from a shared Gaussian mixture over mode centers, so "
  "the robot$\\times$target structure is (approximately) low-rank of rank $d$, as in Section 3. When "
  "robot $i$ engages target $j$ it earns the signed cosine alignment $r_{ij}=\\langle p_i,u_j\\rangle/"
  "(\\lVert p_i\\rVert\\,\\lVert u_j\\rVert)$ of their traits, perturbed by a shared per-engagement effect "
  "noise; the target's health then decreases by the rectified match $\\max(0,\\langle p_i,u_j\\rangle)$ "
  "and the target is neutralized at zero health.</p>")
A("<p class='small'><b>Decentralized, partial, private observation.</b> Robots and targets occupy fixed "
  "2-D positions (a spatial layout, not motion); each robot observes target positions and activity, the "
  "targets its teammates selected (each teammate's selection independently corrupted with probability "
  "$\\sigma_{\\mathrm{obs}}$, an action-identity channel), and a per-observer-noisy reading of each "
  "broadcast engagement outcome (private noise $\\sigma$, added independently per observer on top of the "
  "shared effect noise). There is no communication, and capacity contention is native: if two robots "
  "select the same target in a round they collide.</p>")
A("<p class='small'><b>Protocol and metric.</b> An episode runs for a fixed horizon; each round every "
  "robot selects a target (or no-op), engagements resolve in a random order, and observations are "
  "emitted. Each policy is run over a sequence of $E=16$ episodes and scored by the converged servicing "
  "skill $=(\\text{return}-\\text{random})/(\\text{oracle}-\\text{random})$, averaged over the second "
  "half (the last 8 episodes). SwarmCF is the weighted-ALS "
  "policy: it maintains its online low-rank factors from the per-observer outcome stream and selects the "
  "target it predicts best.</p>")
A("<div class='algo'><div class='cap'>Algorithm 3: LatentSwarm episode (a variant of the setting; each "
  "robot runs a policy, e.g. SwarmCF)</div>"
  "draw traits $p_i$ (robots) and $u_j$ (targets) from a shared Gaussian mixture (rank $d$); fix 2-D positions\n"
  "for episode $=1,\\dots,E$:\n"
  "    reset every target's health and active flag\n"
  "    for round $t=1,\\dots,T_{\\mathrm{ep}}$:\n"
  "        each robot $i$ picks a target $a_i$ (or no-op) from its policy\n"
  "        for each engaging robot $i$ (with $a_i=j$, $j$ active), in random order:\n"
  "            reward $r_i \\leftarrow \\cos(p_i,u_j)$ + shared effect noise\n"
  "            target $j$ health $\\mathrel{-}= \\max(0,\\langle p_i,u_j\\rangle)$; if $\\le 0$, mark $j$ inactive\n"
  "            if another robot also picked $j$ this round, record a collision (contention)\n"
  "        emit each robot's observation: target positions and activity; teammates' picks\n"
  "            (each corrupted w.p. $\\sigma_{\\mathrm{obs}}$); a per-observer-noisy reading of each reward\n"
  "    score the policy by episodic return; skill $=$ (return $-$ random)$/$(oracle $-$ random)</div>")
A("</div></body></html>")
html_str = "\n".join(H)
open(OUT, "w", encoding="utf-8").write(html_str)
# Publish the same paper as the GitHub Pages landing page.
open(os.path.join(ROOT, "docs", "index.html"), "w", encoding="utf-8").write(html_str)
print("wrote", OUT, "and docs/index.html (%d KB)" % (len(html_str.encode("utf-8")) // 1024))
