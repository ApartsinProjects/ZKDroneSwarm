"""Generate the SELF-CONTAINED, graduate-level, step-by-step HTML tutorial covering
the ENTIRE project: motivation, background, the model, baselines, our methods,
theory, metrics, and every experimental result. Figures F2-F11 are base64-embedded
so the page is a single portable artifact suitable for GitHub Pages.
Output: docs/tutorial.html   (regenerable; reads PNGs from docs/figures/).
"""
import base64
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import method_profiles as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures")
OUT = os.path.join(ROOT, "docs", "tutorial.html")


def img(name, alt, w="100%"):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return "<p><em>[missing figure: %s]</em></p>" % name
    b = base64.b64encode(open(p, "rb").read()).decode("ascii")
    return ('<img alt="%s" src="data:image/png;base64,%s" '
            'style="max-width:%s;height:auto;border:1px solid #d7dde3;border-radius:8px;">' % (alt, b, w))


def _bold(s):
    while "**" in s:
        s = s.replace("**", "<b>", 1).replace("**", "</b>", 1)
    return s.strip()


def md_tables(path, drop_h1=True):
    """Render the markdown subset our experiments emit (## headers, | tables |,
    **bold**, inline html kept) as HTML, so the tutorial draws the headline/ablation/
    contention tables from the SAME committed .md files (single source of truth)."""
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
            continue
        if s.startswith("|"):
            rows.append(s); continue
        flush()
        if s.startswith("## "):
            out.append("<h4>%s</h4>" % _bold(s[3:]))
        elif s.startswith("# "):
            if not drop_h1:
                out.append("<h4>%s</h4>" % _bold(s[2:]))
        elif s.strip():
            out.append("<p class='small'>%s</p>" % _bold(s.strip()))
    flush()
    return "\n".join(out)


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
.formal{background:#fcfbf7;border:1px solid var(--ln);border-left:4px solid #b8860b;border-radius:8px;padding:8px 16px;margin:14px 0}
.formal h4{color:#8a6d0b;margin-top:10px}
.algo{background:#fbfdff;border:1px solid var(--ln);border-radius:8px;padding:10px 14px;margin:14px 0;font:12.5px/1.55 SFMono-Regular,Consolas,monospace;white-space:pre-wrap;overflow-x:auto}
.algo .kw{color:#1f5fa8;font-weight:700}.algo .cm{color:#5b6570;font-style:italic}
.katex{font-size:1.02em}.katex-display{margin:10px 0;overflow-x:auto;overflow-y:hidden}
.plain{background:#f0f9f8;border:1px solid #d3e9e6;border-left:4px solid #11a89a;border-radius:8px;padding:9px 16px;margin:13px 0}
.plain .lab{color:#0d8b80;font-weight:700;letter-spacing:.01em}
.note{background:#fafafa;border:1px dashed var(--ln);border-radius:6px;padding:2px 12px;margin:8px 0;font-size:14px;color:var(--mut)}
"""

def plain(s):
    """Layman-friendly 'In plain words' callout, used throughout to keep the tutorial
    self-contained for a non-expert reader."""
    return "<div class='plain'><span class='lab'>In plain words.</span> %s</div>" % s

H = []
A = H.append
A("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
A("<meta name='viewport' content='width=device-width,initial-scale=1'>")
A("<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css'>")
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
toc = [("Notation at a glance", "notation"),
       ("1. Motivation: the problem", "mot"), ("2. Background: the tools", "bg"),
       ("3. The model (world, observability, fairness)", "model"),
       ("4. Baselines (the field)", "base"), ("5. Our methods", "meth"),
       ("6. Metrics", "metrics"), ("7. Theory", "thy"),
       ("8. Results, step by step", "res"), ("9. Novelty and positioning", "nov"),
       ("10. Reproducibility", "repro"), ("11. Glossary", "glo")]
for t, a in toc:
    A("<a href='#%s'>%s</a><br>" % (a, t))
A("</div>")

# NOTATION TABLE (plain-language symbol key, so the tutorial is self-contained)
A("<h2 id='notation'>Notation at a glance</h2>")
A("<p>This tutorial uses a handful of symbols and abbreviations over and over. Here is every one of "
  "them in plain words; keep this table handy and refer back whenever a formula looks dense. None of "
  "the later sections assume you have memorized any of this.</p>")
NOTE_ROWS = [
    ("Counts and indices", ""),
    ("$m$", "number of <b>drones</b> (the agents / decision-makers). In our experiments $m=30$."),
    ("$n$", "number of <b>targets</b> (the tasks a drone can pick). In our experiments $n=240$."),
    ("$T$", "number of <b>rounds</b> (each round, every drone picks one target). We use $T=50$."),
    ("$i$", "index of a particular drone (a row), e.g. 'drone $i$'."),
    ("$j$", "index of a particular target (a column), e.g. 'target $j$'."),
    ("$k$", "index of a teammate (another drone) when we talk about one drone watching another."),
    ("The hidden structure", ""),
    ("$R$", "the <b>reward table</b>: $R_{ij}$ is how well drone $i$ does on target $j$ (its "
            "compatibility / payoff). This is what we are trying to learn. It is never fully observed."),
    ("$R_{ij}$", "one entry of that table: the reward (a number, here in $[-1,1]$) drone $i$ gets from target $j$."),
    ("$\\hat R_{ij}$", "our <b>estimate</b> of $R_{ij}$ (the 'hat' always means 'estimated / guessed', not the true value)."),
    ("$d$", "the <b>true rank</b>: the number of hidden factors that actually explain compatibility. "
            "Small $d$ (here $d=5$) means the table is 'low-rank', i.e. simple underneath."),
    ("$\\hat d$", "the <b>guessed rank</b> the algorithms assume (here $\\hat d=8$). They do NOT know the true $d$."),
    ("$p_i$", "drone $i$'s hidden <b>taste vector</b> (a list of $\\hat d$ numbers). Think 'what this drone is good at'."),
    ("$u_j$", "target $j$'s hidden <b>profile vector</b> (also $\\hat d$ numbers). Think 'what this target needs'."),
    ("$\\langle p_i, u_j\\rangle$", "the <b>dot product</b> of the two vectors: multiply them entry-by-entry and add up. "
            "It is our predicted compatibility $\\hat R_{ij}$ (taste meets profile)."),
    ("Observability (what a drone can see)", ""),
    ("$\\rho$ (rho)", "<b>broadcast probability</b>: the chance a drone sees any given teammate's outcome each "
            "round. $\\rho=1$ means it sees everything; small $\\rho$ means it sees little (heavy masking)."),
    ("$\\sigma_{\\mathrm{obs}}$ (sigma-obs)", "<b>noise</b> on a teammate's observed reward: how garbled the "
            "numbers a drone overhears are. Large $\\sigma_{\\mathrm{obs}}$ = very noisy."),
    ("$\\sigma_{\\mathrm{own}}$", "noise on a drone's OWN reward (smaller, here $0.10$): even your own sensor is imperfect."),
    ("$\\varepsilon$ (epsilon)", "<b>exploration rate</b>: the chance a drone tries a random target instead of its "
            "current best guess (so it keeps learning). It shrinks over time."),
    ("Methods and math tools", ""),
    ("CF", "<b>collaborative filtering</b>: predicting your unknowns by borrowing patterns from others "
           "(the 'people who liked X also liked Y' idea, here for drones and targets)."),
    ("MF", "<b>matrix factorization</b>: approximating the big table $R$ as $p_i$ times $u_j$, i.e. "
           "discovering the hidden taste/profile vectors. The engine under CF."),
    ("ALS", "<b>alternating least squares</b>: a way to fit MF by repeatedly fixing one set of vectors and "
            "solving for the other, back and forth, until they settle."),
    ("EM", "<b>expectation-maximization</b>: a two-step loop that alternates 'guess the hidden thing given "
           "current beliefs' and 'update beliefs given that guess'. Used for our uncertainty-aware variants."),
    ("$\\arg\\max$", "'the option that scores highest'. $\\arg\\max_j \\hat R_{ij}$ = the target $j$ with the "
            "biggest estimated reward for drone $i$."),
    ("$\\Sigma$ (Sigma)", "a <b>covariance / uncertainty</b> bundle: how unsure we are about a vector, and how its "
            "components co-vary. Bigger $\\Sigma$ = less confident."),
    ("$\\gamma_k$ (gamma)", "<b>choice-informativeness</b> of teammate $k$: how much drone trusts that $k$'s "
            "pick reflects real knowledge rather than a random guess (Section 5.6)."),
    ("ZK", "<b>zero-knowledge / zero-communication</b>: agents never message each other or share parameters; "
           "they only passively watch the public outcome stream."),
    ("Scoring", ""),
    ("skill", "a 0-to-1 score: $0$ = no better than random, $1$ = as good as an all-knowing oracle. "
              "Formally $(\\text{method}-\\text{random})/(\\text{oracle}-\\text{random})$. Lets us compare fairly. "
              "This is a STANDARD normalization, not ours: it is the <i>skill score</i> of forecast "
              "verification (Murphy, (forecast-reference)/(perfect-reference)) and the random/human-"
              "<i>normalized score</i> ubiquitous in RL benchmarking (e.g. Atari); an affine rescaling of regret."),
    ("unseen-pair skill", "skill measured ONLY on drone-target pairs the drone never tried, the acid test of "
            "generalizing to the unknown."),
    ("anytime skill", "skill accumulated over ALL rounds (not just the end), rewarding methods that are good EARLY too."),
    ("CI", "<b>confidence interval</b>: the bracket $[\\text{lo},\\text{hi}]$ after a number is the range we are "
           "$95\\%$ sure the true average lies in. Non-overlapping CIs = a real difference, not luck."),
]
A("<table><tr><th class='l'>Symbol / term</th><th class='l'>Plain meaning</th></tr>")
for sym, mean in NOTE_ROWS:
    if mean == "":
        A("<tr><td class='l' colspan='2'><b>%s</b></td></tr>" % sym)
    else:
        A("<tr><td class='l'>%s</td><td class='l'>%s</td></tr>" % (sym, mean))
A("</table>")
A(plain("A <b>matrix</b> is just a table of numbers with rows and columns. Here the rows are drones, "
        "the columns are targets, and each cell is a reward. 'Low-rank' means that big table is secretly "
        "simple: a few hidden dials (the factors) generate the whole thing, so once you learn the dials "
        "you can fill in cells you never measured. That single idea is the heart of everything below."))

# 1 MOTIVATION
A("<h2 id='mot'>1. Motivation: the problem</h2>")
A("<p>Imagine a swarm of autonomous drones that, round after round, must each pick a target to "
  "engage. Three facts make this both hard and interesting:</p>")
A("<ul>"
  "<li><strong>Hidden, heterogeneous compatibility.</strong> Different drones suit different "
  "targets (payload, sensor, geometry). The compatibility is unknown and latent, but structured: "
  "a few hidden factors explain most of it (it is approximately low-rank).</li>"
  "<li><strong>Sample starvation.</strong> There are many more targets than engagement rounds "
  "($n \\gg T$): a learner that treats each drone-target pair as its own unknown can "
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
A(plain("Picture a new employee who can only rate a restaurant after eating there. With thousands of "
        "restaurants and time for only a few meals, they will be clueless about almost all of them, that "
        "is the 'structure-free' learner stuck at its 'error floor' (its dumbest possible guess, the "
        "overall average). The trick we use is the one Netflix uses: notice that people (or drones) fall "
        "into a few taste groups, and predict the un-tried options from what similar others experienced. "
        "<b>Latent</b> = hidden / not directly visible. <b>Low-rank</b> = explained by just a few hidden "
        "factors. <b>$n \\gg T$</b> reads 'far more targets than rounds', i.e. never enough data to "
        "brute-force every cell."))

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
A("<p>Five classical tools combine into our method. Each is given formally (a definition or formula) "
  "and in plain words; Section 5 then specializes them to the decentralized, online, masked setting.</p>")

A("<h3>2.1 Collaborative filtering / low-rank matrix completion</h3>")
A("<p>Predict the unknown entries of a user-item (here drone-target) matrix from a few observed ones "
  "by assuming the matrix is low-rank.</p>")
A("<div class='formal'><h4>Definition (matrix completion)</h4>"
  "<p>Let $R\\in\\mathbb{R}^{m\\times n}$ have rank $d\\ll\\min(m,n)$, and let $\\Omega\\subseteq[m]\\times "
  "[n]$ be the observed entries (with noise). Matrix completion seeks $\\hat R$ of rank $\\le d$ "
  "fitting the observations, e.g. the nuclear-norm (convex) program "
  "$\\min_{X} \\lVert X\\rVert_* \\ \\text{s.t.}\\ \\sum_{(i,j)\\in\\Omega}(X_{ij}-R_{ij})^2\\le\\delta$, "
  "where $\\lVert X\\rVert_*=\\sum_r\\sigma_r(X)$ is the sum of singular values. <b>Guarantee:</b> a "
  "rank-$d$, $\\mu$-incoherent matrix is recovered (exactly, noiseless) from "
  "$|\\Omega| = \\tilde O\\big(d(m+n)\\big)$ entries sampled at random "
  "(Candes-Recht 2009; Keshavan-Montanari-Oh OptSpace 2010; Recht 2011). Incoherence rules out "
  "'spiky' factors so that uniformly random samples are informative.</p></div>")
A(plain("Unpacking the box: $\\mathbb{R}^{m\\times n}$ just means 'a table with $m$ rows and $n$ "
        "columns of real numbers'. $\\Omega$ ('omega') is the SET of cells we have actually observed, "
        "usually a tiny fraction. 'Rank $d$' is the number of hidden dials behind the table. The scary "
        "formula is one precise way to say 'find the SIMPLEST table that agrees with what we saw': "
        "'singular values' measure how many independent dials a table really uses, and the 'nuclear "
        "norm' $\\lVert X\\rVert_*$ adds them up, so making it small forces simplicity (few dials). "
        "'Incoherent' just means the hidden factors are spread out, not concentrated in one freak "
        "row/column, so random peeks are representative. The punchline (the 'guarantee') is the "
        "remarkable part: you can reconstruct the WHOLE table from only about $d(m+n)$ random cells, "
        "far fewer than the $m\\times n$ total. That is why a few observations can teach a drone about "
        "targets it never touched."))
A("<p>We use this as the statistical backbone but in a NON-standard way: decentralized (per drone), "
  "online (one round at a time), and masked (each drone sees a different, biased subset of $\\Omega$).</p>")

A("<h3>2.2 Alternating least squares (ALS), the workhorse solver</h3>")
A("<p>The non-convex but biconvex way to fit $R\\approx PU^\\top$: fix one factor, solve the other in "
  "closed form, alternate. It is the engine of our estimator (Section 5.1).</p>")
A("<div class='formal'><h4>Definition (ridge ALS)</h4>"
  "<p>Minimize $\\sum_{(i,j)\\in\\Omega}(R_{ij}-\\langle p_i,u_j\\rangle)^2 + \\lambda(\\lVert P\\rVert_F^2 "
  "+\\lVert U\\rVert_F^2)$. With one factor fixed the objective is a sum of independent ridge "
  "regressions:</p>"
  "$$ p_i = \\Big(\\textstyle\\sum_{j\\in\\Omega_i} u_j u_j^\\top + \\lambda I\\Big)^{-1} "
  "\\textstyle\\sum_{j\\in\\Omega_i} R_{ij}\\,u_j, \\qquad u_j = \\Big(\\textstyle\\sum_{i\\in\\Omega_j} "
  "p_i p_i^\\top + \\lambda I\\Big)^{-1} \\textstyle\\sum_{i\\in\\Omega_j} R_{ij}\\,p_i, $$"
  "<p>where $\\Omega_i,\\Omega_j$ are the observed columns of row $i$ / rows of column $j$. Each solve "
  "is $O(d^3 + |\\Omega_i| d^2)$. Weighted ALS (WALS) puts a weight $w_{ij}$ on each term; the matrix "
  "inverted is then the per-factor Gaussian posterior precision, the door to confidence (Section 5.3).</p></div>")
A(plain("What the two formulas actually say: to update a drone's taste vector $p_i$, take every target "
        "it has a reward for, and combine those targets' profile vectors $u_j$ weighted by the rewards, "
        "essentially 'you are the average of what you liked'. The $(\\dots)^{-1}$ piece is a fair-share "
        "correction so that targets you have seen many times do not drown out the rest. The same recipe "
        "with rows and columns swapped updates each target's profile $u_j$. 'Alternating' means you do "
        "one, then the other, then back, each step is a tidy textbook least-squares fit, and they "
        "converge. The $\\lambda$ ('lambda') term is <b>regularization</b>: a gentle pull toward zero "
        "that stops the model from over-trusting a handful of noisy observations. The bonus, used "
        "heavily later: the matrix you invert doubles as a measure of how CONFIDENT you are in that "
        "vector (more, more-varied observations, more confident)."))

A("<h3>2.3 Low-rank and bilinear bandits</h3>")
A("<p>Sequential decision-making that exploits low-rank reward structure to act with few samples.</p>")
A("<div class='formal'><h4>Definition (bilinear bandit)</h4>"
  "<p>Each round, choosing an (arm-feature, context) pair $(x,z)$ yields reward "
  "$\\langle x, \\Theta z\\rangle + \\text{noise}$ for an unknown LOW-RANK $\\Theta$ (rank $d$). The goal "
  "minimizes cumulative regret $\\mathrm{Reg}(T)=\\sum_{t}\\big(\\max_{x,z}\\langle x,\\Theta z\\rangle - "
  "\\langle x_t,\\Theta z_t\\rangle\\big)$. Explore-then-commit spectral methods (e.g. ESTR, Jun et al. "
  "2019) spend a probe phase, estimate $\\Theta$ by SVD, then exploit; they are typically CENTRALIZED "
  "and phase-structured. Our setting is the special case where rows are drones and columns are "
  "targets, but ANYTIME (no probe phase) and DISTRIBUTED.</p></div>")
A(plain("A <b>bandit</b> is the classic 'which slot machine to pull' problem: you must choose "
        "repeatedly, learning as you go, balancing trying new options (explore) against cashing in on "
        "the best-known one (exploit). <b>Regret</b> is how much total reward you left on the table "
        "versus a genie who always picked the best, lower is better. 'Bilinear / low-rank' bandit just "
        "adds our twist: the payoffs share hidden structure, so a smart player learns many options at "
        "once. <b>SVD</b> (singular value decomposition) is the standard math tool that reads the hidden "
        "dials out of observed data. Most prior methods use a separate 'probe phase' (waste some rounds "
        "measuring, then stop learning) and a central computer; ours never stops learning ('anytime') "
        "and every drone runs its own copy ('distributed')."))

A("<h3>2.4 Cold-start and fold-in</h3>")
A("<p>Given known item factors $U$, a brand-new user's factor is fit from a few interactions by one "
  "small least-squares solve, no retraining (the WALS 'fold-in').</p>")
A("<div class='formal'><h4>Definition (fold-in)</h4>"
  "<p>A new drone with observed rewards $\\{r_a\\}$ on targets with known factors $\\{u_a\\}_{a=1}^k$ "
  "gets factor $\\hat p = \\big(\\sum_a u_a u_a^\\top + \\lambda I\\big)^{-1}\\sum_a r_a u_a$, after which "
  "$\\hat R_{\\cdot j}=\\langle\\hat p, u_j\\rangle$ is available for ALL targets $j$. Exact once $k\\ge d$ "
  "and $\\{u_a\\}$ span $\\mathbb{R}^d$ (row completion). We reuse this symmetrically for onboarding new "
  "TARGETS (fold-in a $u$ from drones' probes) and welcoming new DRONES.</p></div>")
A(plain("<b>Cold-start</b> is the everyday problem 'a brand-new user (or product) has no history, so "
        "how do we recommend anything?'. <b>Fold-in</b> is the cheap fix: if the swarm already knows the "
        "targets' profile vectors, a newly-arrived drone only has to try a handful of targets, then one "
        "quick least-squares solve places it in the hidden 'taste space', and from that single point it "
        "can instantly score EVERY target, including ones it never tried. 'Span $\\mathbb{R}^d$' means "
        "the few targets it sampled were varied enough to pin down all $d$ dials. No global retraining "
        "needed, and the very same trick onboards a brand-new target."))

A("<h3>2.5 Implicit feedback and choice debiasing</h3>")
A("<p>Treating an observed CHOICE (which target a teammate engaged) as a preference signal, while "
  "correcting for what options were available (exposure).</p>")
A("<div class='formal'><h4>Definition (confidence-weighted implicit feedback)</h4>"
  "<p>Implicit-feedback CF (Hu-Koren-Volinsky 2008) fits preferences $y_{ij}\\in\\{0,1\\}$ with "
  "per-pair CONFIDENCE $c_{ij}$: $\\min \\sum_{i,j} c_{ij}(y_{ij}-\\langle p_i,u_j\\rangle)^2 + "
  "\\lambda(\\dots)$, where a chosen item is a (confident) positive and unchosen offered items are "
  "(less-confident) negatives. We treat a teammate's engaged target as a positive and sample "
  "negatives from the PUBLIC active-target set (the exposure / choice set), so no private menu is "
  "needed (the ZK choice channel). This 'confidence' is exactly what Section 5.5 deepens.</p></div>")
A(plain("<b>Explicit</b> feedback is a star rating; <b>implicit</b> feedback is just an ACTION, you "
        "watched this film, you clicked that link, here: a drone engaged that target. An action is a "
        "weaker hint than a rating (it says 'I preferred this' but not by how much), yet it is "
        "noise-free, you cannot misread WHICH target a teammate picked. The catch is <b>exposure "
        "bias</b>: someone can only choose among options they were shown, so 'not chosen' might mean "
        "'disliked' OR 'never offered'. We correct for it by counting an un-chosen target as a negative "
        "only when it was actually on offer (publicly visible). $y_{ij}\\in\\{0,1\\}$ is just a "
        "chosen/not-chosen flag, and $c_{ij}$ is how much we trust that flag. This is the 'choices "
        "channel' that powers our communication-free teamwork."))

# 3 MODEL
A("<h2 id='model'>3. The model</h2>")
A("<div class='step'>")
A("<h3>Step 3.1 The world: a block model with low-rank compatibility</h3>")
A("<p>There are $m$ drones and $n$ targets. Drones come in $K_1$ latent <em>types</em> and targets in "
  "$K_2$ types (a 'block model'); within a type, agents and targets are similar but not identical. The "
  "hidden compatibility between drone $i$ and target $j$ is the cosine similarity of their latent "
  "factor vectors $p_i, u_j \\in \\mathbb{R}^d$. Everything below is generated once per random seed; no "
  "learner ever sees $P$, $U$, or $R$.</p>")
A("<div class='formal'><h4>Generative model (exact)</h4>"
  "<p>Draw type prototypes $A\\in\\mathbb{R}^{K_1\\times d}$ and $B\\in\\mathbb{R}^{K_2\\times d}$ with "
  "i.i.d. $\\mathcal{N}(0,1)$ entries and unit-normalize each row. Assign each drone a type "
  "$t_i\\in\\{1,\\dots,K_1\\}$ and each target a type $s_j\\in\\{1,\\dots,K_2\\}$. Factor vectors are the "
  "prototype plus a small within-type spread $\\xi$ (default $0.15$), renormalized to the unit sphere:</p>"
  "$$ p_i = \\frac{A_{t_i} + \\xi\\,\\varepsilon_i}{\\lVert A_{t_i} + \\xi\\,\\varepsilon_i\\rVert}, "
  "\\qquad u_j = \\frac{B_{s_j} + \\xi\\,\\eta_j}{\\lVert B_{s_j} + \\xi\\,\\eta_j\\rVert}, "
  "\\qquad \\varepsilon_i,\\eta_j \\sim \\mathcal{N}(0, I_d). $$"
  "<p>The true reward (compatibility) matrix is the bilinear cosine form</p>"
  "$$ R_{ij} = \\langle p_i, u_j\\rangle \\in [-1,1], \\qquad R = P U^\\top \\in \\mathbb{R}^{m\\times n}, "
  "\\qquad \\operatorname{rank}(R) = \\min(d, K_1, K_2). $$"
  "<p>Here $P\\in\\mathbb{R}^{m\\times d}$ stacks the drone factors as rows and "
  "$U\\in\\mathbb{R}^{n\\times d}$ the target factors. Unit norms remove any popularity/scale artifact "
  "(no target is good simply by having a large magnitude); the link is exactly bilinear (no nonlinear "
  "squashing $g(\\cdot)$), so the matrix is GENUINELY rank $d$, the structure CF exploits. We "
  "stress-test violations of both assumptions in Section 8.11.</p></div>")
A("<p><b>In words.</b> A few hidden 'knobs' ($d$ of them) explain every drone's taste for every "
  "target. Two drones of the same type want similar targets; two targets of the same type are wanted "
  "by similar drones. That shared, low-dimensional structure is exactly what lets one drone learn "
  "about a target from <em>other</em> drones' experiences, the engine of collaborative filtering.</p>")
A(plain("Decoding the generative box term by term. A <b>block model</b> just means agents come in a few "
        "groups ('types'): drones of the same type behave alike, like customer segments. Each type has a "
        "<b>prototype</b> direction, and each individual is that prototype nudged a little ($\\xi$ is how "
        "much wiggle room within a group). <b>$\\mathcal{N}(0,1)$</b> is the bell-curve (normal) "
        "distribution; 'i.i.d.' means each number is drawn independently from it. <b>Unit-normalize</b> "
        "means we rescale every vector to length 1 (an arrow on a sphere), so only its DIRECTION "
        "matters, not its size, this prevents a target from looking good just by being 'bigger'. The "
        "reward $\\langle p_i,u_j\\rangle$ is then <b>cosine similarity</b>: $+1$ if the two arrows point "
        "the same way (perfect match), $0$ if perpendicular (irrelevant), $-1$ if opposite (actively bad "
        "fit). Finally $\\operatorname{rank}(R)=\\min(d,K_1,K_2)$ says the effective number of hidden "
        "dials is capped by whichever is smallest: the factor dimension or the number of types."))
A("<h4>Default parameters (used everywhere unless a sweep varies them)</h4>")
A("<table><tr><th class='l'>Symbol</th><th>Meaning</th><th>Default</th></tr>"
  "<tr><td class='l'>$m$</td><td>number of drones (agents)</td><td>30</td></tr>"
  "<tr><td class='l'>$n$</td><td>number of targets (tasks/arms)</td><td>240</td></tr>"
  "<tr><td class='l'>$d$</td><td>true latent rank (hidden)</td><td>5</td></tr>"
  "<tr><td class='l'>$\\hat d$</td><td>GUESSED rank used by every structured method (fair)</td><td>8</td></tr>"
  "<tr><td class='l'>$K_1,K_2$</td><td>drone / target latent types</td><td>10, 10</td></tr>"
  "<tr><td class='l'>$T$</td><td>rounds (engagements per drone)</td><td>50</td></tr>"
  "<tr><td class='l'>$c$</td><td>candidate-set size offered each round</td><td>20</td></tr>"
  "<tr><td class='l'>$\\sigma_{\\mathrm{own}}$</td><td>noise on a drone's OWN observed reward</td><td>0.10</td></tr>"
  "<tr><td class='l'>$\\sigma_{\\mathrm{obs}}$</td><td>noise on an observed TEAMMATE reward</td><td>0.30</td></tr>"
  "<tr><td class='l'>$\\rho$</td><td>fraction of the broadcast each drone senses (masking)</td><td>swept 0..1</td></tr>"
  "<tr><td class='l'>$\\xi$</td><td>within-type factor spread</td><td>0.15</td></tr></table>")
A("<p class='small'>Note $n=240 \\gg cT/\\ldots$: with $T=50$ rounds and $c=20$ offers, a drone can "
  "engage at most $T=50$ distinct targets out of $n=240$, so the regime is SAMPLE-STARVED (most "
  "targets are never tried). This is the regime where generalizing to the unseen matters.</p></div>")
A("<div class='step'>")
A("<h3>Step 3.2 Observability: a public outcome stream with two degradations</h3>")
A("<p>Time is discrete <b>rounds</b> $t=1,\\dots,T$. In each round EVERY drone acts ONCE and "
  "SIMULTANEOUSLY (no turn-taking, no waiting on others): drone $i$ is shown a <b>candidate set</b> "
  "$S_i^t\\subseteq[n]$ of $c$ targets, picks one $a_i^t\\in S_i^t$, and earns that target's true reward "
  "$R_{i,a_i^t}$. A public stream then carries the (action, outcome) of every engagement that round; "
  "drone $i$ senses its OWN outcome cleanly (noise $\\sigma_{\\mathrm{own}}$) and a per-drone-limited, "
  "noisy subset of teammates' outcomes:</p>")
A("<ul><li><strong>Masking ($\\rho$)</strong> (action observability): you either detect a "
  "teammate's engagement or you do not.</li>"
  "<li><strong>Additive noise ($\\sigma_{\\mathrm{obs}}$)</strong> (reward observability): a "
  "detected outcome's value is noisy.</li></ul>")
A("<div class='box'><b>Why a candidate set, and not all $n$ targets every round?</b> The candidate "
  "set models a real per-round constraint: a drone can only reach, sense, or consider a few targets at "
  "a time (think a recommendation 'slate', or a contextual bandit's per-round action set). It also "
  "makes the oracle/random references well-defined (best-in-offer vs mean-in-offer, Section 6). It is "
  "just a KNOB: setting $c=n$ (the entire target set offered every round, the 'peek everything' case) "
  "is a valid special case, and WE RAN IT. Sweeping the training candidate size $c=20\\to n=240$ (8 "
  "seeds), CF's unseen-pair skill stays high at every $c$ ($0.44$ at $c=20$ down to $0.34$ at $c=n$) "
  "while a tabular learner sits at the floor ($\\approx 0$) throughout (Section 8.x / CANDSET). So the "
  "CATEGORICAL result does NOT depend on $c$: it is about the quality of a drone's preference on "
  "targets it never engaged, a property of the learned model, not of how many targets are shown, even "
  "with the full set on offer a tabular learner cannot rank what it never tried. What $c$ DOES affect "
  "is the sample-starvation rate (smaller $c$ relative to the horizon means fewer distinct targets are "
  "tried), which is why we keep $c=20\\ll n=240$ as the interesting, realistic regime.</div>")
A("<div class='formal'><h4>Observation model (exact)</h4>"
  "<p>Let $a_k^t$ be the target drone $k$ engages at round $t$, and let "
  "$M\\in\\{0,1\\}^{m\\times m}$ be the observation mask ($M_{ii}=1$ always). What drone $i$ records "
  "about round $t$ is the noisy value</p>"
  "$$ \\tilde r^{(i)}_{k,t} = \\begin{cases} R_{k,a_k^t} + \\mathcal{N}(0,\\sigma_{\\mathrm{own}}^2) & k=i "
  "\\ \\text{(own, clean)}\\\\[2pt] R_{k,a_k^t} + \\mathcal{N}(0,\\sigma_{\\mathrm{obs}}^2) & k\\neq i,\\ "
  "M_{ik}=1 \\ \\text{(teammate, sensed)}\\\\[2pt] \\text{unobserved} & k\\neq i,\\ M_{ik}=0 \\ "
  "\\text{(masked)} \\end{cases} $$"
  "<p><b>The noise is independent per observer (the superscript $(i)$ is essential).</b> Each drone "
  "$i$ draws its OWN noise on each engagement and applies its OWN mask $M_{ik}$, so for two observers "
  "$i\\neq i'$ the values recorded for the SAME action $a_k^t$ differ: "
  "$\\tilde r^{(i)}_{k,t}=R_{k,a_k^t}+\\eta^{(i)}_{k,t}$ and $\\tilde r^{(i')}_{k,t}=R_{k,a_k^t}+"
  "\\eta^{(i')}_{k,t}$ with $\\eta^{(i)}_{k,t}\\perp\\eta^{(i')}_{k,t}$, both $\\sim\\mathcal{N}(0,"
  "\\sigma_{\\mathrm{obs}}^2)$ and redrawn every round. There is NO single shared 'broadcast number': "
  "the stream is what each drone privately SENSES, not a value any drone transmits. This independent "
  "per-drone sensing (plus the per-drone mask) is exactly why no two drones hold the same internal "
  "state, and why the channel is genuinely passive/communication-free.</p>"
  "<p>The recorded precision of each observation is $\\beta = 1/\\sigma^2$ (so "
  "$\\beta_{\\mathrm{own}} = 1/\\sigma_{\\mathrm{own}}^2$, $\\beta_{\\mathrm{obs}} = "
  "1/\\sigma_{\\mathrm{obs}}^2$). Two masking regimes: PERSISTENT ($M$ drawn once, "
  "$M_{ik}\\sim\\mathrm{Bernoulli}(\\rho)$, fixed for all $t$) gives durably-different per-drone views; "
  "i.i.d. ($M^t_{ik}\\sim\\mathrm{Bernoulli}(\\rho)$ redrawn each round, packet-loss style) gives the "
  "same headline results (the masking-model result, Section 8.7).</p>"
  "<p><b>Data format a learner sees.</b> A stream of tuples $(k, a_k^t, \\tilde r^{(i)}_{k,t})$ for the "
  "$k$ it sensed: the actor id, the discrete action (which target), and the noisy scalar outcome. No "
  "factors, no ids of TYPES, no rank, no teammate parameters, ever.</p></div>")
A(plain("The big curly-brace formula is just three cases for 'what number does drone $i$ jot down about "
        "drone $k$ this round': (1) if $k$ is itself, it records its own reward with a little sensor "
        "noise; (2) if $k$ is a teammate it managed to detect, it records that reward with MORE noise; "
        "(3) if it did not detect $k$, it records nothing. <b>Mask</b> $M_{ik}=1/0$ is the detected / "
        "not-detected switch, and <b>Bernoulli($\\rho$)</b> just means 'flip a biased coin that lands "
        "detected with probability $\\rho$'. <b>Precision</b> $\\beta=1/\\sigma^2$ is simply 'how much "
        "you trust a number': low noise, high precision, trust it; high noise, low precision, take it "
        "with a grain of salt. 'Persistent' masking means each drone is permanently deaf to the same "
        "teammates; 'i.i.d.' means the deafness is re-rolled every round (like dropped radio packets), "
        "both give the same headline result. The little $(i)$ on top matters: each drone hears the same "
        "event through its OWN imperfect ears, so two drones watching the SAME teammate grab the SAME "
        "target jot down DIFFERENT numbers (independent noise), and a drone that did not detect it jots "
        "down nothing. Nobody is broadcasting one agreed value; each just senses for itself, which is "
        "what makes their internal pictures genuinely differ. Crucially, the tuple has no hidden factors "
        "or types in it, only who-did-what-and-got-what, which is why the setting is zero-knowledge."))
A("<div class='box'>Masking is <em>passive sensing of public outcomes</em> (limited detection), NOT "
  "radio transmission, so the setting is genuinely communication-free. Persistent (fixed) per-drone "
  "masks make decentralization durable; i.i.d. per-round masks (packet-loss style) give the same "
  "headline results (Section 8.7).</div></div>")
A("<div class='step'>")
A("<h3>Step 3.3 Fairness and zero-knowledge, for every method</h3>")
A("<p>The same rules bind ours and all baselines: no method ever receives the latent factors, the "
  "true rank, or labels; structured methods all get the SAME <em>guessed</em> rank "
  "$\\hat d=8$ (true $d=5$); every method is an INDEPENDENT per-drone learner "
  "with no parameter sharing and no coordinator; all see the same partial+noisy stream. The "
  "<strong>Oracle</strong> (centralized, complete information) is used only to normalize scores, "
  "never as a competitor.</p></div>")
A("<div class='step'>")
A("<h3>Step 3.4 The simulation loop (the exact environment)</h3>")
A("<p>Putting the world and the observation model together, one episode runs as below. Each drone $i$ "
  "carries a policy $\\pi_i$ and a private estimator; nothing is shared between drones except what the "
  "broadcast passively reveals.</p>")
A("<div class='algo'>"
  "<span class='cm'># inputs: hidden R, horizon T, candidate size c, noises, mask M, policies pi_i</span>\n"
  "<span class='kw'>for</span> t = 1 .. T:\n"
  "    <span class='cm'># 1) each drone is offered c targets, uniform without replacement</span>\n"
  "    <span class='kw'>for</span> i = 1 .. m:  S_i &larr; uniform_subset([n], size=c)\n"
  "    <span class='cm'># 2) each drone picks ONE offered target (its decision rule)</span>\n"
  "    <span class='kw'>for</span> i = 1 .. m:  a_i &larr; pi_i(S_i)              <span class='cm'># a_i in S_i</span>\n"
  "    <span class='cm'># 3) true (noiseless) reward earned -> this is what 'anytime' accumulates</span>\n"
  "    earn R[i, a_i] for every i\n"
  "    <span class='cm'># 4) broadcast of (actor, action, outcome); each drone senses a masked, noisy slice</span>\n"
  "    <span class='kw'>for</span> i = 1 .. m:\n"
  "        obs_i = { (i, a_i, R[i,a_i] + Normal(0, sigma_own^2)) }             <span class='cm'># own, clean</span>\n"
  "        <span class='kw'>for</span> k != i <span class='kw'>with</span> M[i,k] = 1:\n"
  "            obs_i += { (k, a_k, R[k,a_k] + Normal(0, sigma_obs^2)) }        <span class='cm'># sensed teammate</span>\n"
  "        pi_i.estimator.update(obs_i)                                       <span class='cm'># online; precision 1/sigma^2</span>\n"
  "<span class='cm'># after T rounds: freeze each estimator and evaluate (Section 6)</span>"
  "</div>")
A("<p><b>Reading the loop.</b> Step 2 is the only place a method's intelligence enters the EARNED "
  "reward (the anytime metric). Step 4 is the only information a method gets about teammates: passive, "
  "partial, noisy. No message passing, no shared parameters, no coordinator; 'communication-free' is "
  "literal. The estimator update in step 4 is what differs across methods (next two sections).</p></div>")

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
A(plain("The recurring acronyms, once: <b>UCB</b> ('upper confidence bound') is the standard "
        "'optimism' bandit rule, prefer the option whose value COULD plausibly be highest given how "
        "little you have tried it, which automatically explores the under-tested. <b>epsilon-greedy</b> "
        "is the simplest explorer: pick your current best, but a small fraction $\\varepsilon$ of the "
        "time pick at random. <b>SGD</b> ('stochastic gradient descent') nudges the model a little after "
        "each new data point, the default way to learn online. <b>SVD</b> reads the hidden dials out of "
        "a table in one shot (it needs a fairly complete table to work well, which is its weakness under "
        "masking). <b>Thompson sampling</b> explores by acting on a random draw from what it currently "
        "believes, instead of a fixed bonus. <b>kNN</b> ('k nearest neighbours') skips model-fitting "
        "and just averages the rewards of the most similar drones/targets. The split that matters: "
        "<b>no-structure</b> methods learn each cell on its own (cannot generalize to untried targets), "
        "while <b>low-rank</b> methods learn the hidden dials (can). Our methods are in the second camp."))
A("<p class='small'>A 'batch-SVD hybrid' (ESTR, PTF) extracts factors by one SVD of an accumulated "
  "reward table whose unobserved entries are imputed 0; under masking that table is sparse and "
  "biased, which is why these methods decay (Section 8.5). In our harness every baseline runs as an "
  "independent per-drone instance (ESTR's literature default is centralized; we run it per drone for "
  "a fair, fully-distributed comparison).</p>")
A("<h3>4.1 Baseline algorithms in detail (intuition + exact rule + why it fails)</h3>")
A("<p>Every method has the same interface: each round it scores the offered targets and picks "
  "$a=\\arg\\max_{j\\in S}\\hat R_{ij}$ (with probability $\\epsilon$ it explores at random instead), then "
  "updates from the masked broadcast. Notation: $N_{ij}$ = drone $i$'s own pull count of $j$; "
  "$\\bar r_{ij}$ = its empirical mean reward on $j$; $t$ = round; $c$ = UCB constant; $\\hat d$ = "
  "guessed rank. All match the implementations in <code>experiments/pilot_baselines.py</code> and "
  "<code>pilot_noise.py</code>.</p>")

A("<div class='formal'><h4>No-structure class (each (drone,target) is its own unknown)</h4>"
  "<p><b>Random.</b> <i>Intuition:</i> the floor; learns nothing. <i>Rule:</i> "
  "$a\\sim\\mathrm{Unif}(S)$; $\\hat R$ is noise. Skill $\\approx 0$ by construction.</p>"
  "<p><b>Tabular ($\\epsilon$-greedy).</b> <i>Intuition:</i> treat each target independently; average "
  "your OWN rewards on it. <i>Rule:</i> $\\hat R_{ij}=\\bar r_{ij}$ if $N_{ij}>0$, else the global mean "
  "of your observed rewards (a prior constant). <i>Fails because:</i> any never-pulled $j$ stays at the "
  "constant, so it cannot rank the unseen, the tabular floor.</p>"
  "<p><b>UCB-Indep.</b> <i>Intuition:</i> one UCB1 bandit per (drone,target) pair; be optimistic about "
  "rarely-tried arms. It updates ALL arms it senses from the broadcast but selects on its own row. "
  "<i>Rule:</i> $\\text{score}_j=\\bar r_{ij}+c\\sqrt{\\ln(t)/N_{ij}}$, with an untried arm scored "
  "$+\\infty$ (so it is pulled first). <i>Fails because:</i> no link between arms, so unseen stays at "
  "the floor; and with $n\\gg T$ almost every offer contains an untried ($+\\infty$) arm, so it "
  "explores forever and earns $\\approx$ random (anytime $\\approx 0$).</p>"
  "<p><b>UCB-Homo.</b> <i>Intuition:</i> assume all drones are identical, so pool everyone's rewards "
  "into ONE shared per-target table. <i>Rule:</i> same UCB1 but on the pooled "
  "$\\bar r_{\\cdot j}=\\tfrac1m\\sum_k \\bar r_{kj}$. <i>Fails because:</i> pooling estimates "
  "$\\langle\\bar p, u_j\\rangle$ (rank-1 'popularity'), one ranking for everyone, so it gets partial "
  "unseen skill ($\\approx 0.17$) but is confounded on heterogeneous drones.</p></div>")

A("<div class='formal'><h4>Low-rank class (model $R\\approx PU^\\top$)</h4>"
  "<p><b>MFSGD.</b> <i>Intuition:</i> plain SGD matrix factorization; nudge the factors after each "
  "observed reward. <i>Rule:</i> on $(k,j,r)$ with residual $e=\\langle p_k,u_j\\rangle-r$: "
  "$p_k\\mathrel{-}=\\eta(e\\,u_j+\\lambda p_k)$, $u_j\\mathrel{-}=\\eta(e\\,p_k+\\lambda u_j)$; predict "
  "$\\langle p_i,u_j\\rangle$. <i>Fails because:</i> SGD makes tiny steps and there are few observations "
  "($n\\gg T$), so it underfits in the starved regime.</p>"
  "<p><b>ESTR (explore-then-spectral-refit).</b> <i>Intuition:</i> a phase method: explore at random, "
  "then do ONE spectral fit, then commit. <i>Rule:</i> for the first $\\sim 0.4T$ rounds pick randomly "
  "and accumulate the empirical $\\hat M$ (unobserved entries imputed $0$); take its rank-$\\hat d$ SVD "
  "$\\hat M\\approx U_{\\hat d}\\Sigma_{\\hat d}V_{\\hat d}^\\top$, set $\\hat P=U_{\\hat d}\\Sigma^{1/2}$, "
  "$\\hat U=V_{\\hat d}\\Sigma^{1/2}$, then exploit $\\arg\\max\\langle\\hat P_i,\\hat U_j\\rangle$. "
  "<i>Fails because:</i> the explore phase earns $\\approx$ random (anytime cost), it COMMITS (no online "
  "adaptation), and the $0$-imputation biases the SVD as masking grows.</p>"
  "<p><b>PTF (probe-then-fit), the strongest competitor.</b> <i>Intuition:</i> like ESTR but probe with "
  "UCB (targeted, not random), warm-start by SVD, THEN keep fine-tuning online. <i>Rule:</i> probe "
  "phase = own-row UCB1; at probe end, SVD warm-start of $\\hat M$ (clipped to $[-1,1]$ for stable "
  "SVD); thereafter $\\epsilon$-greedy on $\\langle P,U\\rangle$ with online SGD updates. <i>Fails "
  "because:</i> still pays a probe phase on anytime, and the UCB probe biases $\\hat M$ coverage toward "
  "high-reward arms (a real completion trade-off).</p>"
  "<p><b>BPMF (Bayesian PMF, Thompson sampling).</b> <i>Intuition:</i> keep a per-drone Gaussian "
  "posterior over the factors and explore by SAMPLING from it (no $\\epsilon$). <i>Rule:</i> conjugate "
  "updates in natural form, $\\Lambda_{p_k}\\mathrel{+}=u_ju_j^\\top/\\sigma^2$, "
  "$\\eta_{p_k}\\mathrel{+}=r\\,u_j/\\sigma^2$ (and symmetric for $u_j$); select by drawing "
  "$\\tilde p\\sim\\mathcal N(\\mu_{p_i},\\Lambda_{p_i}^{-1})$, $\\tilde u_j\\sim\\mathcal N(\\mu_{u_j},"
  "\\Lambda_{u_j}^{-1})$ and taking $\\arg\\max\\langle\\tilde u_j,\\tilde p\\rangle$. <i>Fails "
  "because:</i> Thompson sampling over-explores in the sample-starved anytime regime. (This is the "
  "closest baseline to our EMCF, but with sampling instead of a predictive-interval UCB.)</p></div>")
A(plain("Symbols in these four rules: the <b>residual</b> $e=\\langle p_k,u_j\\rangle-r$ is simply "
        "'prediction minus truth', the error to shrink. $\\eta$ ('eta') in MFSGD is the <b>step size</b> "
        "(how big a correction to make each time); $\\lambda$ is the same regularization pull-toward-zero "
        "as before. The $+\\infty$ score for an untried arm is just a way to say 'always try the "
        "never-tried option first'. In BPMF the letters describe a running 'belief': $\\mu$ ('mu') is the "
        "best-guess vector (the mean), and $\\Lambda$ ('Lambda', a precision matrix) is the confidence "
        "around it, so $\\Lambda^{-1}$ turns that confidence back into a spread to sample from. "
        "'Conjugate update' is just the clean bookkeeping that keeps this belief Gaussian as data "
        "arrives. The common storyline: all of these CAN learn the hidden dials, but each pays a price, "
        "tiny SGD steps, a wasted probe phase, or too much random exploring, that our method in "
        "Section 5 avoids."))

A("<div class='formal'><h4>Memory-based and additive</h4>"
  "<p><b>SoftImpute (nuclear-norm convex completion).</b> <i>Intuition:</i> the convex way to complete "
  "a low-rank matrix: repeatedly fill the missing entries, SVD, and SOFT-shrink singular values. "
  "<i>Rule:</i> iterate $Z\\leftarrow P_\\Omega(\\bar R)+P_\\Omega^\\perp(X)$, then "
  "$X\\leftarrow\\sum_r \\max(\\sigma_r-\\lambda,0)\\,u_rv_r^\\top$ (threshold chosen to keep "
  "$\\approx\\hat d$ components). <i>Fails because:</i> it is a batch fill-in method (the unobserved "
  "entries are imputed), so under masking the table is sparse and biased and it decays, strongest at "
  "FULL broadcast.</p>"
  "<p><b>kNN-CF (memory-based, model-free).</b> <i>Intuition:</i> 'find similar drones': predict drone "
  "$i$'s reward on $j$ as a similarity-weighted average of OTHER drones who observed $j$. <i>Rule:</i> "
  "center each drone's observed rewards, compute cosine similarity $w_{ik}$ over co-observed targets, "
  "then $\\hat R_{ij}=\\bar r_i+\\big(\\sum_k w_{ik}(\\bar r_{kj}-\\bar r_k)\\big)/\\sum_k w_{ik}$ over "
  "drones $k$ that observed $j$. <i>Fails because:</i> it generalizes but earns little, and needs "
  "neighbors who actually observed $j$; it tests whether explicit factorization is even needed (it "
  "is).</p>"
  "<p><b>BiasModel (additive, the popularity ceiling).</b> <i>Intuition:</i> some targets are generally "
  "easy and some drones generally good, but no personalized interaction. <i>Rule:</i> "
  "$\\hat R_{ij}=\\mu+b_i+c_j$ (fit by alternating means over observed entries). <i>Fails because:</i> "
  "the prediction matrix has rank $\\le 2$, so for a fixed drone the ranking is by $c_j$ alone, a single "
  "shared popularity order, which cannot personalize (the additive-ceiling result).</p></div>")
A("<p class='small'>Common thread: the no-structure methods are at the unseen floor by construction; "
  "the batch low-rank methods (ESTR/PTF/SoftImpute) generalize but pay a probe phase and/or decay under "
  "masking because they impute missing entries; BPMF generalizes but over-explores. Our online "
  "weighted-ALS (Section 5) keeps the generalization while removing the probe phase and the "
  "imputation bias.</p>")
A("<div class='box warn'><b>Why not standard MARL baselines (MAPPO, QMIX, IPPO)?</b> They are "
  "admissible only if they respect our setting, and most do not. The cooperative MARL workhorses "
  "(MAPPO, QMIX, value-decomposition) are CTDE, <em>centralized training</em> with a shared/global "
  "critic and joint-state access, decentralized only at execution. That requires a training-time "
  "coordinator and shared parameters, which our minimal-assumptions setting forbids; including them "
  "would hand a competitor information we deny everyone else. The one fully-decentralized, "
  "communication-free MARL baseline, INDEPENDENT learners (Independent Q-Learning / independent "
  "bandits), is, in our stateless per-round task-selection problem, exactly a per-agent bandit, which "
  "<b>UCB-Indep and Tabular already instantiate</b> (and which sit at the unseen floor). So "
  "the relevant MARL class is covered; the rest are out of scope by construction, not omitted by "
  "choice.</div>")

# 5 METHODS
A("<h2 id='meth'>5. Our methods</h2>")
A("<div class='box key'><b>The whole section in one map: ONE method, a menu of scoped extensions.</b> "
  "Do not let the number of named variants below fool you, there is a single core and everything else "
  "is an optional add-on used only when a stated condition holds. <b>Core:</b> each drone runs online "
  "noise-weighted ridge ALS (low-rank matrix completion) over the public stream and acts "
  "$\\epsilon$-greedily on its completed row; an unobserved event gets zero WEIGHT, not a fake zero "
  "value (5.1-5.2). <b>Scoped extensions, each with a theorem and a 'use it when':</b> "
  "<i>confidence</i> (EMCF: Bayesian posterior for directed exploration / shrinkage, when the "
  "broadcast is noisy or coverage is scarce, 5.3-5.5); <i>rank self-determination</i> (ARD: when you "
  "do not know the rank, 5.7); <i>the choice channel</i> (teammates' actions instead of rewards, when "
  "reward noise is extreme or teammates differ in reliability, 5.6); and the decision-side extensions "
  "covered in results, <i>collective info-directed exploration</i> (8.x) and <i>contention "
  "de-confliction</i> (a fixed private offset, 8.13). Read 5.1-5.2 for the method; treat 5.3-5.7 as the "
  "menu.</div>")
A("<h3>5.0 Every method on the operating axes (who gets more power)</h3>")
A("<p>Before the details, here is every method, the baselines from Section 4 and ours below, on the "
  "axes that decide fairness: distribution, communication, observability, prior knowledge, and "
  "computation. Our methods sit in the HARDEST cell (decentralized, no communication, masked + noisy "
  "observation, only a guessed rank, online); a method that beats us in some regime relaxes one of these "
  "axes (ESTR is centralized; Oracle and CTDE are reference upper bounds, not competitors).</p>")
A(mp.html_profiles())
A("<p>And here is the design space of OUR methods by mechanism, so the named variants read as one core "
  "plus a few scoped switches (signal channel, exploration, confidence, contention, rank, coordination) "
  "rather than a zoo.</p>")
A(mp.html_mechanisms())
A("<p>Each drone runs its own <strong>online weighted alternating least squares</strong> (weighted "
  "ALS) over the events it senses. The crucial trick for masking-robustness: an event the drone did "
  "NOT observe gets zero <em>weight</em>, not an imputed zero <em>value</em>. (Batch 'fill-in' methods "
  "instead pretend a missing entry is a real 0, which is why they decay as more of the broadcast is "
  "masked.) A second, optional refinement weights each OBSERVED event by its precision "
  "$1/\\sigma^2$ (trust clean outcomes more than noisy ones); this helps when the broadcast "
  "is noisy, but at low noise plain uniform weighting actually generalizes a little better to unseen "
  "targets, which we show honestly in the ablation (section 8.12). Estimation is separated from the "
  "decision policy, so the policy can be epsilon-greedy, UCB, or uncertainty-directed.</p>")
A("<dl>"
  "<dt>RewardCF</dt><dd>cross-agent signal = teammates' (noisy) rewards. The simple workhorse; "
  "best on the anytime metric.</dd>"
  "<dt>ChoiceZK</dt><dd>cross-agent signal = teammates' CHOICES only (no rewards), with GLOBAL "
  "negatives and NO menu. A noise-immune fallback: actions are discrete, so no reward noise can "
  "corrupt them.</dd>"
  "</dl>")
A("<div class='box'><b>What 'global negatives, no menu' means (and the masking question).</b> "
  "Treating a teammate's CHOICE as a preference is implicit feedback (Section 2.5): the engaged "
  "target $a_k$ is a POSITIVE example for $k$'s factor, and we need NEGATIVE examples (targets $k$ "
  "implicitly preferred less). Two ways to pick negatives: (i) from the teammate's offered MENU "
  "$S_k$ (the $c$ targets $k$ chose among), which requires observing that menu, or (ii) GLOBAL "
  "negatives, sample uniformly from ALL $n$ targets (within=False). ChoiceZK uses (ii), so it needs "
  "only the single observed fact 'drone $k$ engaged target $a_k$', not the set $k$ was choosing from. "
  "That is the stricter zero-knowledge channel, and the choice-only ablation shows it costs nothing (ChoiceZK $\\approx$ "
  "ChoiceCF), the menu carries no extra value. <b>On masking:</b> a CHOICE is observed exactly when "
  "the engagement is DETECTED (the same $\\rho$ detection that gates the reward); a masked (undetected) "
  "engagement reveals neither action nor reward. So ChoiceZK sees the detected SUBSET of teammates' "
  "actions, not all of them, and the action it does see is clean (discrete), while the co-observed "
  "reward is noisy. The choice channel's value is precisely that detection-without-noise still "
  "carries preference information.</div>"
  "<dl>"
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

A("<h3>5.1 The estimator: online weighted alternating least squares (ALS)</h3>")
A("<p>Every method above shares one estimator. Each drone keeps factor estimates "
  "$P\\in\\mathbb{R}^{m\\times d}$ (its belief about every drone, including itself) and "
  "$U\\in\\mathbb{R}^{n\\times d}$ (every target), fit to the noisy events it has sensed by minimizing a "
  "precision-weighted, ridge-regularized reconstruction error.</p>")
A("<div class='formal'><h4>Weighted-ALS objective and updates (exact)</h4>"
  "<p>Let $\\Omega$ be the set of sensed events $(k,j,r)$ (teammate $k$ engaged target $j$, observed "
  "reward $r$) and let $w_{kj}=1/\\sigma^2$ be the precision of that observation (own events "
  "$1/\\sigma_{\\mathrm{own}}^2$, sensed teammates $1/\\sigma_{\\mathrm{obs}}^2$, masked events ABSENT, "
  "i.e. weight $0$). Minimize</p>"
  "$$ \\mathcal{L}(P,U) = \\sum_{(k,j,r)\\in\\Omega} w_{kj}\\big(r - \\langle p_k, u_j\\rangle\\big)^2 "
  "+ \\lambda\\big(\\lVert P\\rVert_F^2 + \\lVert U\\rVert_F^2\\big). $$"
  "<p>Holding one factor fixed makes this a set of independent ridge regressions, solved in closed "
  "form (the alternating updates), sweeping a few times:</p>"
  "$$ u_j \\leftarrow \\Big(\\textstyle\\sum_{k:(k,j)\\in\\Omega} w_{kj}\\,p_k p_k^\\top + \\lambda I"
  "\\Big)^{-1}\\Big(\\textstyle\\sum_{k:(k,j)\\in\\Omega} w_{kj}\\,r_{kj}\\,p_k\\Big), $$"
  "$$ p_i \\leftarrow \\Big(\\textstyle\\sum_{j:(i,j)\\in\\Omega} w_{ij}\\,u_j u_j^\\top + \\lambda I"
  "\\Big)^{-1}\\Big(\\textstyle\\sum_{j:(i,j)\\in\\Omega} w_{ij}\\,r_{ij}\\,u_j\\Big). $$"
  "<p>The prediction for ANY target (seen or unseen) is then $\\hat R_{ij} = \\langle p_i, u_j\\rangle$. "
  "<b>Why this beats batch fill-in:</b> a masked event simply does not appear in the sums (weight $0$), "
  "so it contributes nothing; batch-SVD methods instead impute the missing entry as the VALUE $0$, "
  "biasing the factors as masking grows. <b>Why precision weighting is subtle:</b> the matrices being "
  "inverted are exactly the per-factor Gaussian posterior precisions (next subsection), so the choice "
  "of $w$ trades noise-filtering against coverage; see 5.3.</p></div>")
A(plain("The objective $\\mathcal{L}(P,U)$ reads: 'choose the hidden taste/profile vectors so that, for "
        "every reward we actually saw, the prediction $\\langle p_k,u_j\\rangle$ is close to the observed "
        "$r$, paying more attention to the rewards we trust ($w_{kj}$ = trust = $1/\\text{noise}^2$), "
        "plus a penalty $\\lambda(\\dots)$ that keeps the vectors small and tame.' The "
        "$\\lVert\\cdot\\rVert_F^2$ ('Frobenius norm, squared') is just 'add up the squares of all the "
        "numbers', a size measure for a whole table. The two arrow-update lines are the closed-form "
        "answer to that objective with everything but one factor held still, the same 'you are the "
        "weighted average of what you saw' recipe from Section 2.2. The single most important design "
        "choice is hiding in plain sight: a target a drone never detected simply is NOT in the sum (it "
        "gets weight $0$), as opposed to being faked as a $0$ reward, that one distinction is why our "
        "estimator keeps working as the swarm goes increasingly blind."))
A("<div class='algo'>"
  "<span class='kw'>class</span> RewardCF:                          <span class='cm'># online weighted-ALS on rewards</span>\n"
  "  state: P[m,d], U[n,d] ~ small random;  buffer B = []\n"
  "  <span class='kw'>def</span> update(observed events):\n"
  "    <span class='kw'>for</span> (k, j, r) in events:  B.append( (k, j, r, w = 1/sigma_kj^2) )\n"
  "    <span class='kw'>if</span> round % refit_every == 0:\n"
  "      <span class='kw'>repeat</span> als_sweeps times:  update all u_j ; then all p_i   <span class='cm'># closed-form ridge above</span>\n"
  "  <span class='kw'>def</span> select(offer S):              <span class='cm'># i = own index</span>\n"
  "    <span class='kw'>with</span> prob eps: return random target in S\n"
  "    <span class='kw'>else</span>: return argmax_{j in S} &lt;p_i, u_j&gt;\n"
  "</div>")

A("<h3>5.2 The decision policies (how confidence drives exploration)</h3>")
A("<p>Estimation (above) is separated from the policy (how we pick). We use three policies, in "
  "increasing use of confidence:</p>")
A("<div class='algo'>"
  "<span class='cm'># (a) eps-greedy (RewardCF): explore at random with prob eps, else exploit the mean prediction</span>\n"
  "select(S):  random in S w.p. eps   else   argmax_{j in S} &lt;p_i, u_j&gt;\n"
  "\n"
  "<span class='cm'># (b) ActiveCFconv: latent-space UCB with a BROADCAST count bonus (collective active learning)</span>\n"
  "<span class='cm'>#   g_j = number of times target j was engaged in the broadcast (everyone's probes count)</span>\n"
  "select(S):  argmax_{j in S}  &lt;p_i, u_j&gt; + c_active / sqrt(g_j + 1)\n"
  "\n"
  "<span class='cm'># (c) HybridCFconv: a short UCB probe + one SVD warm-start, THEN online weighted-ALS</span>\n"
  "<span class='kw'>for</span> first probe_frac*T rounds:  select by UCB1 on own empirical row\n"
  "warm start P,U from rank-d_hat SVD of the empirical table   <span class='cm'># PTF's idea, our estimator</span>\n"
  "<span class='kw'>thereafter</span>:  eps-greedy on the online-ALS factors\n"
  "</div>")
A("<p>Policy (b) is the key idea: because every probe is broadcast, the count $g_j$ is shared, so one "
  "drone probing an under-observed target lowers the uncertainty bonus for EVERYONE, active learning "
  "with zero communication. The count $1/\\sqrt{g_j+1}$ is a cheap proxy for the true posterior "
  "uncertainty of target $j$, which we compute exactly next.</p>")

A("<h3>5.3 Confidence and uncertainty we can compute (and use)</h3>")
A("<p>Under the linear-Gaussian model behind weighted-ALS, the matrices we already invert ARE Gaussian "
  "posterior precisions, so confidence comes essentially for free, the question is how to USE it.</p>")
A("<div class='formal'><h4>Posterior covariance and predictive intervals</h4>"
  "<p>Treating $u_j$ as Gaussian with the per-target normal matrix as its posterior precision,</p>"
  "$$ \\Lambda_j = \\sum_{k} w_{kj}\\, p_k p_k^\\top + \\lambda I, \\qquad \\Sigma_j = \\Lambda_j^{-1} "
  "\\;=\\; \\mathrm{Cov}(u_j \\mid \\text{data}), $$"
  "and symmetrically $\\Sigma_i$ for $p_i$. The PREDICTED reward $\\hat R_{ij}=\\langle p_i,u_j\\rangle$ "
  "then has an explicit predictive variance (independent-Gaussian factors):</p>"
  "$$ \\mathrm{Var}(\\hat R_{ij}) \\approx p_i^\\top \\Sigma_j\\, p_i + u_j^\\top \\Sigma_i\\, u_j "
  "+ \\operatorname{tr}(\\Sigma_i \\Sigma_j), $$"
  "<p>giving a confidence interval $\\hat R_{ij} \\pm z\\sqrt{\\mathrm{Var}(\\hat R_{ij})}$ for every "
  "pair, INCLUDING targets the drone never tried. Three ways we put this to work:</p>"
  "<ul>"
  "<li><b>Confidence-directed exploration:</b> a UCB bonus $\\beta\\sqrt{\\mathrm{Var}(\\hat R_{ij})}$ "
  "(the principled version of ActiveCF's count proxy).</li>"
  "<li><b>Confidence shrinkage:</b> pull an uncertain prediction toward the rank-1 popularity prior "
  "$\\bar R_{\\cdot j} = \\langle \\bar p, u_j\\rangle$ by an amount that grows with its predictive sd "
  "(so sparse-coverage targets default to 'what's popular', confident ones keep their personal "
  "estimate).</li>"
  "<li><b>Confidence weighting of the fit:</b> the observation weights $w$. Naive $w=1/\\sigma^2$ "
  "(full precision) over-trusts a drone's own clean data and UNDER-uses the broadcast that powers "
  "generalization (Section 8.12); we found the right fix is <em>ratio-bounded, scale-normalized</em> "
  "precision (relcap): keep noise-awareness but never let any row dominate or the data-vs-prior balance "
  "drift.</li></ul></div>")
A(plain("The key idea here: not only can the drone GUESS a reward it never measured, it can also say how "
        "SURE it is of that guess, an error bar on the unknown. The variance formula just adds up the "
        "sources of doubt: how unsure we are about the target's hidden profile ($\\Sigma_j$), how unsure "
        "about the drone's own taste ($\\Sigma_i$), and a small interaction term ($\\operatorname{tr}$ "
        "is the 'trace', the sum of a matrix's diagonal). The bigger that total, the wider the bracket "
        "$\\hat R_{ij}\\pm z\\sqrt{\\text{Var}}$ ($z$ just sets the confidence level, $z\\approx 2$ for "
        "95%). Having an error bar on EVERY pair is powerful: the drone can deliberately probe the "
        "targets it is least sure about (UCB), or, when unsure, fall back to 'what's generally popular' "
        "instead of a shaky personal guess (shrinkage)."))
A("<div class='formal'><h4>Variational EM (Bayesian) factorization, with intervals</h4>"
  "<p>The fully Bayesian version (probabilistic matrix factorization, $r_{ij}\\sim "
  "\\mathcal{N}(\\langle p_i,u_j\\rangle, \\sigma_{ij}^2)$, $p_i,u_j\\sim\\mathcal{N}(0,\\lambda^{-1}I)$) "
  "keeps a full posterior $q(p_i)=\\mathcal{N}(\\mu_i,\\Sigma_i)$, $q(u_j)=\\mathcal{N}(\\mu_j,\\Sigma_j)$ "
  "and updates by mean-field VI, the EM updates differ from ALS by propagating uncertainty through the "
  "SECOND MOMENT $\\mathbb{E}[u u^\\top]=\\Sigma_j+\\mu_j\\mu_j^\\top$:</p>"
  "$$ \\Sigma_j = \\Big(\\lambda I + \\textstyle\\sum_k \\beta_{kj}\\,\\mathbb{E}[p_k p_k^\\top]\\Big)^{-1}, "
  "\\quad \\mu_j = \\Sigma_j \\textstyle\\sum_k \\beta_{kj}\\, r_{kj}\\,\\mu_k, \\qquad "
  "\\beta_{kj}=1/\\sigma_{kj}^2 \\ \\ (\\text{and symmetrically for } p_i). $$"
  "<p>A poorly-covered or noisy factor automatically contributes less (its $\\mathbb{E}[uu^\\top]$ is "
  "inflated by $\\Sigma$), WITHOUT us hand-tuning a weight, confidence enters the model, not an ad-hoc "
  "knob. Prediction uses the same predictive-variance formula above for a real interval; we drive a "
  "UCB and optional shrinkage from it. (This is our <code>EMCF</code>.)</p></div>")
A(plain("This box describes the 'fully Bayesian' version, EMCF. The difference from plain ALS in one "
        "sentence: instead of committing to a single best guess for each hidden vector, it carries a "
        "whole CLOUD of plausible vectors (a mean $\\mu$ and a spread $\\Sigma$) and lets that cloud "
        "speak. 'Mean-field VI' (variational inference) is the practical recipe for updating such clouds. "
        "The neat consequence: a target seen only a few times, or seen noisily, has a FAT cloud, and a "
        "fat cloud automatically counts for less in everyone else's updates, no manual 'trust this less' "
        "knob required. Confidence becomes part of the model itself. Second moment / "
        "$\\mathbb{E}[uu^\\top]=\\Sigma+\\mu\\mu^\\top$ is just the bookkeeping that carries the cloud's "
        "spread through the math."))

A("<h3>5.4 Matrix factorization WITH uncertainty: the landscape (what we scouted)</h3>")
A("<p>'Confidence-aware factorization' is a real literature; here is the map, what kind of uncertainty "
  "each method yields, its cost, and what we adopt.</p>")
A("<table><tr><th class='l'>Approach</th><th class='l'>Uncertainty it gives</th><th>Cost</th>"
  "<th class='l'>Here</th></tr>"
  "<tr><td class='l'>PMF (Mnih &amp; Salakhutdinov, 2007)</td><td class='l'>none by itself (MAP point "
  "estimate)</td><td>low</td><td class='l'>= our MFSGD baseline</td></tr>"
  "<tr><td class='l'>Laplace around the MAP / ALS</td><td class='l'>posterior covariance "
  "$\\Sigma=\\Lambda^{-1}$ = the inverse of the matrix ALS already inverts</td><td>~free</td>"
  "<td class='l'><b>used</b> (predictive variance, shrinkage)</td></tr>"
  "<tr><td class='l'>Variational Bayesian MF (Lim &amp; Teh 2007; Raiko et al.)</td><td class='l'>"
  "closed-form factor posteriors via mean-field VI</td><td>moderate</td><td class='l'><b>used</b> "
  "(our EMCF)</td></tr>"
  "<tr><td class='l'>Bayesian PMF / MCMC (Salakhutdinov &amp; Mnih, 2008)</td><td class='l'>full "
  "posterior samples (Gibbs); Thompson sampling</td><td>high</td><td class='l'>= our BPMF baseline</td></tr>"
  "<tr><td class='l'>Online-MF Thompson sampling (Kawale et al., 2015)</td><td class='l'>sequential "
  "(particle/SMC) posterior</td><td>high</td><td class='l'>related; BPMF is the in-harness proxy</td></tr>"
  "<tr><td class='l'>Debiased entrywise CIs for noisy completion (Chen, Fan, Ma &amp; Yan, 2019, PNAS)"
  "</td><td class='l'>non-asymptotic confidence intervals per entry/factor</td><td>batch</td>"
  "<td class='l'>gold standard for completion CIs; centralized</td></tr>"
  "<tr><td class='l'>Low-rank / bilinear bandit ellipsoids (Jun et al., 2019)</td><td class='l'>"
  "frequentist confidence sets on the bilinear parameter</td><td>phase-structured</td>"
  "<td class='l'>related; we are anytime + distributed</td></tr></table>")
A(plain("Vocabulary for the table, in everyday terms. <b>MAP / point estimate</b> = commit to a single "
        "best answer, no error bars. <b>Laplace</b> = a cheap trick that draws an error-bar bubble "
        "around that single answer using math you already did (here, literally free from ALS). "
        "<b>MCMC / Gibbs</b> = the gold-standard but slow way to map uncertainty by drawing many random "
        "samples. <b>Variational (VI)</b> = a faster approximation of that, what our EMCF uses. "
        "<b>Confidence set / ellipsoid</b> = a region the true answer is very likely to sit in. The "
        "bottom line of the whole table: rich uncertainty is available at every price point, and we "
        "deliberately pick the CHEAP options that survive our no-coordinator, see-little setting."))
A("<div class='box key'><b>What we actually compute and use.</b> The cheap Laplace covariance "
  "$\\Sigma_j=\\Lambda_j^{-1}$ (free from ALS) and the full variational posterior (EMCF) both give a "
  "real predictive interval per (drone,target). We use it three ways, exploration (UCB bonus), "
  "decision shrinkage (toward popularity when uncertain), and as a diagnostic for how to weight the "
  "fit. Our honest finding (Section 8.12): in this coverage-scarce, decentralized regime, putting "
  "confidence in the DECISION (shrinkage / UCB) and keeping the FIT broadcast-friendly (uniform or "
  "bounded precision) beats the textbook inverse-variance fit weighting.</div>")

A("<h3>5.5 A deeper view of confidence: three levels (and how to leverage them)</h3>")
A("<p>'Confidence' is not one number. Per-observation precision $1/\\sigma^2$ (Level 1) is the "
  "shallowest kind, and we saw it can even backfire. The useful confidence is SECOND-ORDER: how "
  "well-determined is the learned latent structure, and how much should one agent trust another "
  "agent's revealed CHOICES. Two orthogonal axes organize it: WHOSE confidence (my OWN vs a "
  "TEAMMATE's) and WHICH channel (the REWARD signal vs the CHOICE signal). That gives a 2x2 we use "
  "throughout.</p>")
A("<div class='formal'><h4>The 2x2 of confidence (own vs teammate) x (reward vs choice)</h4>"
  "<table><tr><th></th><th>REWARD channel (continuous outcomes)</th><th>CHOICE channel (discrete "
  "actions)</th></tr>"
  "<tr><td class='l'><b>OWN</b> (about me)</td>"
  "<td class='l'>How well MY factor $p_i$ is pinned down by the rewards I have: posterior covariance "
  "$\\Sigma_{p_i}=(\\sum_j \\beta_{ij}u_ju_j^\\top+\\lambda I)^{-1}$. Large $\\Rightarrow$ explore "
  "(my taste is uncertain).</td>"
  "<td class='l'>Whether I am EXPLOITING a confident model vs still exploring (my own "
  "exploit-state), $1-\\epsilon_i$ or $1-\\mathrm{tr}\\,\\Sigma_{p_i}$-driven.</td></tr>"
  "<tr><td class='l'><b>TEAMMATE</b> (about $k$)</td>"
  "<td class='l'>How much to trust teammate $k$'s REWARD reports: their observation precision "
  "$\\beta_{kj}=1/\\sigma_{\\mathrm{obs}}^2$ AND how well their data pin the target factor "
  "$\\Sigma_{u_j}$. Drives the (bounded) fit weighting and shrinkage.</td>"
  "<td class='l'>How INFORMATIVE / knowledgeable $k$'s CHOICES are: $\\gamma_k=\\Pr(k$ acts on its "
  "model rather than at random$)$, inferred from $k$'s public actions. Drives the choice-channel "
  "weight (ChoiceEM, Section 5.6).</td></tr></table>"
  "<p>Levels 1-3 below populate this grid: Level 1 is per-observation precision (the REWARD cells, "
  "shallow); Level 2 is the posterior structure confidence (own $\\Sigma_{p_i}$ and teammate "
  "$\\Sigma_{u_j}$, the reward column); Level 3 is peer-policy confidence ($\\gamma_k$, the "
  "teammate-choice cell). The own-choice cell is exploit-vs-explore self-assessment. Crucially OUR "
  "confidence and the TEAMMATE's are different quantities, we never see a teammate's posterior, we "
  "INFER its reliability from public behavior.</p></div>")
A(plain("The 2x2 just says 'confidence' is really four different questions, and good behavior needs all "
        "four. Down one axis: is this about ME or about a TEAMMATE? Across the other: is it about the "
        "REWARD numbers or about the CHOICES (actions)? So the four boxes are: (1) how sure am I of my "
        "OWN taste? (if shaky, explore more); (2) am I confidently exploiting or still feeling things "
        "out?; (3) how much should I trust the reward numbers a TEAMMATE reports? (noisier means trust "
        "less); (4) how much should I trust a TEAMMATE's CHOICES, are they acting on real knowledge or "
        "just guessing? The fourth is the subtle one and gets its own method ($\\gamma_k$, Section 5.6). "
        "The deep point: I can directly measure my own confidence, but a teammate's I can only INFER "
        "from watching what they do in public, never by reading their mind."))

A("<div class='formal'><h4>Level 1, observation confidence (a single measurement)</h4>"
  "<p>$\\beta_{kj}=1/\\sigma_{kj}^2$: how noisy is one sensed reward. Local, myopic. Leverage: "
  "noise-aware fit weighting, but only bounded/normalized (relcap), since over-trusting clean OWN "
  "data starves the broadcast that determines the shared structure (Section 8.12).</p></div>")

A("<div class='formal'><h4>Level 2, structure confidence (the posterior over the latent factors)</h4>"
  "<p>As events accumulate, the swarm's belief about the latent geometry sharpens. Two parts, both "
  "available in closed form from the (weighted) ALS / variational fit:</p>"
  "<ul>"
  "<li><b>Confidence in the shared target structure $U$.</b> Per target, the posterior precision "
  "$\\Lambda_{u_j}=\\sum_k \\beta_{kj}\\,p_k p_k^\\top + \\lambda I$ and covariance "
  "$\\Sigma_{u_j}=\\Lambda_{u_j}^{-1}$. Globally, how well the $d$-dimensional subspace is pinned down "
  "is read off the spectrum of the aggregate design $G_U=\\sum_{k,j}\\beta_{kj}\\,p_k p_k^\\top$: its "
  "smallest eigenvalue $\\lambda_{\\min}(G_U)$ is the LEAST-explored latent direction. If the swarm "
  "has only sampled a low-dimensional slice of target-space, $U$ is under-determined no matter how many "
  "rewards were seen.</li>"
  "<li><b>Each drone's confidence in its own factor $p_i$.</b> "
  "$\\Sigma_{p_i}=\\big(\\sum_j \\beta_{ij}\\,u_j u_j^\\top + \\lambda I\\big)^{-1}$. By the row-completion result, "
  "$p_i$ is identified only once drone $i$ has observed $\\ge d$ targets whose factors span "
  "$\\mathbb{R}^d$; $\\Sigma_{p_i}$ quantifies HOW WELL, its top eigenvector is the drone's "
  "worst-known taste direction.</li>"
  "</ul>"
  "<p>These combine into a predictive interval for any pair (seen or unseen): "
  "$\\mathrm{Var}(\\hat R_{ij}) = p_i^\\top\\Sigma_{u_j}p_i + u_j^\\top\\Sigma_{p_i}u_j + "
  "\\operatorname{tr}(\\Sigma_{p_i}\\Sigma_{u_j})$.</p></div>")
A("<p><b>How to leverage Level 2 (three concrete mechanisms).</b></p>"
  "<div class='algo'>"
  "<span class='cm'># (L2-a) Confidence-driven explore/exploit: replace the fixed eps schedule.</span>\n"
  "eps_i(t) = clip( trace(Sigma_{p_i}) / (trace(Sigma_{p_i}) + tau), eps_min, 1 )\n"
  "<span class='cm'>#   explore while your OWN factor is uncertain; exploit once it is pinned down.</span>\n"
  "\n"
  "<span class='cm'># (L2-b) Information-greedy COLLECTIVE exploration (D-optimal, broadcast-shared):</span>\n"
  "<span class='cm'>#   pick the offered target that most reduces uncertainty, not just the count proxy.</span>\n"
  "select(S):  argmax_{j in S}  Rhat_ij + kappa * ( u_j^T (G_U + lambda I)^{-1} u_j )^{1/2}\n"
  "<span class='cm'>#   u_j^T (G_U)^{-1} u_j is large for under-explored latent directions; because every</span>\n"
  "<span class='cm'>#   probe is broadcast, one drone filling a gap lowers G_U-uncertainty for ALL (no comms).</span>\n"
  "\n"
  "<span class='cm'># (L2-c) Predictive-variance shrinkage (implemented as EMshrink):</span>\n"
  "Rhat_ij <- (1-a_ij) * Rhat_ij + a_ij * popularity_j ,   a_ij = sd(Rhat_ij)/(sd+kappa2)\n"
  "</div>")
A(plain("Level 2 is about confidence in the LEARNED STRUCTURE, not a single number. An <b>eigenvalue</b> "
        "of a covariance measures how spread-out (uncertain) the belief is along one direction; the "
        "SMALLEST eigenvalue $\\lambda_{\\min}$ flags the latent direction the swarm has explored LEAST, "
        "its biggest blind spot. The <b>trace</b> (sum of the diagonal) is a handy 'total uncertainty' "
        "score. The three mechanisms turn that into action: (a) explore while your own taste is still "
        "fuzzy and settle down once it sharpens; (b) <b>collective active learning</b>, deliberately "
        "probe the targets that would shrink the swarm's blind spot the most, and since probes are "
        "public, one drone's probe educates everyone (D-optimal is just the formal name for 'pick the "
        "measurement that cuts uncertainty most'); (c) <b>shrinkage</b>, when unsure about a target, "
        "blend your personal guess with the crowd-favorite, leaning on the crowd exactly in proportion "
        "to your doubt."))

A("<div class='formal'><h4>Level 3, peer-policy confidence (is a teammate's CHOICE model-based or "
  "random?)</h4>"
  "<p>A teammate $k$'s observed action $a_k^t$ is a useful PREFERENCE signal only to the extent $k$ "
  "chose it by EXPLOITING a confident model; if $k$ is still exploring, $a_k^t$ is noise. We cannot "
  "see $k$'s posterior (no parameter sharing), so we INFER $k$'s exploit-confidence from its public "
  "behavior. Let $\\gamma_k^t = \\Pr(k \\text{ is exploiting at } t \\mid k\\text{'s observed actions})$. "
  "Two estimable signals (ZK: actions only):</p>"
  "<ul>"
  "<li><b>Choice predictability under our model of $k$:</b> if $k$ tends to pick the target OUR "
  "estimate $\\hat p_k$ says it prefers, $k$ is likely exploiting a model consistent with the shared "
  "structure. Score $k$'s round-$t$ choice by "
  "$\\pi^t_k = \\dfrac{\\exp(\\langle\\hat p_k,u_{a_k^t}\\rangle/\\tau)}{\\sum_{j\\in S_k^t}\\exp("
  "\\langle\\hat p_k,u_j\\rangle/\\tau)}$ (high = unsurprising = model-based; near $1/c$ = looks "
  "random).</li>"
  "<li><b>Temporal consistency:</b> $k$'s chosen factors $u_{a_k^t}$ stabilizing over time (low "
  "drift) signals convergence to exploit-mode (this is the 'competence' weight we already ramp).</li>"
  "</ul>"
  "<p>Use $\\gamma_k^t$ to WEIGHT teammate $k$'s choice observation in the choice channel: "
  "$w^{\\text{choice}}_{k} \\propto \\gamma_k^t$. Early random choices are discounted; late model-based "
  "choices are trusted, automatically.</p></div>")
A("<div class='box key'><b>The virtuous cycle (why this is the deep structure).</b> The three levels "
  "are coupled: my confidence in $U$ depends on the quality of every drone's $p_k$; the value of $k$'s "
  "CHOICE signal depends on $k$'s confidence in its own $p_k$; and I estimate $k$'s confidence from how "
  "model-consistent $k$'s choices look to ME. So the swarm bootstraps a shared latent geometry over the "
  "broadcast, and as it grows more confident, its public choices become SHARPER preference signals, "
  "which further sharpen everyone's structure. This predicts a regime shift: EARLY, choices are near-"
  "random, so rely on the (noisy) reward channel; LATE, choices are crisp and noise-immune, so the "
  "choice channel should dominate. It is the principled justification for ramping the choice weight "
  "with confidence over time, and it suggests the strongest method is one that fuses the reward and "
  "choice channels with weights set by Levels 2 and 3, not by a fixed schedule.</p></div>")
A("<div class='box'><b>Result: how to use the posterior for exploration (info-directed exploration, "
  "tested).</b> We compared exploration rules on anytime sample efficiency (rho=0.25, 8 seeds): "
  "eps-greedy, the count-bonus (ActiveCF), and posterior UCBs on EMCF. <b>Honest finding:</b> a "
  "FULL-predictive-variance UCB at a big bonus OVER-explores (cumulative skill $\\approx 0$ at round 10 "
  "vs $0.07$ for the count-bonus) because the own-factor uncertainty term is uniformly large early; "
  "using only the COLLECTIVE latent term $p_i^\\top\\Sigma_{u_j}p_i$ with a BIG bonus does not fix it. "
  "But the COLLECTIVE term as a SMALL exploit-bias ($\\beta=0.3$) gives the HIGHEST final anytime skill "
  "($0.356$ vs count-bonus $0.333$, eps-greedy $0.324$), the informative probes pay off by convergence "
  "though it rises a touch slower early. So the right way to 'reduce collective latent uncertainty' is "
  "a gentle tie-break toward shared-structure-informative targets layered on exploitation, not a large "
  "exploration bonus (which the anytime metric, charging every probe, penalizes). The count-bonus "
  "remains the fastest EARLY explorer. (Confirms Proposition 6's caveat.)</div>")

A("<h3>5.6 Joint estimation of latent structure and choice-informativeness (an EM framework)</h3>")
A("<p>Level 3 above raises a real failure of the naive choice channel. When drone $i$ learns from a "
  "teammate $k$'s CHOICE $a_k$ (treating the chosen target as a revealed positive), it implicitly "
  "assumes $k$ chose because $k$'s model says $a_k$ is good. But early on, $k$ has LOW confidence in "
  "its own latent structure, so $k$ is largely EXPLORING, its choices are almost random and carry "
  "little preference information. Folding such choices in as positives corrupts the latent estimate. "
  "We want to estimate, JOINTLY with the factors, how INFORMATIVE each teammate's choices are, so "
  "non-informative choices automatically get a small weight.</p>")
A("<div class='box key'><b>The framework: teammates as noisy preference 'annotators'.</b> This is "
  "exactly the structure of <em>learning the truth from unreliable annotators</em> (the Dawid-Skene "
  "model), fused with matrix factorization and a rational-choice (inverse-RL) likelihood. Each "
  "teammate is an annotator whose 'label' is a discrete CHOICE; its unknown RELIABILITY "
  "$\\gamma_k\\in[0,1]$ is how often it acts on its model rather than at random; and the shared "
  "ground truth being recovered is the latent geometry $(P,U)$. Reliability and structure are "
  "estimated TOGETHER by EM, so unreliable annotators are discounted automatically.</p></div>")
A("<div class='formal'><h4>Generative model of a choice (mixture of rational and random)</h4>"
  "<p>Teammate $k$ offered set $S_k^t$ chooses target $a_k^t$ via a latent indicator "
  "$z_k^t\\in\\{\\text{informative},\\text{random}\\}$ with $\\Pr(z=\\text{informative})=\\gamma_k$:</p>"
  "$$ a_k^t \\mid z_k^t=\\text{informative} \\ \\sim\\ \\mathrm{softmax}_{j\\in S_k^t}\\!\\Big(\\tfrac{1}{\\tau}"
  "\\langle p_k, u_j\\rangle\\Big), \\qquad a_k^t \\mid z_k^t=\\text{random}\\ \\sim\\ \\mathrm{Unif}(S_k^t). $$"
  "<p>The informative component is the Boltzmann-rational / conditional-logit choice (the standard "
  "inverse-RL and discrete-choice model): a confident agent picks high-preference targets with high "
  "probability; the temperature $\\tau$ sets how sharp. The marginal choice likelihood is the mixture "
  "$\\;p(a_k^t) = \\gamma_k\\,\\mathrm{softmax}_{a_k^t}(\\cdot) + (1-\\gamma_k)/|S_k^t|$.</p></div>")
A("<div class='formal'><h4>EM (each drone runs this locally over the teammates it senses)</h4>"
  "<p><b>E-step</b> (per-choice responsibility = posterior that the choice was informative):</p>"
  "$$ r_k^t = \\frac{\\gamma_k\\,\\mathrm{softmax}_{a_k^t}\\!\\big(\\langle \\hat p_k, u\\rangle/\\tau\\big)}"
  "{\\gamma_k\\,\\mathrm{softmax}_{a_k^t}\\!\\big(\\langle \\hat p_k, u\\rangle/\\tau\\big) + (1-\\gamma_k)/|S_k^t|}. $$"
  "<p>$r_k^t\\to 1$ when $k$ picked what our current model predicts it prefers (looks model-based); "
  "$r_k^t\\to 0$ when the choice looks no better than random. <b>M-step</b>:</p>"
  "$$ \\gamma_k \\leftarrow \\tfrac{1}{T_k}\\textstyle\\sum_t r_k^t \\quad(\\text{closed form}); \\qquad "
  "(P,U) \\leftarrow \\arg\\max \\ \\underbrace{\\textstyle\\sum_{\\text{rewards}} \\beta\\,\\text{(reward fit)}}"
  "_{\\text{reward channel}} + \\underbrace{\\textstyle\\sum_{k,t} r_k^t\\,\\log\\mathrm{softmax}_{a_k^t}"
  "(\\langle p_k,u_j\\rangle/\\tau)}_{\\text{choice channel, weighted by informativeness}}. $$"
  "<p>The choice channel's contribution is scaled by $r_k^t$, so a random-looking choice barely moves "
  "the factors, exactly the desired effect. In practice the choice log-likelihood is optimized by a "
  "few gradient steps, or replaced by the implicit-feedback least-squares surrogate (positive at "
  "$a_k^t$, sampled negatives from the public active set) with weight $r_k^t$, so it slots into the "
  "same weighted-ALS solver (Section 5.1).</p></div>")
A(plain("The machinery in everyday terms. <b>Softmax</b> turns scores into pick-probabilities: the "
        "higher a target scores, the likelier it is chosen, and the <b>temperature</b> $\\tau$ sets how "
        "decisive, low $\\tau$ means 'almost always take the top one', high $\\tau$ means 'pick fuzzily'. "
        "We model each teammate's choice as a COIN FLIP between two stories: with probability $\\gamma_k$ "
        "they chose thoughtfully (softmax on real preferences), otherwise they chose at random (uniform). "
        "<b>EM</b> then untangles the two without ever being told which is which: the <b>E-step</b> asks "
        "'given my current model, does this particular choice look thoughtful or random?' and assigns a "
        "<b>responsibility</b> $r_k^t$ between 0 and 1; the <b>M-step</b> then (1) sets each teammate's "
        "reliability $\\gamma_k$ to their average responsibility, and (2) re-fits the hidden factors "
        "while letting each choice pull only as hard as its responsibility. Round and round until it "
        "settles. The payoff: a teammate who is just exploring gets near-zero responsibility, so their "
        "near-random choices cannot corrupt the shared map, precisely the failure we set out to fix."))
A("<div class='algo'>"
  "<span class='kw'>class</span> ChoiceEM:        <span class='cm'># joint latent + per-teammate informativeness</span>\n"
  "  state: P,U (factors);  gamma[k] in [0,1] init 0.5  <span class='cm'># reliability of each teammate</span>\n"
  "  <span class='kw'>def</span> refit():\n"
  "    <span class='kw'>repeat</span> em_sweeps:\n"
  "      <span class='cm'># E-step: responsibility each sensed choice is model-based, not random</span>\n"
  "      <span class='kw'>for</span> each sensed (k, S, a):  r[k,t] = gamma[k]*soft(k,a) / (gamma[k]*soft(k,a) + (1-gamma[k])/|S|)\n"
  "      <span class='cm'># M-step: update reliabilities, then factors weighting choices by r</span>\n"
  "      <span class='kw'>for</span> k:  gamma[k] = mean_t r[k,t]\n"
  "      weighted-ALS on  rewards(precision beta)  +  choices(positive a, neg ~ active set, weight r[k,t])\n"
  "</div>")
A("<p><b>Why this is the right tool, and what it predicts.</b> (i) It DEGRADES GRACEFULLY: a teammate "
  "that is exploring contributes $\\approx 0$ to the latent update (low $r$), so the choice channel "
  "never poisons the structure, the core fix you asked for. (ii) $\\gamma_k$ is INTERPRETABLE, drone "
  "$i$'s estimate of how 'expert' teammate $k$ currently is, learned purely from public actions (ZK). "
  "(iii) It is SELF-CALIBRATING over time: early, everyone explores so all $\\gamma_k$ are low and the "
  "swarm relies on the reward channel; late, $\\gamma_k$ rise and the (noise-immune) choice channel "
  "takes over, the explore-to-exploit regime shift falls out of the model rather than being "
  "scheduled. (iv) It unifies our heuristics: ChoiceCF's competence ramp is a crude proxy for "
  "$\\gamma_k$, and the predictive-variance shrinkage (EMCF) is the structure-confidence half; "
  "ChoiceEM is the principled joint version.</p>")
A("<div class='box'><b>Scouted ideas it builds on.</b> Dawid-Skene EM for annotator reliability + "
  "true labels (and its matrix-factorization variants, coupled/symmetric NMF; spectral-meets-EM); "
  "inverse RL from heterogeneous suboptimal demonstrators via a Boltzmann-rationality model of "
  "expertise (IRLEED, MaxEnt-IRL); McFadden conditional-logit / Plackett-Luce discrete choice; and "
  "crowdsourcing reliability estimation. Our novelty is the DECENTRALIZED, ZK, ONLINE instance: each "
  "agent infers its teammates' choice-informativeness from public actions alone and fuses it with "
  "low-rank completion in one EM loop, no labels, no communication, no shared parameters.</p></div>")
A("<div class='box warn'><b>Honest result (we implemented, rescued, and re-tested it): ChoiceEM does "
  "NOT win on skill, but the rescue confirms exactly WHY, and reveals where the choice channel does "
  "win.</b> Across reward noise $\\sigma_{\\mathrm{obs}}\\in\\{0.6,1.0,2.0\\}$ (8 seeds, bootstrap 95% "
  "CI), the naive ChoiceEM ($\\gamma_0=0.5$, no warm-up) DEADLOCKS, unseen $0.012$, anytime $0.163$, "
  "below even the fixed-ramp ChoiceCF (unseen $0.093$, anytime $0.219$). The principled fix we then "
  "tested, ChoiceEM-rescue, starts skeptical ($\\gamma_0=0.1$) and freezes the EM gate for the first "
  "$30\\%$ of rounds so the reward channel can fit a usable model first. It WORKS as a deadlock fix: "
  "anytime climbs $0.163\\to0.217$, statistically tied with the ramp's $0.219$. But the learned "
  "per-teammate gate still buys NO skill edge over the crude fixed ramp (rescue unseen $0.031$ vs "
  "ChoiceCF $0.093$). Two lessons: (1) the EM gating genuinely DEADLOCKS at cold-start, the E-step "
  "needs a good model to tell a model-based choice from a random one, but down-weighting choices keeps "
  "the model from improving; the rescue's reward-warm-start breaks exactly this chicken-and-egg, "
  "validating the root cause. (2) Once the deadlock is gone, a learned gate and a scheduled ramp are "
  "indistinguishable here, so the extra machinery is not worth its complexity.</p>"
  "<p><b>The positive that fell out:</b> the choice channel is not useless, it is extreme-noise "
  "insurance. At $\\sigma_{\\mathrm{obs}}=2.0$ the noise-immune ChoiceCF (an action's argmax is "
  "unaffected by reward noise) overtakes RewardCF on BOTH metrics, unseen $0.093$ vs $0.042$ and "
  "anytime $0.219$ vs $0.179$ (non-overlapping CIs), whereas at $\\sigma_{\\mathrm{obs}}=0.6$ RewardCF "
  "still dominates ($0.295$ vs $0.093$). The choice methods are flat in $\\sigma_{\\mathrm{obs}}$ by "
  "construction (they never read the noisy reward), so they cross over RewardCF as rewards degrade. "
  "Practical reading: fuse the channels and lean on choices only when your own reward signal is "
  "drowning, do not pay for learned per-teammate gating, the schedule is enough. We keep this as a "
  "documented negative-with-a-niche, the kind a tutorial should show.</p></div>")

A("<h4>Making it work: held-out informativeness, and a sanity check (the follow-up)</h4>"
  "<p>The null above is for a HOMOGENEOUS swarm, where every teammate explores then exploits on the "
  "same schedule, so there is no per-teammate difference for $\\gamma_k$ to exploit and it can only "
  "match the fixed ramp. The interesting case is HETEROGENEOUS teammates: some discover the structure "
  "earlier (and their choices are worth trusting sooner), some are unreliable. We built that test "
  "(<code>pilot_choicehetero.py</code>): real learners alongside SPECIAL teammates that are either "
  "ORACLE choosers (always engage their true best, maximally informative) or RANDOM choosers (faulty / "
  "off-objective). It doubles as a SANITY check, a working informativeness estimator MUST drive "
  "$\\gamma$(oracle) high and $\\gamma$(random) low; if it cannot, it is broken.</p>")
A(plain("This is a good habit for any negative result: build an experiment whose answer is OBVIOUS "
        "(oracles should be trusted, random teammates should not) and check the method delivers it. "
        "Here it did more than validate, it diagnosed the bug."))
A("<div class='formal'><h4>Why naive ChoiceEM fails the sanity, and the fix (Proposition 9)</h4>"
  "<p>A uniform-random teammate is the WORST case for the mixture model: its choice is equally likely "
  "to land anywhere, so the 'is this informative?' posterior averages to $\\gamma$ itself, no force "
  "pushes it down. Worse, the naive E-step scores each choice against a factor $\\hat p_k$ FIT to that "
  "same teammate's choices, which OVERFITS and makes even random choices look informative. Result "
  "(8 seeds): in-sample ChoiceEM gives random teammates $\\gamma\\approx 0.70$ (should be low), barely "
  "below oracle's $0.95$, it FAILS the sanity. The fix is a HELD-OUT (predictive) responsibility: "
  "score each choice ONCE against the model BEFORE the refit incorporates it. A random teammate's "
  "choices are then independent of the scorer, so its $\\gamma$ stays at its low prior; a genuine "
  "informer's choices are predicted even held-out, so its $\\gamma$ rises. Measured: predictive "
  "$\\gamma$(oracle) $\\approx 0.49 \\gg \\gamma$(random) $\\approx 0.10$ (about $5\\times$, "
  "non-overlapping CIs), it PASSES the sanity, and beats in-sample ChoiceEM on skill everywhere.</p></div>")
A("<div class='box key'><b>The positive that matters: knowledge propagates through the choice channel "
  "(your teammates' competence is contagious).</b> When knowledgeable teammates are present, the GOOD "
  "drones' unseen-pair skill LEAPS, from $0.089$ with all-ordinary teammates to $0.55$ with $50\\%$ "
  "oracle teammates, learned purely from watching their CHOICES, with zero communication and no "
  "teammate rewards. So a swarm where some agents discover the latent structure earlier can lift the "
  "rest through actions alone, and held-out $\\gamma_k$ is what lets a drone preferentially absorb the "
  "knowledgeable teammates while ignoring the noisy ones.</div>")
A(plain("Honest scope of the fix. Held-out $\\gamma_k$ is a correct ESTIMATOR of who-knows-what (it "
        "passes the sanity and beats the in-sample version), and it is more robust than the fixed ramp "
        "when many teammates are unreliable. But it does NOT make the choice channel beat the simple "
        "ramp on raw skill in every regime: when teammates are mostly GOOD, the ramp's blanket trust "
        "slightly edges held-out's caution (oracle unseen $0.55$ vs $0.49$), and if you can observe "
        "teammate REWARDS at all, the reward channel beats every choice-only method (oracle $0.62$). So "
        "per-teammate informativeness is the right tool for the choice-only, MIXED-reliability regime; "
        "the next refinement (a reward-improvement gradient, to also down-weight teammates that are "
        "consistent but WRONG, which held-out still trusts) is in the backlog."))

A("<h3>5.7 Do we even need to guess the rank? (and how to adapt it)</h3>")
A("<p>A natural worry: every structured method uses a GUESSED rank $\\hat d=8$ while the true rank is "
  "$d=5$, and every drone starts from the SAME guess (it is a shared hyperparameter, not learned or "
  "shared state, each drone's factors stay private). What if the guess is wrong, and can the method "
  "learn the rank from data?</p>")
A("<div class='formal'><h4>Effect of a wrong guess (and why over-guessing is safe)</h4>"
  "<p>Write $\\hat d$ for the guessed rank, $d$ for the truth. Two regimes:</p>"
  "<ul>"
  "<li><b>Under-guess</b> ($\\hat d < d$): the model literally cannot represent the rank-$d$ structure; "
  "it projects $R$ onto its best rank-$\\hat d$ subspace and loses the rest, so unseen skill drops. "
  "This is the only harmful case.</li>"
  "<li><b>Over-guess</b> ($\\hat d > d$): the extra $\\hat d - d$ latent directions carry no signal; the "
  "ridge term $\\lambda\\lVert\\cdot\\rVert^2$ shrinks them toward $0$ and the data places $\\approx 0$ "
  "energy there, so prediction is essentially unchanged. Over-guessing costs a little compute, not "
  "accuracy.</li>"
  "</ul>"
  "<p>So the safe rule is GUESS GENEROUSLY ($\\hat d \\ge d$); the regularizer absorbs the slack. "
  "Measured (Section 8.12, $\\hat d$ sweep, true $d=5$): unseen skill is $0.222$ at $\\hat d=2$ "
  "(under-guess, hurts) and $\\approx 0.33$-$0.36$ for $\\hat d \\in \\{5,8,12,20\\}$ (flat, robust). "
  "We deliberately run $\\hat d=8 > 5$ to show no oracle knowledge of the rank is used.</p></div>")
A(plain("Think of the rank as the number of DIALS the model is allowed to use to explain compatibility. "
        "The true world uses $d=5$ dials. If you give the model too few ($\\hat d<5$, 'under-guess') it "
        "physically cannot capture reality and accuracy suffers, the one dangerous mistake. If you give "
        "it too many ($\\hat d>5$, 'over-guess'), the spare dials simply find nothing to do and the "
        "gentle regularizer parks them at zero, costing a sliver of compute but not accuracy. Hence the "
        "safe habit: when unsure, guess a bit HIGH. 'Projecting onto a rank-$\\hat d$ subspace' is just "
        "the formal way to say 'forced to explain everything with only $\\hat d$ dials'."))
A("<p><b>Can the method ADAPT the rank as data grows? Yes, several ways (scouted).</b> The key signal "
  "is already in hand: the structure-confidence spectrum (Level 2, Section 5.5). The number of latent "
  "directions the data CONFIDENTLY supports is the number of large eigenvalues of the aggregate design "
  "$G_U=\\sum_{k,j}\\beta_{kj}p_kp_k^\\top$; early on only a few are well-determined, and more become "
  "supportable as coverage grows, so the EFFECTIVE rank should start small and rise toward $d$.</p>")
A("<ul>"
  "<li><b>Automatic Relevance Determination (ARD).</b> Put a per-column precision prior "
  "$\\alpha_r$ on the factor columns (a Gamma hyperprior in the variational EMCF); a column with no "
  "signal has its posterior precision $\\alpha_r\\to\\infty$ and is pruned. The rank self-determines, "
  "no hard choice, and grows with data. (Bayesian/VB matrix and tensor factorization with automatic "
  "rank determination; Nakajima et al.; Zhao et al.)</li>"
  "<li><b>Nuclear-norm / singular-value thresholding.</b> Penalize $\\lVert\\hat R\\rVert_*$ (sum of "
  "singular values); the threshold $\\lambda$ implicitly selects rank by zeroing small singular values "
  "(this is what our SoftImpute baseline already does).</li>"
  "<li><b>Spectral-gap estimate.</b> Read the rank off the eigenvalue spectrum of the accumulated "
  "$\\hat R$: keep the components above the noise floor (the gap between signal and noise singular "
  "values).</li>"
  "<li><b>Online greedy rank growth.</b> Start with small $\\hat d$; when held-out OWN-reward fit "
  "plateaus, add one latent dimension and keep it only if it reduces validation error, decentralized "
  "self-validation (the StackCF idea) drives the schedule.</li>"
  "</ul>"
  "<div class='box key'><b>Result: ARD-EMCF removes the rank hyperparameter.</b> We extended the "
  "variational EMCF (Section 5.3) with ARD per-column precisions (each drone runs it locally, ZK) and "
  "tested it (8 seeds, true $d=5$). It recovers a STABLE effective rank $\\approx 3.2$ that is "
  "essentially IDENTICAL whether the guessed rank is $\\hat d=8$ or $\\hat d=20$, so you never have to "
  "know or tune the rank; skill is unchanged across $\\hat d$ (no overfit from the 15 extra dimensions) "
  "and anytime reward actually IMPROVES ($0.46$ vs EMCF $0.42$, the pruning regularizes). Two honest "
  "caveats: (i) ARD prunes to the strongly-identified $\\approx 3$ directions, BELOW the generative "
  "$d=5$ (the two weakest latent directions are not identifiable under within-type spread + masking), "
  "which costs a little unseen skill ($0.57$ vs EMCF's $0.63$); (ii) the Bayesian prior already "
  "prevents overfit at large $\\hat d$ on its own, so ARD's real value is REPORTING the effective rank "
  "and the anytime gain, not rescuing accuracy. Net: a principled, communication-free way to let the "
  "data set the rank.</div>")

# 6 METRICS
A("<h2 id='metrics'>6. Metrics (and why each one matters)</h2>")
A("<p>To compare methods fairly we put every score on the same 0-to-1 ruler, called <b>skill</b>: the "
  "fraction of the gap between random guessing and a cheating oracle that a method closes. Below each "
  "metric is given both formally (a formula) and in plain words.</p>")
A("<div class='formal'><h4>Skill (normalized reward), formal definition</h4>"
  "<p>Fix drone $i$ and a uniform random size-$c$ offer $S\\subseteq[n]$. A policy that selects "
  "$a_i\\in S$ earns expected reward $\\rho_i = \\mathbb{E}_S[R_{i,a_i}]$. Define two references from "
  "the SAME offer distribution:</p>"
  "$$ \\mu_i = \\mathbb{E}_S\\Big[\\tfrac{1}{c}\\textstyle\\sum_{j\\in S} R_{ij}\\Big] \\;(\\text{random, "
  "mean-in-offer}), \\qquad \\rho^{\\mathrm{or}}_i = \\mathbb{E}_S\\big[\\max_{j\\in S} R_{ij}\\big] "
  "\\;(\\text{oracle, best-in-offer}). $$"
  "$$ \\mathrm{skill}_i(\\rho_i) = \\frac{\\rho_i - \\mu_i}{\\rho^{\\mathrm{or}}_i - \\mu_i} "
  "\\in (-\\infty, 1], \\qquad 0=\\text{random},\\quad 1=\\text{oracle}. $$"
  "<p>Skill is an affine rescaling of expected reward (equivalently of per-round regret "
  "$\\rho^{\\mathrm{or}}_i-\\rho_i$), so it keeps reward's meaning while being comparable across worlds "
  "of different difficulty. The oracle is the per-round best-in-offer (the centralized, "
  "complete-information ceiling in the non-contention setting); under contention it becomes the "
  "Hungarian matching optimum (Section 8.13).</p></div>")
A(plain("Reading the formula: $\\mathbb{E}_S[\\cdots]$ means 'the average over all the random handfuls "
        "of targets a drone might be offered'. $\\rho_i$ is what the method actually earns on average, "
        "$\\mu_i$ is what blind guessing earns (the average target in the offer), and "
        "$\\rho^{\\mathrm{or}}_i$ is what a cheating oracle that always grabs the best-in-offer earns. "
        "Skill just measures 'how far up from guessing toward the oracle did you get', as a fraction. "
        "'Affine rescaling' means we only stretched and shifted the reward axis (so $0$ and $1$ land on "
        "guessing and oracle), without distorting comparisons. 'Regret' is the flip side, how far SHORT "
        "of the oracle you fell; less regret means more skill."))
A("<p>So <b>skill = 0</b> means 'no better than guessing' and <b>skill = 1</b> means 'as good as the "
  "cheating oracle'. A method at 0.4 has closed 40% of that gap. A difference counts only if the error "
  "bars (confidence intervals, defined at the end of this section) do not overlap. We use five flavors "
  "of skill, each answering a different question:</p>")

def metric(name, what, why, look, formal=None):
    A("<div class='step'><h3>%s</h3>" % name)
    A("<p><b>What it is.</b> %s</p>" % what)
    if formal:
        A("<div class='formal'>%s</div>" % formal)
    A("<p><b>Why it matters.</b> %s</p>" % why)
    A("<p><b>What to look for.</b> %s</p></div>" % look)

metric("Overall skill",
       "How good the final learned policy is when offered a random handful of targets.",
       "It is the all-round 'did you learn the task' score.",
       "Higher is better; but it can be flattered by methods that explored a lot 'for free' (see "
       "anytime).",
       formal="<p>$\\mathrm{skill}$ of the frozen policy $\\hat\\pi_i(S)=\\arg\\max_{j\\in S}\\hat "
       "R_{ij}$ over offers $S$ drawn from ALL targets $[n]$: "
       "$\\rho_i = \\mathbb{E}_S[R_{i,\\hat\\pi_i(S)}]$, plugged into the skill formula above.</p>")
metric("Unseen-pair skill (the categorical one)",
       "The same score, but ONLY on targets the drone never tried itself, the 'unseen' pairs.",
       "This is the make-or-break test of GENERALIZATION. A learner with no shared structure literally "
       "cannot do better than guessing here (it has no information), so it scores ~0 by construction. "
       "Any score clearly above 0 means the method is genuinely transferring knowledge to things it "
       "never experienced.",
       "We want our methods well above 0 and the structure-free baselines stuck at 0. The GAP between "
       "them is the headline result.",
       formal="<p>Let $\\mathcal{U}_i = \\{j : \\text{drone } i \\text{ never engaged } j\\}$ be its "
       "unseen set after training. Restrict offers to $\\mathcal{U}_i$: "
       "$\\rho_i = \\mathbb{E}_{S\\subseteq\\mathcal{U}_i}[R_{i,\\hat\\pi_i(S)}]$ and the references "
       "$\\mu_i,\\rho^{\\mathrm{or}}_i$ to $\\mathcal{U}_i$. A structure-free learner has "
       "$\\hat R_{ij}=\\text{const}$ on $\\mathcal{U}_i$, so $\\rho_i=\\mu_i$ and "
       "$\\mathrm{skill}=0$ exactly (the tabular floor).</p>")
metric("Anytime / AUC skill (the operational one)",
       "The reward actually EARNED while learning, summed over the rounds ('how many targets did you "
       "destroy by round K'), not just the quality of the policy at the end.",
       "In the real world you care about results DURING the mission, not only after. This metric "
       "CHARGES a method for any time it spends just exploring or probing. It is the most honest, "
       "practical score.",
       "Methods that explore politely while still exploiting (ours) win; methods that pause to probe, "
       "or that never stop exploring, lose here even if their final policy looks fine.",
       formal="<p>Sum the REALIZED reward as the episode runs and normalize the cumulative quantity. "
       "With per-round totals $\\text{real}_t=\\sum_i R_{i,a_i^t}$, $\\text{orac}_t=\\sum_i\\max_{j\\in "
       "S_i^t}R_{ij}$, $\\text{rand}_t=\\sum_i\\frac1c\\sum_{j\\in S_i^t}R_{ij}$, the anytime "
       "trajectory is</p>"
       "$$ \\mathrm{skill}^{\\mathrm{any}}_t = \\frac{\\sum_{\\tau\\le t}\\text{real}_\\tau - "
       "\\sum_{\\tau\\le t}\\text{rand}_\\tau}{\\sum_{\\tau\\le t}\\text{orac}_\\tau - "
       "\\sum_{\\tau\\le t}\\text{rand}_\\tau}, $$"
       "<p>and the headline anytime skill is $\\mathrm{skill}^{\\mathrm{any}}_T$ (the whole-episode "
       "value). A probe/explore phase earns $\\approx\\text{rand}$, so it is charged in full here, "
       "unlike overall skill.</p>")
metric("State-uniqueness",
       "How DIFFERENT the drones' learned internal models are from each other.",
       "It checks that decentralization is REAL: if every drone ended up with the same model, 'no "
       "communication' would be meaningless. It is also the quantity that distinguishes the two "
       "masking models in the theory.",
       "It should rise as observation gets more limited (drones see different things), and stay high "
       "over time under persistent masking (durable) but fade under i.i.d. masking (transient).",
       formal="<p>Compare drones' full predicted matrices $\\hat R^{(i)}=P^{(i)}(U^{(i)})^\\top$ "
       "(frame-invariant, unlike a single row). Mean-center each, normalize, and take the average "
       "pairwise cosine $\\bar c$; report $1-\\bar c$:</p>"
       "$$ \\text{uniq} = 1 - \\frac{2}{m(m-1)}\\sum_{i<i'} \\frac{\\langle \\tilde R^{(i)}, "
       "\\tilde R^{(i')}\\rangle}{\\lVert \\tilde R^{(i)}\\rVert\\,\\lVert \\tilde R^{(i')}\\rVert}, "
       "\\quad \\tilde R^{(i)} = \\hat R^{(i)} - \\overline{\\hat R^{(i)}}. $$"
       "<p>$\\text{uniq}\\to 0$ when all drones converge to the same model; bounded away from $0$ when "
       "each keeps a permanent blind set (persistent masking).</p>")
metric("Onboarding / newcomer skill vs number of probes",
       "How good the prediction for a brand-new target (or a brand-new drone) gets, as a function of "
       "how many trial engagements ('probes') it has received.",
       "It measures sample efficiency for cold-start: how cheaply can the swarm absorb something new?",
       "Our methods should reach high skill after only about d probes (the number of hidden traits), "
       "while a structure-free learner needs roughly one probe per drone or per target.",
       formal="<p>For a new target with factor $u_\\star$ seen via $k$ shared probes "
       "$\\{(p_{a},r_a)\\}_{a=1}^k$ and a basis $U$, the fold-in solves a $d$-dim least squares: "
       "$\\hat u_\\star=(\\sum_a p_a p_a^\\top+\\lambda I)^{-1}\\sum_a r_a p_a$, exact once $k\\ge d$ and "
       "the $p_a$ span $\\mathbb{R}^d$ (row completion); tabular needs $k\\approx m$ (one probe per drone). "
       "We plot newcomer skill vs $k$. <b>Strict ZK note:</b> for a new DRONE the basis is the "
       "targets' factors $U$, which the newcomer does NOT receive from any peer; it RECOVERS $\\hat U$ "
       "itself by running its own factorization on the public broadcast it passively observed (under "
       "its own mask $\\rho$). So the only thing handed to a newcomer is the same partial, noisy "
       "broadcast everyone hears.</p>")

A("<h3>6.6 The evaluation procedure (how every number is produced)</h3>")
A("<p>So the metrics are reproducible, here is the exact protocol used after the $T$-round episode.</p>")
A("<div class='algo'>"
  "<span class='cm'># after training: freeze every drone's estimator (no more updates)</span>\n"
  "<span class='kw'>for</span> each evaluation offer (120 repeats x m drones):\n"
  "    overall:  draw S ~ uniform([n], c);      record R[i, argmax_{j in S} Rhat_ij], max_{S} R, mean_{S} R\n"
  "    unseen :  draw S ~ uniform(U_i, c)        <span class='cm'># U_i = targets i never engaged; skip if |U_i|&lt;c</span>\n"
  "skill = (mean earned - mean random) / (mean oracle - mean random)        <span class='cm'># per the formula above</span>\n"
  "anytime = computed online DURING the episode (cumulative, formula above)\n"
  "<span class='cm'># repeat the whole episode for seeds 0..S-1 (S = 8, 12, or 20 depending on the panel)</span>\n"
  "report MEAN over seeds and a BOOTSTRAP 95% CI:\n"
  "    resample the per-seed values with replacement B=10^4 times; CI = [2.5th, 97.5th] percentile of the resample means\n"
  "</div>")
A("<p>A claim 'A beats B' is only made when their bootstrap CIs do not overlap. All per-seed values "
  "are saved to <code>results/pilots/*.json</code> so any number can be recomputed without re-running "
  "(Section 10).</p>")

# 7 THEORY
A("<h2 id='thy'>7. Theory, step by step (why the win is categorical)</h2>")
A("<p>The empirical results are not luck: a short chain of arguments PREDICTS them. We give each "
  "result as: <em>statement</em>, <em>why it matters</em>, <em>proof (key steps)</em>, and "
  "<em>confirmed by</em> (which experiment). Full formal proofs: <code>docs/THEORY_FORMAL.md</code>.</p>")
A(plain("The whole theory in everyday language, before the symbols. <b>The tabular floor:</b> a learner "
        "that treats each target separately is HELPLESS on any target it never tried, its best guess is "
        "just the overall average, so it scores zero on the unseen, period. <b>Row completion:</b> a "
        "learner that uses the shared hidden structure can fill in its ENTIRE row of preferences from only "
        "about $d$ tries (one per hidden dial), a few samples buy opinions on everything. <b>Anytime "
        "separation:</b> when there are far more targets than rounds, the separate-targets learner cannot "
        "even earn well WHILE learning, no matter how long it runs; the structured learner is near-perfect "
        "after ~$d$ rounds. <b>Masking dichotomy:</b> whether each drone is permanently or just "
        "occasionally blind to teammates does not change the headline, it only changes whether the drones' "
        "views stay permanently different. <b>Later results</b> formalize the confidence and contention "
        "findings. "
        "If you read nothing else here: the advantage is CATEGORICAL (a zero-versus-nonzero gap), not a "
        "matter of tuning."))
A("<h4>Setup and notation (recap of Sections 3 and 6)</h4>")
A("<p>True reward $R = P U^\\top$ of rank $d$, unit factors so $R_{ij}=\\langle p_i,u_j\\rangle\\in"
  "[-1,1]$. Drone $i$ is offered a uniform size-$c$ set each round; $\\mu_i$ is its mean-in-offer "
  "(random reference), $\\rho^{\\mathrm{or}}_i=\\mathbb{E}_S[\\max_{j\\in S}R_{ij}]$ the oracle, and "
  "$\\mathrm{skill}=(\\rho-\\mu_i)/(\\rho^{\\mathrm{or}}_i-\\mu_i)$. A learner is "
  "<b>structure-free</b> (Definition) if its estimate $\\hat R_{ij}$ depends only on drone $i$'s own "
  "past pulls of target $j$, and equals a fixed prior constant on any $j$ it never pulled (the per-arm "
  "tabular class: UCB-Indep, Tabular).</p>")
A(plain("A few proof shorthands you will meet below. <b>$\\Omega(1)$, $\\Theta(d)$, $O(\\cdot)$</b> are "
        "'big-O' size notation: $\\Theta(d)$ means 'grows in proportion to $d$', $\\Omega(1)$ means 'at "
        "least some fixed positive amount', $O(d/T)$ means 'no bigger than about $d/T$'. <b>Order "
        "statistics</b> just means 'the maximum (or k-th largest) of a few random draws', here, the best "
        "target in a small random offer. <b>Jensen / concavity</b> is the fact that the average of a "
        "curved-downward function is below the function of the average, which we use to bound how much a "
        "blind explorer can luck into. <b>Borel-Cantelli</b> is the probability fact that an event with "
        "a steady positive chance each round is guaranteed to happen infinitely often given enough "
        "rounds. None of these change the story; they are just the tools that make it airtight."))

A("<div class='step'><h3>The tabular floor (EXACT)</h3>")
A("<p><b>Statement.</b> For a structure-free learner and any target j drone i never pulled, the "
  "estimate is the prior constant, so its squared error is at least the row variance "
  "$\\mathrm{Var}_j(R_{ij}) = \\Omega(1)$, and its expected unseen-pair skill is exactly 0.</p>")
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
  "estimates the column mean $\\langle \\bar p, u_j\\rangle$ = the rank-1 'popularity' term, "
  "so it gets PARTIAL unseen skill (~0.17), not zero and not full, exactly as measured.</p></div>")

A("<div class='step'><h3>CF completes a whole row from O(d) observations (EXACT)</h3>")
A("<p><b>Statement.</b> If the target factors U are known (rank d) and drone i observes its true "
  "rewards on any set Omega with |Omega| &gt;= d whose factors {u_j} span R^d, then p_i is the unique "
  "least-squares solution and $R_{ij} = \\langle p_i, u_j\\rangle$ is recovered EXACTLY for "
  "ALL j, including never-pulled ones.</p>")
A("<p><b>Why it matters.</b> This is the engine of generalization and of onboarding/cold-start: a "
  "few observations plus the shared structure determine the entire row. Per-drone sample complexity "
  "drops from $\\Theta(n)$ (tabular) to $\\Theta(d)$, a categorical, not incremental, change.</p>")
A("<p><b>Proof (key steps).</b> (1) Stacking $j\\in\\Omega$ gives the linear system "
  "$R_{i,\\Omega} = U_\\Omega\\, p_i$. (2) Spanning means $U_\\Omega$ has rank $d$, so "
  "$U_\\Omega^\\top U_\\Omega$ is invertible and the solution $p_i = (U_\\Omega^\\top U_\\Omega)^{-1} "
  "U_\\Omega^\\top R_{i,\\Omega}$ is unique and equals the true $p_i$. (3) Then "
  "$\\langle p_i, u_j\\rangle$ reproduces $R_{ij}$ for every $j$. (With noise, the ridge "
  "estimate has error O(sigma^2 d / lambda_min), vanishing as data grows.) U itself is identified from "
  "the pooled broadcast by standard matrix completion once about d(m+n) entries are seen.</p>")
A("<p><b>Confirmed by.</b> Target onboarding at ~d shared probes (F3) and newcomer cold-start at ~d "
  "own probes (F10), both vs tabular's ~n; and unseen skill rising as the rank falls (F4).</p></div>")

A("<div class='step'><h3>Anytime separation under starvation (ORDER)</h3>")
A("<p><b>Statement.</b> With $n$ targets, offers of size $c$, horizon $T$, ANY structure-free learner "
  "has anytime (cumulative-reward) skill at most $g(cT/n)$, which $\\to 0$ when $cT = o(n)$, EVEN with "
  "full broadcast (here $g(x)=\\mathbb{E}[\\max \\text{ of } \\lceil x\\rceil \\text{ i.i.d. draws}]-\\mu$ "
  "is the expected-order-statistic gap). A CF learner reaches near-oracle after about $d$ rounds, so "
  "its anytime skill is $1 - O(d/T)$.</p>")
A("<p><b>Why it matters.</b> 'Final-policy quality' can flatter a method that explores a lot for "
  "free. The operationally honest metric is reward EARNED while learning (targets destroyed by round "
  "K). This theorem says structure-free methods cannot earn under starvation.</p>")
A("<p><b>Proof (key steps).</b> (1) A structure-free learner only earns above the mean on a round "
  "whose offer contains an already-pulled target; its pulled set has size &lt;= t-1. (2) Because new "
  "targets are chosen blind to their reward, the pulled set is reward-blind, so the offered-and-pulled "
  "rewards are like a few i.i.d. draws; by concavity of expected order statistics (Jensen) the "
  "per-round surplus is &lt;= g(c(t-1)/n). (3) Summing over t and using g increasing gives "
  "&lt;= g(cT/n) -&gt; 0 for cT = o(n). The broadcast does not change the pulled set (the tabular floor). "
  "(UCB is even lower: its infinite untried-arm bonus makes it explore forever while n &gt; t.)</p>")
A("<p><b>Confirmed by.</b> UCBIndep stuck at ~0 anytime even at T=200 with n=240 (F9); online CF "
  "earns from round one and dominates at every horizon (F6).</p></div>")

A("<div class='step'><h3>Masking-model dichotomy (EXACT limits)</h3>")
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
  "(3) The categorical results use only own pulls (the tabular floor) and U-identification (row completion), which "
  "both hold under either mask.</p>")
A("<p><b>Confirmed by.</b> i.i.d. vs persistent give the same unseen/anytime curves, but "
  "state-uniqueness is flat in T under persistent and DECAYS under i.i.d. (F8), exactly as "
  "predicted.</p></div>")
A("<div class='step'><h3>An additive model cannot personalize (EXACT)</h3>")
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
  "measured value, with honest tensions: the anytime constant is loose though its mechanism is exact; the "
  "row-completion result idealizes 'U known' vs finite-data recovery) is in <code>docs/THEORY_FORMAL.md</code>.</p>")
A("<div class='step'><h3>Recent findings, formalized</h3>"
  "<p>The later confidence, contention, and rank results each have a matching statement (full versions + "
  "justifications in <code>THEORY_FORMAL.md</code>):</p>"
  "<p><b>Confidence.</b> For an UNSEEN pair, $\\hat u_j$ comes only from the broadcast, "
  "so inverse-own-variance ('precision') fit weighting, which over-weights a drone's own rewards that "
  "carry no information about $\\hat u_j$, inflates $\\mathrm{Var}(\\hat u_j)$ and is SUBOPTIMAL vs "
  "uniform; the Bayesian posterior (EMCF) is the consistent full-information estimator with a valid "
  "predictive interval. Caveat: the FULL predictive-variance UCB over-explores (the own-factor term is "
  "uniformly large early), so info-directed exploration should use the COLLECTIVE term "
  "$p_i^\\top\\Sigma_{u_j}p_i$. (Confirms 8.12, the precision sweep, and the EM win.)</p>"
  "<p><b>Contention symmetry-breaking.</b> Under a shared pool with no communication, "
  "deterministic argmax makes each same-type group of size $g$ lose $g-1$ engagements (the matching "
  "floor); a FIXED private offset $\\hat R_{ij}+\\varepsilon h_i[j]$ makes same-type picks a.s. "
  "distinct (collisions $\\to 0$) while preserving any target with margin $>2\\varepsilon\\lVert "
  "h\\rVert_\\infty$, and only a fixed-AND-private offset is stable (per-round-random re-collides, "
  "shared-signal re-synchronizes). (Confirms the 8.13 contention win and the 4-design meta-finding.)</p>"
  "<p><b>ARD rank.</b> ARD retains a latent column iff the masked design excites it above "
  "the prior floor, so the recovered effective rank = the data-IDENTIFIABLE rank $\\le d$ (strictly "
  "$<d$ under masking), and it does not depend on the guessed $\\hat d$, removing the rank "
  "hyperparameter. (Confirms 5.7: effective rank $\\approx 3.2$ at both $\\hat d=8,20$.)</p></div>")
A(plain("These three in plain words. <b>Confidence:</b> to judge a target you never tried, ALL your "
        "useful information comes from teammates, so leaning extra-hard on your own (here irrelevant) "
        "readings actually HURTS, the humble 'treat all data evenly' rule beats the clever "
        "'trust-my-own-clean-data' rule. <b>Contention:</b> if look-alike drones all grab their "
        "identical favorite they keep colliding; giving each drone its own small, FIXED tie-breaking "
        "nudge makes them pick different near-ties and stop colliding, and it has to be both private and "
        "fixed (re-rolling it each round, or sharing it, undoes the benefit). <b>Rank:</b> the "
        "automatic-rank method keeps exactly as many hidden dials as the data can actually support and "
        "ignores the rest, so you never have to guess the number of dials yourself."))
A("<div class='step'><h3>Newest theory (a self-audit pass): scope corrected, precision fixed, a regret reason, and the honest keystone</h3>"
  "<p><b>Precise scope.</b> The boxed '3-condition iff' was loose. Correctly: the CATEGORICAL "
  "unseen gap vs a tabular learner needs only a shared channel ($\\rho>0$) and some structure (it holds "
  "even at $d=1$, even when sample-rich); $d>1$ is what additionally beats a POPULARITY baseline; and "
  "sample-starvation is what makes the unseen edge OPERATIONALLY dominant. Three conditions, three "
  "distinct roles, not one AND against one baseline.</p>"
  "<p><b>Precision (corrected).</b> Our earlier claim ('uniform beats precision') over-reached. "
  "Under HETEROGENEOUS teammate noise, bounded ratio-capped precision WINS (it down-weights the noisy "
  "sources yet preserves coverage), while UNBOUNDED $1/\\sigma^2$ over-concentrates on the clean ones and "
  "starves coverage. So noise-aware weighting helps iff sources differ in reliability, and only in a "
  "bounded form, matching the heterogeneous-noise sanity (8.12).</p>"
  "<p><b>Why directed exploration helps.</b> Given the recovered target factors, EMCF's "
  "predictive-variance UCB is exactly LinUCB on the latent features, so per-drone regret is "
  "$\\tilde O(d\\sqrt T)$, the formal reason confidence-directed probing beats a fixed schedule (and is "
  "what wins under churn). The JOINT case (learning the factors WHILE exploring) is open.</p>"
  "<p><b>The honest keystone.</b> The self-audit found that our headline 'a few samples complete the "
  "whole row' leans on a CITED step, recovering the shared structure $U$ from the MASKED broadcast, "
  "that we have NOT proven for structured (persistent) masking, where uniform-sampling completion theory "
  "does not directly apply. It is the central open problem; we label it plainly. (Full statements, "
  "proofs, and the critical audit: <code>docs/THEORY_FORMAL.md</code>.)</p></div>")
A(plain("Why this matters: good research polices its own claims. This pass DEMOTED the results that were "
        "textbook or over-labeled, FIXED the precision claim that the experiments actually contradicted, ADDED "
        "the missing regret reason for the exploration win, and NAMED the single gap the categorical story "
        "still rests on. The theory section came out smaller, sharper, and honest about its edges, which "
        "is the point."))

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
  "Both are $\\Theta(d)$ vs $\\Theta(n)$ separations.</p>")
A(plain("<b>Strictly zero-knowledge newcomer (no peer hand-off).</b> A subtle point of fairness: the "
        "newcomer does not RECEIVE the learned target factors from anyone (that would be a hidden form "
        "of communication). It is a passive listener that, over the warm-up rounds, hears the same "
        "public broadcast everyone else does (under its own personal masking $\\rho$) and recovers its "
        "OWN copy of the target basis $\\hat U$ by running the same factorization the swarm uses. We "
        "re-ran F10 this way over $\\rho\\in\\{1.0,0.5,0.25\\}$: the categorical gap over tabular "
        "SURVIVES at every $\\rho$ (tabular stays at the floor). At full visibility the fold-in "
        "personalizes well above the population average; under heavy masking ($\\rho=0.25$) the "
        "probe-efficiency curve FLATTENS, because the newcomer's self-recovered $\\hat U$ (built from "
        "little data) becomes the bottleneck rather than the number of probes. Honest takeaway: the "
        "$\\Theta(d)$-vs-$\\Theta(n)$ separation is real and ZK-clean, and its slope degrades "
        "gracefully as the newcomer hears less."))
A("<figure>%s<figcaption><strong>F3.</strong> Target onboarding: CF onboards from about d probes; "
  "tabular needs about one per drone.</figcaption></figure>" % img("F3_onboard.png", "F3"))
A("<figure>%s<figcaption><strong>F10.</strong> Newcomer cold-start (strictly ZK): a new drone "
  "recovers the target basis from its OWN passively-observed masked broadcast and folds in its "
  "factor; the tabular newcomer stays at the floor at every masking level &rho;.</figcaption></figure>"
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
  "$n \\gg T$, it never stops exploring untried arms, and PTF/ESTR pay a probe "
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
  "$\\sigma_{\\mathrm{obs}}$ rises while the clean choice channel (ChoiceZK) is flat. The crossover is "
  "at $\\sigma_{\\mathrm{obs}}\\approx 1$ (noise = half the signal range): for realistic noise the reward "
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
A("<p>For the camera-ready, the headline panels are confirmed at <b>20 seeds with bootstrap "
  "95%% confidence intervals</b> (a CI is a range we are 95%% confident the true mean lies in; "
  "non-overlapping intervals mean a real difference, not noise). Skill: 0 = random, 1 = oracle; "
  "ours in bold.</p>")
A(md_tables("docs/HEADLINE_TABLE.md"))

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
  "BiasModel stays near the additive ceiling (it cannot personalize, exactly as the additive-ceiling result says). "
  "Under masking (the regime that matters) and on the anytime metric, our methods beat the best of "
  "these new baselines with non-overlapping error bars. No competitor wins in the regime that defines "
  "the problem.</p>")

A("<h3>8.10 External validation: ZK-MRTA-Sim, an independently-built simulator (transfer, not 'realism')</h3>")
A("<div class='box warn'><b>A note on naming.</b> We call this <b>ZK-MRTA-Sim</b>, an independently-built "
  "simulator of the same Zero-Knowledge / Zero-prior / Zero-communication MRTA setting (its PettingZoo "
  "package is <code>tabula_drone</code>; not to be confused with our <i>Tabular</i> baseline). We "
  "previously called it 'a realistic simulator', which over-claims, so we are precise: it is still a "
  "SIMULATOR (a "
  "PettingZoo environment), not real hardware or field data, and it is itself a LATENT-compatibility "
  "world, so it does not independently prove the world is low-rank. What it genuinely provides is "
  "<b>out-of-harness transfer</b>: a SEPARATELY-built codebase, which we did not design for this "
  "experiment, with DIFFERENT dynamics our generative model lacks, spatial geometry (drones/targets "
  "with positions and distances), target HEALTH that depletes as it is engaged (a soft CONTENTION / "
  "matching pressure), episodic resets, and its own reward/observation noise. Its oracle is a Hungarian "
  "OPTIMAL-ASSIGNMENT matcher. So this is a test of TRANSFER and partial-contention robustness, not of "
  "realism.</div>")
A("<p>We drop our estimator in as a drop-in policy (WeightedALS = RewardCF in the env's policy "
  "interface) and benchmark it (3 seeds, learning over 16 episodes) against the environment's OWN "
  "policies: its SGD matrix-factorization, a per-arm UCB-Indep, random, and the optimal-assignment "
  "oracle. Score = efficiency (mean reward per step over the converged second half of episodes), "
  "normalized to skill = (policy &minus; random)/(oracle &minus; random) (Figure F13).</p>")
A("<figure>%s<figcaption><strong>F13.</strong> ZK-MRTA-Sim, the independent simulator (spatial, depleting "
  "HP, episodic; matching oracle): converged skill (left) and per-episode learning curves "
  "(right).</figcaption></figure>" % img("F13_realsim.png", "F13"))
A("<p><b>Takeaway.</b> The advantage TRANSFERS out of our own harness: our method reaches skill ~0.81 "
  "(low variance), clearly beating the environment's own SGD matrix-factorization (~0.25) and the "
  "per-arm UCB-Indep (~0.72), and approaching the matching oracle. So the result is not an artifact of "
  "our specific generative code: it survives a different implementation with spatial, depleting, "
  "episodic dynamics and a matching (contention) objective. <b>What it does NOT show:</b> real-world "
  "realism, or an independent confirmation of low-rankness (the sim is also latent-based); for the "
  "latter see the assumption-stress test next.</p>")

A("<h3>8.11 Stress-testing the low-rank assumption</h3>")
A("<p>A fair worry: what if the world is not exactly low-rank? We deliberately break the assumption "
  "two ways and watch what happens (Figure F14). First, a nonlinear reward link (a power curve) that "
  "raises the effective rank; here our methods actually do a touch BETTER, because the curve sharpens "
  "the differences between targets. Second, and more importantly, we add random noise to every entry "
  "of the reward table, which drives the effective rank from 5 up toward 29 (nearly full-rank, i.e. "
  "almost no structure left).</p>")
A("<figure>%s<figcaption><strong>F14.</strong> Skill as we leave exact low-rank (er = realized "
  "effective rank): nonlinear link (right) and entrywise-noise approximate rank (left).</figcaption></figure>"
  % img("F14_stress.png", "F14"))
A("<p><b>Takeaway.</b> The advantage degrades GRACEFULLY, not off a cliff: as the structure is "
  "progressively destroyed (effective rank 5 to 29), skill slides down smoothly and our methods stay "
  "the best the whole way, only meeting the floor when there is essentially no low-rank structure "
  "left to exploit. So the result does not hinge on the world being perfectly low-rank, only on there "
  "being SOME usable structure, which real problems have.</p>")

A("<h3>8.12 Which ingredient matters? (one honest ablation)</h3>")
A("<p>Our recommended method, ActiveCFconv, has four design choices stacked together. To avoid a "
  "confusing 'zoo' of methods, we turn each one OFF in turn and measure the cost, all under one "
  "identical setup (12 seeds, 95%% CIs). This shows what each piece actually buys.</p>")
A(md_tables("docs/ABLATION_TABLE.md"))
A("<p><b>Takeaway.</b> (i) Learning ONLINE (updating every round) beats batch 'probe-then-commit' "
  "methods (PTF, ESTR) under masking and on earned reward. (ii) An HONEST surprise: weighting "
  "observations by PRECISION (1/sigma^2) does NOT help at the default noise; turning it OFF (uniform "
  "weights) actually gives the BEST unseen skill in the table (0.584), because uniform weighting uses "
  "the plentiful broadcast more fully, and the broadcast is exactly what lets a drone judge targets it "
  "never tried. Precision weighting only earns its keep once the broadcast is genuinely noisy (we "
  "chart the crossover separately). (iii) the SVD warm-start mostly helps when the broadcast is full, "
  "at a small cost to early earned reward; (iv) ACTIVE exploration (chase the most uncertain target) "
  "beats plain random exploration on earned reward and dense-broadcast unseen; and (v) the method is "
  "robust to a wrong guessed rank as long as the guess is at least the true rank (5 to 20 all behave "
  "the same; only guessing 2, below the true 5, hurts), so you do not need to know the rank "
  "exactly.</p>")
A("<p>Because finding (ii) was a surprise, we chased it down: here is precision weighting ON vs OFF as "
  "we crank up the broadcast noise (full broadcast, so only the noise changes). It shows exactly when "
  "each choice wins.</p>")
A(md_tables("docs/PRECISION_SWEEP.md"))
A("<p><b>Takeaway.</b> The two choices cross over: when the broadcast is clean, uniform weighting wins "
  "(it trusts and uses all of it); when the broadcast is noisy, precision weighting wins (it correctly "
  "stops trusting the garbage). So 'weight by precision' is not free wisdom, it is the right call only "
  "in the high-noise regime; the honest default is uniform, switching to precision when you know the "
  "shared signal is noisy.</p>")
A("<h4>Sanity check: the precision negative, scoped exactly (heterogeneous sources)</h4>"
  "<p>To pin down WHEN precision is the right call, we ran an obvious-expected sanity test: hold the "
  "MEAN broadcast noise fixed and vary only its HETEROGENEITY across teammates. <em>homog</em> = every "
  "teammate equally noisy ($\\sigma_{\\mathrm{obs}}=1.0$); <em>hetero</em> = half the teammates clean "
  "($\\sigma=0.1$), half garbage ($\\sigma=1.9$), same mean. If precision weighting works, it MUST "
  "help in hetero (it can down-weight the garbage half) and need not help in homog.</p>")
A(md_tables("docs/PRECISION_HETERO.md"))
A("<p><b>Takeaway (a sharper statement of the negative).</b> The sanity passes, with an instructive "
  "twist (8 seeds, CIs). (i) Under HOMOGENEOUS noise, precision does NOT help (uniform best, the known "
  "result: coverage, not noise, binds). (ii) Under HETEROGENEOUS noise, BOUNDED precision "
  "(<code>relcap</code>, ratio-capped) WINS clearly, unseen $0.483$ vs uniform $0.390$ and anytime "
  "$0.356$ vs $0.267$ (both non-overlapping CIs). (iii) But UNBOUNDED precision (<code>full</code>, "
  "$1/\\sigma^2$) LOSES even in hetero ($-0.10$ unseen): correctly identifying the clean sources is not "
  "enough, raw $1/\\sigma^2$ weights pile almost all mass on the few clean rows and the drone's own "
  "data ($\\sigma=0.1\\Rightarrow$ weight $100$) and STARVE coverage. So the precise scope is: "
  "noise-aware weighting is the right call exactly when observation sources DIFFER in reliability, "
  "<em>and only in a ratio-bounded form</em> that stays coverage-friendly. This is the same lesson as "
  "the confidence bake-off (below): the win comes from confidence that does not let any source dominate "
  "the broadcast.</p>")
A("<h4>Can we get the best of both? (the right way to use confidence)</h4>")
A("<p>Rather than settle for a trade-off, we asked: is there a confidence rule that matches uniform "
  "when the broadcast is clean AND beats it when it is noisy? The diagnosis above is the clue: naive "
  "precision weighting failed not because confidence is useless, but because its raw weights "
  "(11 to 100) dwarfed the model's regularizer, so it quietly over-fit and under-used the broadcast. "
  "We bake off the fixes against plain uniform and plain precision, across clean and noisy broadcasts: "
  "bounded/normalized precision (<code>relcap</code>) and, crucially, the BAYESIAN factorization with "
  "confidence intervals (<code>EM</code> = variational EMCF: a predictive-interval UCB; "
  "<code>EMshrink</code> additionally shrinks uncertain predictions toward popularity).</p>")
A(md_tables("docs/CONFIDENCE.md"))
A("<p><b>Takeaway.</b> Confidence, done right, WINS, vindicating the instinct not to give up on it. "
  "<code>EM</code> (put the uncertainty in the MODEL, via a Bayesian posterior, and use its predictive "
  "interval to explore) dominates plain uniform: as good when the broadcast is clean and clearly better "
  "when it is noisy (unseen 0.33 vs 0.20 at the noisy full-broadcast point), because a confident "
  "posterior knows which predictions to trust. <code>EMshrink</code> pushes the FINAL-policy unseen "
  "quality highest of all (0.69) by also shrinking uncertain predictions toward popularity, at some "
  "cost to early earned reward, a clean Pareto choice (use it when you care about the converged policy, "
  "EM when you care about anytime reward). The shallow rules (bounded precision <code>relcap</code>) "
  "merely match uniform; the gain comes from SECOND-ORDER confidence (the posterior), not from "
  "reweighting single observations.</p>")

A("<h3>8.13 Does it survive contention? (when targets cannot be shared)</h3>")
A("<p>Our main results assume two drones CAN engage the same target. What if they cannot, so engaging "
  "a target uses it up (a 'matching' problem)? We test this directly: all drones see the same small "
  "pool of targets, each target goes to ONE random drone that wanted it, and the losers earn nothing "
  "that round. We shrink the pool to crank up the competition, and we normalize earned reward by the "
  "best possible assignment (the Hungarian matching optimum).</p>")
A(md_tables("docs/CONTENTION.md"))
A("<p><b>Takeaway (three honest findings from the numbers).</b></p>"
  "<ul>"
  "<li><b>The CATEGORICAL result survives.</b> Preference quality (unseen-pair skill, measured "
  "contention-free after training) stays high for CF, RewardCFconv holds $0.29$-$0.38$ across every "
  "pool size, while structure-free UCB-Indep sits at $\\approx 0$ throughout. Preference quality is a "
  "property of the learned model, not of how rewards were collected, so contention does not erase it.</li>"
  "<li><b>The OPERATIONAL gap narrows under heavy contention.</b> On earned reward (normalized by the "
  "Hungarian matching optimum), CF leads clearly while there is slack (pool $\\ge 30$: e.g. RewardCFconv "
  "$0.44$ vs PTF $0.24$ at pool $240$), but as the pool shrinks to $15$ (2:1 contention) collisions, "
  "not preferences, become the bottleneck and GREEDY (argmax) CF collapses toward the matching floor. "
  "But this is a DECISION failure, not an estimation one, and a one-line decentralized de-confliction "
  "(ContentionCF, below) recovers it, roughly doubling earned reward at the most severe contention with "
  "no communication.</li>"
  "<li><b>A new, honest surprise: active exploration BACKFIRES under contention.</b> ActiveCFconv's "
  "collective count-bonus is great in the non-contention world (one drone's probe lowers everyone's "
  "uncertainty), but with a SHARED pool it SYNCHRONIZES probes, every drone is drawn to the same "
  "under-counted target, which collides more (collision rate $0.21$ vs RewardCF $0.15$ at pool $240$) "
  "and narrows coverage, hurting both its earned reward and its unseen quality there. Plain "
  "$\\epsilon$-greedy RewardCFconv, whose exploration is de-synchronized by independent randomness, is "
  "the more contention-robust choice. The fix for active exploration under contention is to "
  "de-synchronize it (per-drone jitter or a coordination term), a concrete next step.</li>"
  "</ul>")
A("<h4>Winning under contention: keep the estimate, change the decision (ContentionCF)</h4>"
  "<p>The right lesson is not 'CF loses under contention' but 'argmax is the wrong DECISION rule under "
  "contention'. The CF ESTIMATE of who-suits-what is exactly as good as ever (the categorical result "
  "above); what fails is everyone greedily grabbing their single best target. So we keep the estimator "
  "and swap the policy for a contention-aware, decentralized (still communication-free) one. Three "
  "ideas, in increasing sophistication, all ZK:</p>"
  "<ul>"
  "<li><b>Randomized (softmax) selection.</b> Instead of $\\arg\\max_j \\hat R_{ij}$, sample "
  "$j\\sim\\mathrm{softmax}(\\hat R_{ij}/\\tau)$. Drones with similar tastes still diverge "
  "probabilistically, so collisions drop; $\\tau$ trades raw quality against spread. Pure symmetry "
  "breaking, the standard decentralized anti-coordination tool.</li>"
  "<li><b>Contention-discounted value.</b> From the public broadcast each drone estimates each target's "
  "POPULARITY (how many drones tend to want it) and discounts expected reward by the chance it loses "
  "the collision: $\\tilde R_{ij} = \\hat R_{ij}\\cdot \\Pr(\\text{win } j)$, then "
  "$\\arg\\max_j \\tilde R_{ij}$. This routes drones off crowded high-popularity targets onto their "
  "personal-but-uncontested favorites, where CF's PERSONALIZATION (the $d>1$ part) is precisely the "
  "edge: structure-free learners only know popularity, so they all pile onto the same few targets.</li>"
  "<li><b>De-synchronized exploration.</b> Replace the shared count-bonus (which synchronizes probes) "
  "with a per-drone private bonus / jittered tie-break, so exploration spreads across targets.</li>"
  "</ul>"
  "<div class='box key'><b>Result: we WIN at severe contention.</b> We tested all three decision ideas "
  "(8 seeds, CIs). The honest finding on HOW: only PRIVATE, FIXED randomness works. Random-per-round "
  "softmax BACKFIRES (it re-randomizes every round, so drones re-collide and lose value), and "
  "shared-signal (popularity) routing BACKFIRES (every drone routes off the same crowded target onto "
  "the same empty one, re-synchronizing). The winner is a FIXED private per-target offset "
  "$\\hat R_{ij}+\\varepsilon h_i[j]$ (each drone $i$ has its own fixed $h_i$): same-type drones, who "
  "would all grab their identical $\\arg\\max$, instead consistently prefer DIFFERENT near-tied "
  "targets, de-conflicting stably while clear winners stay greedy. This <code>ContentionCF</code> "
  "earns $0.105\\,[0.096,0.114]$ at the most severe contention (pool $=15$, $2{:}1$) versus $\\approx "
  "0.05$ for argmax-CF and PTF, non-overlapping CIs, roughly DOUBLE, lifting the swarm off the matching "
  "floor; it also leads at pool $=30$.</div>")
A("<div class='box key'><b>Making it self-tuning (ContentionAdaCF, built and tested, 8 seeds).</b> The "
  "fixed offset has one knob set by hand. We made it ADAPTIVE: each drone scales the offset by its own "
  "observed loss rate (lose a lot, spread harder) and gates it by a ZK SCARCITY signal (offer size vs "
  "swarm size), so the offset engages only when targets are actually scarce. This EXTENDS the win from "
  "the single severe point to the whole contested range: ContentionAdaCF beats the fixed offset across "
  "pool $\\le 60$ (earned skill $0.153$ vs $0.134$ at pool$=30$; $0.205$ vs $0.178$ at pool$=60$) and "
  "ties it at the most severe contention (pool$=15$, $0.100$ vs $0.105$), while every no-offset method "
  "sits at $\\le 0.06$ there. <b>And we closed the no-contention gap.</b> Earlier the offset policies "
  "trailed plain CF at pool$=240$ because, like the fixed offset, they used a pure argmax and dropped "
  "the $\\epsilon$-greedy exploration that drives COVERAGE when there is nothing to de-conflict (a hard "
  "scarcity gate confirmed the gap was missing exploration, not the offset). The fix is one line: when "
  "the scarcity gate is OFF, fall back to $\\epsilon$-greedy CF. With it, ContentionAdaCF now earns "
  "$0.448\\,[0.422,0.472]$ at pool$=240$, matching plain CF ($0.439$) with the LOWEST collision rate "
  "($0.126$), and its unseen-pair quality recovers to $0.320$. <b>Bottom line:</b> a SINGLE self-tuning, "
  "scarcity-gated, communication-free policy is now best-or-statistically-tied at EVERY contention level "
  "on earned reward, it matches plain CF when targets are plentiful (pool$=240$: $0.448$ vs $0.439$) and "
  "WINS by $\\approx 2\\times$ at severe contention (pool$=15$: $0.100$ vs $\\le 0.06$ for every "
  "no-offset method, non-overlapping CIs). The contention regime, where we started by losing, is now a "
  "clean win with no communication.</div>")

A("<h3>8.14 Staying adapted under churn (non-stationarity), a negative that became a win</h3>")
A("<p>Real swarms face CHURN: targets come and go over time. We test continuous turnover, a "
  "fixed-size active set where a batch of targets DEPARTS and an equal batch of FRESH ones ARRIVES "
  "every few rounds, and measure steady-state skill on the active set and on the very freshest "
  "arrivals. This is a clean illustration of the research process: a negative, diagnosed, turned into "
  "a win.</p>")
A(md_tables("docs/CHURN.md"))
A("<p><b>The negative (and its diagnosis).</b> Plain exploitative CF (RewardCF) does NOT win under "
  "fast churn: it ties the structure-free UCBIndep on the active set and TRAILS it on the freshest "
  "arrivals ($0.074$ vs $0.132$). Why: a newcomer's factor needs about $d$ collective probes before "
  "fold-in can place it, a latency that rapid churn outpaces, and a purely exploitative policy never "
  "probes the new targets. UCBIndep, by contrast, is drawn to untried arms by its optimism, so it at "
  "least samples the arrivals.</p>")
A(plain("The fix follows directly from the diagnosis: combine CF's structure (fold-in) WITH directed "
        "exploration of the uncertain (fresh) targets, so the swarm both PROBES newcomers and PROPAGATES "
        "what it learns about them to everyone. We already have exactly those methods: ActiveCFconv "
        "(probe the under-observed via a broadcast count-bonus) and EMCF (probe by predictive-variance "
        "UCB). Both DOMINATE under churn, on the active set EMCF reaches $0.842$ vs UCBIndep's $0.619$, "
        "and on the freshest arrivals ActiveCFconv $0.363$ and EMCF $0.371$ vs UCBIndep $0.132$ and plain "
        "CF $0.074$ (all non-overlapping CIs). So non-stationarity IS handled, but only by the variant "
        "that UNITES low-rank fold-in with confidence-directed newcomer-probing: structure-free optimism "
        "(UCBIndep) lacks the propagation, and exploitative CF lacks the probing; you need both."))

A("<h3>8.15 When does collaborative filtering actually help? (the honest scope)</h3>")
A("<p>It would be dishonest to claim CF always wins. It does not. Pinning down EXACTLY when it beats a "
  "plain learner is itself a useful result, and it comes down to three conditions that must ALL hold "
  "at once.</p>")
A("<div class='box key'><b>CF beats a tabular learner if and only if all three hold:</b>"
  "<ol>"
  "<li><b>The reward is low-rank but PERSONALIZED</b> (1 &lt; d &lt;&lt; number of drones/targets). "
  "Think of every reward as 'popularity + personal taste'. The <em>popularity</em> part (some targets "
  "are just good for everyone) is a single shared ranking, and a simple popularity/average model "
  "already nails it, no CF needed. The <em>personal taste</em> part is everything left over: different "
  "drones genuinely prefer different targets. That part only exists when the rank d is bigger than 1. "
  "So if d=1 (pure popularity), CF has nothing extra to offer; if d is huge (no structure at all), "
  "there is nothing to learn; CF wins in the sweet spot in between, where tastes differ but still ride "
  "on a few hidden factors.</li>"
  "<li><b>The setting is SAMPLE-STARVED</b> (far more targets than any drone can try). If instead "
  "every drone could try every target many times, a plain learner would just measure each target "
  "directly and CF's whole trick, guessing about targets you NEVER tried, becomes pointless. This is "
  "why our early experiments in 'sample-rich' settings showed no CF advantage: there were no unseen "
  "targets left to be smart about.</li>"
  "<li><b>The reward channel is SHARED</b> (there is a broadcast to observe). CF learns the hidden "
  "factors from OTHER drones' outcomes. Cut the sharing and each drone is alone with its own data, "
  "which is exactly the tabular case again.</li>"
  "</ol></div>")
A("<p><b>Why this matters.</b> These three conditions are precisely the situation our drone swarm is "
  "in (hidden compatibility with a few factors, way more targets than rounds, a public outcome "
  "stream), which is why CF wins here, and they honestly explain the cases where it would NOT, "
  "pre-empting the reasonable question 'is this always better?' with a clear no-and-here-is-when.</p>")

A("<h3>8.16 The menu composes into ONE method (UnifiedCF capstone)</h3>")
A("<p>Sections 5 and 8.12-8.15 presented a single core estimator plus a MENU of scoped refinements, "
  "each helping only under a stated condition. A fair question: must a practitioner pick the right "
  "extension per situation? No. The extensions COMPOSE into one self-tuning policy, <b>UnifiedCF</b>, "
  "because each one gates itself on a quantity the drone can observe, and switches itself off when its "
  "condition is absent.</p>")
A("<ul>"
  "<li><b>Confidence core (EMCF):</b> predictive-variance exploration + optional ARD rank "
  "self-determination, always on (it is the core).</li>"
  "<li><b>De-confliction offset (8.13):</b> a fixed PRIVATE per-target offset whose strength scales "
  "with the drone's OWN observed loss rate, so it is exactly 0 until the drone actually starts losing "
  "contests, and engages only under real contention (the private-offset result).</li>"
  "<li><b>Exploration gate:</b> the same loss signal damps exploration when the drone is losing "
  "(exploring wastes scarce capacity), and an ABUNDANCE gate also damps it when the offer is plentiful "
  "($|S|>4m$, nothing left to explore-for-coverage), so it exploits when targets are abundant.</li>"
  "</ul>")
A("<p>On the SAME 8 seeds, with ONE configuration and no per-regime tuning, UnifiedCF is "
  "best-or-statistically-tied in every regime-metric we measure:</p>")
A("<table><tr><th class='l'>Regime (metric)</th><th>UnifiedCF</th><th>per-regime specialist</th><th>verdict</th></tr>"
  "<tr><td class='l'>Standard (anytime skill)</td><td>0.437</td><td>EMCF 0.433</td><td>tie</td></tr>"
  "<tr><td class='l'>Churn (active-set skill)</td><td>0.851</td><td>EMCF 0.842</td><td>tie / edge</td></tr>"
  "<tr><td class='l'>Churn (recent arrivals)</td><td>0.347</td><td>EMCF 0.371</td><td>tie</td></tr>"
  "<tr><td class='l'>Contention $|S|{=}15$ (earned)</td><td>0.104</td><td>AdaCF 0.100 / greedy 0.059</td><td>tie / $\\approx2\\times$ greedy</td></tr>"
  "<tr><td class='l'>Contention $|S|{=}240$ (earned)</td><td>0.425</td><td>greedy 0.439 / AdaCF 0.448</td><td>tie (overlapping CIs)</td></tr>"
  "</table>")
A(plain("The last row is the one that took work. Without the abundance gate, UnifiedCF kept exploring "
        "even when there was no contention and plenty of targets, and that exploration cost earned "
        "reward it had no time to recoup, so it scored only $0.344$ there while a plain greedy policy "
        "got $\\sim0.44$. The fix is the abundance gate: when the offer is much larger than the swarm "
        "($|S|>4m$) there is nothing to spread out over, so the method just exploits its best target. "
        "Crucially that gate fires ONLY in that no-contention regime (everywhere else the offer is "
        "small, so the other four rows are byte-for-byte unchanged), which is why we close the one gap "
        "WITHOUT disturbing the four ties."))
A("<p><b>Why it matters.</b> The design space does not merely reduce to 'a core plus a menu', it "
  "collapses to a SINGLE deployable policy that adapts itself to the regime it finds itself in "
  "(plentiful vs contested targets, stationary vs churning, clean vs noisy), with no knob the operator "
  "must set per situation. That is the strongest form of the consolidation argument.</p>")

A("<h3>8.17 Sensing-grounded observability (a robotics check)</h3>")
A(plain("<b>What are the hidden factors, physically?</b> Think of $p_i$ as robot $i$'s CAPABILITY "
        "profile (payload, sensors, speed, endurance, skills) and $u_j$ as task $j$'s REQUIREMENT "
        "profile along the same traits; the reward $\\langle p_i,u_j\\rangle$ is how well that robot "
        "suits that task. Neither is known in advance, the swarm learns them from outcomes. And the "
        "'broadcast' is just LIMITED-RANGE SENSING: a robot notices what a teammate did only when that "
        "engagement happens close enough to see, more fuzzily the farther away. That is where the "
        "masking and noise come from in reality, not a dial we turn."))
A("<p>To make sure our results are not an artifact of an abstract mask, we placed the robots and tasks "
  "in a 2-D arena and DERIVED observability from geometry: a robot senses a teammate's engagement only "
  "if the task is within its sensing radius $R$, with noise growing with distance "
  "($\\sigma(d)=\\sigma_0(1+d/R)$). This is a persistent, structured, per-robot visibility, exactly the "
  "kind a real swarm has, now emerging from physics. Sweeping the sensing radius:</p>")
A("<table><tr><th>sensing radius $R$</th><th>coverage</th><th>RewardCF unseen</th><th>Tabular unseen</th></tr>"
  "<tr><td>0.20</td><td>0.11</td><td>0.030</td><td>~0</td></tr>"
  "<tr><td>0.35</td><td>0.29</td><td>0.147</td><td>~0</td></tr>"
  "<tr><td>0.50</td><td>0.50</td><td>0.229</td><td>~0</td></tr>"
  "<tr><td>0.80</td><td>0.86</td><td>0.304</td><td>~0</td></tr></table>")
A(plain("Read: once a robot can sense more than about a third of the engagements, CF generalizes to "
        "tasks it never tried (unseen skill $0.15$-$0.30$) while a structure-free learner stays pinned "
        "at the floor ($\\approx0$) at EVERY radius. The categorical win is real under physically-grounded, "
        "geometry-limited, distance-noisy sensing, the regime an actual decentralized drone swarm "
        "operates in. It degrades only when sensing is very sparse (~10% coverage), which is the honest "
        "and expected limit: with almost no shared observations there is no shared structure to recover."))

A("<figure>%s<figcaption><strong>F16.</strong> Sensing-grounded observability: unseen-pair skill vs "
  "effective sensing coverage when masking and noise are DERIVED from 2-D sensing geometry (range + "
  "distance-noise). The low-rank CF win survives geometry-limited sensing; structure-free learners stay "
  "at the floor at every coverage.</figcaption></figure>" % img("F16_sensing.png", "F16"))

A("<h3>8.18 An operational mission (what the abstract numbers mean)</h3>")
A(plain("<b>What are the hidden factors, concretely?</b> They are TRAITS. $p_i$ is robot $i$'s "
        "capability/trait vector (sensor modalities, payload, effector type, speed, endurance); $u_j$ is "
        "task $j$'s requirement/trait-demand vector; the reward $\\langle p_i,u_j\\rangle$ is how well the "
        "robot's traits meet the task's needs. It is LOW-RANK because only a handful of traits govern fit "
        "(d ~ a few). This is the standard 'trait-based' / capability-vs-requirement view of multi-robot "
        "task allocation; our twist is that the traits are HIDDEN and learned online, decentralized, with "
        "no communication, instead of being given."))
A("<p>Framed as an operational target-servicing / dispatch MISSION (each robot repeatedly services an "
  "offered target and earns the trait-match, under range-limited distance-noisy sensing), and measured "
  "on our SAME servicing skill, our method beats the WHOLE competing field, the structured low-rank "
  "methods (PTF, ESTR, BPMF, SoftImpute, MFSGD) AND the structure-free ones, under limited observability "
  "(rho=0.25): ours $\\approx0.36$ vs the best of the entire field $\\approx0.29$ (non-overlapping error "
  "bars), while structure-free learners sit at the random-dispatch floor. At full broadcast the "
  "structured methods catch up; the gap opens precisely under the limited observability that defines the "
  "operational regime. So the swarm dispatches the right asset to targets it never personally serviced. "
  "Applications: limited-resource servicing under degraded comms (firefighting, medical-evac, precision "
  "agriculture), search-and-service, and sensor-target matching.</p>")

A("<figure>%s<figcaption><strong>F17.</strong> Operational target-servicing mission: servicing skill "
  "(0=random dispatch, 1=oracle) for OURS (blue) vs the structured low-rank field (orange) vs "
  "structure-free (gray), at full broadcast and under limited observability. The separation over the "
  "whole field opens at $\\rho=0.25$.</figcaption></figure>" % img("F17_mission.png", "F17"))

A("<h3>8.19 Why a swarm at all? Collaboration value and positive scaling</h3>")
A(plain("<b>What is the broadcast actually worth?</b> Turn it off and see. We sweep the broadcast rate "
        "from rho=0 (each drone is ISOLATED, sees only its own outcomes) up to rho=1 (it passively senses "
        "every engagement). At rho=0 a drone holds just ONE row of the reward matrix, its own, and from a "
        "single row you cannot recover a structure shared across drones, so its skill on pairs it never "
        "tried is essentially zero. Switching the broadcast on lifts RewardCF's unseen skill by +0.39. "
        "The structure-free learners gain NOTHING (UCBIndep &minus;0.00, Tabular +0.01): they have no "
        "model linking one task to another, so observations about other targets are useless to them. (The "
        "batch-refit baseline PTF actually gains the most, +0.51, and overtakes us at full broadcast, "
        "because a one-shot refit on a near-fully-observed matrix is the easy case for batch low-rank "
        "completion; in the masked regime that matters operationally, our online method leads. This "
        "crossover near rho about 0.6 is not a fluke: it reproduces across the bake-off and crossover "
        "sweeps with near-identical numbers, and is exactly what matrix-completion theory predicts.) That "
        "single contrast is the cleanest 'why share?' answer we have: communication-free sensing converts "
        "m isolated, near-useless learners into one effective swarm."))
A(plain("<b>Does the swarm get SMARTER as it gets bigger?</b> Yes, and this is the surprising part. "
        "Usually adding agents makes coordination HARDER (more interference, more collisions). Here we fix "
        "the number of targets, the per-drone horizon, and the broadcast rate, and simply grow the swarm "
        "from m=5 to m=80 drones. RewardCF's unseen skill climbs steadily (0.08, 0.22, 0.36, 0.41, 0.43 "
        "as m grows), because more drones pour more broadcast observations (about m&times;T&times;rho of "
        "them) into the ONE shared low-rank structure everyone is estimating, so every drone's prediction "
        "on targets it never serviced improves as teammates are added (the collective-recovery speedup). "
        "The structure-free learner stays FLAT, because m cannot help an agent that only ever learns its "
        "own row. A swarm that gets smarter as it grows is exactly the property you want from a swarm, and "
        "here it is a direct consequence of sharing structure."))
A("<figure>%s<figcaption><strong>F18.</strong> (a) Collaboration value: unseen-pair skill vs broadcast "
  "rate (the left edge rho=0 is isolated, self-only); the comms-free broadcast unlocks generalization "
  "for low-rank CF (RewardCF, PTF) and is worthless to structure-free (UCBIndep, Tabular). (b) Positive "
  "scaling: unseen skill RISES with swarm size m for CF (the swarm gets smarter as it grows), and is "
  "flat for structure-free.</figcaption></figure>" % img("F18_collab_scaling.png", "F18"))
A("<h3>8.20 Operational metrics: what the swarm value looks like in mission terms</h3>")
A(plain("The same runs, restated in the language a mission planner cares about. <b>Time-to-competence</b> "
        "(how fast is the swarm useful?): under limited observability our online CF reaches a quarter of "
        "oracle dispatch in about 35 rounds on every seed, while structure-free learners and the batch "
        "competitors mostly never reach that bar within the mission. <b>Cumulative lost value</b> (how much "
        "mission value is left on the table?): adding up oracle-minus-earned over the whole run, our "
        "methods waste the least. <b>New-asset readiness</b> (how quickly is a NEW drone useful?): a drone "
        "that just joined becomes effective on targets it has never personally serviced after only a "
        "handful of engagements, on the order of the number of hidden traits and independent of how many "
        "targets exist, whereas a structure-free newcomer never becomes ready. <b>Resilience to "
        "attrition</b> (does it survive turnover?): when drones are continually lost and replaced, our "
        "confidence-directed CF keeps the highest skill both on the standing fleet and, especially, on the "
        "freshly arrived drones."))
A(plain("Two more metrics need careful reading rather than a raw number, and once treated correctly both "
        "go our way. <b>Redundancy:</b> raw collision rate is minimized trivially by random dispatch "
        "(spreading out avoids collisions but earns nothing), the same effectiveness-versus-coverage "
        "tension we met in the inspection mission, so we read it as a frontier instead. Among the methods "
        "that actually chase reward, at severe contention our private-offset variants are the top earners "
        "AND the lowest-collision method overall, so they DOMINATE the auction-backoff, musical-chairs, "
        "and batch baselines on the earned-versus-collision frontier, a real coordination win. "
        "<b>Information efficiency</b> splits in two: for OFFLINE estimation (skill per observed entry) all "
        "low-rank methods turn a tiny budget into real skill while structure-free turns it into nothing, "
        "with the batch PTF marginally ahead (the same batch-on-dense-data edge as the full-broadcast "
        "crossover); for ONLINE earning (reward per round while learning) we win, because the batch methods "
        "pay a probe phase, so our cumulative lost value is the lowest."))
A("<figure>%s<figcaption><strong>F19.</strong> Operational metrics (a re-analysis of existing runs, no "
  "new simulation). (a) Time-to-competence: rounds to reach 25%% of oracle dispatch under partial "
  "broadcast; ours reach it fastest, most competitors never do. (b) Cumulative lost mission value "
  "(lower is better). (c) New-asset readiness: a new drone's skill on never-tried targets versus its own "
  "engagements; a structure-free newcomer never becomes ready. (d) Resilience to attrition under "
  "continuous turnover. ESTR is a centralized method (reference, not directly comparable); every other "
  "baseline is decentralized.</figcaption></figure>" % img("F19_operational.png", "F19"))
A("<h2 id='nov'>9. Novelty and honest positioning</h2>")
A("<p><strong>Novelty.</strong> (1) The decentralized, online, broadcast-only, per-drone-masked "
  "formulation of CF for MRTA, with the unseen-pair / onboarding categorical separations and a "
  "matching per-drone theory; standard matrix-completion theory is centralized and about estimation "
  "error, not online decision reward. (2) The anytime (cumulative-reward) separation under sample "
  "starvation. (3) The masking-model dichotomy (durable vs transient decentralization). (4) Collective "
  "active exploration via the shared broadcast.</p>")
A("<p><b>Where we sit among MRTA and decentralized-learning paradigms.</b> The table places the major "
  "families on the four axes that define our problem; each established paradigm relaxes at least one axis "
  "we hold fixed, and our cell (low-rank with only a guessed rank, no communication, decentralized, "
  "masked and noisy) is the one left open.</p>")
A(mp.html_paradigms())
A(plain("<b>Multi-agent-RL view (where this sits).</b> In MARL language this is a cooperative, "
        "decentralized, COMMUNICATION-FREE multi-agent bandit with hidden low-rank structure, not a "
        "sequential Markov game (the reward has no state to transition). That placement is useful: "
        "INDEPENDENT learners (the classic IQL / independent-UCB) are exactly our tabular baseline and "
        "sit at the unseen floor (no shared structure to generalize with); CENTRALIZED-training methods "
        "(MAPPO/QMIX/VDN) and LEARNED-communication methods (CommNet/TarMAC) are not admissible here "
        "because our broadcast is passive sensing, not message-passing; and the legitimate no-comms "
        "matchers (SIC-MMAB / musical chairs) are the de-confliction baselines we beat (8.13). Seen "
        "positively, our method is decentralized MODEL-BASED MARL whose shared 'world model' is the "
        "low-rank reward matrix, learned collectively but estimated privately, with coordination "
        "(exploration division-of-labor, contention de-confliction) EMERGING without any messages. The "
        "thing standard MARL does not give you here is generalization to UNSEEN tasks, which is the "
        "whole point."))
A("<figure>%s<figcaption><strong>F15.</strong> De-confliction under capacity-1 contention: our "
  "proactive private offset beats the no-comms field primitives (CBBA auction-with-backoff, "
  "SIC-MMAB/musical-chairs re-seating) and plain greedy at severe contention, while matching them "
  "when targets are plentiful.</figcaption></figure>" % img("F15_deconfliction.png", "F15"))
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
A("<p>Plain one-line definitions of the recurring terms. For every mathematical symbol, see the "
  "<a href='#notation'>Notation at a glance</a> table at the top.</p>")
A("<dl>"
  "<dt>Collaborative filtering (CF)</dt><dd>predicting your unknowns by borrowing patterns from "
  "others who share hidden structure with you (the 'people who liked X also liked Y' idea).</dd>"
  "<dt>Low rank / latent factors</dt><dd>the reward table $R=PU^\\top$ is generated by a few hidden "
  "dials; compatibility is the inner product of two short vectors (a drone's taste, a target's "
  "profile). 'Latent' = hidden; 'rank' = how many dials.</dd>"
  "<dt>Matrix completion</dt><dd>filling in the unobserved cells of a partly-seen table by assuming "
  "it is low-rank, the statistical backbone of CF.</dd>"
  "<dt>Weighted ALS</dt><dd>alternating least squares: solve for $P$ given $U$, then $U$ given $P$, "
  "back and forth; each observation weighted by how much we trust it; unseen entries get zero weight "
  "(not a fake zero value).</dd>"
  "<dt>EM (expectation-maximization)</dt><dd>a loop that alternates 'guess the hidden thing given "
  "current beliefs' (E-step) and 'update beliefs given that guess' (M-step); used to jointly learn "
  "the factors and how trustworthy each teammate's choices are.</dd>"
  "<dt>Posterior / predictive interval</dt><dd>the model's belief after seeing data, including an "
  "error bar; lets a drone say not just 'this target is good' but 'how sure' it is.</dd>"
  "<dt>Fold-in</dt><dd>place a brand-new drone or target into the hidden space from a few "
  "observations, given the existing factors, no full retraining (the cold-start fix).</dd>"
  "<dt>Unseen pair</dt><dd>a (drone, target) the drone never engaged or observed; a structure-free "
  "learner is stuck at the floor here, CF is not. The headline test of generalization.</dd>"
  "<dt>Skill</dt><dd>$(\\text{method}-\\text{random})/(\\text{oracle}-\\text{random})$: $0$ = "
  "guessing, $1$ = all-knowing oracle. A single 0-to-1 ruler for fair comparison.</dd>"
  "<dt>Anytime / AUC skill</dt><dd>reward actually earned WHILE learning, summed over rounds; charges "
  "a method for any time it wastes just exploring.</dd>"
  "<dt>Masking ($\\rho$)</dt><dd>limited DETECTION: each drone passively senses only a fraction "
  "$\\rho$ of teammates' public outcomes (not radio silence, just partial sight).</dd>"
  "<dt>Zero-knowledge (ZK)</dt><dd>no messages, no shared parameters, no coordinator; drones learn "
  "only from the public stream of who-did-what-and-got-what.</dd>"
  "<dt>Contention</dt><dd>when a target can be taken by only one drone, so look-alike drones collide "
  "and lose reward; fixed by private tie-breaking 'offsets' (Section 8.13).</dd>"
  "<dt>Choice-informativeness ($\\gamma_k$)</dt><dd>how much a teammate's pick reflects real "
  "knowledge versus a random guess; learned from public actions to weight the choice channel.</dd>"
  "</dl>")

A("<p class='small' style='margin-top:30px'>Generated from the project repository; every figure and "
  "number is regenerable from saved data. See the paper draft for the full write-up.</p>")
A("</div>")
A("<script defer src='https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js'></script>")
A("<script defer src='https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js' "
  "onload='renderMathInElement(document.body,{delimiters:[{left:\"$$\",right:\"$$\",display:true},"
  "{left:\"$\",right:\"$\",display:false},{left:\"\\\\(\",right:\"\\\\)\",display:false},"
  "{left:\"\\\\[\",right:\"\\\\]\",display:true}],throwOnError:false})'></script>")
A("</body></html>")

html = "\n".join(H)
open(OUT, "w", encoding="utf-8").write(html)
print("wrote %s (%d KB)" % (OUT, len(html.encode("utf-8")) // 1024))
