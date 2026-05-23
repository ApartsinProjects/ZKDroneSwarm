"""SINGLE SOURCE OF TRUTH for how every method/paradigm sits on the axes that
matter for this problem, so the paper, tutorial, figures, and docs all agree and
our wins read honestly (who gets MORE power is explicit).

Three tables:
  A. METHOD OPERATING PROFILES  -- every method we run, on:
        distribution (decentralized / centralized),
        communication (none / broadcast-params / full),
        observability (full / masked rho / masked+noisy / self-only),
        prior knowledge (none / low-rank with guessed rank / true rank / oracle factors),
        computation (online / batch-refit / explore-then-commit / memory).
     Our flagship sits in the HARDEST cell: decentralized, no comms, masked+noisy,
     guessed-rank, online. Anything that beats us somewhere relaxes one of these.
  B. MRTA / DECENTRALIZED-LEARNING PARADIGMS IN CONTEXT -- the literature families
     on the same four axes, locating our (unfilled) cell.
  C. OUR METHODS BY MECHANISM -- signal channel, exploration, confidence, contention,
     rank, coordination type (implicit vs explicit-no-comms).

Compact per-method badge (the notation):
    [dist | comm | obs | prior | compute]
  e.g. RewardCF (ours) = [D | 0 | rho,sig | d-hat | online]
       ESTR            = [C | full | rho | d-hat | ETC]   (centralized reference)
       Oracle          = [C | full | full | U* | -]       (reference upper bound)

Renderers: html_*( ) for the HTML paper/tutorial, md_*( ) for docs/METHOD_PROFILES.md.
Run as a script to (re)write docs/METHOD_PROFILES.md.
"""
import os

# ---- axis vocabularies (long form for tables; short form for the compact badge) ----
DIST = {"D": "decentralized", "C": "centralized"}
COMM = {"0": "none (passive sensing of a public outcome stream)",
        "B": "broadcast of parameters / gradients", "full": "full message-passing / data sharing"}
OBS = {"full": "full (every engagement, noiseless)", "rho": "masked (sees a fraction &rho;)",
       "rho,sig": "masked + noisy (fraction &rho;, per-observer noise &sigma;)",
       "self": "self-only (isolated)", "-": "n/a"}
PRIOR = {"none": "none (structure-free)", "dhat": "low-rank, guessed rank d&#770;",
         "d": "low-rank, true rank d", "U*": "true factors U* (oracle)"}
COMPUTE = {"online": "online (anytime)", "batch": "batch refit", "ETC": "explore-then-commit",
           "memory": "memory-based", "-": "n/a"}
SHORT = {"rho,sig": "&rho;&sigma;", "rho": "&rho;", "full": "full", "self": "self", "-": "&ndash;",
         "dhat": "d&#770;", "none": "&ndash;", "d": "d", "U*": "U*", "0": "0", "B": "B",
         "online": "online", "batch": "batch", "ETC": "ETC", "memory": "mem"}

KLASS = {"ours": "ours", "online-lr": "low-rank (online)", "batch-lr": "low-rank (batch)",
         "struct-free": "structure-free", "central-ref": "centralized reference",
         "ceiling": "reference ceiling"}

# ---- TABLE A: method operating profiles ----
# name: (klass, dist, comm, obs, prior, compute, one-line blurb)
PROFILES = [
    ("RewardCF",        "ours",       "D", "0",   "rho,sig", "dhat", "online", "online weighted-ALS on the public broadcast (our core)"),
    ("EMCF",            "ours",       "D", "0",   "rho,sig", "dhat", "online", "variational Bayesian PMF + predictive-variance exploration"),
    ("BothCF",          "ours",       "D", "0",   "rho,sig", "dhat", "online", "fuses reward + competence-weighted choice signal"),
    ("ChoiceCF",        "ours",       "D", "0",   "rho,sig", "dhat", "online", "uses only who-engaged-what (noise-immune channel)"),
    ("ContentionAdaCF", "ours",       "D", "0",   "rho,sig", "dhat", "online", "scarcity-gated private offset for capacity-1 contention"),
    ("UnifiedCF",       "ours",       "D", "0",   "rho,sig", "dhat", "online", "one method; refinements activate only on their condition"),
    ("HybridCF",        "ours",       "D", "0",   "rho,sig", "dhat", "online", "UCB probe then SVD warm-start then online ALS"),
    ("MFSGD",           "online-lr",  "D", "0",   "rho,sig", "dhat", "online", "SGD matrix factorization"),
    ("KNNCF",           "online-lr",  "D", "0",   "rho,sig", "none", "memory", "neighborhood (memory) collaborative filtering"),
    ("BiasModel",       "online-lr",  "D", "0",   "rho,sig", "none", "online", "global + row + col bias (popularity, rank&le;2)"),
    ("PTF",             "batch-lr",   "D", "0",   "rho,sig", "dhat", "batch",  "probe-then-fit: SVD warm-start + finetune (batch refit)"),
    ("BPMF",            "batch-lr",   "D", "0",   "rho,sig", "dhat", "batch",  "Bayesian PMF (Gibbs sampling, batch)"),
    ("SoftImpute",      "batch-lr",   "D", "0",   "rho,sig", "dhat", "batch",  "nuclear-norm soft-threshold completion (batch convex)"),
    ("CLUB",            "batch-lr",   "D", "0",   "rho,sig", "none", "batch",  "clustering of bandits (hard drone clusters)"),
    ("ESTR",            "central-ref", "C", "full", "rho",    "dhat", "ETC",    "centralized explore-then-spectral-commit (reference)"),
    ("UCBIndep",        "struct-free", "D", "0",   "rho,sig", "none", "online", "per-(drone,target) UCB1; no cross-arm generalization"),
    ("UCBHomo",         "struct-free", "D", "0",   "rho,sig", "none", "online", "shared arm table; assumes drone homogeneity"),
    ("Tabular",         "struct-free", "D", "0",   "rho,sig", "none", "online", "eps-greedy own-row table"),
    ("Random",          "struct-free", "D", "0",   "-",       "none", "-",      "uniform random selection (floor)"),
    ("CTDE-ceiling",    "ceiling",    "C", "full", "full",    "dhat", "batch",  "1 shared model + Hungarian assignment (full-comms ceiling)"),
    ("Oracle",          "ceiling",    "C", "full", "full",    "U*",   "-",      "best-in-offer under the true reward (=1 by definition)"),
]


def badge(name):
    """Compact operating-profile badge for a method, e.g. '[D | 0 | rho,sig | dhat | online]'."""
    for n, k, dist, comm, obs, prior, comp, _ in PROFILES:
        if n == name:
            return "[%s | %s | %s | %s | %s]" % (dist, comm, obs, prior, comp)
    return "[?]"


def klass(name):
    for n, k, *_ in PROFILES:
        if n == name:
            return k
    return None


def is_reference(name):
    """True for methods that are NOT directly comparable (centralized / oracle ceilings)."""
    return klass(name) in ("central-ref", "ceiling")


# ---- TABLE B: literature paradigms in context ----
# paradigm: (prior, communication, distribution, observability, note)
PARADIGMS = [
    ("Auction / CBBA (consensus bundle)", "known task values/costs", "message-passing (bids)",
     "decentralized", "full task info", "needs communication AND known utilities"),
    ("DCOP / consensus MRTA", "known constraints/utilities", "message-passing",
     "decentralized", "full", "needs communication AND a known objective"),
    ("Cooperative MARL (CTDE: MAPPO/QMIX/VDN)", "none (learned)", "centralized training",
     "centralized-train / decentralized-exec", "full (in training)", "central critic; not comms-free"),
    ("Learned-communication MARL (CommNet/TarMAC)", "none (learned)", "learned messages",
     "decentralized", "partial + messages", "broadcast is learned message-passing, not passive"),
    ("No-comms multiplayer bandits (SIC-MMAB, musical chairs)", "none (per-arm)", "none",
     "decentralized", "own pulls + collisions", "comms-free but STRUCTURE-FREE (no unseen generalization)"),
    ("Matrix completion (nuclear norm, spectral)", "low-rank", "centralized (one matrix)",
     "centralized", "partial (uniform)", "centralized estimation, not online/decision"),
    ("Low-rank / bilinear bandits (ESTR, etc.)", "low-rank", "centralized",
     "centralized", "partial", "centralized and/or explore-then-commit"),
    ("Federated / gossip CF", "low-rank", "broadcast of factors/gradients",
     "decentralized", "partial", "shares PARAMETERS, not a passive stream"),
    ("Multi-user RL, low-rank rewards (Nagaraj-Agarwal)", "low-rank", "centralized aggregation",
     "centralized", "partial", "closest prior; centralizes trajectory aggregation"),
    ("Trait-based MRTA (Prorok et al.)", "KNOWN traits", "varies",
     "decentralized", "full", "capability/requirement traits are GIVEN, not learned"),
    ("OURS (broadcast CF for ZK-MRTA)", "low-rank, GUESSED rank only", "none (passive sensing)",
     "decentralized", "masked + noisy", "the hardest cell: no comms, no known utilities/traits, guessed rank"),
]

# ---- TABLE C: our methods by mechanism ----
# name: (signal channel, exploration, confidence, contention, rank, coordination)
MECHANISMS = [
    ("RewardCF",        "reward",        "eps-greedy",        "none",            "none",            "fixed d&#770;", "implicit"),
    ("ChoiceCF",        "choice",        "eps-greedy",        "none",            "none",            "fixed d&#770;", "implicit"),
    ("BothCF",          "reward+choice", "eps-greedy",        "competence-weight", "none",          "fixed d&#770;", "implicit"),
    ("EMCF",            "reward",        "collective-UCB",    "Bayesian posterior", "none",         "fixed d&#770;", "implicit"),
    ("ActiveCF",        "reward",        "collective-UCB",    "Bayesian posterior", "none",         "fixed d&#770;", "explicit (exploration division)"),
    ("CoordCF",         "reward",        "neg-correlated-UCB", "Bayesian posterior", "none",        "fixed d&#770;", "explicit (no-comms division of labor)"),
    ("ContentionCF",    "reward",        "eps-greedy",        "none",            "fixed private offset", "fixed d&#770;", "explicit de-confliction (no comms)"),
    ("ContentionAdaCF", "reward",        "eps-greedy",        "none",            "scarcity-gated offset", "fixed d&#770;", "explicit de-confliction (no comms)"),
    ("ARD-EMCF",        "reward",        "collective-UCB",    "Bayesian posterior", "none",         "ARD (self-tuned)", "implicit"),
    ("HybridCF",        "reward",        "probe-then-exploit", "none",           "none",            "fixed d&#770;", "implicit"),
    ("UnifiedCF",       "reward",        "gated collective-UCB", "Bayesian posterior", "gated offset", "fixed d&#770;", "both (conditionally)"),
]


# ============================ renderers ============================
def _legend_html():
    return ("<p class='small'><b>Notation</b> [dist | comm | obs | prior | compute]: "
            "<b>dist</b> D=decentralized, C=centralized; <b>comm</b> 0=none (passive), B=broadcast params, "
            "full=message-passing; <b>obs</b> full / &rho;=masked / &rho;&sigma;=masked+noisy / self; "
            "<b>prior</b> &ndash;=none, d&#770;=guessed rank, d=true rank, U*=oracle factors; "
            "<b>compute</b> online / batch / ETC=explore-then-commit / mem. Our flagship sits in the "
            "hardest cell [D | 0 | &rho;&sigma; | d&#770; | online]; methods that beat us somewhere relax one axis.</p>")


def html_profiles():
    rows = ["<table><tr><th class='l'>Method</th><th class='l'>Class</th><th>Distribution</th>"
            "<th>Communication</th><th>Observability</th><th>Prior</th><th>Compute</th>"
            "<th class='l'>Profile</th></tr>"]
    for n, k, dist, comm, obs, prior, comp, blurb in PROFILES:
        nm = "<b>%s</b>" % n if k == "ours" else n
        rows.append("<tr><td class='l'>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
                    "<td>%s</td><td>%s</td><td class='l'><code>%s</code></td></tr>"
                    % (nm, KLASS[k], DIST[dist], SHORT[comm], SHORT[obs], SHORT[prior], SHORT[comp],
                       "%s|%s|%s|%s|%s" % (dist, SHORT[comm], SHORT[obs], SHORT[prior], SHORT[comp])))
    rows.append("</table>")
    return _legend_html() + "\n" + "\n".join(rows)


def html_paradigms():
    rows = ["<table><tr><th class='l'>Paradigm</th><th class='l'>Prior knowledge</th>"
            "<th class='l'>Communication</th><th class='l'>Distribution</th>"
            "<th class='l'>Observability</th><th class='l'>Note</th></tr>"]
    for para, prior, comm, dist, obs, note in PARADIGMS:
        ours = para.startswith("OURS")
        cell = ("<b>%s</b>" % para) if ours else para
        tr = "<tr style='background:#eef6ff'>" if ours else "<tr>"
        rows.append(tr + "<td class='l'>%s</td><td class='l'>%s</td><td class='l'>%s</td>"
                    "<td class='l'>%s</td><td class='l'>%s</td><td class='l'>%s</td></tr>"
                    % (cell, prior, comm, dist, obs, note))
    rows.append("</table>")
    return "\n".join(rows)


def html_mechanisms():
    rows = ["<table><tr><th class='l'>Method</th><th>Signal channel</th><th>Exploration</th>"
            "<th>Confidence</th><th>Contention</th><th>Rank</th><th>Coordination</th></tr>"]
    for n, ch, ex, cf, ct, rk, co in MECHANISMS:
        rows.append("<tr><td class='l'><b>%s</b></td><td>%s</td><td>%s</td><td>%s</td>"
                    "<td>%s</td><td>%s</td><td>%s</td></tr>" % (n, ch, ex, cf, ct, rk, co))
    rows.append("</table>")
    return "\n".join(rows)


def _md_row(cells):
    return "| " + " | ".join(cells) + " |"


def md_all():
    L = ["# Method operating profiles and comparison tables\n",
         "Single source of truth (experiments/method_profiles.py) for how every method and paradigm "
         "sits on the axes that matter, so our wins read honestly: who gets more power is explicit.\n",
         "**Notation** `[dist | comm | obs | prior | compute]`: dist D=decentralized / C=centralized; "
         "comm 0=none(passive) / B=broadcast params / full=message-passing; obs full / rho=masked / "
         "rho,sig=masked+noisy / self; prior none / d-hat=guessed rank / d=true rank / U*=oracle factors; "
         "compute online / batch / ETC=explore-then-commit / mem. Our flagship is the hardest cell "
         "`[D | 0 | rho,sig | d-hat | online]`.\n",
         "## A. Method operating profiles\n",
         _md_row(["Method", "Class", "Dist", "Comm", "Observability", "Prior", "Compute", "Profile"]),
         _md_row(["---"] * 8)]
    for n, k, dist, comm, obs, prior, comp, blurb in PROFILES:
        nm = "**%s**" % n if k == "ours" else n
        prof = "%s|%s|%s|%s|%s" % (dist, comm, obs, prior, comp)
        L.append(_md_row([nm, KLASS[k], dist, comm, obs, prior, comp, "`%s`" % prof]))
    L.append("\n## B. MRTA / decentralized-learning paradigms in context\n")
    L.append(_md_row(["Paradigm", "Prior knowledge", "Communication", "Distribution", "Observability", "Note"]))
    L.append(_md_row(["---"] * 6))
    for para, prior, comm, dist, obs, note in PARADIGMS:
        nm = "**%s**" % para if para.startswith("OURS") else para
        L.append(_md_row([nm, prior, comm, dist, obs, note]))
    L.append("\n## C. Our methods by mechanism\n")
    L.append(_md_row(["Method", "Signal channel", "Exploration", "Confidence", "Contention", "Rank", "Coordination"]))
    L.append(_md_row(["---"] * 7))
    for n, ch, ex, cf, ct, rk, co in MECHANISMS:
        L.append(_md_row(["**%s**" % n, ch, ex, cf, ct, rk.replace("&#770;", "-hat"), co]))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "docs", "METHOD_PROFILES.md")
    open(out, "w", encoding="utf-8").write(md_all())
    print("wrote", out)
    print("\nsample badges:")
    for m in ("RewardCF", "PTF", "ESTR", "UCBIndep", "Oracle"):
        print("  %-14s %s" % (m, badge(m)))
