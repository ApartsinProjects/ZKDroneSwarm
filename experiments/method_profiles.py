"""SINGLE SOURCE OF TRUTH for how every method/paradigm sits on the axes that
matter, so the paper, tutorial, figures, and docs agree and the comparison reads
honestly. Framing (confirmed): the ZK-MRTA setting is itself new, so this is not
"our method vs external rivals" but a CONTROLLED SWEEP across the low-rank design
space we instantiate in the setting, against the genuinely external structure-free
paradigm and full-information reference ceilings.

Provenance is explicit:
  - ours          : the CF family we built (RewardCF, EMCF, BothCF, contention/unified, ...)
  - ours (hybrid) : PTF -- a strong batch-flavored low-rank method we built from standard parts
  - standard      : named prior-work estimators we ADAPT to the setting (MFSGD, ESTR, BPMF,
                    SoftImpute, KNNCF, CLUB, BiasModel) -- algorithm not ours, application is
  - structure-free: the external paradigm we beat (per-arm UCB, tabular, random)
  - reference     : full-information upper bounds, NOT competitors (Oracle, CTDE ceiling)

Harness fact (clarified): run_masked builds ONE estimator PER robot, each seeing only the
broadcast it observes and acting on its own row -- so EVERY bake-off method is decentralized
and communication-free in-harness. ESTR is spectral/centralized in ORIGIN (Kang-Hsieh-Lee) but
runs here as a per-robot reduction; its distinguishing feature in-harness is explore-then-commit,
not centralization. Genuine centralization/full-info applies only to the Oracle and CTDE ceilings.

Compact badge: [dist | comm | obs | prior | compute].
Three tables (A profiles, B paradigms, C mechanisms) + a performance SCORECARD.
Run as a script to (re)write docs/METHOD_PROFILES.md.
"""
import os, glob, json

DIST = {"D": "decentralized", "C": "centralized"}
OBS = {"full": "full (every engagement, noiseless)", "fullsig": "full but noisy (no masking, noise &sigma;)",
       "rho": "masked (sees a fraction &rho;)",
       "rho,sig": "masked + noisy (fraction &rho;, per-observer noise &sigma;)",
       "self": "self-only (isolated)", "-": "n/a"}
SHORT = {"rho,sig": "&rho;&sigma;", "rho": "&rho;", "full": "full", "fullsig": "full&sigma;", "self": "self", "-": "&ndash;",
         "dhat": "d&#770;", "ard": "ARD", "none": "&ndash;", "d": "d", "U*": "U*", "0": "0", "B": "B",
         "online": "online", "batch": "batch", "ETC": "ETC", "memory": "mem"}
PROV = {"ours": "ours", "ours-hybrid": "ours (hybrid)", "standard": "standard, adapted",
        "structfree": "structure-free baseline", "reference": "reference (upper bound)"}

# Our invented methods are ONE family, SwarmCF, differing only in the menu axes (Table C).
# Single display name per method (no code-name parentheticals): code class names unchanged.
FAMILY = {"RewardCF": "SwarmCF", "ChoiceCF": "SwarmCF-Ch", "BothCF": "SwarmCF-RC",
          "EMCF": "SwarmCF-B", "ActiveCF": "SwarmCF-X", "CoordCF": "SwarmCF-Xc",
          "ContentionCF": "SwarmCF-D", "ContentionAdaCF": "SwarmCF-D+", "UnifiedCF": "SwarmCF-U",
          "HybridCF": "SwarmCF-H", "PTF": "SwarmCF-batch", "ARD-EMCF": "SwarmCF-B-ARD"}

# Clean single display names for the literature / structure-free / reference methods (cite by name).
DISPLAY_EXT = {"MFSGD": "MF-SGD", "UCBIndep": "Independent-UCB", "UCBHomo": "Homogeneous-UCB",
               "CentralClean-ceiling": "Centralized (clean)", "CTDE-ceiling": "Centralized (CTDE)"}


def disp(name):
    """Single display name per method: ours as the SwarmCF family, external methods by their
    (clean) literature name."""
    if name in FAMILY:
        return FAMILY[name]
    return DISPLAY_EXT.get(name, name)

# ---- TABLE A: method operating profiles ----
# name: (provenance, dist, comm, obs, prior, compute, blurb)
PROFILES = [
    ("RewardCF",        "ours",        "D", "0",   "rho,sig", "dhat", "online", "online weighted-ALS on the public broadcast (our core)"),
    ("EMCF",            "ours",        "D", "0",   "rho,sig", "dhat", "online", "variational Bayesian PMF + predictive-variance exploration"),
    ("BothCF",          "ours",        "D", "0",   "rho,sig", "dhat", "online", "fuses reward + competence-weighted choice signal"),
    ("ChoiceCF",        "ours",        "D", "0",   "rho,sig", "dhat", "online", "uses only who-engaged-what (noise-immune channel)"),
    ("ContentionAdaCF", "ours",        "D", "0",   "rho,sig", "dhat", "online", "scarcity-gated private offset for capacity-1 contention"),
    ("UnifiedCF",       "ours",        "D", "0",   "rho,sig", "dhat", "online", "one method; refinements activate only on their condition"),
    ("HybridCF",        "ours",        "D", "0",   "rho,sig", "dhat", "online", "UCB probe then SVD warm-start then online ALS"),
    ("ARD-EMCF",        "ours",        "D", "0",   "rho,sig", "ard",  "online", "EMCF + ARD rank self-determination: LEARNS the rank, removing the guessed d-hat"),
    ("PTF",             "ours-hybrid", "D", "0",   "rho,sig", "dhat", "batch",  "probe-then-fit hybrid WE built: SVD warm-start + finetune (batch refit)"),
    ("MFSGD",           "standard",    "D", "0",   "rho,sig", "dhat", "online", "SGD matrix factorization (standard)"),
    ("KNNCF",           "standard",    "D", "0",   "rho,sig", "none", "memory", "neighborhood (memory) CF (standard)"),
    ("BiasModel",       "standard",    "D", "0",   "rho,sig", "none", "online", "global + row + col bias (popularity, rank&le;2)"),
    ("BPMF",            "standard",    "D", "0",   "rho,sig", "dhat", "batch",  "Bayesian PMF, Salakhutdinov-Mnih (Gibbs, batch)"),
    ("SoftImpute",      "standard",    "D", "0",   "rho,sig", "dhat", "batch",  "nuclear-norm completion, Mazumder et al. (batch convex)"),
    ("CLUB",            "standard",    "D", "0",   "rho,sig", "none", "batch",  "clustering of bandits, Gentile et al. (hard robot clusters)"),
    ("ESTR",            "standard",    "D", "0",   "rho",     "dhat", "ETC",    "explore-then-spectral-commit, Kang-Hsieh-Lee (centralized in ORIGIN; run per-robot)"),
    ("UCBIndep",        "structfree",  "D", "0",   "rho,sig", "none", "online", "per-(robot,target) UCB1; no cross-arm generalization"),
    ("UCBHomo",         "structfree",  "D", "0",   "rho,sig", "none", "online", "shared arm table; assumes robot homogeneity"),
    ("Tabular",         "structfree",  "D", "0",   "rho,sig", "none", "online", "eps-greedy own-row table"),
    ("Random",          "structfree",  "D", "0",   "-",       "none", "-",      "uniform random selection (floor)"),
    ("CentralClean-ceiling", "reference", "C", "full", "full",  "dhat", "online", "centralized low-rank + Hungarian on NOISELESS, unmasked observation (no priors; the clean centralized ceiling)"),
    ("CTDE-ceiling",    "reference",   "C", "full", "fullsig", "dhat", "online", "1 shared model + Hungarian assignment (full comms, but noisy observation)"),
    ("Oracle",          "reference",   "C", "full", "full",   "U*",   "-",      "best-in-offer under the true reward (=1 by definition)"),
]


def badge(name):
    for n, prov, dist, comm, obs, prior, comp, _ in PROFILES:
        if n == name:
            return "[%s | %s | %s | %s | %s]" % (dist, comm, obs, prior, comp)
    return "[?]"


def provenance(name):
    for n, prov, *_ in PROFILES:
        if n == name:
            return prov
    return None


def is_reference(name):
    return provenance(name) == "reference"


# ---- TABLE B: literature paradigms in context ----
PARADIGMS = [
    ("Auction / consensus / DCOP MRTA", "task utilities or costs", "message-passing",
     "decentralized", "full task info", "needs communication and known utilities"),
    ("Cooperative MARL (CTDE)", "none (learned)", "central training or messages",
     "central train, decentral. exec", "full (in training)", "needs a central critic or messages"),
    ("No-communication multiplayer bandits", "none (per-arm)", "none",
     "decentralized", "own pulls + collisions", "structure-free: no unseen generalization"),
    ("Low-rank completion / bandits", "low-rank", "centralized",
     "centralized", "partial (uniform)", "centralized and/or explore-then-commit"),
    ("Federated / gossip CF", "low-rank", "parameter exchange",
     "decentralized", "partial", "shares parameters, not a passive stream"),
    ("Trait-based MRTA", "known traits", "varies",
     "decentralized", "full", "traits given, not learned"),
    ("Ours: communication-free MRTA (this paper)", "low-rank, guessed rank only", "none (passive sensing)",
     "decentralized", "masked + per-observer noisy", "none: the open cell (hardest)"),
]

# ---- TABLE C: our methods by mechanism ----
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
    ("PTF",             "reward",        "probe-then-exploit", "none",           "none",            "fixed d&#770;", "implicit (batch refit)"),
    ("UnifiedCF",       "reward",        "gated collective-UCB", "Bayesian posterior", "gated offset", "fixed d&#770;", "both (conditionally)"),
]


# ============================ renderers ============================
def _legend_html():
    return ("<p class='small'><b>Notation</b> [distribution | communication | observability | prior | "
            "computation]: <b>distribution</b> D=decentralized, C=centralized; <b>communication</b> "
            "0=none (passive sensing), full=message-passing; <b>observability</b> full / &rho;=masked "
            "(fraction &rho; seen) / &rho;&sigma;=masked and per-observer-noisy / self-only; <b>prior</b> "
            "&ndash;=none, d&#770;=guessed rank, d=true rank, U*=oracle factors; <b>computation</b> "
            "online / batch / ETC=explore-then-commit. In our harness every method runs decentralized and "
            "communication-free (one estimator per robot on the passive broadcast); the low-rank methods "
            "differ only in the update rule. <b>SwarmCF</b> (our online method) and its batch variant "
            "<b>SwarmCF-batch</b> are ours; the rest are standard low-rank estimators we adapt, the "
            "structure-free paradigm, or reference ceilings.</p>")


def html_profiles(subset=None):
    rows = ["<table><tr><th class='l'>Method</th><th class='l'>Provenance</th><th>Dist</th>"
            "<th>Comm</th><th>Observability</th><th>Prior</th><th>Compute</th>"
            "<th class='l'>Profile</th></tr>"]
    for n, prov, dist, comm, obs, prior, comp, blurb in PROFILES:
        if subset is not None and n not in subset:
            continue
        nm = "<b>%s</b>" % disp(n) if prov.startswith("ours") else disp(n)
        rows.append("<tr><td class='l'>%s</td><td class='l'>%s</td><td>%s</td><td>%s</td><td>%s</td>"
                    "<td>%s</td><td>%s</td><td class='l'><code>%s</code></td></tr>"
                    % (nm, PROV[prov], DIST[dist], SHORT[comm], SHORT[obs], SHORT[prior], SHORT[comp],
                       "%s|%s|%s|%s|%s" % (dist, SHORT[comm], SHORT[obs], SHORT[prior], SHORT[comp])))
    rows.append("</table>")
    return _legend_html() + "\n" + "\n".join(rows)


def html_paradigms():
    rows = ["<table><tr><th class='l'>Paradigm</th><th class='l'>Prior knowledge</th>"
            "<th class='l'>Communication</th><th class='l'>Decisions</th>"
            "<th class='l'>Observation</th><th class='l'>Constraint it relaxes vs. ours</th></tr>"]
    for para, prior, comm, dist, obs, note in PARADIGMS:
        ours = para.startswith("Ours")
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
                    "<td>%s</td><td>%s</td><td>%s</td></tr>" % (disp(n), ch, ex, cf, ct, rk, co))
    rows.append("</table>")
    return "\n".join(rows)


# ============================ performance scorecard ============================
def _latest(root, pat):
    fs = sorted(glob.glob(os.path.join(root, pat)))
    return fs[-1] if fs else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


# scorecard method order (grouped by provenance), restricted to the masked-harness family
SCORE_ORDER = ["RewardCF", "PTF", "MFSGD", "BPMF", "ESTR",
               "UCBIndep", "Tabular", "Random"]


def scorecard_rows(root):
    """Pull headline metrics from the canonical masked harness (bake-off c14) + operational
    metrics (anytime). Returns list of dicts; methods absent in the data are skipped."""
    out = []
    fc = _latest(root, "results/pilots/c14_compare_*.json")
    fo = _latest(root, "results/pilots/opmetrics_*.json")
    c14 = json.load(open(fc))["raw"] if fc else {}
    op = json.load(open(fo))["raw"]["anytime"] if fo else {}
    for nm in SCORE_ORDER:
        u025 = _mean(c14.get("0.25", {}).get(nm, {}).get("unseen", [])) if c14 else None
        u100 = _mean(c14.get("1.0", {}).get(nm, {}).get("unseen", [])) if c14 else None
        reg = op.get("regret", {}).get("0.25", {}).get(nm, {}).get("mean") if op else None
        ttc = op.get("ttc", {}).get("0.25", {}).get(nm, {}).get("0.25") if op else None
        if u025 is None and reg is None:
            continue
        ttc_s = "never"
        if ttc and ttc.get("rounds_mean") and ttc.get("frac_reached", 0) >= 0.5:
            ttc_s = "%.0f" % ttc["rounds_mean"]
        out.append({"name": nm, "prov": provenance(nm), "badge": badge(nm),
                    "unseen025": u025, "unseen100": u100, "regret": reg, "ttc": ttc_s})
    return out


def html_scorecard(root):
    rows = ["<table><tr><th class='l'>Method</th><th class='l'>Provenance</th>"
            "<th>unseen skill<br>(&rho;=0.25, masked)</th><th>unseen skill<br>(&rho;=1, full)</th>"
            "<th>cumulative regret<br>(&rho;=0.25, lower=better)</th>"
            "<th>rounds to 25%<br>of oracle</th></tr>"]
    for r in scorecard_rows(root):
        nm = "<b>%s</b>" % disp(r["name"]) if r["prov"].startswith("ours") else disp(r["name"])
        f = lambda v, p="%.3f": (p % v) if v is not None else "&ndash;"
        rows.append("<tr><td class='l'>%s</td><td class='l'>%s</td><td>%s</td><td>%s</td><td>%s</td>"
                    "<td>%s</td></tr>"
                    % (nm, PROV[r["prov"]], f(r["unseen025"]), f(r["unseen100"]),
                       f(r["regret"], "%.1f"), r["ttc"]))
    rows.append("</table>")
    note = ("<p class='small'>One canonical masked harness (m=30, n=240, 8 seeds): unseen skill from the "
            "bake-off, regret and time-to-competence from the anytime trajectories. SwarmCF leads the "
            "masked column and the operational columns; the batch variant SwarmCF-batch wins only the "
            "full-broadcast column; structure-free learners sit at the floor.</p>")
    return "\n".join(rows) + "\n" + note


def _md_row(cells):
    return "| " + " | ".join(cells) + " |"


def md_all(root=None):
    L = ["# Method operating profiles, paradigms, and performance scorecard\n",
         "Single source of truth (experiments/method_profiles.py). The ZK-MRTA setting is itself new, so "
         "this is a CONTROLLED SWEEP across the low-rank design space we instantiate in the setting, against "
         "the external structure-free paradigm and full-information reference ceilings, not 'our method vs "
         "rivals'. Provenance is explicit: PTF and the CF family are ours; MFSGD/ESTR/BPMF/SoftImpute/KNNCF/"
         "CLUB are standard estimators we adapt (cited); UCB/Tabular/Random are the structure-free paradigm; "
         "Oracle/CTDE are upper bounds.\n",
         "**Harness fact:** run_masked builds ONE estimator per robot, so every bake-off method is "
         "decentralized and communication-free in-harness; ESTR is spectral/centralized only in ORIGIN and "
         "runs here as a per-robot explore-then-commit reduction. Genuine full information applies only to "
         "Oracle and the CTDE ceiling.\n",
         "**Notation** `[dist | comm | obs | prior | compute]`.\n",
         "## A. Method operating profiles\n",
         _md_row(["Method", "Provenance", "Dist", "Comm", "Observability", "Prior", "Compute", "Profile"]),
         _md_row(["---"] * 8)]
    for n, prov, dist, comm, obs, prior, comp, blurb in PROFILES:
        nm = "**%s**" % disp(n) if prov.startswith("ours") else disp(n)
        prof = "%s|%s|%s|%s|%s" % (dist, comm, obs, prior, comp)
        L.append(_md_row([nm, PROV[prov], dist, comm, obs, prior, comp, "`%s`" % prof]))
    L.append("\n## B. MRTA / decentralized-learning paradigms in context\n")
    L.append(_md_row(["Paradigm", "Prior knowledge", "Communication", "Distribution", "Observability", "Note"]))
    L.append(_md_row(["---"] * 6))
    for para, prior, comm, dist, obs, note in PARADIGMS:
        nm = "**%s**" % para if para.startswith("Ours") else para
        L.append(_md_row([nm, prior, comm, dist, obs, note]))
    L.append("\n## C. Our methods by mechanism\n")
    L.append(_md_row(["Method", "Signal channel", "Exploration", "Confidence", "Contention", "Rank", "Coordination"]))
    L.append(_md_row(["---"] * 7))
    for n, ch, ex, cf, ct, rk, co in MECHANISMS:
        L.append(_md_row(["**%s**" % disp(n), ch, ex, cf, ct, rk.replace("&#770;", "-hat"), co]))
    if root:
        L.append("\n## D. Performance scorecard (one canonical masked harness)\n")
        L.append(_md_row(["Method", "Provenance", "unseen@rho=0.25", "unseen@rho=1.0", "regret@0.25", "rounds-to-25%-oracle", "profile"]))
        L.append(_md_row(["---"] * 7))
        for r in scorecard_rows(root):
            nm = "**%s**" % disp(r["name"]) if r["prov"].startswith("ours") else disp(r["name"])
            f = lambda v, p="%.3f": (p % v) if v is not None else "-"
            L.append(_md_row([nm, PROV[r["prov"]], f(r["unseen025"]), f(r["unseen100"]),
                              f(r["regret"], "%.1f"), r["ttc"], "`%s`" % r["badge"].strip("[]").replace(" | ", "|")]))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    open(os.path.join(root, "docs", "METHOD_PROFILES.md"), "w", encoding="utf-8").write(md_all(root))
    print("wrote docs/METHOD_PROFILES.md")
    print("\nscorecard:")
    for r in scorecard_rows(root):
        print("  %-10s %-16s u025=%s u100=%s reg=%s ttc=%s" % (
            r["name"], r["prov"], r["unseen025"], r["unseen100"], r["regret"], r["ttc"]))
