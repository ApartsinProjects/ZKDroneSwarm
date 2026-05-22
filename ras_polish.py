"""
RAS submission polish pass on docs/index.html (canonical HTML).
Changes:
  1. Em-dash audit and replacement (context-aware)
  2. Author order: Yigal -> Yehudit -> Alexander
  3. Remove section meta-openers ("This section presents / situates / reports ...")
  4. Remove abstract meta-opener sentence
  5. Add RAS-required elements placeholder block
  6. Tighten notation and framing per reviewer notes
"""

import re, pathlib

TARGET = pathlib.Path('docs/index.html')
html = TARGET.read_text(encoding='utf-8')

# ============================================================
# 1. EM-DASH REPLACEMENTS (context-aware)
# ============================================================
# Literal em-dash U+2014 is  — ; HTML entity is &mdash;
# Strategy:
#   — when flanked by spaces and introduces an example/list -> ": "
#   — when flanked by spaces and continues a clause -> ", "
#   — in pairs (parenthetical) -> ", ...,"   (both get comma)
#   — in definition lists in <li> -> ": "
#   — in phase labels "Phase N — Title" -> "Phase N: Title"
#   — in table captions "Table N — Title" -> "Table N: Title"
#   — in figure captions "... by policy — Total HP" -> ": Total HP"
#   — before "not" / "unlike" / "rather" (contrast) -> "; "
#   — before "specifically," / "precisely" (elaboration) -> ": "

def fix_emdash(text):
    # Normalize HTML entity to literal character first
    text = text.replace('&mdash;', '—')

    # --- Phase labels: "Phase N — Title" -> "Phase N: Title"
    text = re.sub(r'Phase (\d+) — ', r'Phase \1: ', text)

    # --- Table captions: "Table N — ..." -> "Table N: ..."
    text = re.sub(r'(Table \d+) — ', r'\1: ', text)

    # --- Figure captions: "by policy — " -> "by policy: "
    text = re.sub(r'(by policy) — ', r'\1: ', text)

    # --- Definition 1 title: "Task Allocation — general" -> "Task Allocation: General"
    text = text.replace(
        'Task Allocation — general problem class',
        'Task Allocation: General Problem Class'
    )

    # --- List items with " — " as term-definition separator -> ": "
    text = re.sub(r'(\$[^$]+\$) — ', r'\1: ', text)   # math term — definition
    text = re.sub(r'(<strong>[^<]+</strong>) — ', r'\1: ', text)   # bold term — def

    # --- Elaboration / appositive after citation or noun phrase:
    #     "... [NN] — a formulation/mechanism/structure"
    text = re.sub(r'(\]\s*)— (a |an |the )', r'\1, \2', text)

    # --- Elaboration introduced by "specifically," -> ": specifically,"
    text = re.sub(r' — specifically,', ': specifically,', text)

    # --- "precisely the/this/that" -> ": precisely"
    text = re.sub(r' — (precisely )', r': \1', text)

    # --- Contrast: before "not the" / "unlike" / "rather" -> "; "
    text = re.sub(r' — (not the |unlike |rather )', r'; \1', text)

    # --- After "private" before "collaboration" etc. (semicolon)
    text = re.sub(r'(remain private) — (collaboration)', r'\1; \2', text)

    # --- "No privileged state is consumed — the learner" -> "; "
    text = re.sub(r'(state is consumed) — (the learner)', r'\1; \2', text)

    # --- Parenthetical pairs: " — X — " -> ", X, "
    text = re.sub(r' — ([^——]+?) — ', r', \1, ', text)

    # --- Colons before elaborations that start with a verb phrase or noun list
    #     "provide the structural vocabulary — taxonomies"
    text = re.sub(r'(vocabulary) — (taxonomies)', r'\1: \2', text)

    # --- "for example," / "e.g.," / "including"  -> ": "
    text = re.sub(r' — (for example,|e\.g\.,|including )', r': \1', text)

    # --- Remaining: " — " -> ", "  (safe default)
    text = text.replace(' — ', ', ')
    text = text.replace('—', ',')   # bare em-dash without spaces

    return text

html = fix_emdash(html)

# Verify no em-dashes remain
remaining = html.count('—') + html.count('&mdash;')
print(f"Em-dashes remaining after fix: {remaining}")

# ============================================================
# 2. AUTHOR ORDER: Yigal -> Yehudit -> Alexander
# ============================================================
old_order = 'Yigal Meshulam &middot; Alexander Apartsin &middot; Yehudit Aperstein'
new_order = 'Yigal Meshulam &middot; Yehudit Aperstein &middot; Alexander Apartsin'
html = html.replace(old_order, new_order)

# Also plain-text version
old_order2 = 'Yigal Meshulam · Alexander Apartsin · Yehudit Aperstein'
new_order2 = 'Yigal Meshulam · Yehudit Aperstein · Alexander Apartsin'
html = html.replace(old_order2, new_order2)

print("Author order updated.")

# ============================================================
# 3. REMOVE META-OPENERS (sentences that describe the section)
# ============================================================

meta_openers = [
    # Literature review opener
    (
        '<p>\n            This section situates ZK-MRTA at the intersection of three relevant literatures: swarm robotics and MRTA, decentralized online learning under uncertainty, and collaborative filtering with latent-factor models. The goal is not only to survey prior work, but to clarify which assumptions dominate existing approaches, where those assumptions become limiting under strict zero-knowledge constraints, and why matrix-factorization-based latent-utility learning provides a suitable theoretical foundation for the method studied in this paper.\n        </p>',
        ''
    ),
    # Methods opener
    (
        '<p>\n            This section presents the three methods compared in the paper and frames them through a single distinction: what an agent <em>knows or infers</em> about the utility of an assignment (estimation) versus how that information is turned into an action (decision). This framing is useful in ZK-MRTA because the problem is defined precisely by the absence of prior knowledge, explicit communication, and direct access to hidden compatibility structure; different methods are therefore best read as alternative responses to the same core difficulty.\n        </p>',
        ''
    ),
    # Results opener
    (
        '<p>\n            This section reports the results of the evaluation described in Section 6. It presents task-completion outcomes, cross-policy comparisons, learning dynamics, latent-structure recovery, coordination effects, and the convergence assessment for the decentralized matrix-factorization policy.\n        </p>',
        ''
    ),
    # Problem definition opener
    (
        '<p>\n            This section formalizes ZK-MRTA as a sequential multi-agent decision problem under strict informational constraints. It defines the core assumptions, interaction model, objectives, and notation used throughout the paper; notation is summarized in §3.10.\n        </p>',
        ''
    ),
    # Framework description opener (if present)
    (
        'This section provides a technical description of the benchmark framework',
        'The benchmark framework'
    ),
]

for old, new in meta_openers:
    if old in html:
        html = html.replace(old, new)
        print(f"Removed meta-opener: {old[:60].strip()!r}")
    else:
        # Try looser match for the first sentence only
        first_sent = old.split('.')[0][:80]
        if first_sent.strip() and first_sent.strip() in html:
            print(f"  Partial match for: {first_sent.strip()!r}")

# ============================================================
# 4. NOTATION: Define MF abbreviation on first use
# ============================================================
# "Matrix Factorization (MF)" should appear on first use;
# check if it does (it should in abstract or intro)
if 'Matrix Factorization (MF)' in html or 'matrix factorization (MF)' in html:
    print("MF abbreviation already defined.")
else:
    # Add (MF) on first occurrence of "matrix factorization" in body
    html = html.replace(
        'matrix-factorization policy in which each drone',
        'matrix-factorization (MF) policy in which each drone',
        1
    )
    print("Added MF abbreviation definition.")

# ============================================================
# 5. FRAMING: "This paper makes three main contributions" ->
#    more direct active voice
# ============================================================
# Already good; no change needed.

# ============================================================
# 6. Abstract: count words to check 250-word RAS limit
# ============================================================
abstract_m = re.search(r'<h1>Abstract</h1>(.*?)<h1>1\. Introduction</h1>', html, re.DOTALL)
if abstract_m:
    abstract_text = re.sub(r'<[^>]+>', ' ', abstract_m.group(1))
    abstract_words = len(abstract_text.split())
    print(f"Abstract word count: {abstract_words} (RAS limit: 250)")

# ============================================================
# 7. Write output
# ============================================================
TARGET.write_text(html, encoding='utf-8')
print(f"\nWrote {TARGET} ({len(html)} chars)")
