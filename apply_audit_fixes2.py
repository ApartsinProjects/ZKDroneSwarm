"""
apply_audit_fixes2.py - Fix the 4 misses from apply_audit_fixes.py
(blank lines in HTML have trailing 8-space indent, not bare newlines)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

with open('docs/index.html', encoding='utf-8') as f:
    html = f.read()

changes = []
SP = '        '  # 8-space indent

def rep(old, new, label):
    global html
    if old in html:
        html = html.replace(old, new, 1)
        changes.append('FIXED: ' + label)
        return True
    else:
        changes.append('MISS:  ' + label)
        return False

# ============================================================
# FIX 3.2 - §3.2 nested list → prose
# Blank lines in HTML = '\n' + 8-spaces + '\n'
# ============================================================
OLD_32 = (
    '        <p>The ZK-MRTA framework is defined by strict informational constraints:</p>\n'
    + SP + '\n'
    + '        <ul>\n'
    + '            <li><strong>No prior knowledge of tasks</strong>: Agents have no access to task attributes such as difficulty, type, or required resources</li>\n'
    + '            <li><strong>No self-knowledge</strong>: Agents do not know their own capabilities or effectiveness</li>\n'
    + '            <li><strong>No knowledge of other agents</strong>: Agents cannot access the capabilities or internal states of others</li>\n'
    + '            <li><strong>No communication</strong>: Direct communication between agents is not permitted</li>\n'
    + '            <li><strong>Outcome-based learning only</strong>: Agents can observe only:\n'
    + '                <ul>\n'
    + '                    <li>the outcomes of their own actions, and</li>\n'
    + '                    <li>a shared <em>public summary</em> of the previous step:\n'
    + '                        <ul>\n'
    + '                            <li>the joint action vector</li>\n'
    + '                            <li>the per-agent reward vector</li>\n'
    + '                        </ul>\n'
    + '                    </li>\n'
    + '                </ul>\n'
    + '                <p style="margin-left: 2rem;">which carries the actions and outcomes of other agents but contains no capability or identity information (see §3.6 for the exact structure).</p>\n'
    + '            </li>\n'
    + '        </ul>'
)

NEW_32 = (
    '        <p>The ZK-MRTA framework imposes four informational constraints. '
    'Agents have no prior knowledge of task attributes (difficulty, type, or required resources); '
    'they do not know their own capabilities or effectiveness; '
    'they cannot access the capabilities or internal states of other agents; '
    'and direct inter-agent communication is not permitted. '
    'The sole input available is each agent\'s per-step observation: the outcomes of its own actions '
    'together with a shared <em>public summary</em> of the previous step, comprising the joint action '
    'vector and the per-agent reward vector. This broadcast conveys the actions and outcomes of all '
    'agents but carries no capability or identity information (§3.6).</p>'
)

rep(OLD_32, NEW_32, '3.2 nested list -> prose')

# ============================================================
# FIX 3.3 - 2-item list → inline prose
# ============================================================
OLD_33 = (
    '        <p>\n'
    '            Classical Multi-Robot Task Allocation is typically formulated as an optimization problem defined by:\n'
    '        </p>\n'
    + SP + '\n'
    + '        <ul>\n'
    + '            <li>a cost function $C(a_i, t_j)$, and</li>\n'
    + '            <li>a feasibility function $G(a_i, t_j) \\in \\{0, 1\\}$</li>\n'
    + '        </ul>\n'
    + SP + '\n'
    + '        <p>with the objective of finding an assignment'
)

NEW_33 = (
    '        <p>\n'
    '            Classical Multi-Robot Task Allocation is typically formulated as an optimization '
    'over a cost function $C(a_i, t_j)$ and a feasibility function $G(a_i, t_j) \\in \\{0, 1\\}$, '
    'with the objective of finding an assignment'
)

rep(OLD_33, NEW_33, '3.3 2-item list -> inline')

# ============================================================
# FIX 3.5 - h3 subsections + lists → prose run-in heads
# ============================================================
OLD_35 = (
    '        <h3>3.5.1 Target (Task) Types</h3>\n'
    + SP + '\n'
    + '        <ul>\n'
    + '            <li><strong>Passive targets</strong>: Static tasks whose internal state evolves only as a consequence of incoming actions.</li>\n'
    + '            <li><strong>Active targets</strong>: Dynamic tasks that may react, move, or otherwise change in response to interaction.</li>\n'
    + '        </ul>\n'
    + SP + '\n'
    + '        <p>The benchmark studied in this paper instantiates passive targets with fixed resilience (HP), so neutralization is a cumulative-damage process.</p>\n'
    + SP + '\n'
    + '        <h3>3.5.2 Agent Resource Constraints</h3>\n'
    + SP + '\n'
    + '        <ul>\n'
    + '            <li><strong>Infinite-capacity</strong>: Agents are not subject to a hard resource budget over the episode horizon.</li>\n'
    + '            <li><strong>Finite-capacity</strong>: Agents possess limited expendable resources, such as ammunition or energy.</li>\n'
    + '        </ul>\n'
    + SP + '\n'
    + '        <p>The experiments reported in this paper use the <strong>infinite-capacity</strong> variant: no per-agent ammunition limit is enforced, although ammunition consumption is still tracked diagnostically.</p>'
)

NEW_35 = (
    '        <p><strong>Task dynamics.</strong> Targets may be <em>passive</em> (static; their state evolves only under incoming actions) '
    'or <em>active</em> (dynamic; they may move or react). The benchmark studied here uses passive targets with fixed resilience (HP), '
    'so neutralization is a cumulative-damage process.</p>\n'
    + SP + '\n'
    + '        <p><strong>Agent resource constraints.</strong> Agents may be <em>infinite-capacity</em> (no hard resource budget over the episode horizon) '
    'or <em>finite-capacity</em> (limited expendable resources such as ammunition or energy). '
    'The experiments reported here use the infinite-capacity variant; ammunition consumption is tracked diagnostically but does not constrain engagement.</p>'
)

rep(OLD_35, NEW_35, '3.5 h3 + lists -> prose run-in heads')

# ============================================================
# FIX 3.9 - two category/metric lists → one prose sentence
# ============================================================
OLD_39_LISTS = (
    '        <p>\n'
    '            At an abstract level, these metrics fall into four categories:\n'
    '        </p>\n'
    + SP + '\n'
    + '        <ul>\n'
    + '            <li><strong>Time-based metrics</strong>: measures of how quickly the system reaches a desired level of task completion</li>\n'
    + '            <li><strong>Completion or progress metrics</strong>: measures of how much of the task set has been completed over the episode horizon</li>\n'
    + '            <li><strong>Efficiency metrics</strong>: measures of resource use relative to achieved task progress</li>\n'
    + '            <li><strong>Coordination metrics</strong>: measures of swarm-level redundancy, contention, and wasted effort</li>\n'
    + '        </ul>\n'
    + SP + '\n'
    + '        <p>The specific metrics used in the empirical evaluation are:</p>\n'
    + SP + '\n'
    + '        <ul>\n'
    + '            <li><strong>Steps to completion</strong> and <strong>total ammo consumed</strong> (time-based / efficiency)</li>\n'
    + '            <li><strong>Shots per target</strong> and <strong>targets neutralized</strong> (efficiency / completion)</li>\n'
    + '            <li><strong>Total collisions</strong> and <strong>total overkill</strong> (coordination)</li>\n'
    + '            <li><strong>Average latent match quality</strong> and <strong>latent mismatch</strong> (learning quality)</li>\n'
    + '        </ul>'
)

NEW_39_LISTS = (
    '        <p>\n'
    '            These metrics span four categories: <em>time-based</em> (steps to completion, total ammo consumed), '
    '<em>progress</em> (targets neutralized, shots per target), <em>coordination</em> (total target contention, total overkill), '
    'and <em>learning quality</em> (average latent match quality $\\bar{q}$, latent mismatch in HP).\n'
    '        </p>'
)

rep(OLD_39_LISTS, NEW_39_LISTS, '3.9 category+metric lists -> prose')

# ============================================================
# FIX 5.2 - intro sentence "Table 2" → "Table B1 in Appendix B"
# ============================================================
rep(
    'Table 2 summarizes the top-level configuration groups and the axes of variation they control; this enumerates the surface over which the framework is reusable.',
    'Table&nbsp;B1 in Appendix&nbsp;B summarizes the top-level configuration groups and the axes of variation they control.',
    '5.2 intro sentence: Table 2 -> Table B1'
)

# ============================================================
# Diagnostics
# ============================================================
checks = [
    ('3.2 prose',          'four informational constraints' in html),
    ('3.2 list gone',      '<li><strong>No prior knowledge' not in html),
    ('3.3 inline',         'optimization over a cost function' in html),
    ('3.3 list gone',      '<li>a cost function' not in html),
    ('3.5 run-in heads',   '<strong>Task dynamics.</strong>' in html),
    ('3.5 h3 gone',        '<h3>3.5.1' not in html),
    ('3.9 prose',          'four categories:' in html),
    ('3.9 cat list gone',  '<li><strong>Time-based metrics' not in html),
    ('5.2 Table B1 ref',   'Table&nbsp;B1 in Appendix' in html),
]

changes.append('')
for label, ok in checks:
    changes.append(('OK  : ' if ok else 'FAIL: ') + label)

changes.append(f'\nFile: {len(html)} chars')

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

for c in changes:
    print(c)
