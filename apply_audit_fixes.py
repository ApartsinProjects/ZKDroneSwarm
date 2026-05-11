"""
apply_audit_fixes.py
Apply all academic style/format improvements from the audit:
 1. §2.3.x h3 headings → prose run-in heads
 2. §2.7 "Summary" heading removal
 3. §3.2 nested list → prose
 4. §3.3 2-item list → inline prose
 5. §3.5 h3 subsections + lists → prose run-in heads
 6. §3.9 two category/metric lists → one prose sentence
 7. §3.9 crowding/overkill list → prose paragraphs
 8. §3.10 Notation table → Appendix A (add cross-ref in §3.1)
 9. §4.4.1 / §4.5 order swap (structural bug) + cross-ref fix
10. §5.2 config table → Appendix B
11. §6.3 hyperparameter list → condense, table in Appendix B
12. §8.6 + §8.7 → merge into single §8.6
"""
import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

with open('docs/index.html', encoding='utf-8') as f:
    html = f.read()

changes = []

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
# 1. §2.3.x h3 headings → bold prose run-in heads
# ============================================================
rep(
    '        <h3>2.3.1 Multi-Armed Bandits and Exploration–Exploitation</h3>\n\n        <p>\n            A complementary',
    '        <p>\n            <strong>Multi-Armed Bandits and Exploration–Exploitation.</strong> A complementary',
    '2.3.1 h3 -> run-in head'
)
rep(
    '        <h3>2.3.2 Cooperative and Multiplayer Bandits</h3>\n\n        <p>\n            When multiple agents',
    '        <p>\n            <strong>Cooperative and Multiplayer Bandits.</strong> When multiple agents',
    '2.3.2 h3 -> run-in head'
)
rep(
    '        <h3>2.3.3 Dec-POMDPs and Independent Learning</h3>\n\n        <p>\n            The decentralized',
    '        <p>\n            <strong>Dec-POMDPs and Independent Learning.</strong> The decentralized',
    '2.3.3 h3 -> run-in head'
)
rep(
    '        <h3>2.3.4 Mean-Field Approximations for Large Swarms</h3>\n\n        <p>\n            As swarm size grows',
    '        <p>\n            <strong>Mean-Field Approximations for Large Swarms.</strong> As swarm size grows',
    '2.3.4 h3 -> run-in head'
)
rep(
    '        <h3>2.3.5 Emergent Roles and Heterogeneous-Agent MARL</h3>\n\n        <p>\n            A strand of recent',
    '        <p>\n            <strong>Emergent Roles and Heterogeneous-Agent MARL.</strong> A strand of recent',
    '2.3.5 h3 -> run-in head'
)

# ============================================================
# 2. §2.7 "Summary" heading removal (keep paragraphs)
# ============================================================
rep(
    '        <h2>2.7 Summary</h2>\n\n        <p>\n            Taken together',
    '        <p>\n            Taken together',
    '2.7 heading removed'
)

# ============================================================
# 3. §3.2 nested list → prose
# ============================================================
OLD_32 = '''        <p>The ZK-MRTA framework is defined by strict informational constraints:</p>

        <ul>
            <li><strong>No prior knowledge of tasks</strong>: Agents have no access to task attributes such as difficulty, type, or required resources</li>
            <li><strong>No self-knowledge</strong>: Agents do not know their own capabilities or effectiveness</li>
            <li><strong>No knowledge of other agents</strong>: Agents cannot access the capabilities or internal states of others</li>
            <li><strong>No communication</strong>: Direct communication between agents is not permitted</li>
            <li><strong>Outcome-based learning only</strong>: Agents can observe only:
                <ul>
                    <li>the outcomes of their own actions, and</li>
                    <li>a shared <em>public summary</em> of the previous step:
                        <ul>
                            <li>the joint action vector</li>
                            <li>the per-agent reward vector</li>
                        </ul>
                    </li>
                </ul>
                <p style="margin-left: 2rem;">which carries the actions and outcomes of other agents but contains no capability or identity information (see §3.6 for the exact structure).</p>
            </li>
        </ul>'''

NEW_32 = '''        <p>The ZK-MRTA framework imposes four informational constraints. Agents have no prior knowledge of task attributes (difficulty, type, or required resources); they do not know their own capabilities or effectiveness; they cannot access the capabilities or internal states of other agents; and direct inter-agent communication is not permitted. The sole input available is each agent's per-step observation: the outcomes of its own actions together with a shared <em>public summary</em> of the previous step, comprising the joint action vector and the per-agent reward vector. This broadcast conveys the actions and outcomes of all agents but carries no capability or identity information (§3.6).</p>'''

rep(OLD_32, NEW_32, '3.2 nested list -> prose')

# ============================================================
# 4. §3.3 2-item list → inline prose
# ============================================================
OLD_33 = '''        <p>
            Classical Multi-Robot Task Allocation is typically formulated as an optimization problem defined by:
        </p>

        <ul>
            <li>a cost function $C(a_i, t_j)$, and</li>
            <li>a feasibility function $G(a_i, t_j) \\in \\{0, 1\\}$</li>
        </ul>

        <p>with the objective of finding an assignment'''

NEW_33 = '''        <p>
            Classical Multi-Robot Task Allocation is typically formulated as an optimization over a cost function $C(a_i, t_j)$ and a feasibility function $G(a_i, t_j) \\in \\{0, 1\\}$, with the objective of finding an assignment'''

rep(OLD_33, NEW_33, '3.3 2-item list -> inline')

# ============================================================
# 5. §3.5 h3 subsections + bullet lists → prose run-in heads
# ============================================================
OLD_35 = '''        <h3>3.5.1 Target (Task) Types</h3>

        <ul>
            <li><strong>Passive targets</strong>: Static tasks whose internal state evolves only as a consequence of incoming actions.</li>
            <li><strong>Active targets</strong>: Dynamic tasks that may react, move, or otherwise change in response to interaction.</li>
        </ul>

        <p>The benchmark studied in this paper instantiates passive targets with fixed resilience (HP), so neutralization is a cumulative-damage process.</p>

        <h3>3.5.2 Agent Resource Constraints</h3>

        <ul>
            <li><strong>Infinite-capacity</strong>: Agents are not subject to a hard resource budget over the episode horizon.</li>
            <li><strong>Finite-capacity</strong>: Agents possess limited expendable resources, such as ammunition or energy.</li>
        </ul>

        <p>The experiments reported in this paper use the <strong>infinite-capacity</strong> variant: no per-agent ammunition limit is enforced, although ammunition consumption is still tracked diagnostically.</p>'''

NEW_35 = '''        <p><strong>Task dynamics.</strong> Targets may be <em>passive</em> (static; their state evolves only under incoming actions) or <em>active</em> (dynamic; they may move or react). The benchmark studied here uses passive targets with fixed resilience (HP), so neutralization is a cumulative-damage process.</p>

        <p><strong>Agent resource constraints.</strong> Agents may be <em>infinite-capacity</em> (no hard resource budget over the episode horizon) or <em>finite-capacity</em> (limited expendable resources such as ammunition or energy). The experiments reported here use the infinite-capacity variant; ammunition consumption is tracked diagnostically but does not constrain engagement.</p>'''

rep(OLD_35, NEW_35, '3.5 h3 + lists -> prose run-in heads')

# ============================================================
# 6. §3.9 two category/metric bullet lists → one prose sentence
# ============================================================
OLD_39_LISTS = '''        <p>
            At an abstract level, these metrics fall into four categories:
        </p>

        <ul>
            <li><strong>Time-based metrics</strong>: measures of how quickly the system reaches a desired level of task completion</li>
            <li><strong>Completion or progress metrics</strong>: measures of how much of the task set has been completed over the episode horizon</li>
            <li><strong>Efficiency metrics</strong>: measures of resource use relative to achieved task progress</li>
            <li><strong>Coordination metrics</strong>: measures of swarm-level redundancy, contention, and wasted effort</li>
        </ul>

        <p>The specific metrics used in the empirical evaluation are:</p>

        <ul>
            <li><strong>Steps to completion</strong> and <strong>total ammo consumed</strong> (time-based / efficiency)</li>
            <li><strong>Shots per target</strong> and <strong>targets neutralized</strong> (efficiency / completion)</li>
            <li><strong>Total collisions</strong> and <strong>total overkill</strong> (coordination)</li>
            <li><strong>Average latent match quality</strong> and <strong>latent mismatch</strong> (learning quality)</li>
        </ul>'''

NEW_39_LISTS = '''        <p>
            These metrics span four categories: <em>time-based</em> (steps to completion, total ammo consumed), <em>progress</em> (targets neutralized, shots per target), <em>coordination</em> (total target contention, total overkill), and <em>learning quality</em> (average latent match quality $\\bar{q}$, latent mismatch in HP).
        </p>'''

rep(OLD_39_LISTS, NEW_39_LISTS, '3.9 category+metric lists -> prose')

# ============================================================
# 7. §3.9 crowding/overkill list → prose paragraphs
# ============================================================
OLD_39_DEF = '''        <ul>
            <li>
                <strong>Crowding</strong> (also called <em>collision</em> in the logged metrics): at timestep $t$, a <em>collision</em> is recorded when two or more agents select the same target $t_j$ and that target\'s HP is still positive at the start of the step. Crowding refers to the episodic pattern of elevated collision rates produced by independent agents converging on the same high-compatibility targets as their estimated utilities align. Crowding is <em>not</em> a safety failure; it is a coordination efficiency loss: shots that could have been directed at distinct targets are instead concentrated on one, potentially eliminating it faster but leaving others untouched.
            </li>
            <li>
                <strong>Overkill</strong>: at timestep $t$, overkill is recorded for target $t_j$ if the total damage applied to $t_j$ within that timestep exceeds $t_j$\'s remaining HP at the start of the step. Total overkill HP per episode sums these excesses. Overkill arises because agents engage targets simultaneously without knowledge of remaining HP and without intra-step coordination; shots landing on a target whose HP has already been reduced to zero within the same timestep are entirely wasted.
            </li>
        </ul>'''

NEW_39_DEF = '''        <p>
            <strong>Target contention</strong> (logged as <em>collision</em>): at timestep $t$, a contention event is recorded when two or more agents select the same target $t_j$ and that target\'s HP is still positive at the start of the step. Target contention refers to the episodic pattern of elevated co-selection rates produced by independent agents converging on the same high-compatibility targets as their estimated utilities align. It is a coordination efficiency loss rather than a safety failure: shots that could have been directed at distinct targets are concentrated on one, potentially accelerating its neutralization but leaving others untouched.
        </p>
        <p>
            <strong>Overkill</strong>: at timestep $t$, overkill is recorded for target $t_j$ if the total damage applied to $t_j$ within that timestep exceeds its remaining HP at the start of the step. Total overkill HP per episode sums these excesses. Overkill arises because agents engage targets simultaneously without knowledge of remaining HP and without intra-step coordination; shots landing on a target whose HP has already been reduced to zero within the same timestep are wasted.
        </p>'''

rep(OLD_39_DEF, NEW_39_DEF, '3.9 crowding/overkill list -> prose paragraphs')

# ============================================================
# 8. §3.10 Notation table → extract then remove (append to Appendix)
#    Add cross-reference in §3.1
# ============================================================

# Extract the notation table block exactly (we will paste into appendix)
NOTATION_BLOCK_START = '        <h2>3.10 Notation Summary</h2>'
NOTATION_BLOCK_END = '\n        \n        <h2>3.11 Formal Definition of ZK-MRTA</h2>'
idx_ns = html.find(NOTATION_BLOCK_START)
idx_ne = html.find(NOTATION_BLOCK_END)
if idx_ns > 0 and idx_ne > idx_ns:
    notation_content = html[idx_ns + len(NOTATION_BLOCK_START):idx_ne]
    changes.append(f'EXTRACT: §3.10 notation block ({len(notation_content)} chars)')
    # Remove §3.10 from its current position
    html = html[:idx_ns] + html[idx_ne:]
    changes.append('REMOVED: §3.10 from main body')
else:
    notation_content = ''
    changes.append(f'MISS: §3.10 block not found (start={idx_ns}, end={idx_ne})')

# Add cross-reference at end of §3.1 (before §3.2 heading)
rep(
    '\n        \n        <h2>3.2 Zero-Knowledge Assumptions</h2>',
    '\n        <p>Notation used throughout this paper is compiled in Appendix&nbsp;A.</p>\n        \n        <h2>3.2 Zero-Knowledge Assumptions</h2>',
    '3.1 notation cross-ref added'
)

# ============================================================
# 9a. Fix cross-ref error §4.5.1 → §4.4.1 in §4.4 body text
# ============================================================
rep(
    'decentralized SGD update architecture (§4.5.1) rather than',
    'decentralized SGD update architecture (§4.4.1) rather than',
    '4.4 cross-ref §4.5.1 -> §4.4.1 fixed'
)

# ============================================================
# 9b. §4.5 / §4.4.1 order swap
# §4.5 currently appears before §4.4.1 — structural bug
# ============================================================
BLOCK_45 = '''        <h2>4.5 Action Selection</h2>

        <p>
            At decision time, each agent $a_i$ uses only its own latent row $P_{i,:}^{(a_i)}$ to evaluate currently active targets. Action selection follows an $\\varepsilon$-greedy mechanism. With probability $\\varepsilon$, the agent explores by selecting a random active target. Otherwise, it exploits by selecting the target with the highest predicted utility. After each action, the exploration rate decays multiplicatively until it reaches a predefined floor. The initial exploration rate, decay factor, and floor are configurable hyperparameters; their values for the reported experiments are listed in §5.
        </p>

        <p>
            Formally, if $\\mathcal{T}_t(a_i)$ is the set of active targets observed by agent $a_i$ at time $t$, then exploitation selects:
        </p>

        $$a_t(a_i) = \\arg\\max_{j \\in \\mathcal{T}_t(a_i)} \\hat{r}_{ij}^{(a_i)}$$

        <p>while exploration samples uniformly from $\\mathcal{T}_t(a_i)$.</p>
        '''

BLOCK_441 = '''        <h3>4.4.1 Learning from Shared Interaction Data</h3>

        <p>
            Learning relies on the public-observation assumption introduced in §3: after each step, every agent sees which drone engaged which target and the resulting reward signal, even though the underlying compatibility structure that generated those rewards remains hidden. Concretely, the public step summary exposes two arrays that drive the learner: the joint action vector identifies the engaged drone-target pairs $(a_i, t_j)$; and the per-agent reward vector provides the supervision target $y_{ij}$ in direct mode, or a single sample feeding the running mean in integration-matrix mode. No privileged state is consumed; the learner sees only what the environment makes public.
        </p>

        <p>
            This is what makes the method collaborative despite being decentralized: no parameters are exchanged, but the observation stream is shared. Two supervision modes are supported: <strong>direct mode</strong>, in which the policy regresses predicted utility against each observed reward directly (including events on already-inactive targets, whose reward still reflects compatibility); and <strong>integration-matrix mode</strong>, in which the policy first updates a running-mean interaction matrix and then uses that accumulated estimate as the supervision target. The experiments reported in this paper use integration-matrix mode.
        </p>

        <p>
            Each agent $a_k$ therefore updates its local model for every observed event $(a_i, t_j)$, not only its own engagements. Let the prediction error for an observed event $(a_i, t_j)$, as computed by agent $a_k$, be denoted by:
        </p>

        $$e_{ij}^{(a_k)} = \\hat{r}_{ij}^{(a_k)} - y_{ij}$$

        <p>
            where $y_{ij}$ is either the immediate observed reward or the current running-mean entry of the integration matrix. The local embeddings minimize the regularized squared-error loss:
        </p>

        $$\\mathcal{L}_{ij}^{(a_k)} = \\left( e_{ij}^{(a_k)} \\right)^2 + \\tfrac{\\lambda}{2} \\left( \\| P_{i,:}^{(a_k)} \\|^2 + \\| U_{:,j}^{(a_k)} \\|^2 \\right)$$

        <p>
            <strong>Note on loss convention.</strong> The data term uses $(e)^2$ rather than $\\frac{1}{2}(e)^2$, so the gradient carries an explicit factor of 2 in the data term relative to the regularizer. Practitioners comparing with published CF implementations using the half-squared convention should scale $\\eta$ accordingly.
        </p>
        <p>
            The SGD updates applied after each observed event are:
        </p>

        $$P_{i,:}^{(a_k)} \\leftarrow P_{i,:}^{(a_k)} - \\eta \\left( 2 e_{ij}^{(a_k)} U_{:,j}^{(a_k)} + \\lambda P_{i,:}^{(a_k)} \\right)$$

        $$U_{:,j}^{(a_k)} \\leftarrow U_{:,j}^{(a_k)} - \\eta \\left( 2 e_{ij}^{(a_k)} P_{i,:}^{(a_k)} + \\lambda U_{:,j}^{(a_k)} \\right)$$

        <p>
            where $\\eta$ is the learning rate and $\\lambda$ is the regularization coefficient. Learning persists across episodes: the embedding matrices are carried forward at episode boundaries rather than re-initialized, so knowledge accumulated in earlier episodes continues to shape future decisions.
        </p>

        <p>
            <strong>Computational cost.</strong> Each agent $a_k$ maintains two matrices: $P^{(a_k)} \\in \\mathbb{R}^{m \\times d_f}$ and $U^{(a_k)} \\in \\mathbb{R}^{d_f \\times n}$, totaling $(m + n) \\cdot d_f$ parameters per agent. For the benchmark configuration ($m = 9$, $n = 27$, $d_f = 3$) this is 108 parameters per agent, or 972 for the full swarm. Per-step cost is $O(m \\cdot n \\cdot d_f)$; the algorithm requires no matrix inversion or eigendecomposition.
        </p>
'''

# Find the combined block (§4.5 then §4.4.1) and swap
OLD_ORDER = BLOCK_45 + BLOCK_441
NEW_ORDER = BLOCK_441 + '\n' + BLOCK_45

if OLD_ORDER in html:
    html = html.replace(OLD_ORDER, NEW_ORDER, 1)
    changes.append('FIXED: §4.5/§4.4.1 order swapped')
else:
    changes.append('MISS:  §4.5/§4.4.1 swap - combined block not found; trying split approach')
    # Try to find just the boundary and swap
    idx_45 = html.find('        <h2>4.5 Action Selection</h2>')
    idx_441 = html.find('        <h3>4.4.1 Learning from Shared Interaction Data</h3>')
    idx_46 = html.find('        <h2>4.6 Summary of the Compared Methods</h2>')
    changes.append(f'  §4.5 at idx {idx_45}, §4.4.1 at idx {idx_441}, §4.6 at idx {idx_46}')
    if 0 < idx_45 < idx_441 < idx_46:
        seg_45  = html[idx_45:idx_441]
        seg_441 = html[idx_441:idx_46]
        html = html[:idx_45] + seg_441 + seg_45 + html[idx_46:]
        changes.append('FIXED: §4.5/§4.4.1 order swapped (fallback)')
    else:
        changes.append('MISS:  §4.5/§4.4.1 swap - fallback also failed')

# ============================================================
# 10. §5.2 config table → Appendix B
# ============================================================
# Extract the §5.2 table block
TABLE2_START = '        <p><strong>Table 2: Framework configuration surface.</strong></p>'
TABLE2_AFTER = '\n        \n        <p>\n            Scenario construction follows'

idx_t2s = html.find(TABLE2_START)
idx_t2e = html.find(TABLE2_AFTER)
if idx_t2s > 0 and idx_t2e > idx_t2s:
    table2_content = html[idx_t2s + len(TABLE2_START):idx_t2e]
    changes.append(f'EXTRACT: §5.2 config table ({len(table2_content)} chars)')
    # Replace with a reference
    html = html[:idx_t2s] + '        <p>The full configuration surface is documented in Appendix&nbsp;B, Table&nbsp;B1.</p>' + html[idx_t2e:]
    changes.append('FIXED: §5.2 table replaced with appendix reference')
else:
    table2_content = ''
    changes.append(f'MISS: §5.2 table block not found (start={idx_t2s}, end={idx_t2e})')

# Also fix the intro sentence that references "Table 2"
rep(
    'Table&nbsp;2 summarizes the top-level configuration groups and the axes of variation they control; this enumerates the surface over which the framework is reusable.',
    'Table&nbsp;B1 in Appendix&nbsp;B summarizes the top-level configuration groups and the axes of variation they control.',
    '5.2 intro sentence Table 2 -> Table B1'
)

# ============================================================
# 11. §6.3 hyperparameter list → condense to one sentence
# ============================================================
OLD_63_LIST = '''        <ul>
            <li>$\\eta \\in \\{0.001, 0.005, 0.01, 0.05\\}$; selected 0.01 (fastest stable convergence)</li>
            <li>$\\lambda \\in \\{0.01, 0.02, 0.05\\}$; selected 0.02 (minimal over-regularization)</li>
            <li>$d_f \\in \\{1, 2, 3, 4, 6\\}$; selected 3 (matches ground-truth latent dimension; sensitivity reported in §7.8)</li>
            <li>$\\varepsilon_0 \\in \\{0.2, 0.3, 0.5\\}$; selected 0.3 (sufficient early exploration)</li>
            <li>$\\delta \\in \\{0.999, 0.9995, 0.9998\\}$; selected 0.9995 (reaches low $\\varepsilon$ by episode 35)</li>
        </ul>'''

NEW_63_LIST = '''        <p>
            The selected values were $\\eta = 0.01$, $\\lambda = 0.02$, $d_f = 3$, $\\varepsilon_0 = 0.3$, and $\\delta = 0.9995$. Candidate ranges and selection rationale are documented in Appendix&nbsp;B, Table&nbsp;B2.
        </p>'''

rep(OLD_63_LIST, NEW_63_LIST, '6.3 hyperparameter list -> one sentence')

# ============================================================
# 12. §8.6 + §8.7 → merge into single §8.6
# ============================================================
OLD_86_87 = '''        <h2>8.6 Comparison with Emergent-Role MARL</h2>

        <p>
            The ZK-MRTA setting shares a structural property with ROMA [42] and RODE [43]: agents develop task specializations without predefined capability labels, driven entirely by interaction outcomes. The critical difference is that ROMA and RODE operate with shared training infrastructure and explicit reward feedback, whereas ZK-MRTA agents update private models from a shared observation stream with no gradient sharing. The convergence guarantees provided by Zhong et al. [45] for heterogeneous cooperative agents without parameter sharing provide theoretical grounding for why private local models can converge to efficient collective behavior; the multi-seed consistency observed here is an empirical instantiation of that guarantee in a decentralized, noise-corrupted setting.
        </p>

        <h2>8.7 Exclusion of MARL Baselines</h2>

        <p>
            Standard cooperative MARL algorithms, such as IQL [12], QMIX [44], or MAPPO, were not included as baselines. This exclusion is deliberate and principled, not a gap. MARL algorithms of these families require: (i) a centralized training phase (CTDE) with access to global state or joint reward signals, or (ii) direct inter-agent communication, or (iii) explicit capability disclosure for role assignment. All three requirements violate the ZK-MRTA constraints formalized in §3.2. The purpose of this paper is precisely to evaluate what is achievable when these resources are unavailable; including a baseline that requires them would conflate two different problem settings.
        </p>
        <p>
            An independent Q-learner (IQL) that uses the same restricted public observation stream as the MF policy would satisfy ZK-constraints, and is a natural candidate for future comparison. IQL was not included in the present evaluation because (a) implementing a correct IQL with the same observation-space factorization required to make it ZK-compliant introduces non-trivial design choices that merit their own study, and (b) the UCB-Indep baseline already captures the key dimension of interest: can a per-arm learner that exploits no latent structure match the latent-exploiting MF policy? It cannot (§7.3). An IQL baseline would be expected to occupy a position between UCB-Indep and MF, and characterizing that gap is deferred to future work.
        </p>'''

NEW_86_87 = '''        <h2>8.6 MARL Comparisons and Exclusion Rationale</h2>

        <p>
            The ZK-MRTA setting shares a structural property with ROMA [42] and RODE [43]: agents develop task specializations without predefined capability labels, driven entirely by interaction outcomes. The critical difference is that ROMA and RODE operate with shared training infrastructure and explicit reward feedback, whereas ZK-MRTA agents update private models from a shared observation stream with no gradient sharing. The convergence guarantees proved by Zhong et al. [45] for heterogeneous cooperative agents without parameter sharing provide theoretical grounding for why private local models can converge to efficient collective behavior; the multi-seed consistency observed here is an empirical instantiation of that guarantee in a decentralized, noise-corrupted setting.
        </p>

        <p>
            Standard cooperative MARL algorithms such as IQL [12], QMIX [44], or MAPPO were not included as baselines. This exclusion is deliberate and principled. MARL algorithms of these families require (i) a centralized training phase (CTDE) with access to global state or joint reward signals, (ii) direct inter-agent communication, or (iii) explicit capability disclosure for role assignment. All three requirements violate the ZK-MRTA constraints formalized in §3.2; including such baselines would conflate two qualitatively different problem settings. A ZK-compliant IQL variant that uses the same restricted public observation stream as the MF policy is a natural candidate for future comparison, but its correct formulation requires non-trivial observation-space factorization choices that merit independent study. The UCB-Indep baseline already isolates the key structural question: can a per-arm learner with no latent-structure exploitation match the MF policy? It cannot (§7.3). An IQL baseline would be expected to fall between UCB-Indep and MF; characterizing that gap is deferred to future work.
        </p>'''

rep(OLD_86_87, NEW_86_87, '8.6+8.7 merged')

# ============================================================
# Build appendix HTML
# ============================================================

APPENDIX_HTML = '''\n<h1>Appendix</h1>\n\n        <h2>Appendix A: Notation Summary</h2>\n\n        <p>Table&nbsp;A1 lists the mathematical symbols used throughout this paper.</p>\n''' + notation_content + '''\n        <h2>Appendix B: Implementation Details</h2>\n\n        <h3>B.1 Framework Configuration Surface</h3>\n\n        <p>\n            A benchmark instance is specified by a single JSON configuration file loaded through a typed configuration layer. Table&nbsp;B1 summarizes the top-level configuration groups and the axes of variation they control.\n        </p>\n''' + table2_content + '''\n        <h3>B.2 Hyperparameter Search Details</h3>\n\n        <p>\n            Table&nbsp;B2 shows the candidate ranges evaluated during the coordinate-search procedure on held-out seed&nbsp;314 and the value selected for each hyperparameter.\n        </p>\n\n        <table>\n            <thead>\n                <tr>\n                    <th>Hyperparameter</th>\n                    <th>Candidates</th>\n                    <th>Selected</th>\n                    <th>Selection criterion</th>\n                </tr>\n            </thead>\n            <tbody>\n                <tr><td>$\\eta$ (learning rate)</td><td>0.001, 0.005, 0.01, 0.05</td><td>0.01</td><td>Fastest stable convergence</td></tr>\n                <tr><td>$\\lambda$ (regularization)</td><td>0.01, 0.02, 0.05</td><td>0.02</td><td>Minimal over-regularization</td></tr>\n                <tr><td>$d_f$ (factorization dim.)</td><td>1, 2, 3, 4, 6</td><td>3</td><td>Matches ground-truth latent dimension; sensitivity in §7.8</td></tr>\n                <tr><td>$\\varepsilon_0$ (initial exploration)</td><td>0.2, 0.3, 0.5</td><td>0.3</td><td>Sufficient early exploration</td></tr>\n                <tr><td>$\\delta$ (exploration decay)</td><td>0.999, 0.9995, 0.9998</td><td>0.9995</td><td>Reaches floor $\\varepsilon_{\\min}$ by episode 35</td></tr>\n            </tbody>\n        </table>\n\n        <p><em>Table&nbsp;B2. MF policy hyperparameter search and selection on held-out seed&nbsp;314.</em></p>\n\n'''

# Insert appendix before Bibliography
rep(
    '\n<h1>Bibliography</h1>',
    APPENDIX_HTML + '\n<h1>Bibliography</h1>',
    'Appendix section inserted before Bibliography'
)

# ============================================================
# Diagnostics
# ============================================================
checks = [
    ('2.3.1 run-in head present',       '<strong>Multi-Armed Bandits' in html),
    ('2.7 heading absent',               '<h2>2.7 Summary</h2>' not in html),
    ('3.2 prose present',                'four informational constraints' in html),
    ('3.3 list absent',                  '<li>a cost function' not in html),
    ('3.5 task dynamics prose present',  '<strong>Task dynamics.</strong>' in html),
    ('3.9 prose metric list present',    'four categories:' in html and '<ul>' not in html[html.find('four categories:'):html.find('four categories:')+500]),
    ('3.9 crowding prose present',       '<strong>Target contention</strong>' in html),
    ('3.10 removed from body',           '<h2>3.10 Notation Summary</h2>' not in html),
    ('Appendix A present',               '<h2>Appendix A: Notation Summary</h2>' in html),
    ('Appendix B present',               '<h2>Appendix B: Implementation Details</h2>' in html),
    ('Table B1 ref present',             'Table&nbsp;B1' in html),
    ('Table B2 ref present',             'Table&nbsp;B2' in html),
    ('4.4.1 before 4.5 in HTML',         html.find('<h3>4.4.1 Learning') < html.find('<h2>4.5 Action Selection')),
    ('5.2 table removed from body',      '<p><strong>Table 2: Framework' not in html),
    ('6.3 list absent',                  r'\\eta \\in \\{0.001' not in html),
    ('8.7 heading absent',               '<h2>8.7 Exclusion of MARL' not in html),
    ('8.6 merged heading present',       '8.6 MARL Comparisons' in html),
    ('Notation cross-ref in §3.1',       'Notation used throughout this paper is compiled in Appendix' in html),
]

changes.append('')
changes.append('=== CHECKS ===')
all_ok = True
for label, ok in checks:
    status = 'OK  ' if ok else 'FAIL'
    if not ok:
        all_ok = False
    changes.append(f'  {status}: {label}')

changes.append('')
changes.append(f'File length: {len(html)} chars')
changes.append('All checks passed: ' + str(all_ok))

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

for c in changes:
    print(c)
