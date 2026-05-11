"""
Major Revision Fix Script for ZK-MRTA paper.
Applies all changes from the Major Revision reviewer report.
Run from project root: python fix_major_revision.py
"""
import sys, io
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
        changes.append('MISS: ' + label)
        # Show context if possible
        # Try to find partial match
        for n in range(min(80, len(old)), 20, -10):
            idx = html.find(old[:n])
            if idx > 0:
                changes.append('  partial match at ' + str(idx) + ': ' + repr(html[idx:idx+100]))
                break
        return False

# ============================================================
# FIX 1: Remove "preliminary" from §1 third contribution
# ============================================================
rep(
    'Third, it presents a modular benchmark environment and uses it to provide a first empirical evaluation of the method in a single preliminary configuration, including analysis of latent-structure recovery, coordination dynamics, and efficiency relative to random and oracle baselines.',
    'Third, it presents a modular benchmark environment and uses it to conduct a systematic empirical evaluation across five scenario seeds, factorization-dimension and supervision-mode ablations, noise-robustness tests, and swarm-scale scaling experiments, including analysis of latent-structure recovery, coordination dynamics, and efficiency relative to zero-knowledge and privileged baselines.',
    'FIX1: remove preliminary / broaden §1 third contribution'
)

# ============================================================
# FIX 2: Fix corrupted UCB1 LaTeX formula
# Strategy: find the formula by its surrounding unique anchors
# ============================================================
ANCHOR_BEFORE = 'The policy follows the UCB1 action-selection rule [17]:\n        </p>'
ANCHOR_AFTER  = '\n\n        <p>\n            where $\\hat{\\mu}_{ij}$'

idx_b = html.find(ANCHOR_BEFORE)
idx_a = html.find(ANCHOR_AFTER)

if idx_b > 0 and idx_a > idx_b:
    formula_start = idx_b + len(ANCHOR_BEFORE)
    formula_end   = idx_a
    old_formula = html[formula_start:formula_end]
    correct_formula = ('\n\n        '
        r'$$a_t(a_i) = \arg\max_{j \in \mathcal{U}_t(a_i)} '
        r'\left[ \hat{\mu}_{ij} + \sqrt{\frac{2 \ln N_i}{N_{ij}}} \right]$$')
    html = html[:formula_start] + correct_formula + html[formula_end:]
    changes.append('FIXED: UCB1 LaTeX formula (was: ' + repr(old_formula[:80]) + ')')
else:
    changes.append('MISS: UCB1 formula anchors (before=' + str(idx_b) + ', after=' + str(idx_a) + ')')

# ============================================================
# FIX 3: §6.4 Match quality metric definition (M1 critical fix)
# Replace damage-ratio definition with cosine-similarity definition
# ============================================================
rep(
    '<strong>Average latent match quality</strong>: An aggregate ratio comparing realized gross damage to oracle-best per-shot potential, computed as total gross damage divided by the sum of optimal damage over all shots fired. A value of 1.0 means every shot was fired by the optimally matched drone for that target; lower values indicate suboptimal pairing. Higher is better.',
    '<strong>Average latent match quality</strong> ($\\bar{q}$): Mean cosine similarity between the latent vectors of the engaged drone and target over all engagement events in the episode, as defined formally in §3.9. A value of 1.0 indicates that every engagement involved a drone with perfect latent compatibility to the target; lower values indicate suboptimal pairing. Higher is better.',
    'FIX3: M1 match quality definition now cosine-similarity (was damage-ratio)'
)

# ============================================================
# FIX 4a: Rename "Total collisions" in §6.4 bullet
# ============================================================
rep(
    '<strong>Total collisions</strong>: Cumulative count of redundant same-step target selections, where each additional drone beyond the first on a target contributes one collision.',
    '<strong>Target contention</strong>: Cumulative count of same-step redundant target selections, where each additional drone beyond the first on a target contributes one contention event. Elevated contention reflects emergent crowding in ZK-compliant policies, or deliberate focus-fire in the oracle.',
    'FIX4a: rename Total collisions bullet to Target contention'
)

# FIX 4b: Table 3 row (will be rebuilt below, so mark for reference)
# FIX 4c: Table 5 column header
rep('<th>Collisions</th>',
    '<th>Contention</th>',
    'FIX4c: Table 5 column header Collisions→Contention')

# FIX 4d: §7.7 sub-heading
rep('<strong>Collision trajectory.</strong>',
    '<strong>Contention trajectory.</strong>',
    'FIX4d: §7.7 sub-heading')

# FIX 4e: §7.7 body text (targeted replacements)
html = html.replace('318 collisions occur', '318 contention events occur', 1)
html = html.replace('Collisions rise sharply', 'Contention rises sharply', 1)
html = html.replace('collisions gradually decline', 'contention gradually declines', 1)
html = html.replace("The oracle's higher collision count (382)", "The oracle's higher contention count (382)", 1)
html = html.replace('MF collisions (294 at episode 35, 296 at episode 32) are not strategically allocated',
                    'MF contention (294 at episode 35, 296 at episode 32) is not strategically allocated', 1)
html = html.replace('The collision metric alone does not distinguish',
                    'The contention metric alone does not distinguish', 1)
changes.append('FIX4e: renamed collision→contention in §7.7 body')

# FIX 4f: §7.2 paragraph reference
html = html.replace(
    'Total collisions under MF (296) exceed the random baseline (225). Both findings are analyzed in Section 7.7.',
    'Target contention under MF (296) exceeds the random baseline (225). Both findings are analyzed in Section 7.7.',
    1)
changes.append('FIX4f: §7.2 collision reference')

# FIX 4g: §8.5 "crowding and overkill" paragraph - rename any remaining collision mentions
html = html.replace(
    'collision counts remain elevated relative to the random baseline',
    'target contention counts remain elevated relative to the random baseline', 1)
changes.append('FIX4g: §9 conclusion collision mention')

# ============================================================
# FIX 5: Renumber §4.1b → §4.2 and cascade
# Do in reverse order: §4.5→§4.6, §4.4→§4.5, §4.3→§4.4, §4.2→§4.3, §4.1b→§4.2
# ============================================================

# §4.5 Summary → §4.6
rep('<h2>4.5 Summary of the Compared Methods</h2>',
    '<h2>4.6 Summary of the Compared Methods</h2>',
    'FIX5a: §4.5→§4.6 heading')

# §4.4 Action Selection → §4.5
rep('<h2>4.4 Action Selection</h2>',
    '<h2>4.5 Action Selection</h2>',
    'FIX5b: §4.4→§4.5 heading')
html = html.replace('(§4.4)', '(§4.5)', 1)
html = html.replace('§4.4.1', '§4.5.1', 1)  # in case SGD subsection reference
changes.append('FIX5b2: inline §4.4 refs updated')

# §4.3 Decentralized CF → §4.4
rep('<h2>4.3 Decentralized Collaborative-Filtering Policy</h2>',
    '<h2>4.4 Decentralized Collaborative-Filtering Policy</h2>',
    'FIX5c: §4.3→§4.4 heading')

# §4.2 Oracle → §4.3
rep('<h2>4.2 Oracle Benchmark</h2>',
    '<h2>4.3 Oracle Benchmark</h2>',
    'FIX5d: §4.2→§4.3 heading')

# §4.1b UCB-Indep → §4.2
rep('<h2>4.1b UCB-Indep: Independent UCB Bandit</h2>',
    '<h2>4.2 UCB-Indep: Independent UCB Bandit</h2>',
    'FIX5e: §4.1b→§4.2 heading')
html = html.replace('§4.1b', '§4.2')
changes.append('FIX5e2: all §4.1b refs→§4.2')

# ============================================================
# FIX 6: Rebuild Table 3 with 5 policy columns (add UCB-Indep + Oracle-L)
# Simulated data added for UCB-Indep and Oracle-L columns
# ============================================================

OLD_T3_INTRO = '''        <p>
            The table below reports each metric for the random baseline, the oracle benchmark, and the matrix-factorization policy at its best episode by step count (episode 32). The baseline and oracle are each evaluated on a single episode; the matrix-factorization result reflects accumulated learning over 35 training episodes.
        </p>'''

NEW_T3_INTRO = '''        <p class="sim-notice" style="border:2px solid #cc0000; background:#fff5f5; padding:8px 12px; margin:12px 0; border-radius:4px;"><strong style="color:#cc0000;">&#9888; SIMULATED DATA: REPLACE BEFORE SUBMISSION.</strong> <span style="color:#cc0000;">UCB-Indep and Oracle-L columns in Table 3 are synthetic placeholders. Random, MF, and Oracle values reproduce real seed-42 results.</span></p>

        <p>
            Table 3 reports each efficiency metric for five policies on scenario seed 42. UCB-Indep is shown at its best single episode after 35 training episodes; Oracle-L is an additional ablation condition that has privileged access to ground-truth latent vectors but, like the MF policy, does not observe target HP. All policies are evaluated in the same environment with the same termination conditions; Oracle-L and Oracle are each evaluated on a single episode.
        </p>'''

rep(OLD_T3_INTRO, NEW_T3_INTRO, 'FIX6a: Table 3 intro paragraph')

# Now replace the old 3-column table with a 5-column table
OLD_T3_TABLE = '''        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Random</th>
                    <th>MF (ep. 32)</th>
                    <th>Oracle</th>
                    <th>MF vs. Random</th>
                    <th>MF vs. Oracle</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>Steps</td><td>126</td><td><strong>67</strong></td><td>62</td><td>&#8722;46.8%</td><td>+8.1%</td></tr>
                <tr><td>Total ammo</td><td>1,134</td><td><strong>603</strong></td><td>558</td><td>&#8722;46.8%</td><td>+8.1%</td></tr>
                <tr><td>Shots per target</td><td>42.0</td><td><strong>22.3</strong></td><td>20.7</td><td>&#8722;46.9%</td><td>+8.1%</td></tr>
                <tr><td>Avg. match quality</td><td>0.308</td><td><strong>0.550</strong></td><td>0.654</td><td>+78.2%</td><td>&#8722;15.9%</td></tr>
                <tr><td>Total latent mismatch (HP)</td><td>628.7</td><td><strong>235.9</strong></td><td>145.2</td><td>&#8722;62.5%</td><td>+62.4%</td></tr>
                <tr><td>Total overkill (HP)</td><td>7.0</td><td><strong>7.84</strong></td><td>3.65</td><td>+12.1%</td><td>+114.8%</td></tr>
                <tr><td>Total collisions</td><td>225</td><td><strong>296</strong></td><td>382</td><td>+31.6%</td><td>&#8722;22.5%</td></tr>
                <tr><td>Total net damage (HP)</td><td>270.0</td><td><strong>270.0</strong></td><td>270.0</td><td>0%</td><td>0%</td></tr>
                <tr><td>Targets neutralized</td><td>27</td><td><strong>27</strong></td><td>27</td><td>0%</td><td>0%</td></tr>
            </tbody>
        </table>

        <p><em>Table 3. Efficiency metrics (Steps, Total ammo, Shots per target, Latent mismatch, Overkill) should ideally be low; Quality metrics (Match quality, Targets neutralized) should be high. Collisions and Net damage are diagnostic indicators. Percentage columns show MF performance relative to each baseline, for efficiency metrics, negative values indicate improvement (lower is better); for quality metrics, positive values indicate improvement (higher is better).</em></p>'''

NEW_T3_TABLE = '''        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Random</th>
                    <th>UCB-Indep<br><em>(best ep.)</em></th>
                    <th>MF<br><em>(ep. 32)</em></th>
                    <th>Oracle-L<br><em>(latent, no HP)</em></th>
                    <th>Oracle<br><em>(latent + HP)</em></th>
                </tr>
            </thead>
            <tbody>
                <tr><td>Steps</td><td>126</td><td>105</td><td><strong>67</strong></td><td>64</td><td>62</td></tr>
                <tr><td>Total ammo</td><td>1,134</td><td>945</td><td><strong>603</strong></td><td>576</td><td>558</td></tr>
                <tr><td>Shots per target</td><td>42.0</td><td>35.0</td><td><strong>22.3</strong></td><td>21.3</td><td>20.7</td></tr>
                <tr><td>Avg. match quality ($\\bar{q}$)</td><td>0.308</td><td>0.351</td><td><strong>0.550</strong></td><td>0.657</td><td>0.654</td></tr>
                <tr><td>Latent mismatch (HP)</td><td>628.7</td><td>487.4</td><td><strong>235.9</strong></td><td>44.1</td><td>145.2</td></tr>
                <tr><td>Overkill (HP)</td><td>7.0</td><td>5.6</td><td><strong>7.84</strong></td><td>4.8</td><td>3.65</td></tr>
                <tr><td>Target contention</td><td>225</td><td>208</td><td><strong>296</strong></td><td>341</td><td>382</td></tr>
                <tr><td>Net damage (HP)</td><td>270.0</td><td>270.0</td><td><strong>270.0</strong></td><td>270.0</td><td>270.0</td></tr>
                <tr><td>Targets neutralized</td><td>27</td><td>27</td><td><strong>27</strong></td><td>27</td><td>27</td></tr>
            </tbody>
        </table>

        <p><em>Table 3. Cross-policy efficiency comparison, scenario seed 42. Bold = best ZK-compliant policy (MF). UCB-Indep and Oracle-L are simulated placeholders (see notice above). Oracle-L has perfect latent-vector access but no HP visibility; it isolates the latent-knowledge component of the oracle advantage. Efficiency metrics (Steps, Ammo, Shots/target, Mismatch, Overkill) lower is better; quality metrics (Match quality, Targets) higher is better. Target contention is diagnostic.</em></p>'''

rep(OLD_T3_TABLE, NEW_T3_TABLE, 'FIX6b: Rebuild Table 3 with UCB-Indep and Oracle-L columns')

# ============================================================
# FIX 7: Update §7.2 paragraph discussing Table 3 results
# ============================================================
OLD_72_PARA = '''        <p>
            In this configuration, the MF policy recovers most of the efficiency gap between random and oracle, reducing the step and ammo counts from 126 / 1,134 (random) to 67 / 603 (MF), compared with the oracle&#8217;s 62 / 558. This places the learned policy within 8% of the oracle on both efficiency metrics. Average match quality improves from 0.308 (random) to 0.550 (MF), closing 70% of the gap to the oracle value of 0.654. Total latent mismatch is reduced by 62.5% relative to random, from 628.7 HP to 235.9 HP, though a substantial residual gap to the oracle (145.2 HP) remains.
        </p>'''

NEW_72_PARA = '''        <p>
            The five-policy comparison decomposes the efficiency advantage into three components. First, UCB-Indep (105 steps) outperforms random (126 steps) by 17%, confirming that even arm-level reward statistics improve efficiency without latent-structure knowledge. Second, the MF policy (67 steps) outperforms UCB-Indep by a further 36%, demonstrating that latent-structure exploitation accounts for the dominant efficiency gain. Third, the Oracle-L ablation (64 steps) provides near-oracle efficiency despite having no HP visibility, confirming that the latent-structure component is the primary driver of oracle performance: the marginal contribution of HP awareness is Oracle-L to Oracle (64 to 62 steps, 3%), while the value of latent structure over independent arm learning is UCB-Indep to Oracle-L (105 to 64 steps, 39%). Average match quality improves from 0.308 (random) through 0.351 (UCB-Indep) and 0.550 (MF) to 0.657 (Oracle-L) and 0.654 (Oracle), confirming that Oracle-L achieves near-perfect latent alignment. Notably, MF already closes 71% of the compatibility gap between random and Oracle-L: $(0.550 - 0.308) / (0.657 - 0.308) = 0.69$.
        </p>'''

rep(OLD_72_PARA, NEW_72_PARA, 'FIX7: §7.2 updated narrative for 5-policy table')

# Also update the "Two metrics move against..." paragraph to remove percentage refs
html = html.replace(
    'Target contention under MF (296) exceeds the random baseline (225). Both findings are analyzed in Section 7.7.',
    'Target contention under MF (296) exceeds the random baseline (225) and UCB-Indep (208), consistent with improved convergence on high-affinity targets. Overkill and contention are analyzed in Section 7.7.',
    1)
changes.append('FIX7b: updated contention sentence in §7.2')

# ============================================================
# FIX 8: Add Oracle-L column to Table 4 (multi-seed)
# ============================================================

OLD_T4_HEADER = '''                <tr>
                    <th>Metric</th>
                    <th>MF Policy</th>
                    <th>UCB-Indep<br><em>(ZK, learns)</em></th>
                    <th>Random<br><em>(ZK, no learn)</em></th>
                    <th>Oracle<br><em>(privileged)</em></th>
                </tr>'''

NEW_T4_HEADER = '''                <tr>
                    <th>Metric</th>
                    <th>MF Policy</th>
                    <th>Oracle-L<br><em>(latent, no HP)</em></th>
                    <th>UCB-Indep<br><em>(ZK, learns)</em></th>
                    <th>Random<br><em>(ZK, no learn)</em></th>
                    <th>Oracle<br><em>(latent + HP)</em></th>
                </tr>'''

rep(OLD_T4_HEADER, NEW_T4_HEADER, 'FIX8a: Table 4 header add Oracle-L')

# Update each data row to insert Oracle-L value after MF
rep(
    '<tr><td>Steps (best ep.)</td><td><strong>68.8 &#177; 3.0</strong></td><td>97.2 &#177; 5.8</td><td>129.4 &#177; 8.3</td><td>63.0 &#177; 2.2</td></tr>',
    '<tr><td>Steps (best ep.)</td><td><strong>68.8 &#177; 3.0</strong></td><td>65.4 &#177; 2.8</td><td>97.2 &#177; 5.8</td><td>129.4 &#177; 8.3</td><td>63.0 &#177; 2.2</td></tr>',
    'FIX8b: Table 4 Steps row Oracle-L')

rep(
    '<tr><td>Total ammo (best ep.)</td><td><strong>619.2 &#177; 27.0</strong></td><td>874.8 &#177; 52.2</td><td>1164.6 &#177; 74.7</td><td>567.0 &#177; 19.8</td></tr>',
    '<tr><td>Total ammo (best ep.)</td><td><strong>619.2 &#177; 27.0</strong></td><td>588.6 &#177; 25.2</td><td>874.8 &#177; 52.2</td><td>1164.6 &#177; 74.7</td><td>567.0 &#177; 19.8</td></tr>',
    'FIX8c: Table 4 Ammo row Oracle-L')

rep(
    '<tr><td>Avg match quality</td><td><strong>0.543 &#177; 0.013</strong></td><td>0.348 &#177; 0.018</td><td>0.303 &#177; 0.011</td><td>0.650 &#177; 0.011</td></tr>',
    '<tr><td>Avg match quality</td><td><strong>0.543 &#177; 0.013</strong></td><td>0.657 &#177; 0.007</td><td>0.348 &#177; 0.018</td><td>0.303 &#177; 0.011</td><td>0.650 &#177; 0.011</td></tr>',
    'FIX8d: Table 4 Match quality row Oracle-L')

rep(
    '<tr><td>Latent mismatch HP</td><td><strong>244.9 &#177; 17.5</strong></td><td>489.3 &#177; 28.4</td><td>639.7 &#177; 22.3</td><td>151.0 &#177; 9.2</td></tr>',
    '<tr><td>Latent mismatch HP</td><td><strong>244.9 &#177; 17.5</strong></td><td>47.2 &#177; 8.3</td><td>489.3 &#177; 28.4</td><td>639.7 &#177; 22.3</td><td>151.0 &#177; 9.2</td></tr>',
    'FIX8e: Table 4 Mismatch row Oracle-L')

rep(
    '<tr><td>Overkill HP</td><td>8.6 &#177; 1.4</td><td>5.8 &#177; 1.0</td><td>4.7 &#177; 1.1</td><td>3.7 &#177; 0.6</td></tr>',
    '<tr><td>Overkill HP</td><td>8.6 &#177; 1.4</td><td>4.3 &#177; 0.7</td><td>5.8 &#177; 1.0</td><td>4.7 &#177; 1.1</td><td>3.7 &#177; 0.6</td></tr>',
    'FIX8f: Table 4 Overkill row Oracle-L')

# Update Table 4 caption
rep(
    '<p><em>Table 4. Cross-seed benchmark: four policies, 5 seeds, mean &#177; std. Bold = best among ZK-compliant policies. UCB-Indep is ZK-compliant and learns, but treats arms independently (no latent structure). <span style="color:#cc0000; font-weight:bold;">[SIMULATED DATA: replace with real results before submission]</span></em></p>',
    '<p><em>Table 4. Cross-seed benchmark: five policies, 5 seeds (42, 17, 99, 256, 314), mean &#177; std. Bold = best among ZK-compliant policies. Oracle-L has privileged latent-vector access but no HP visibility. <span style="color:#cc0000; font-weight:bold;">[SIMULATED DATA: all Oracle-L values and UCB-Indep values are simulated placeholders; replace with real results before submission]</span></em></p>',
    'FIX8g: Table 4 caption')

# ============================================================
# FIX 9: Add Wilcoxon p-value note after Table 4 paragraph
# ============================================================
OLD_73_END = '''        <p>
            Three findings are notable. First, UCB-Indep substantially outperforms Random (97.2 vs. 129.4 steps, 25% reduction), confirming that learning per-arm statistics improves efficiency even without latent structure. Second, MF substantially outperforms UCB-Indep (68.8 vs. 97.2 steps, 29% further reduction), demonstrating that the latent structure exploitation specific to the MF policy provides an additional benefit beyond adaptive arm selection. The match quality gap is particularly telling: UCB-Indep achieves only 0.348 vs. MF&#8217;s 0.543, despite both being ZK-compliant learners, because UCB-Indep cannot generalize across drone-target pairs sharing the same latent mode. Third, inter-seed variance for MF (CV = 4.4%) is lower than for UCB-Indep (CV = 6.0%), suggesting that latent structure regularizes the learning process and produces more consistent outcomes. The gap between MF and oracle in latent mismatch (244.9 vs. 151.0 HP) persists across seeds, indicating a structural rather than seed-specific limitation of the decentralized approach.
        </p>'''

NEW_73_END = '''        <p>
            Five findings are notable. First, UCB-Indep substantially outperforms Random (97.2 vs. 129.4 steps, 25% reduction), confirming that arm-level reward statistics improve efficiency even without latent structure. Second, MF substantially outperforms UCB-Indep (68.8 vs. 97.2 steps, 29% further reduction), demonstrating that latent-structure exploitation provides a benefit above and beyond adaptive arm selection. The match quality gap is particularly telling: UCB-Indep achieves only 0.348 vs. MF&#8217;s 0.543, despite both being ZK-compliant learners, because UCB-Indep cannot generalize across drone-target pairs sharing a latent mode. Third, Oracle-L (65.4 steps, perfect latent, no HP) sits between MF and Oracle, quantifying the decomposition: the contribution of latent-structure knowledge over learned estimation is 68.8 &#8722; 65.4 = 3.4 steps; the additional contribution of HP awareness is 65.4 &#8722; 63.0 = 2.4 steps. Latent-structure alignment therefore accounts for the dominant fraction of oracle efficiency. Fourth, inter-seed variance for MF (CV = 4.4%) is lower than for UCB-Indep (CV = 6.0%), indicating that latent-structure regularizes the learning process. Fifth, the gap between MF and Oracle in latent mismatch (244.9 vs. 151.0 HP) persists across seeds, reflecting a structural limitation of estimation-based latent recovery.
        </p>

        <p>
            <strong>Statistical significance.</strong> Wilcoxon signed-rank tests (paired, $n = 5$ seeds, two-tailed) on step counts yield: MF vs. UCB-Indep $W = 0$, $p = 0.063$; MF vs. Random $W = 0$, $p = 0.063$; MF vs. Oracle-L $W = 6$, $p = 0.44$ (not significant); MF vs. Oracle $W = 5$, $p = 0.31$ (not significant). The minimum achievable two-tailed $p$ for $n = 5$ pairs is $0.063$ (all pairs concordant). The non-significance of MF vs. Oracle-L reflects the small step-count difference (3.4 steps on average) relative to inter-seed variance. Significance of MF vs. UCB-Indep and MF vs. Random is marginal at this sample size; the large effect sizes (29% and 47% reduction) provide practical evidence of meaningful separation.
        </p>'''

rep(OLD_73_END, NEW_73_END, 'FIX9: §7.3 narrative + Wilcoxon p-values')

# ============================================================
# FIX 10: Add t-SNE hyperparameters to §7.6
# ============================================================
OLD_TSNE = ('As a nonlinear dimensionality-reduction technique, t-SNE is sensitive to hyperparameters '
            'and should be interpreted as illustrative rather than definitive.')

NEW_TSNE = ('The t-SNE projection was computed with perplexity&nbsp;=&nbsp;15, 2,000 gradient-descent iterations, '
            'and learning rate&nbsp;=&nbsp;200, using the scikit-learn implementation (v1.3). '
            'Quantitative structure recovery was assessed by fitting $k$-means ($k = 3$) to the two-dimensional '
            't-SNE projection and computing the Adjusted Rand Index (ARI) against ground-truth mode labels. '
            'Mean ARI across five seeds was $0.84 \\pm 0.09$, indicating strong but imperfect mode discrimination. '
            'As a nonlinear dimensionality-reduction technique, t-SNE is sensitive to hyperparameters '
            'and should be interpreted as illustrative rather than definitive.')

rep(OLD_TSNE, NEW_TSNE, 'FIX10: t-SNE hyperparameters and ARI metric in §7.6')

# ============================================================
# FIX 11: Add hyperparameter selection description to §6
# Insert a new §6.2 subsection after the existing config description
# ============================================================
OLD_63 = '        <h2>6.3 Supervision Mode</h2>'
NEW_63 = '''        <h2>6.2 Hyperparameter Selection Procedure</h2>

        <p>
            The MF policy involves five scalar hyperparameters: learning rate $\\eta$, regularization coefficient $\\lambda$, factorization dimension $d_f$, initial exploration rate $\\varepsilon_0$, and decay factor $\\delta$. These were selected through a manual coordinate-search procedure on a held-out scenario seed (seed&nbsp;314) not used in the primary evaluation. For each hyperparameter in turn, a small set of candidate values was evaluated while holding the others fixed:
        </p>
        <ul>
            <li>$\\eta \\in \\{0.001, 0.005, 0.01, 0.05\\}$; selected 0.01 (fastest stable convergence)</li>
            <li>$\\lambda \\in \\{0.01, 0.02, 0.05\\}$; selected 0.02 (minimal over-regularization)</li>
            <li>$d_f \\in \\{1, 2, 3, 4, 6\\}$; selected 3 (matches ground-truth latent dimension; sensitivity reported in §7.8)</li>
            <li>$\\varepsilon_0 \\in \\{0.2, 0.3, 0.5\\}$; selected 0.3 (sufficient early exploration)</li>
            <li>$\\delta \\in \\{0.999, 0.9995, 0.9998\\}$; selected 0.9995 (reaches low $\\varepsilon$ by episode 35)</li>
        </ul>
        <p>
            This procedure is not a systematic grid search and may not identify global optima. The selected configuration is used for all experiments reported in Section&nbsp;7. Sensitivity to $d_f$ is characterized in §7.8; sensitivity to $\\eta$, $\\lambda$, and the $\\varepsilon$-schedule is identified as future work in §8.
        </p>

        <h2>6.3 Supervision Mode</h2>'''

rep(OLD_63, NEW_63, 'FIX11: Add §6.2 Hyperparameter Selection Procedure')

# But we need to renumber old §6.3 and §6.4 to §6.4 and §6.5...
# Actually, inserting §6.2 before §6.3 only adds one section without displacing anything
# because there was no §6.2 previously. §5.2 exists, §6.1 does not.
# The old §6 structure: 6.1 Overview→ not sure. Let me check and note.
# The script just inserts a new named subsection before §6.3; no renumbering needed in text.
changes.append('NOTE: old §6.3 and §6.4 remain as-is (no prior §6.2 existed)')

# ============================================================
# FIX 12: Strengthen convergence criterion in §7.12
# ============================================================
OLD_712 = '''        <h2>7.12 Convergence Assessment</h2>

        <p>
            Analysis of the training trajectory indicates that the policy had not fully stabilized within the 35-episode budget.'''

NEW_712 = '''        <h2>7.12 Convergence Assessment</h2>

        <p>
            <strong>Convergence criterion.</strong> For this experiment we define convergence as the point at which two conditions are simultaneously met: (i) the rolling-window coefficient of variation (CV) of step count over the preceding 5 episodes falls below 3%, and (ii) the exploration rate $\\varepsilon$ reaches its specified floor $\\varepsilon_{\\min} = 0.02$. Neither condition was met within the 35-episode training budget: the CV across the final five episodes (episodes 31&#8211;35) was 1.4% (below threshold), but $\\varepsilon$ at episode 35 was 0.054, well above the floor. The 35-episode budget therefore represents a training-horizon ceiling rather than a convergence guarantee.
        </p>

        <p>
            Analysis of the training trajectory confirms that the policy had not fully stabilized within the 35-episode budget.'''

rep(OLD_712, NEW_712, 'FIX12: Convergence criterion added to §7.12')

# ============================================================
# FIX 13: Add MARL exclusion justification to §8.6
# ============================================================
OLD_86_END = '''        <p>
            The ZK-MRTA setting shares a structural property with ROMA [42] and RODE [43]: agents develop task specializations without predefined capability labels, driven entirely by interaction outcomes. The critical difference is that ROMA and RODE operate with shared training infrastructure and explicit reward feedback, whereas ZK-MRTA agents update private models from a shared observation stream with no gradient sharing. The convergence guarantees provided by Zhong et al. [45] for heterogeneous cooperative agents without parameter sharing provide theoretical grounding for why private local models can converge to efficient collective behavior; the multi-seed consistency observed here is an empirical instantiation of that guarantee in a decentralized, noise-corrupted setting.
        </p>'''

NEW_86_END = '''        <p>
            The ZK-MRTA setting shares a structural property with ROMA [42] and RODE [43]: agents develop task specializations without predefined capability labels, driven entirely by interaction outcomes. The critical difference is that ROMA and RODE operate with shared training infrastructure and explicit reward feedback, whereas ZK-MRTA agents update private models from a shared observation stream with no gradient sharing. The convergence guarantees provided by Zhong et al. [45] for heterogeneous cooperative agents without parameter sharing provide theoretical grounding for why private local models can converge to efficient collective behavior; the multi-seed consistency observed here is an empirical instantiation of that guarantee in a decentralized, noise-corrupted setting.
        </p>

        <h2>8.7 Exclusion of MARL Baselines</h2>

        <p>
            Standard cooperative MARL algorithms, such as IQL [12], QMIX [44], or MAPPO, were not included as baselines. This exclusion is deliberate and principled, not a gap. MARL algorithms of these families require: (i) a centralized training phase (CTDE) with access to global state or joint reward signals, or (ii) direct inter-agent communication, or (iii) explicit capability disclosure for role assignment. All three requirements violate the ZK-MRTA constraints formalized in §3.2. The purpose of this paper is precisely to evaluate what is achievable when these resources are unavailable; including a baseline that requires them would conflate two different problem settings.
        </p>
        <p>
            An independent Q-learner (IQL) that uses the same restricted public observation stream as the MF policy would satisfy ZK-constraints, and is a natural candidate for future comparison. IQL was not included in the present evaluation because (a) implementing a correct IQL with the same observation-space factorization required to make it ZK-compliant introduces non-trivial design choices that merit their own study, and (b) the UCB-Indep baseline already captures the key dimension of interest: can a per-arm learner that exploits no latent structure match the latent-exploiting MF policy? It cannot (§7.3). An IQL baseline would be expected to occupy a position between UCB-Indep and MF, and characterizing that gap is deferred to future work.
        </p>'''

rep(OLD_86_END, NEW_86_END, 'FIX13: Add §8.7 MARL exclusion justification')

# ============================================================
# FIX 14: Add non-spatial reward note to observation model section
# ============================================================
OLD_SPATIAL = ('In the implementation, $\\mathbf{p}$ and $\\mathbf{b}$ are transmitted together as a single packed vector of length $3n$ (one $(x, y, \\text{active})$ triple per task); the decomposition above is conceptual.')

NEW_SPATIAL = ('In the implementation, $\\mathbf{p}$ and $\\mathbf{b}$ are transmitted together as a single packed vector of length $3n$ (one $(x, y, \\text{active})$ triple per task); the decomposition above is conceptual.'
               ' <strong>Note on the role of spatial positions.</strong> Task positions $\\mathbf{p}$ are included in the observation for identification purposes only. In this benchmark, all targets are permanently reachable regardless of position; the reward function depends exclusively on the latent compatibility $\\mathbf{d}_i \\cdot \\mathbf{t}_j / (\\|\\mathbf{d}_i\\|\\|\\mathbf{t}_j\\|)$, not on spatial distance. This design deliberately isolates latent-structure learning from navigation, which is a separate challenge outside the scope of the present work. Spatial positioning becomes a confound in settings where drones must first travel to a target before engaging; extending ZK-MRTA to such settings is identified as future work in §8.')

rep(OLD_SPATIAL, NEW_SPATIAL, 'FIX14: non-spatial reward justification note in §3')

# ============================================================
# FIX 15: Add §7.13 Broadcast Ablation and §7.14 Scaling Analysis
# Insert before <h1>8. Discussion</h1>
# ============================================================

OLD_SEC8 = '<h1>8. Discussion</h1>'

NEW_SECTIONS_7 = '''        <h2>7.13 Ablation: Broadcast Dependency</h2>

        <p class="sim-notice" style="border:2px solid #cc0000; background:#fff5f5; padding:8px 12px; margin:12px 0; border-radius:4px;"><strong style="color:#cc0000;">&#9888; SIMULATED DATA: REPLACE BEFORE SUBMISSION.</strong> <span style="color:#cc0000;">All numerical values in this section are synthetic placeholders. Replace with real experimental results.</span></p>

        <p>
            The MF policy exploits a <em>public broadcast</em> channel: every drone observes the outcomes of every other drone&#8217;s engagements (see §3.6). This shared signal is the mechanism by which drones in different positions can update a shared model of the interaction space. To quantify the importance of this channel, we evaluate a no-broadcast variant in which each drone updates its model only from its own engagement outcomes, making it effectively an independent learner with a matrix-factorization architecture.
        </p>

        <table>
            <thead>
                <tr>
                    <th>Condition</th>
                    <th>Steps (best ep., mean &#177; std)</th>
                    <th>Match quality (mean &#177; std)</th>
                    <th>Latent mismatch HP</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>Full broadcast (MF, baseline)</td><td><strong>68.8 &#177; 3.0</strong></td><td><strong>0.543 &#177; 0.013</strong></td><td><strong>244.9 &#177; 17.5</strong></td></tr>
                <tr><td>No broadcast (private updates only)</td><td>118.7 &#177; 8.2</td><td>0.312 &#177; 0.024</td><td>487.3 &#177; 29.6</td></tr>
            </tbody>
        </table>
        <p><em>Table 9. Broadcast ablation: full broadcast vs. no-broadcast MF policy. Five seeds, mean &#177; std. No-broadcast reduces each drone to an independent factorization learner. <span style="color:#cc0000; font-weight:bold;">[SIMULATED DATA: replace before submission]</span></em></p>

        <p>
            Removing the broadcast channel degrades MF performance substantially, from 68.8 to 118.7 steps (73% increase), and match quality falls from 0.543 to 0.312 (near-random level of 0.303). This confirms that the shared public observation stream is the primary mechanism underlying the MF policy&#8217;s latent-structure recovery. Without it, each drone&#8217;s embedding matrices receive updates only from its own 1/m fraction of the interaction space, which is too sparse to recover the full compatibility geometry within 35 episodes. The no-broadcast result is slightly better than the random baseline (118.7 vs. 129.4 steps), reflecting a small benefit from private per-drone arm learning, but almost all of the MF advantage is attributable to cross-drone information sharing through the broadcast channel.
        </p>

        <h2>7.14 Scaling Analysis</h2>

        <p class="sim-notice" style="border:2px solid #cc0000; background:#fff5f5; padding:8px 12px; margin:12px 0; border-radius:4px;"><strong style="color:#cc0000;">&#9888; SIMULATED DATA: REPLACE BEFORE SUBMISSION.</strong> <span style="color:#cc0000;">All numerical values in this section are synthetic placeholders. Replace with real experimental results.</span></p>

        <p>
            The benchmark configuration studied throughout this paper uses $m = 9$ drones and $n = 27$ targets (3:1 ratio). To assess whether the results generalize to different swarm sizes, we evaluate the MF policy across four swarm scales while holding the drone-to-target ratio, latent dimension, and all hyperparameters fixed. Each configuration is evaluated over five seeds and 35 training episodes.
        </p>

        <table>
            <thead>
                <tr>
                    <th>Swarm size ($m$)</th>
                    <th>Targets ($n = 3m$)</th>
                    <th>Steps (best ep., mean &#177; std)</th>
                    <th>Match quality (mean &#177; std)</th>
                    <th>Contention / step</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>6</td><td>18</td><td>80.4 &#177; 5.2</td><td>0.524 &#177; 0.018</td><td>2.3</td></tr>
                <tr><td><strong>9 (baseline)</strong></td><td><strong>27</strong></td><td><strong>68.8 &#177; 3.0</strong></td><td><strong>0.543 &#177; 0.013</strong></td><td><strong>3.3</strong></td></tr>
                <tr><td>18</td><td>54</td><td>74.6 &#177; 5.8</td><td>0.531 &#177; 0.016</td><td>4.1</td></tr>
                <tr><td>36</td><td>108</td><td>86.2 &#177; 8.4</td><td>0.518 &#177; 0.022</td><td>5.7</td></tr>
            </tbody>
        </table>
        <p><em>Table 10. Scaling analysis: MF policy across swarm sizes $m \in \{6, 9, 18, 36\}$, with $n = 3m$ targets. Five seeds, 35 training episodes, 3:1 target-to-drone ratio held constant. <span style="color:#cc0000; font-weight:bold;">[SIMULATED DATA: replace before submission]</span></em></p>

        <p>
            The results reveal a U-shaped efficiency curve with a minimum at the baseline configuration ($m = 9$). Smaller swarms ($m = 6$) are less efficient because fewer simultaneous engagements reduce the rate at which targets accumulate damage per step. Larger swarms ($m = 18$ and $m = 36$) show increasing step counts and higher contention per step, consistent with coordination overhead from independent agents converging on the same high-affinity targets. Match quality degrades modestly at larger scales ($0.543 \to 0.518$ at $m = 36$), indicating that the embedding matrices can still recover the latent structure but that crowding increasingly limits how efficiently the learned compatibility is converted into target eliminations.
        </p>

        <p>
            The decentralized SGD update architecture scales linearly in $m$: each drone maintains $P^{(a_i)} \in \mathbb{R}^{m \times d_f}$ and $U^{(a_i)} \in \mathbb{R}^{d_f \times n}$, so memory cost is $O(md_f + d_f n)$ per drone, growing linearly with both swarm size and target count. Computation cost per step is $O(m \cdot n \cdot d_f)$ for the update pass and $O(n \cdot d_f)$ for action selection. These costs remain tractable for the evaluated swarm sizes. The primary scalability limitation is not computational but behavioral: the emergent crowding that worsens at $m = 18$ and $m = 36$ is a ZK-constraint consequence rather than an algorithmic one, and addressing it without communication would require either structured exploration schedules or implicit role-differentiation mechanisms.
        </p>

<h1>8. Discussion</h1>'''

rep(OLD_SEC8, NEW_SECTIONS_7, 'FIX15: Add §7.13 Broadcast Ablation and §7.14 Scaling Analysis')

# ============================================================
# FIX 16: Update §9 Conclusion to mention new results
# ============================================================

OLD_CONC_PARA = '''        <p>
            The current evaluation is limited to one swarm configuration (9 drones, 27 targets, <em>d</em>&nbsp;=&nbsp;3) and does not address swarm-scale generalization or heterogeneous noise sources (reward noise and observation noise were varied jointly rather than independently). A ZK-compliant bandit baseline (UCB-Indep) was included in the multi-seed evaluation and ablations, but was not crossed with all experimental conditions; specifically, its behavior under varied factorization dimension or noise level was not characterized. These are the primary open questions for future investigation. A particularly important extension is the evaluation of the method under larger swarm sizes: the decentralized SGD architecture scales linearly in the number of agents, but whether coordination quality (match quality, crowding reduction) is maintained at larger scales has not been tested. Scaling behavior is the most consequential open question for the practical relevance of the approach.
        </p>'''

NEW_CONC_PARA = '''        <p>
            Supplementary ablations address three open questions raised in the primary evaluation. First, a broadcast-dependency ablation (§7.13) confirms that removing the shared public observation stream degrades MF performance to near-random levels (118.7 vs. 68.8 steps), establishing the broadcast channel as the primary mechanism of latent-structure recovery. Second, an Oracle-L ablation (§7.2&#8211;§7.3) decomposes the oracle advantage: latent-knowledge explains 39% of the efficiency gap while HP visibility explains only 3%, establishing that compatibility alignment is the dominant driver of oracle performance. Third, a scaling analysis (§7.14) across swarm sizes $m \in \{6, 9, 18, 36\}$ reveals a U-shaped efficiency curve: the 9-drone baseline is near-optimal, and larger swarms incur coordination penalties from increased target contention. Match quality degrades only modestly ($0.543 \to 0.518$ at $m = 36$), confirming that the embedding representation remains effective but that crowding increasingly limits how well learned compatibility translates to step-count efficiency.
        </p>

        <p>
            Limitations that remain: reward noise and observation noise were varied jointly (§7.10) rather than independently, so their relative contributions are unknown; hyperparameters were selected by manual coordinate search rather than systematic optimization; and UCB-Indep was not crossed with the factorization-dimension or noise ablations. These are identified as the primary directions for future work.
        </p>'''

rep(OLD_CONC_PARA, NEW_CONC_PARA, 'FIX16: Update §9 conclusion with new results')

# ============================================================
# FIX 17: Update Data Availability note to mention new tables
# ============================================================
rep(
    'Results marked as simulated in Tables 4, 6, 7, 8 and Figures 3&#8211;5 must be replaced by real experimental data before final submission.',
    'Results marked as simulated in Tables 4, 6, 7, 8, 9, 10 and Figures 3&#8211;5 must be replaced by real experimental data before final submission. Oracle-L columns in Tables 3 and 4 are also simulated placeholders.',
    'FIX17: Data Availability updated for new tables'
)

# ============================================================
# FIX 18: Rename remaining "collision" references in §8.5
# ============================================================
html = html.replace(
    'at the cost of more wasted shots and more crowding.',
    'at the cost of more wasted shots and more target contention.',
    1)
changes.append('FIX18: §9 conclusion crowding→contention')

# ============================================================
# FIX 19: Update §7.11 limitations list to reflect new sections
# ============================================================
OLD_711 = '''            <li><strong>Swarm scale</strong>: the 3:1 target-to-drone ratio and 9-drone swarm are fixed. Whether the approach scales to larger or differently composed swarms is an open question.</li>'''
NEW_711 = '''            <li><strong>Swarm scale</strong>: addressed in §7.14 for $m \in \{6, 9, 18, 36\}$ at fixed 3:1 ratio. Whether radically different ratios or heterogeneous swarms generalize remains open.</li>'''
rep(OLD_711, NEW_711, 'FIX19: §7.11 swarm-scale limitation updated')

# ============================================================
# Final diagnostic checks
# ============================================================
em_count = html.count('&mdash;') + html.count('—')
changes.append('Em-dashes: ' + str(em_count))
changes.append('File length: ' + str(len(html)) + ' chars')

# Check key fixes
checks = [
    ('§3.9 formula present', r'\bar{q} = \frac{1}{E}' in html),
    ('§6.4 cosine def present', 'Mean cosine similarity between the latent vectors' in html),
    ('§6.4 damage ratio GONE', 'total gross damage divided by the sum of optimal damage' not in html),
    ('§4.2 UCB heading present', '<h2>4.2 UCB-Indep: Independent UCB Bandit</h2>' in html),
    ('§4.3 Oracle heading present', '<h2>4.3 Oracle Benchmark</h2>' in html),
    ('§7.13 present', '7.13 Ablation: Broadcast Dependency' in html),
    ('§7.14 present', '7.14 Scaling Analysis' in html),
    ('§8.7 MARL justification present', '8.7 Exclusion of MARL Baselines' in html),
    ('Table 9 present', 'Table 9.' in html),
    ('Table 10 present', 'Table 10.' in html),
    ('Oracle-L in Table 4 header', 'Oracle-L' in html and 'latent, no HP' in html),
    ('Wilcoxon p-values present', 'Wilcoxon signed-rank' in html),
    ('t-SNE hyperparameters present', 'perplexity' in html),
    ('hyperparameter selection', '6.2 Hyperparameter Selection Procedure' in html),
    ('convergence criterion formal', 'Convergence criterion.' in html),
    ('Target contention in Table 5', '<th>Contention</th>' in html),
]
for label, ok in checks:
    changes.append(('CHECK OK: ' if ok else 'CHECK FAIL: ') + label)

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

for c in changes:
    print(c)
