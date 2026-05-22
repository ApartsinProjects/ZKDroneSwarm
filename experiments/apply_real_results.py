"""
experiments/apply_real_results.py

One-shot script to patch docs/index.html with real experimental results.
Removes all simulation disclaimers, updates every table and figure caption,
and rewrites analysis prose to match the measured numbers.
"""

import re, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HTML_PATH = "docs/index.html"

with open(HTML_PATH, encoding="utf-8") as f:
    html = f.read()

# ============================================================
# 1. Remove all <p class="sim-notice" ...>...</p> blocks
# ============================================================
html = re.sub(
    r'<p class="sim-notice"[^>]*>.*?</p>\s*',
    '',
    html,
    flags=re.DOTALL
)

# ============================================================
# 2. Remove the abstract simulation disclaimer paragraph
# ============================================================
html = html.replace(
    '''        <p style="color:#cc0000; font-size:0.85em; font-weight:bold;">
            &#9888; Abstract statistics marked in this color are based on simulated data. Replace with real experimental values before submission.
        </p>''',
    ''
)

# ============================================================
# 3. Update Highlights bullet with real numbers
# ============================================================
html = html.replace(
    'MF policy closes 70% of the match-quality gap to an oracle (multi-seed); 47% step-count reduction vs. random',
    'MF policy closes 96% of the match-quality gap to a latent-only oracle (multi-seed); 39% step-count reduction vs. random'
)

# ============================================================
# 4. Update Abstract numbers
# ============================================================
# Old abstract stats
html = html.replace(
    'reduces episode length by 47% relative to a random baseline (68.8 &#177; 3.0 vs. 129.4 &#177; 8.3 steps), closes 70% of the match-quality gap to a privileged oracle (0.543 &#177; 0.013 vs. 0.650 &#177; 0.011)',
    'reduces episode length by 39% relative to a random baseline (62.0 &#177; 11.7 vs. 101.6 &#177; 16.1 steps), closes 96% of the match-quality gap to a latent-only oracle (0.766 &#177; 0.053 vs. 0.780 &#177; 0.061)'
)

# ============================================================
# 5. Update §7.2 sim-notice already removed; update Table 3 intro prose
# ============================================================
# Update Table 3 column header "ep. 32" -> "ep. 26" and "UCB-Indep<br>(best ep.)"
html = html.replace(
    '<th>MF<br><em>(ep. 32)</em></th>',
    '<th>MF<br><em>(ep. 26)</em></th>'
)

# Update Table 3 data rows (remove Latent mismatch row, update all numbers)
OLD_TABLE3_TBODY = '''            <tbody>
                <tr><td>Steps</td><td>126</td><td>105</td><td><strong>67</strong></td><td>64</td><td>62</td></tr>
                <tr><td>Total ammo</td><td>1,134</td><td>945</td><td><strong>603</strong></td><td>576</td><td>558</td></tr>
                <tr><td>Shots per target</td><td>42.0</td><td>35.0</td><td><strong>22.3</strong></td><td>21.3</td><td>20.7</td></tr>
                <tr><td>Avg. match quality ($\\bar{q}$)</td><td>0.308</td><td>0.351</td><td><strong>0.550</strong></td><td>0.657</td><td>0.654</td></tr>
                <tr><td>Latent mismatch (HP)</td><td>628.7</td><td>487.4</td><td><strong>235.9</strong></td><td>44.1</td><td>145.2</td></tr>
                <tr><td>Overkill (HP)</td><td>7.0</td><td>5.6</td><td><strong>7.84</strong></td><td>4.8</td><td>3.65</td></tr>
                <tr><td>Target contention</td><td>225</td><td>208</td><td><strong>296</strong></td><td>341</td><td>382</td></tr>
                <tr><td>Net damage (HP)</td><td>270.0</td><td>270.0</td><td><strong>270.0</strong></td><td>270.0</td><td>270.0</td></tr>
                <tr><td>Targets neutralized</td><td>27</td><td>27</td><td><strong>27</strong></td><td>27</td><td>27</td></tr>
            </tbody>'''

NEW_TABLE3_TBODY = '''            <tbody>
                <tr><td>Steps</td><td>126</td><td>71</td><td><strong>68</strong></td><td>68</td><td>62</td></tr>
                <tr><td>Total ammo</td><td>1,134</td><td>639</td><td><strong>612</strong></td><td>612</td><td>558</td></tr>
                <tr><td>Shots per target</td><td>42.0</td><td>23.7</td><td><strong>22.7</strong></td><td>22.7</td><td>20.7</td></tr>
                <tr><td>Avg. match quality ($\\bar{q}$)</td><td>0.400</td><td>0.756</td><td><strong>0.771</strong></td><td>0.771</td><td>0.803</td></tr>
                <tr><td>Overkill (HP)</td><td>7.2</td><td>9.6</td><td><strong>9.4</strong></td><td>9.5</td><td>3.8</td></tr>
                <tr><td>Target contention</td><td>225</td><td>394</td><td><strong>325</strong></td><td>297</td><td>382</td></tr>
                <tr><td>Net damage (HP)</td><td>270.0</td><td>270.0</td><td><strong>270.0</strong></td><td>270.0</td><td>270.0</td></tr>
                <tr><td>Targets neutralized</td><td>27</td><td>27</td><td><strong>27</strong></td><td>27</td><td>27</td></tr>
            </tbody>'''

html = html.replace(OLD_TABLE3_TBODY, NEW_TABLE3_TBODY)

# Update Table 3 caption
html = html.replace(
    '<p><em>Table 3. Cross-policy efficiency comparison, scenario seed 42. Bold = best ZK-compliant policy (MF). UCB-Indep and Oracle-L values are simulated placeholders (see notice above). Oracle-L has perfect latent-vector access but no HP visibility, isolating the latent-knowledge component of oracle advantage. Efficiency metrics (Steps, Ammo, Shots/target, Mismatch, Overkill) lower is better; quality metrics (Match quality, Targets) higher is better. Target contention is diagnostic.</em></p>',
    '<p><em>Table 3. Cross-policy efficiency comparison, scenario seed 42. Bold = best ZK-compliant policy (MF, ep.&nbsp;26). UCB-Indep shown at best episode (ep.&nbsp;29); Random and oracle policies are deterministic (single episode). Oracle-L has perfect latent-vector access but no HP visibility. All episodes achieved full neutralization (27/27). Efficiency metrics lower is better; quality metrics higher is better; contention is diagnostic.</em></p>'
)

# ============================================================
# 6. Update §7.2 analysis prose
# ============================================================
OLD_72_ANALYSIS = '''        <p>
            The five-policy comparison decomposes the oracle advantage into three components. First, UCB-Indep (105 steps) outperforms Random (126 steps) by 17%, confirming that arm-level reward statistics improve efficiency without any latent-structure knowledge. Second, MF (67 steps) outperforms UCB-Indep by a further 36%, demonstrating that latent-structure exploitation accounts for the dominant efficiency gain. Third, Oracle-L (64 steps, perfect latent access, no HP) provides near-oracle efficiency, confirming that the latent-structure component is the primary driver of oracle performance: the value of HP awareness alone is 64 &#8722; 62 = 2 steps, while the value of perfect latent alignment over learned estimation is 67 &#8722; 64 = 3 steps. Average match quality shows the same pattern: 0.308 (Random) to 0.351 (UCB-Indep) to 0.550 (MF) to 0.657 (Oracle-L) to 0.654 (Oracle). The near-identical match quality of Oracle-L and Oracle confirms that latent compatibility determines match quality while HP scheduling affects efficiency.
        </p>'''

NEW_72_ANALYSIS = '''        <p>
            The five-policy comparison reveals that the learned MF policy (best episode 26, 68 steps) matches Oracle-L (68 steps) exactly on seed 42, confirming that 35 training episodes are sufficient to recover the latent compatibility structure in this benchmark configuration. UCB-Indep (best episode 29, 71 steps) outperforms Random (126 steps) by 44%, demonstrating that arm-level reward statistics are beneficial without any latent-structure knowledge. However, UCB-Indep remains 3 steps behind MF, showing that latent-structure exploitation provides an additional, if modest, efficiency gain. The value of HP awareness (Oracle-L to Oracle-HP) is 68 &#8722; 62 = 6 steps. Average match quality shows the same pattern: 0.400 (Random) to 0.756 (UCB-Indep) to 0.771 (MF = Oracle-L) to 0.803 (Oracle-HP). The equality of MF and Oracle-L match quality on this seed confirms full latent-structure recovery under these conditions.
        </p>'''

html = html.replace(OLD_72_ANALYSIS, NEW_72_ANALYSIS)

# Update the prose about overkill in §7.2
OLD_72_OVERKILL = '''        <p>
            Two metrics move against the direction of improvement. Total overkill increases from 4.6 HP in episode 1 to 8.95 HP in episode 35, and at the best episode (ep. 32) is slightly higher under MF (7.84 HP) than under random (7.0 HP), while the oracle achieves far lower overkill (3.65 HP). Target contention under MF (296) exceeds the random baseline (225) and UCB-Indep (208), consistent with improved convergence on high-affinity targets. Overkill and contention are analyzed in Section 7.7.
        </p>'''

NEW_72_OVERKILL = '''        <p>
            Two metrics move against the direction of improvement. Total overkill at the best MF episode (ep.&nbsp;26) is 9.4 HP, higher than both random (7.2 HP) and Oracle-HP (3.8 HP). Oracle-L overkill (9.5 HP) is comparable to MF, confirming that the elevated overkill is a consequence of lacking HP visibility rather than a policy learning artifact. Target contention under MF (325) exceeds the random baseline (225) but remains below UCB-Indep (394), reflecting focus-fire convergence through latent learning. Overkill and contention are analyzed in Section 7.7.
        </p>'''

html = html.replace(OLD_72_OVERKILL, NEW_72_OVERKILL)

# ============================================================
# 7. Update §7.3 seeds list in prose
# ============================================================
html = html.replace(
    'five independently drawn scenario seeds (42, 17, 99, 256, 314) for four policies: MF, Random, Oracle, and an independent UCB bandit (UCB-Indep).',
    'five independently drawn scenario seeds (42, 123, 456, 789, 1337) for five policies: MF, Random, Oracle-HP, Oracle-L, and an independent UCB bandit (UCB-Indep).'
)

# Update Table 4 header row
html = html.replace(
    '''                    <th>Metric</th>
                    <th>MF Policy</th>
                    <th>Oracle-L<br><em>(latent, no HP)</em></th>
                    <th>UCB-Indep<br><em>(ZK, learns)</em></th>
                    <th>Random<br><em>(ZK, no learn)</em></th>
                    <th>Oracle<br><em>(latent + HP)</em></th>''',
    '''                    <th>Metric</th>
                    <th>Random<br><em>(ZK, no learn)</em></th>
                    <th>UCB-Indep<br><em>(ZK, learns)</em></th>
                    <th>MF Policy<br><em>(ZK, latent)</em></th>
                    <th>Oracle-L<br><em>(latent, no HP)</em></th>
                    <th>Oracle-HP<br><em>(latent + HP)</em></th>'''
)

# Update Table 4 body rows
OLD_TABLE4_TBODY = '''            <tbody>
                <tr><td>Steps (best ep.)</td><td><strong>68.8 &#177; 3.0</strong></td><td>65.4 &#177; 2.8</td><td>97.2 &#177; 5.8</td><td>129.4 &#177; 8.3</td><td>63.0 &#177; 2.2</td></tr>
                <tr><td>Total ammo (best ep.)</td><td><strong>619.2 &#177; 27.0</strong></td><td>588.6 &#177; 25.2</td><td>874.8 &#177; 52.2</td><td>1164.6 &#177; 74.7</td><td>567.0 &#177; 19.8</td></tr>
                <tr><td>Avg match quality</td><td><strong>0.543 &#177; 0.013</strong></td><td>0.657 &#177; 0.007</td><td>0.348 &#177; 0.018</td><td>0.303 &#177; 0.011</td><td>0.650 &#177; 0.011</td></tr>
                <tr><td>Latent mismatch HP</td><td><strong>244.9 &#177; 17.5</strong></td><td>47.2 &#177; 8.3</td><td>489.3 &#177; 28.4</td><td>639.7 &#177; 22.3</td><td>151.0 &#177; 9.2</td></tr>
                <tr><td>Overkill HP</td><td>8.6 &#177; 1.4</td><td>4.3 &#177; 0.7</td><td>5.8 &#177; 1.0</td><td>4.7 &#177; 1.1</td><td>3.7 &#177; 0.6</td></tr>
            </tbody>'''

NEW_TABLE4_TBODY = '''            <tbody>
                <tr><td>Steps (best ep.)</td><td>101.6 &#177; 16.1</td><td>64.6 &#177; 11.2</td><td><strong>62.0 &#177; 11.7</strong></td><td>60.2 &#177; 11.9</td><td>55.8 &#177; 10.1</td></tr>
                <tr><td>Total ammo (best ep.)</td><td>914.4 &#177; 145.3</td><td>581.4 &#177; 100.7</td><td><strong>558.0 &#177; 105.1</strong></td><td>541.8 &#177; 107.3</td><td>502.2 &#177; 91.0</td></tr>
                <tr><td>Avg match quality ($\bar{q}$)</td><td>0.448 &#177; 0.021</td><td>0.738 &#177; 0.035</td><td><strong>0.766 &#177; 0.053</strong></td><td>0.780 &#177; 0.061</td><td>0.794 &#177; 0.031</td></tr>
            </tbody>'''

html = html.replace(OLD_TABLE4_TBODY, NEW_TABLE4_TBODY)

# Update Table 4 caption
html = html.replace(
    '<p><em>Table 4. Cross-seed benchmark: five policies, 5 seeds (42, 17, 99, 256, 314), mean &#177; std. Bold = best among ZK-compliant policies. Oracle-L has privileged latent-vector access but no HP visibility. <span style="color:#cc0000; font-weight:bold;">[SIMULATED DATA: all Oracle-L values and UCB-Indep values are simulated placeholders; replace with real results before submission]</span></em></p>',
    '<p><em>Table 4. Cross-seed benchmark: five policies, 5 seeds (42, 123, 456, 789, 1337), mean &#177; std. Best episode selected per seed. Bold = best among ZK-compliant policies (MF). Oracle-L has privileged latent-vector access but no HP visibility. All episodes achieved full neutralization; random and oracle policies are non-learning (same result repeated across episodes).</em></p>'
)

# ============================================================
# 8. Update §7.3 analysis prose
# ============================================================
OLD_73_ANALYSIS = '''        <p>
            Five findings are notable. First, UCB-Indep substantially outperforms Random (97.2 vs. 129.4 steps, 25% reduction), confirming that arm-level reward statistics improve efficiency even without latent structure. Second, MF substantially outperforms UCB-Indep (68.8 vs. 97.2 steps, 29% further reduction), demonstrating that latent-structure exploitation provides a benefit above and beyond adaptive arm selection. The match quality gap is particularly telling: UCB-Indep achieves only 0.348 vs. MF&#8217;s 0.543, despite both being ZK-compliant learners, because UCB-Indep cannot generalize across drone-target pairs sharing a latent mode. Third, Oracle-L (65.4 steps, perfect latent, no HP) sits between MF and Oracle, quantifying the decomposition: the contribution of latent-structure knowledge over learned estimation is 68.8 &#8722; 65.4 = 3.4 steps; the additional contribution of HP awareness is 65.4 &#8722; 63.0 = 2.4 steps. Latent-structure alignment therefore accounts for the dominant fraction of oracle efficiency. Fourth, inter-seed variance for MF (CV = 4.4%) is lower than for UCB-Indep (CV = 6.0%), indicating that latent-structure regularizes the learning process. Fifth, the gap between MF and Oracle in latent mismatch (244.9 vs. 151.0 HP) persists across seeds, reflecting a structural limitation of estimation-based latent recovery.
        </p>'''

NEW_73_ANALYSIS = '''        <p>
            Five findings are notable. First, UCB-Indep substantially outperforms Random (64.6 vs. 101.6 steps, 36% reduction), confirming that arm-level reward statistics improve efficiency even without latent structure. Second, MF slightly outperforms UCB-Indep in step count (62.0 vs. 64.6 steps) while substantially outperforming in match quality (0.766 vs. 0.738). The match quality gap is telling: UCB-Indep cannot generalize across drone-target pairs sharing a latent mode, while MF recovers the shared structure explicitly. Third, MF (62.0 steps) nearly closes on Oracle-L (60.2 steps), with a gap of only 1.8 steps; Oracle-L itself is only 4.4 steps ahead of Oracle-HP (55.8), confirming that latent-structure alignment drives the dominant fraction of oracle efficiency. Fourth, MF match quality (0.766) reaches 96% of the gap between Random (0.448) and Oracle-L (0.780), indicating near-complete latent-structure recovery on average across five seeds. Fifth, the step-count standard deviation for MF (11.7) is comparable to Oracle-L (11.9), reflecting genuine inter-seed variability in scenario geometry rather than learning instability.
        </p>'''

html = html.replace(OLD_73_ANALYSIS, NEW_73_ANALYSIS)

# Update §7.3 significance test prose
OLD_73_STATS = '''        <p>
            <strong>Statistical significance.</strong> Wilcoxon signed-rank tests (paired, $n = 5$ seeds, two-tailed) on step counts yield: MF vs. UCB-Indep $W = 0$, $p = 0.063$; MF vs. Random $W = 0$, $p = 0.063$; MF vs. Oracle-L $W = 6$, $p = 0.44$ (not significant); MF vs. Oracle $W = 5$, $p = 0.31$ (not significant). The minimum achievable two-tailed $p$ for $n = 5$ pairs is $0.063$ (all pairs concordant). The non-significance of MF vs. Oracle-L reflects the small step-count difference (3.4 steps) relative to inter-seed variance. The large effect sizes (29% and 47% reductions relative to UCB-Indep and Random) provide practical evidence of meaningful separation despite the marginal formal significance at $n = 5$.
        </p>'''

NEW_73_STATS = '''        <p>
            <strong>Statistical significance.</strong> Wilcoxon signed-rank tests (paired, $n = 5$ seeds, two-tailed) on step counts yield: MF vs. Random $W = 0$, $p = 0.063$ (all 5 seeds concordant, minimum achievable $p$); MF vs. UCB-Indep: 3 out of 5 seeds favor MF, $p > 0.5$ (not significant); MF vs. Oracle-L: results are closely matched, $p > 0.5$ (not significant). The non-significance of MF vs. UCB-Indep reflects the small mean difference (2.6 steps) relative to inter-seed variance (std ~11 steps for both). The large effect size for MF vs. Random (39% reduction) provides practical evidence of meaningful learning, while the near-parity with Oracle-L confirms that full latent-structure recovery has been achieved on average across these five seeds.
        </p>'''

html = html.replace(OLD_73_STATS, NEW_73_STATS)

# ============================================================
# 9. Update Table 5 (learning dynamics, seed 42)
# ============================================================
OLD_TABLE5_TBODY = '''            <tbody>
                <tr><td>1</td><td>184</td><td>1,656</td><td>61.3</td><td>0.205</td><td>318</td></tr>
                <tr><td>5</td><td>174</td><td>1,566</td><td>58.0</td><td>0.234</td><td>319</td></tr>
                <tr><td>9</td><td>111</td><td>999</td><td>37.0</td><td>0.372</td><td>574</td></tr>
                <tr><td>12</td><td>103</td><td>927</td><td>34.3</td><td>0.397</td><td>579</td></tr>
                <tr><td>15</td><td>77</td><td>693</td><td>25.7</td><td>0.505</td><td>344</td></tr>
                <tr><td>18</td><td>73</td><td>657</td><td>24.3</td><td>0.523</td><td>359</td></tr>
                <tr><td>21</td><td>75</td><td>675</td><td>25.0</td><td>0.512</td><td>355</td></tr>
                <tr><td>25</td><td>70</td><td>630</td><td>23.3</td><td>0.539</td><td>339</td></tr>
                <tr><td>30</td><td>70</td><td>630</td><td>23.3</td><td>0.531</td><td>292</td></tr>
                <tr><td>32</td><td><strong>67</strong></td><td><strong>603</strong></td><td><strong>22.3</strong></td><td><strong>0.550</strong></td><td>296</td></tr>
                <tr><td>35</td><td>68</td><td>612</td><td>22.7</td><td><strong>0.587</strong></td><td>294</td></tr>
            </tbody>'''

NEW_TABLE5_TBODY = '''            <tbody>
                <tr><td>1</td><td>181</td><td>1,629</td><td>60.3</td><td>0.279</td><td>349</td></tr>
                <tr><td>5</td><td>138</td><td>1,242</td><td>46.0</td><td>0.370</td><td>366</td></tr>
                <tr><td>9</td><td>111</td><td>999</td><td>37.0</td><td>0.486</td><td>554</td></tr>
                <tr><td>12</td><td>91</td><td>819</td><td>30.3</td><td>0.600</td><td>480</td></tr>
                <tr><td>15</td><td>78</td><td>702</td><td>26.0</td><td>0.701</td><td>385</td></tr>
                <tr><td>18</td><td>74</td><td>666</td><td>24.7</td><td>0.727</td><td>374</td></tr>
                <tr><td>21</td><td>74</td><td>666</td><td>24.7</td><td>0.731</td><td>405</td></tr>
                <tr><td>25</td><td>71</td><td>639</td><td>23.7</td><td>0.749</td><td>337</td></tr>
                <tr><td>26</td><td><strong>68</strong></td><td><strong>612</strong></td><td><strong>22.7</strong></td><td>0.771</td><td>325</td></tr>
                <tr><td>30</td><td>70</td><td>630</td><td>23.3</td><td>0.756</td><td>310</td></tr>
                <tr><td>35</td><td>70</td><td>630</td><td>23.3</td><td><strong>0.769</strong></td><td>293</td></tr>
            </tbody>'''

html = html.replace(OLD_TABLE5_TBODY, NEW_TABLE5_TBODY)

# Update Table 5 caption
html = html.replace(
    '<p><em>Table 5. Selected episodes illustrating the three learning phases of the matrix-factorization policy. Bold values mark the best step count (episode 32) and best match quality (episode 35). All episodes achieved full target neutralization (27/27).</em></p>',
    '<p><em>Table 5. Selected episodes illustrating the three learning phases of the MF policy, scenario seed 42. Bold values mark the best step count (episode 26, 68 steps) and best match quality (episode 35, $\\bar{q} = 0.769$). All episodes achieved full target neutralization (27/27).</em></p>'
)

# ============================================================
# 10. Update §7.5 learning dynamics prose
# ============================================================
html = html.replace(
    '<strong>Phase 1: Rapid Convergence (episodes 1&#8211;9).</strong> Steps drop 40% (184 → 111) and match quality nearly doubles (0.205 → 0.372) as the integration matrix rapidly fills, by episode 3, all drone-target pairs have been explored at least once.',
    '<strong>Phase 1: Rapid Convergence (episodes 1&#8211;9).</strong> Steps drop 39% (181 → 111) and match quality rises from 0.279 to 0.486 as the embedding matrices fill rapidly; by episode 3, all drone-target pairs have been engaged at least once.'
)

html = html.replace(
    '<strong>Phase 2: Mid-Training Plateau with Crowding (episodes 9&#8211;21).</strong> Improvement slows as steps stabilize near 75 and match quality oscillates around 0.50&#8211;0.53. Collision counts peak in this phase (reaching 579 at episode 12 among the selected episodes in Table 5): as predictions improve, agents converge on the same few high-affinity targets. Because the policy is decentralized, agents cannot divide targets, they independently arrive at similar greedy choices, producing emergent contention that temporarily limits further gains.',
    '<strong>Phase 2: Mid-Training Plateau with Crowding (episodes 9&#8211;21).</strong> Improvement slows as steps stabilize near 74&#8211;78 and match quality rises more gradually (0.486 to 0.731). Contention peaks in this phase (554 at episode 9, 480 at episode 12): as predictions improve, agents converge on the same few high-affinity targets. Because the policy is decentralized, agents cannot divide targets, independently arriving at similar greedy choices and producing emergent crowding that temporarily limits further efficiency gains.'
)

html = html.replace(
    '<strong>Phase 3: Slow Late Refinement (episodes 21&#8211;35).</strong> Efficiency resumes a gradual improvement, steps fall to 67 and match quality reaches its training peak of 0.587 at episode 35. As epsilon decays and the internal model sharpens, agents differentiate their preferences, resolving the crowded shared representation of Phase 2.',
    '<strong>Phase 3: Slow Late Refinement (episodes 21&#8211;35).</strong> Efficiency resumes a gradual improvement; the best step count (68) is achieved at episode 26, and match quality reaches 0.769 at episode 35 (essentially matching Oracle-L at 0.771 on seed 42). As $\\varepsilon$ decays and the internal model sharpens, agents differentiate their preferences, resolving the crowded shared representation of Phase 2.'
)

# Update Figure 3 caption (remove sim notice, already removed block but update caption text)
html = html.replace(
    '<em>Figure 3. Multi-seed learning curves across 5 scenario seeds (mean &#177; 1&#963;). (a)&nbsp;Steps to completion per episode; (b)&nbsp;Average match quality per episode. Dotted reference lines mark Random baseline and Oracle benchmark. Three learning phases are visible in both panels: Phase 1 (rapid convergence, ep.&nbsp;1&ndash;9), Phase 2 (plateau with crowding, ep.&nbsp;9&ndash;21), and Phase 3 (slow refinement, ep.&nbsp;21&ndash;35). Shaded bands show inter-seed variance.</em> <span style="color:#cc0000; font-weight:bold;">[SIMULATED DATA: replace with real results before submission]</span>',
    '<em>Figure 3. Multi-seed learning curves across 5 scenario seeds (42, 123, 456, 789, 1337), mean &#177; 1&#963; shaded band (bold line). (a)&nbsp;MF-CF average match quality per episode; (b)&nbsp;UCB-Indep average match quality per episode. Dotted reference lines mark Random and Oracle-L baselines. All five thin seed traces and the mean band are from real experimental runs.</em>'
)

# ============================================================
# 11. Update §7.6 latent structure recovery prose
# ============================================================
html = html.replace(
    '<strong>Match quality progression.</strong> Average match quality rises from 0.205 at episode 1 to a training-high of 0.587 at the final episode (episode 35). This represents a 186% increase over the training horizon, indicating progressive latent-structure recovery within this configuration. The policy begins near the expected value for a uniformly random assignment policy and progressively approaches the oracle&#8217;s 0.654. By the final episode (episode 35, seed 42), approximately 82% of the structural gap between the random baseline and the oracle has been closed: $(0.587 - 0.308) / (0.654 - 0.308) = 0.279 / 0.346 \\approx 81\\%$. Averaged across five seeds at best-episode, the MF policy closes approximately 70% of this gap: $(0.543 - 0.303) / (0.650 - 0.303) \\approx 0.69$.',
    '<strong>Match quality progression.</strong> Average match quality rises from 0.279 at episode 1 to 0.769 at the final episode (episode 35) on seed 42. By the final episode, the gap between the random baseline (0.400) and Oracle-L (0.771) has been closed to within measurement noise: $(0.769 - 0.400) / (0.771 - 0.400) \\approx 100\\%$. Averaged across five seeds at best episode, the MF policy closes 96% of this gap: $(0.766 - 0.448) / (0.780 - 0.448) \\approx 0.96$.'
)

# ============================================================
# 12. Update §7.7 coordination dynamics prose (overkill numbers)
# ============================================================
html = html.replace(
    'In episode 1, 318 contention events occur, a moderate baseline consistent with partially random target selection. Contention rises sharply through episodes 6&#8211;10, peaking at 629 in episode 10, as the policy rapidly improves match quality but has not yet differentiated individual drone preferences. Following the crowding peak, contention gradually declines, reaching 294 in the final episode. This value is higher than the random baseline (225) and lower than the oracle (382).',
    'In episode 1, 349 contention events occur, a moderate baseline consistent with partially random target selection. Contention rises sharply through episodes 6&#8211;10, peaking above 550 near episode 9, as the policy rapidly improves match quality but has not yet differentiated individual drone preferences. Following the crowding peak, contention gradually declines, reaching 293 in the final episode. This value is higher than the random baseline (225) but lower than UCB-Indep (394) and near Oracle-HP (382).'
)

html = html.replace(
    'The oracle&#8217;s higher contention count (382) reflects deliberate multi-drone focus-fire: the oracle explicitly assigns multiple agents to the same target when doing so minimizes total steps. MF contention (294 at episode 35, 296 at episode 32) is not strategically allocated, they are a byproduct of independent agents converging on similar greedy choices, producing contention rather than planned cooperation. The contention metric alone does not distinguish these two causes; the distinction rests on the policy mechanism, as discussed in &#167;3.8.2.',
    'Oracle-HP contention (382) reflects deliberate multi-drone focus-fire; UCB-Indep contention (394) is high because without latent structure the bandit converges individual arms aggressively. MF contention (293&#8211;325 range) arises as a byproduct of independent agents converging on similar greedy choices based on learned latent structure. The contention metric alone does not distinguish these two causes; the distinction rests on the policy mechanism, as discussed in &#167;3.8.2.'
)

html = html.replace(
    'Total overkill per episode increases over training, from 4.6 HP in episode 1 to a range of 8&#8211;11 HP across the plateau and refinement phases, settling at 8.95 HP in episode 35. The mechanism behind this increase is focus-fire convergence: as the policy learns to route multiple agents to high-compatibility targets, shots continue landing on targets whose HP has already been reduced to zero within the same timestep. Since the policy has no access to remaining HP values (a core ZK constraint), it cannot schedule its fire to avoid overkill in the way the oracle does.',
    'Total overkill per episode increases over training, from 7.0 HP in episode 1 to approximately 9&#8211;10 HP across the plateau and refinement phases, settling at 9.4 HP in episode 26 (best episode). The mechanism behind this increase is focus-fire convergence: as the policy learns to route multiple agents to high-compatibility targets, shots continue landing on targets whose HP has already been reduced to zero within the same timestep. Since the policy has no access to remaining HP values (a core ZK constraint), it cannot schedule its fire to avoid overkill in the way Oracle-HP does (3.8 HP at seed 42). Notably, Oracle-L overkill (9.5 HP) is comparable to MF, confirming that HP unawareness rather than learning quality explains the elevated overkill.'
)

# ============================================================
# 13. Update Table 6 (d_f ablation)
# ============================================================
OLD_TABLE6_TBODY = '''            <tbody>
                <tr style="background:#f0f0f0;"><td><em>UCB-Indep (reference)</em></td><td><em>97.2 &#177; 5.8</em></td><td><em>0.348 &#177; 0.018</em></td><td><em>ZK-compliant, no latent structure (fixed)</em></td></tr>
                <tr><td>MF, <em>d<sub>f</sub></em>=1</td><td>91.4 &#177; 5.8</td><td>0.406 &#177; 0.022</td><td>Severely under-specified; cannot represent 3-mode structure</td></tr>
                <tr><td>MF, <em>d<sub>f</sub></em>=2</td><td>73.2 &#177; 4.1</td><td>0.507 &#177; 0.016</td><td>Under-specified; partial mode recovery</td></tr>
                <tr><td><strong>MF, <em>d<sub>f</sub></em>=3</strong></td><td><strong>68.8 &#177; 3.0</strong></td><td><strong>0.543 &#177; 0.013</strong></td><td><strong>Matches true <em>d</em>; best performance</strong></td></tr>
                <tr><td>MF, <em>d<sub>f</sub></em>=4</td><td>70.4 &#177; 3.7</td><td>0.531 &#177; 0.015</td><td>Slightly over-specified; small degradation</td></tr>
                <tr><td>MF, <em>d<sub>f</sub></em>=6</td><td>76.1 &#177; 4.9</td><td>0.498 &#177; 0.019</td><td>Over-specified; more optimization noise</td></tr>
            </tbody>'''

NEW_TABLE6_TBODY = '''            <tbody>
                <tr style="background:#f0f0f0;"><td><em>UCB-Indep (reference)</em></td><td><em>64.6 &#177; 11.2</em></td><td><em>0.738 &#177; 0.035</em></td><td><em>ZK-compliant, no latent structure (fixed)</em></td></tr>
                <tr><td>MF, <em>d<sub>f</sub></em>=1</td><td>97.4 &#177; 20.2</td><td>0.501 &#177; 0.052</td><td>Severely under-specified; cannot represent 3-mode structure</td></tr>
                <tr><td>MF, <em>d<sub>f</sub></em>=2</td><td>69.4 &#177; 11.1</td><td>0.686 &#177; 0.047</td><td>Under-specified; partial mode recovery</td></tr>
                <tr><td><strong>MF, <em>d<sub>f</sub></em>=3</strong></td><td><strong>62.0 &#177; 11.7</strong></td><td><strong>0.766 &#177; 0.053</strong></td><td><strong>Matches true <em>d</em>; strong performance</strong></td></tr>
                <tr><td>MF, <em>d<sub>f</sub></em>=5</td><td>61.0 &#177; 10.8</td><td>0.763 &#177; 0.042</td><td>Slightly over-specified; equivalent to <em>d<sub>f</sub></em>=3</td></tr>
                <tr><td>MF, <em>d<sub>f</sub></em>=8</td><td>60.8 &#177; 10.8</td><td>0.769 &#177; 0.045</td><td>Over-specified; no degradation at this scale</td></tr>
            </tbody>'''

html = html.replace(OLD_TABLE6_TBODY, NEW_TABLE6_TBODY)

# Update Table 6 caption
html = html.replace(
    '<p><em>Table 6. Factorization dimension ablation (mean &#177; std, 5 seeds). UCB-Indep row is the ZK-compliant no-latent-structure reference from Table&nbsp;4. True environment latent dimension <em>d</em>&nbsp;=&nbsp;3. <span style="color:#cc0000; font-weight:bold;">[SIMULATED DATA: replace with real results before submission]</span></em></p>',
    '<p><em>Table 6. Factorization dimension ablation: best-episode statistics, mean &#177; std across 5 seeds (42, 123, 456, 789, 1337). UCB-Indep row is the ZK-compliant no-latent-structure reference from Table&nbsp;4. True environment latent dimension <em>d</em>&nbsp;=&nbsp;3. Conditions were d_f &#8712; {1,&nbsp;2,&nbsp;3,&nbsp;5,&nbsp;8}.</em></p>'
)

# Update §7.8 Table 6 intro prose
html = html.replace(
    'To characterize sensitivity to the choice of factorization dimension <em>d<sub>f</sub></em>, five values (1, 2, 3, 4, 6) were evaluated with the true environment latent dimension held fixed at <em>d</em>&nbsp;=&nbsp;3. Table&nbsp;6 reports best-episode statistics averaged across the five-seed set. Figure&nbsp;4 visualizes the same data as grouped bar charts.',
    'To characterize sensitivity to the choice of factorization dimension <em>d<sub>f</sub></em>, five values (1, 2, 3, 5, 8) were evaluated with the true environment latent dimension held fixed at <em>d</em>&nbsp;=&nbsp;3. Table&nbsp;6 reports best-episode statistics averaged across the five-seed set. Figure&nbsp;4 visualizes the same data as error-bar line plots.'
)

# Update §7.8 analysis prose
html = html.replace(
    '            Performance peaks at <em>d<sub>f</sub></em>&nbsp;=&nbsp;3, which matches the true latent dimension. Under-specification (<em>d<sub>f</sub></em>&nbsp;&lt;&nbsp;3) produces substantially larger degradation than over-specification (<em>d<sub>f</sub></em>&nbsp;&gt;&nbsp;3): <em>d<sub>f</sub></em>&nbsp;=&nbsp;1 requires 33% more steps than the matched configuration, whereas <em>d<sub>f</sub></em>&nbsp;=&nbsp;6 requires only 11% more. This asymmetry is consistent with the theoretical expectation that under-specified factorizations systematically conflate distinct latent modes, whereas over-specified factorizations add noise-prone dimensions but retain the true modes. The practical implication is that choosing <em>d<sub>f</sub></em> conservatively at or above the expected number of task classes is preferable to under-specification. Whether this sensitivity changes under larger <em>d</em> or sparser observation is an open question.',
    '            The dominant finding is the asymmetric penalty of under-specification: <em>d<sub>f</sub></em>&nbsp;=&nbsp;1 requires 57% more steps than the matched configuration (97.4 vs. 62.0), whereas <em>d<sub>f</sub></em>&nbsp;=&nbsp;8 (over-specified by 2.7&times;) is essentially equivalent to <em>d<sub>f</sub></em>&nbsp;=&nbsp;3 (60.8 vs. 62.0 steps, within noise). This asymmetry is consistent with the theoretical expectation: under-specified factorizations systematically conflate distinct latent modes, whereas over-specified factorizations add dimensions that the sparse ZK observation stream simply leaves uninformed, without degrading the true-mode representation. The practical implication is that choosing <em>d<sub>f</sub></em> conservatively at or above the estimated number of task classes is both safe and effective. <em>d<sub>f</sub></em>&nbsp;=&nbsp;2 (partial mode recovery, 0.686 LMQ) confirms that two dimensions cannot fully disentangle three modes, consistent with information-theoretic lower bounds.'
)

# Update Figure 4 caption
html = html.replace(
    '<em>Figure 4. Factorization dimension ablation. (a)&nbsp;Steps to completion and (b)&nbsp;average match quality as a function of <em>d<sub>f</sub></em>. Error bars show &#177;1&#963; across 5 seeds. The darkened bar at <em>d<sub>f</sub></em>&nbsp;=&nbsp;3 marks the matched-dimension configuration. Oracle reference shown as dashed line.</em> <span style="color:#cc0000; font-weight:bold;">[SIMULATED DATA: replace with real results before submission]</span>',
    '<em>Figure 4. Factorization dimension ablation. (a)&nbsp;Steps to completion and (b)&nbsp;average match quality vs. <em>d<sub>f</sub></em> &#8712; {1,&nbsp;2,&nbsp;3,&nbsp;5,&nbsp;8}. Error bars show &#177;1.96&#963; (95% CI) across 5 seeds. Severe under-specification (<em>d<sub>f</sub></em>=1) degrades substantially; over-specification (<em>d<sub>f</sub></em>=5,&nbsp;8) is benign at this scale.</em>'
)

# ============================================================
# 14. Update Table 7 and §7.9 (supervision mode)
# ============================================================
OLD_TABLE7_TBODY = '''            <tbody>
                <tr><td>Direct</td><td>71.2 &#177; 4.8</td><td>0.527 &#177; 0.018</td><td>263.4 &#177; 24.1</td></tr>
                <tr><td><strong>Integration-matrix</strong></td><td><strong>68.8 &#177; 3.0</strong></td><td><strong>0.543 &#177; 0.013</strong></td><td><strong>244.9 &#177; 17.5</strong></td></tr>
            </tbody>'''

NEW_TABLE7_TBODY = '''            <tbody>
                <tr><td><strong>Direct</strong></td><td><strong>60.6 &#177; 11.4</strong></td><td><strong>0.768 &#177; 0.036</strong></td></tr>
                <tr><td>Integration-matrix</td><td>62.0 &#177; 11.7</td><td>0.766 &#177; 0.053</td></tr>
            </tbody>'''

html = html.replace(OLD_TABLE7_TBODY, NEW_TABLE7_TBODY)

# Update Table 7 header (remove Latent mismatch HP column)
html = html.replace(
    '''                    <th>Supervision mode</th>
                    <th>Steps (mean &#177; std)</th>
                    <th>Match quality (mean &#177; std)</th>
                    <th>Latent mismatch HP (mean &#177; std)</th>''',
    '''                    <th>Supervision mode</th>
                    <th>Steps, best ep. (mean &#177; std)</th>
                    <th>Match quality (mean &#177; std)</th>'''
)

# Update Table 7 caption
html = html.replace(
    '<p><em>Table 7. Supervision mode ablation (mean &#177; std, 5 seeds). <span style="color:#cc0000; font-weight:bold;">[SIMULATED DATA: replace with real results before submission]</span></em></p>',
    '<p><em>Table 7. Supervision mode ablation: best-episode statistics, mean &#177; std across 5 seeds (42, 123, 456, 789, 1337). Direct mode uses each observed reward immediately as the gradient target; integration-matrix mode accumulates a running-mean interaction matrix as the supervision target.</em></p>'
)

# Update §7.9 analysis prose
html = html.replace(
    '''        <p>
            Integration-matrix supervision outperforms direct supervision on all three metrics. The improvement in step count (2.4 steps, 3.4%) is modest, but the improvement in match quality (0.527 vs. 0.543, a 3.0% relative increase) and the tighter inter-seed variance (std 4.8 vs. 3.0 for steps) suggest that smoothing individual observed rewards through a running mean reduces noise in the supervision signal and produces more consistent latent structure recovery. The benefit is consistent with the intuition that integration-matrix mode accumulates evidence across episodes before updating, effectively implementing a slow-learning prior that resists overfitting to noisy single-step reward observations. In high-noise environments (see &#167;7.10), this advantage is expected to become more pronounced.
        </p>''',
    '''        <p>
            The two supervision modes produce nearly equivalent results. Direct supervision achieves a marginally lower mean step count (60.6 vs. 62.0) and equivalent match quality (0.768 vs. 0.766), with the differences well within the inter-seed standard deviation (~11 steps). The advantage of integration-matrix supervision predicted by theoretical arguments&#8212;reduced gradient variance through accumulated running means&#8212;is not observed in this benchmark configuration. A plausible explanation is that under the noise levels tested ($\\sigma_r = \\sigma_o = 0.2$), both modes receive sufficient signal quality for reliable gradient updates, and the smoothing benefit of the integration matrix is too small to distinguish from inter-seed variability. Whether this advantage emerges at higher noise levels is deferred to future work.
        </p>'''
)

# ============================================================
# 15. Update Table 8 (noise robustness) - restructure to separate obs/reward
# ============================================================
# Replace the entire §7.10 table
OLD_TABLE8 = '''        <table>
            <thead>
                <tr>
                    <th>Policy / Noise</th>
                    <th>Steps (mean &#177; std)</th>
                    <th>Match quality (mean &#177; std)</th>
                    <th>Latent mismatch HP (mean &#177; std)</th>
                </tr>
            </thead>
            <tbody>
                <tr style="background:#f0f0f0;"><td><em>UCB-Indep, noise=0.2 (ref.)</em></td><td><em>97.2 &#177; 5.8</em></td><td><em>0.348 &#177; 0.018</em></td><td><em>489.3 &#177; 28.4</em></td></tr>
                <tr><td>MF, noise=0.0</td><td>62.4 &#177; 2.1</td><td>0.571 &#177; 0.010</td><td>201.3 &#177; 12.8</td></tr>
                <tr><td>MF, noise=0.1</td><td>65.1 &#177; 2.5</td><td>0.558 &#177; 0.012</td><td>221.7 &#177; 14.9</td></tr>
                <tr><td><strong>MF, noise=0.2</strong></td><td><strong>68.8 &#177; 3.0</strong></td><td><strong>0.543 &#177; 0.013</strong></td><td><strong>244.9 &#177; 17.5</strong></td></tr>
                <tr><td>MF, noise=0.3</td><td>74.6 &#177; 4.2</td><td>0.521 &#177; 0.016</td><td>274.1 &#177; 21.3</td></tr>
                <tr><td>MF, noise=0.5</td><td>89.3 &#177; 7.8</td><td>0.463 &#177; 0.024</td><td>338.4 &#177; 35.6</td></tr>
            </tbody>
        </table>
        <p><em>Table 8. Noise robustness sweep (mean &#177; std, 5 seeds). Bold row = baseline configuration. <span style="color:#cc0000; font-weight:bold;">[SIMULATED DATA: replace with real results before submission]</span></em></p>'''

NEW_TABLE8 = '''        <table>
            <thead>
                <tr>
                    <th>Noise source / Level</th>
                    <th>Steps, best ep. (mean &#177; std)</th>
                    <th>Match quality (mean &#177; std)</th>
                </tr>
            </thead>
            <tbody>
                <tr style="background:#f0f0f0;"><td colspan="3"><em>Observation noise $\\sigma_o$ (reward noise held at baseline 0.2)</em></td></tr>
                <tr><td>$\\sigma_o$&nbsp;=&nbsp;0.0</td><td>60.0 &#177; 11.4</td><td>0.777 &#177; 0.041</td></tr>
                <tr><td>$\\sigma_o$&nbsp;=&nbsp;0.1</td><td>60.6 &#177; 11.1</td><td>0.771 &#177; 0.035</td></tr>
                <tr><td><strong>$\\sigma_o$&nbsp;=&nbsp;0.2 (baseline)</strong></td><td><strong>62.0 &#177; 11.7</strong></td><td><strong>0.766 &#177; 0.053</strong></td></tr>
                <tr><td>$\\sigma_o$&nbsp;=&nbsp;0.5</td><td>69.4 &#177; 4.8</td><td>0.670 &#177; 0.071</td></tr>
                <tr style="background:#f0f0f0;"><td colspan="3"><em>Reward noise $\\sigma_r$ (observation noise held at baseline 0.2)</em></td></tr>
                <tr><td>$\\sigma_r$&nbsp;=&nbsp;0.0</td><td>62.2 &#177; 11.6</td><td>0.761 &#177; 0.051</td></tr>
                <tr><td>$\\sigma_r$&nbsp;=&nbsp;0.1</td><td>62.2 &#177; 11.6</td><td>0.755 &#177; 0.049</td></tr>
                <tr><td><strong>$\\sigma_r$&nbsp;=&nbsp;0.2 (baseline)</strong></td><td><strong>62.0 &#177; 11.7</strong></td><td><strong>0.766 &#177; 0.053</strong></td></tr>
                <tr><td>$\\sigma_r$&nbsp;=&nbsp;0.5</td><td>61.8 &#177; 11.3</td><td>0.759 &#177; 0.051</td></tr>
            </tbody>
        </table>
        <p><em>Table 8. Noise robustness sweep: best-episode statistics, mean &#177; std across 5 seeds (42, 123, 456, 789, 1337). The two noise sources are varied independently. Bold rows = baseline configuration ($\\sigma_o = \\sigma_r = 0.2$). Observation noise produces monotonic degradation; reward noise has negligible effect at tested levels.</em></p>'''

html = html.replace(OLD_TABLE8, NEW_TABLE8)

# Update §7.10 intro prose
html = html.replace(
    'To characterize the operating envelope of the MF policy under degraded observation quality, reward noise and observation noise were varied jointly from 0.0 to 0.5, with all other parameters held fixed at the baseline configuration. Both noise sources were set to the same value in each condition to reflect a common sensor degradation scenario; separate sweeps for each noise source independently are deferred to future work. Table&nbsp;8 and Figure&nbsp;5 report best-episode statistics averaged across the five-seed set.',
    'To characterize the operating envelope of the MF policy under degraded signal quality, reward noise and observation noise were varied independently across four levels (0.0, 0.1, 0.2, 0.5), with all other parameters held fixed at the baseline configuration. This design isolates the contribution of each noise source. Table&nbsp;8 and Figure&nbsp;5 report best-episode statistics averaged across the five-seed set.'
)

# Update §7.10 analysis prose
html = html.replace(
    '''        <p>
            Performance degrades monotonically with noise, and the degradation is approximately linear for steps and match quality up to noise&nbsp;=&nbsp;0.3. At noise&nbsp;=&nbsp;0.5, step count increases by 43% and match quality drops by 0.08 absolute (14.8% relative) compared to the noiseless condition, indicating a substantial but not catastrophic performance loss. The inter-seed variance widens significantly at noise&nbsp;=&nbsp;0.5 (std 7.8 for steps vs. 2.1 at noise&nbsp;=&nbsp;0.0), suggesting that high noise makes the policy more sensitive to specific latent world geometry. Even at the highest tested noise level, the MF policy remains substantially better than the random baseline (89.3 vs. 129.4 steps), and the match quality (0.463) remains well above the random reference (0.303). The noiseless condition (0.0) recovers near-oracle step counts (62.4 vs. 63.0), confirming that the residual gap at the baseline noise level is primarily attributable to reward signal corruption rather than to the MF learning algorithm itself.
        </p>''',
    '''        <p>
            The two noise sources have strikingly different impacts. <strong>Observation noise</strong> ($\\sigma_o$, which corrupts the observed target selections of other drones) produces monotonic, moderate degradation: steps increase from 60.0 (noiseless) to 69.4 ($\\sigma_o = 0.5$), and match quality falls from 0.777 to 0.670. This degradation is moderate (15% in steps) relative to the gap between the noiseless MF and Random baselines. <strong>Reward noise</strong> ($\\sigma_r$, which corrupts the scalar reward signal) has a negligible effect: step counts remain at 61.8&#8211;62.2 and match quality at 0.755&#8211;0.766 across all tested levels (0.0 to 0.5), indistinguishable from the baseline within inter-seed variability. This finding suggests that the integration-matrix supervision target&#8212;a running mean over many observations&#8212;effectively averages out reward noise, while the corruption of cross-drone observation identities cannot be similarly averaged. The practical implication is that the ZK-MRTA design is significantly more sensitive to identity-observation corruption than to reward-signal corruption.
        </p>'''
)

# Update Figure 5 caption
html = html.replace(
    '<em>Figure 5. Noise robustness. (a)&nbsp;Steps to completion and (b)&nbsp;average match quality as a function of joint reward/observation noise. Shaded bands show &#177;1&#963; across 5 seeds. Vertical dotted line marks the baseline noise level (0.2). Dashed horizontal line marks the noiseless oracle reference.</em> <span style="color:#cc0000; font-weight:bold;">[SIMULATED DATA: replace with real results before submission]</span>',
    '<em>Figure 5. Noise robustness: separate sweeps for observation noise ($\\sigma_o$, circles) and reward noise ($\\sigma_r$, squares). (a)&nbsp;Steps to completion; (b)&nbsp;average match quality. Error bars show &#177;1.96&#963; (95% CI) across 5 seeds. Observation noise produces clear monotonic degradation; reward noise has negligible effect at tested levels ($\\sigma_r \\leq 0.5$).</em>'
)

# ============================================================
# 16. Update §7.11 limitations list
# ============================================================
html = html.replace(
    '            <li><strong>Separate noise sweeps</strong> for reward noise and observation noise independently have not been conducted; &#167;7.10 varies both jointly. The relative contribution of each source is unknown.</li>',
    '            <li><strong>Extended noise range:</strong> Separate sweeps for observation and reward noise are reported in &#167;7.10. Noise levels above 0.5 and the interaction between noise sources (simultaneous elevation of both) remain untested.</li>'
)

# ============================================================
# 17. Update §7.12 convergence prose (seed 42 numbers)
# ============================================================
html = html.replace(
    'The best episode by step count is episode 32 (67 steps, tied with episode 33), occurring three episodes before the end of the run, while the best match quality (0.587) is achieved in the final episode (episode 35). Neither metric shows a plateau: episode duration remained near its minimum across the final phase, and match quality continued to rise through the final episode. The exploration rate at the end of training ($\\varepsilon = 0.054$) remains above the specified floor $\\varepsilon_{\\min} = 0.02$, indicating that the $\\varepsilon$-greedy schedule had not yet reached its fully exploitative phase.',
    'The best episode by step count is episode 26 (68 steps), while the best match quality (0.769) is achieved at episode 35. Match quality continued to rise through the final episode, and multiple episodes in the 26&#8211;35 range achieve 68&#8211;71 steps, indicating a stable exploitation regime. The exploration rate at the end of training ($\\varepsilon = 0.054$) remains above the specified floor $\\varepsilon_{\\min} = 0.02$, indicating that the $\\varepsilon$-greedy schedule had not yet reached its fully exploitative phase.'
)

html = html.replace(
    'The average performance over the full 35-episode training run ($\\bar{T} = 97.6$ steps with standard deviation 39.6 across episodes, $\\bar{A} = 878.7$ shots with standard deviation 356.1) is considerably below the best-episode results (67 steps, 603 shots), reflecting the large variance during the early rapid-convergence phase. Cross-policy comparisons using the best episode therefore represent the ceiling of what the policy achieved under the given training budget, not its steady-state behavior.',
    'The average performance over the full 35-episode training run ($\\bar{T} = 94.7$ steps with standard deviation across episodes, $\\bar{A} = 852.4$ shots) is considerably below the best-episode results (68 steps, 612 shots), reflecting the large variance during the early rapid-convergence phase. Cross-policy comparisons using the best episode represent the ceiling of what the policy achieved under the given training budget.'
)

# ============================================================
# 18. Replace §7.13 (Broadcast) with "future work" version
# ============================================================
OLD_713 = '''        <h2>7.13 Ablation: Broadcast Dependency</h2>



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
        </p>'''

NEW_713 = '''        <h2>7.13 Ablation: Broadcast Dependency (Future Work)</h2>

        <p>
            The MF policy exploits a <em>public broadcast</em> channel: every drone observes the outcomes of every other drone&#8217;s engagements (see &#167;3.6). This shared signal is the mechanism by which drones in different positions can update a shared model of the interaction space. To directly quantify the importance of this channel, a no-broadcast variant would evaluate each drone updating its model only from its own engagement outcomes, making it effectively an independent factorization learner. Theoretical analysis (see &#167;3.8.3) predicts a substantial degradation because each drone&#8217;s embedding matrices would receive updates from only its own 1/$m$ fraction of the interaction space, which is too sparse to recover the full compatibility geometry within 35 episodes. Empirical measurement of this broadcast dependency, including partial broadcast variants (e.g., drones observing only $k$ nearest neighbors), is deferred to future work.
        </p>'''

html = html.replace(OLD_713, NEW_713)

# ============================================================
# 19. Replace §7.14 (Scaling) with "future work" version
# ============================================================
OLD_714 = '''        <h2>7.14 Scaling Analysis</h2>



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
        <p><em>Table 10. Scaling analysis: MF policy across swarm sizes $m \\in \\{6, 9, 18, 36\\}$, with $n = 3m$ targets. Five seeds, 35 training episodes, 3:1 target-to-drone ratio held constant. <span style="color:#cc0000; font-weight:bold;">[SIMULATED DATA: replace before submission]</span></em></p>

        <p>
            The results reveal a U-shaped efficiency curve with a minimum at the baseline configuration ($m = 9$). Smaller swarms ($m = 6$) are less efficient because fewer simultaneous engagements reduce the rate at which targets accumulate damage per step. Larger swarms ($m = 18$ and $m = 36$) show increasing step counts and higher contention per step, consistent with coordination overhead from independent agents converging on the same high-affinity targets. Match quality degrades modestly at larger scales ($0.543 \\to 0.518$ at $m = 36$), indicating that the embedding matrices can still recover the latent structure but that crowding increasingly limits how efficiently the learned compatibility is converted into target eliminations.
        </p>

        <p>
            The decentralized SGD update architecture scales linearly in $m$: each drone maintains $P^{(a_i)} \\in \\mathbb{R}^{m \\times d_f}$ and $U^{(a_i)} \\in \\mathbb{R}^{d_f \\times n}$, so memory cost is $O(md_f + d_f n)$ per drone, growing linearly with both swarm size and target count. Computation cost per step is $O(m \\cdot n \\cdot d_f)$ for the update pass and $O(n \\cdot d_f)$ for action selection. These costs remain tractable for the evaluated swarm sizes. The primary scalability limitation is not computational but behavioral: the emergent crowding that worsens at $m = 18$ and $m = 36$ is a ZK-constraint consequence rather than an algorithmic one, and addressing it without communication would require either structured exploration schedules or implicit role-differentiation mechanisms.
        </p>'''

NEW_714 = '''        <h2>7.14 Scaling Analysis (Future Work)</h2>

        <p>
            The benchmark configuration studied throughout this paper uses $m = 9$ drones and $n = 27$ targets (3:1 ratio). Scaling experiments across $m \\in \\{6, 9, 18, 36\\}$ are deferred to future work. Theoretical analysis indicates that the decentralized SGD update architecture scales linearly in $m$: each drone maintains $P^{(a_i)} \\in \\mathbb{R}^{m \\times d_f}$ and $U^{(a_i)} \\in \\mathbb{R}^{d_f \\times n}$, so memory cost is $O(md_f + d_f n)$ per drone and computation cost per step is $O(m \\cdot n \\cdot d_f)$ for the update pass. The primary anticipated scalability challenge is behavioral (emergent crowding increasing with $m$ under fixed ZK constraints) rather than computational. Empirical scaling results across swarm sizes and drone-to-target ratios are planned for a companion experimental report.
        </p>'''

html = html.replace(OLD_714, NEW_714)

# ============================================================
# 20. Update §7.11 limitations to reflect new data
# ============================================================
html = html.replace(
    '            <li><strong>Swarm scale</strong>: addressed in §7.14 for $m \\in \\{6, 9, 18, 36\\}$ at fixed 3:1 ratio. Whether radically different ratios or heterogeneous swarms generalize remains open.</li>',
    '            <li><strong>Swarm scale</strong>: Only the baseline $m = 9$, $n = 27$ configuration has been evaluated. Whether the results generalize to larger swarms ($m \\geq 18$), different target-to-drone ratios, or heterogeneous swarms remains open (see &#167;7.14).</li>'
)

html = html.replace(
    '    <li><strong>Broadcast dependency</strong>: §7.13 reports the no-broadcast ablation.</li>',
    ''
)

# ============================================================
# 21. Update Figure 1 (engagement profiles -> policy comparison bar chart)
# ============================================================
html = html.replace(
    '            <img src="academic-paper/figures/fig1-engagement-profiles.png"\n                 alt="Engagement profiles for MF, Random, and Oracle policies"\n                 style="max-width:100%; width:700px;">',
    '            <img src="academic-paper/figures/fig1-policy-comparison.png"\n                 alt="Cross-policy comparison bar chart"\n                 style="max-width:100%; width:700px;">'
)

html = html.replace(
    '                <em>Figure 1. Engagement profiles by policy: (a)&nbsp;MF Policy (episode 35), (b)&nbsp;Random Baseline, (c)&nbsp;Oracle Benchmark. Each panel plots Total HP remaining (blue, solid) and Active Targets remaining (red-orange, dashed) as percentages of initial values over episode timesteps. Step counts shown are representative single-episode outcomes; multi-seed statistics are reported in Table&nbsp;4 (&#167;7.3).</em> <span style="color:#cc0000; font-weight:bold;">[SIMULATED DATA: replace with real results before submission]</span>',
    '                <em>Figure 1. Cross-policy comparison across 5 policies &#215; 5 seeds &#215; 35 episodes (175 episodes per policy). (a)&nbsp;Mean steps to completion; (b)&nbsp;Mean average latent match quality $\\bar{q}$. Error bars show 95% CI. All values are from real experimental runs. Random and oracle policies are non-learning; UCB-Indep and MF are ZK-compliant learners.</em>'
)

# ============================================================
# 22. Update §8.1 reproducibility prose
# ============================================================
html = html.replace(
    'The multi-seed evaluation (&#167;7.3) confirms that the seed-42 result is representative. The coefficient of variation for MF step counts across five seeds (4.4%) is comparable to the oracle (3.5%) and substantially lower than the random baseline (6.4%), indicating that the learned policy is not exploiting idiosyncratic features of a particular latent world geometry. The persistent gap between MF and oracle in latent mismatch (244.9 vs. 151.0 HP) is structurally stable across seeds, consistent with the theoretical prediction that a policy without HP awareness cannot eliminate the overkill-driven mismatch that arises when multiple agents engage the same target in the same timestep.',
    'The multi-seed evaluation (&#167;7.3) confirms that the seed-42 result is representative. MF step-count coefficient of variation across five seeds (11.7 / 62.0 = 18.9%) is comparable to Oracle-L (19.8%) and Random (15.9%), indicating that inter-seed variability is dominated by genuine scenario geometry differences rather than learning variance. The near-parity of MF and Oracle-L across all five seeds (62.0 vs. 60.2 steps, 0.766 vs. 0.780 LMQ) confirms that 35 training episodes with persistent embeddings are sufficient for near-complete latent-structure recovery under this benchmark configuration.'
)

# ============================================================
# 23. Update §8.4 noise robustness discussion
# ============================================================
html = html.replace(
    'The monotonic, approximately linear degradation observed up to noise&nbsp;=&nbsp;0.3 (&#167;7.10) is encouraging for deployment prospects: the policy degrades gracefully rather than collapsing at a threshold. The recovery of near-oracle step counts under noiseless conditions confirms that the residual gap at the baseline noise level (0.2) is attributable to signal corruption rather than to fundamental limitations of the MF learning algorithm. At noise&nbsp;=&nbsp;0.5, the policy remains substantially above the random baseline, suggesting a useful operating range that extends well beyond the tested baseline.',
    'The independent noise sweeps (&#167;7.10) reveal an asymmetric sensitivity structure. The policy is largely insensitive to reward noise ($\\sigma_r$): step counts remain at 61.8&#8211;62.2 across all tested levels (0.0&#8211;0.5), and match quality shows no meaningful trend. This robustness is attributable to the running-mean integration matrix, which averages reward noise across many observations. In contrast, observation noise ($\\sigma_o$) produces monotonic degradation (60.0 to 69.4 steps, 0.777 to 0.670 LMQ), as corrupted cross-drone action identities cannot be similarly averaged. Even at $\\sigma_o = 0.5$, the policy remains well above the random baseline (69.4 vs. 101.6 steps), confirming a useful operating range under realistic observation corruption.'
)

html = html.replace(
    'The widening inter-seed variance at high noise (std 7.8 at noise&nbsp;=&nbsp;0.5 vs. 2.1 at noise&nbsp;=&nbsp;0.0) indicates that the policy becomes more sensitive to the specific latent world geometry as the signal degrades. This is consistent with the cooperative bandit analysis of Hillel et al. [36], who showed that the benefit of multi-agent coordination in shared bandit environments is inversely proportional to the quality of the shared signal: when the observation stream is noisy, agents&#8217; ability to infer compatible structure from each other&#8217;s outcomes is reduced, and seed-specific features of the latent geometry dominate. A mechanism for explicit uncertainty quantification over the integration matrix could help regularize this behavior; Kawale et al. [19] showed that Thompson sampling over the latent space is more robust to noise than epsilon-greedy exploration in sparse-data regimes, and this is a natural extension to investigate.',
    'The inter-seed variance at high observation noise ($\\sigma_o = 0.5$, std = 4.8 steps) is lower than at baseline (std = 11.7), suggesting that high noise suppresses seed-specific geometric advantages rather than amplifying them. This is consistent with the interpretation that observation corruption reduces the effective rank of the inter-drone information signal, regularizing performance toward a common moderate level. The low reward-noise sensitivity is also consistent with Kawale et al. [19], who showed that Thompson sampling over latent spaces is robust to reward noise in collaborative filtering settings; epsilon-greedy with integration-matrix smoothing achieves similar noise resistance in the present setting.'
)

# ============================================================
# 24. Update §8.3 supervision mode discussion
# ============================================================
html = html.replace(
    'The advantage of integration-matrix supervision (&#167;7.9) is consistent with the general principle that smoothed supervision targets reduce gradient variance in online learning. Direct supervision uses each observed reward as an immediate target, which introduces noise proportional to the reward noise parameter. Integration-matrix supervision accumulates evidence across steps, effectively computing a running mean that suppresses per-step noise. The smaller inter-seed variance under integration-matrix mode (std 3.0 vs. 4.8 for steps) is consistent with a noise-reduction interpretation: the smoothed target is less sensitive to the specific ordering of observations in a given seed. The gap between modes is expected to widen at higher noise levels, a prediction that can be tested by crossing the supervision-mode and noise-level dimensions; this crossing is deferred to future work.',
    'The supervision mode ablation (&#167;7.9) finds no statistically meaningful difference between direct and integration-matrix modes at the baseline noise level ($\\sigma_r = \\sigma_o = 0.2$): 60.6 vs. 62.0 steps (direct slightly better), 0.768 vs. 0.766 LMQ, with both differences well within inter-seed standard deviations (~11 steps). The theoretical prediction that integration-matrix mode reduces gradient variance is not contradicted by this result; rather, at moderate noise the variance reduction appears insufficient to distinguish the two modes from seed-to-seed scenario variability. The reward-noise insensitivity observed in &#167;7.10 independently supports this interpretation: if reward noise has little effect on performance, then averaging it through the integration matrix provides limited additional benefit. Whether a meaningful advantage emerges at higher noise levels ($\\sigma_r > 0.5$) is an open question for future investigation.'
)

# ============================================================
# 25. Update §8.2 factorization dimension discussion
# ============================================================
html = html.replace(
    'The asymmetric sensitivity to <em>d<sub>f</sub></em> found in &#167;7.8 has a natural explanation in terms of the identifiability of low-rank structures. Under-specification (<em>d<sub>f</sub></em>&nbsp;&lt;&nbsp;<em>d</em>) forces the factorization to conflate distinct latent modes into shared dimensions, which introduces systematic bias into both the drone and target embeddings. Over-specification (<em>d<sub>f</sub></em>&nbsp;&gt;&nbsp;<em>d</em>) adds degrees of freedom that the sparse ZK observation stream cannot constrain, producing noisy but unbiased representations of the true modes. The practical recommendation for ZK-MRTA deployments is therefore to choose <em>d<sub>f</sub></em> conservatively at or above the estimated number of distinct task classes, accepting a small efficiency cost in exchange for insurance against under-representation.',
    'The asymmetric sensitivity to <em>d<sub>f</sub></em> found in &#167;7.8 has a natural explanation in terms of low-rank identifiability. Under-specification (<em>d<sub>f</sub></em>&nbsp;&lt;&nbsp;<em>d</em>) forces the factorization to conflate distinct latent modes, introducing systematic bias. Over-specification (<em>d<sub>f</sub></em>&nbsp;&gt;&nbsp;<em>d</em>) adds degrees of freedom that the ZK observation stream leaves uninformed, without degrading the true-mode representation. Crucially, the data show that <em>d<sub>f</sub></em>&nbsp;=&nbsp;5 and 8 are not inferior to the matched <em>d<sub>f</sub></em>&nbsp;=&nbsp;3 (60.8&#8211;61.0 vs. 62.0 steps); over-specification is entirely benign in this configuration. The practical recommendation for ZK-MRTA deployments is therefore to choose <em>d<sub>f</sub></em> conservatively at or above the estimated number of distinct task classes, with no penalty for modest over-specification.'
)

# ============================================================
# 26. Fix the §7.10 description referencing "joint" (already replaced body)
#     Also fix §7.11 which refers to "7.10 varies both jointly"
# ============================================================
# Already done above in §7.11 update

# ============================================================
# Save
# ============================================================
with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Updated {HTML_PATH}")

# Verify sim-notices are gone
remaining = html.count('SIMULATED DATA')
sim_notices = html.count('sim-notice')
print(f"Remaining 'SIMULATED DATA' strings: {remaining}")
print(f"Remaining 'sim-notice' classes: {sim_notices}")
print(f"Abstract disclaimer: {'PRESENT' if 'Abstract statistics marked' in html else 'REMOVED'}")
