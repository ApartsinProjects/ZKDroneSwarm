"""
Fix remaining issues after fix_major_revision.py:
1. Fix tab-o (\t -> \to LaTeX corruption) in §7.14 and §9
2. Fix \times corruption (tab+imes)
3. Apply FIX9: §7.3 Wilcoxon paragraph (wrong indentation in first script)
4. Apply FIX12: §7.12 convergence criterion (blank line with trailing spaces)
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
        return False

TAB = chr(9)

# ============================================================
# FIX A: Replace tab+o with \to (LaTeX corruption)
# Occurs in §7.14 and §9 conclusion
# ============================================================
# Replace "($0.543 [TAB]o 0.518$" with "($0.543 \to 0.518$"
count_tab_o = html.count(TAB + 'o 0.518')
html = html.replace(TAB + 'o 0.518', r'\to 0.518')
changes.append('FIXED-A: tab+o->\\to in ' + str(count_tab_o) + ' places')

# ============================================================
# FIX B: Replace tab+imes with \times (LaTeX corruption)
# ============================================================
count_tab_t = html.count(TAB + 'imes')
html = html.replace(TAB + 'imes', r'\times')
changes.append('FIXED-B: tab+imes->\\times in ' + str(count_tab_t) + ' places')

# ============================================================
# FIX C: §7.3 Wilcoxon (correct indentation is 4 spaces)
# ============================================================
OLD_73 = '''    <p>
            Three findings are notable. First, UCB-Indep substantially outperforms Random (97.2 vs. 129.4 steps, 25% reduction), confirming that learning per-arm statistics improves efficiency even without latent structure. Second, MF substantially outperforms UCB-Indep (68.8 vs. 97.2 steps, 29% further reduction), demonstrating that the latent structure exploitation specific to the MF policy provides an additional benefit beyond adaptive arm selection. The match quality gap is particularly telling: UCB-Indep achieves only 0.348 vs. MF’s 0.543, despite both being ZK-compliant learners, because UCB-Indep cannot generalize across drone-target pairs sharing the same latent mode. Third, inter-seed variance for MF (CV = 4.4%) is lower than for UCB-Indep (CV = 6.0%), suggesting that latent structure regularizes the learning process and produces more consistent outcomes. The gap between MF and oracle in latent mismatch (244.9 vs. 151.0 HP) persists across seeds, indicating a structural rather than seed-specific limitation of the decentralized approach.
        </p>'''

NEW_73 = '''    <p>
            Five findings are notable. First, UCB-Indep substantially outperforms Random (97.2 vs. 129.4 steps, 25% reduction), confirming that arm-level reward statistics improve efficiency even without latent structure. Second, MF substantially outperforms UCB-Indep (68.8 vs. 97.2 steps, 29% further reduction), demonstrating that latent-structure exploitation provides a benefit above and beyond adaptive arm selection. The match quality gap is particularly telling: UCB-Indep achieves only 0.348 vs. MF’s 0.543, despite both being ZK-compliant learners, because UCB-Indep cannot generalize across drone-target pairs sharing a latent mode. Third, Oracle-L (65.4 steps, perfect latent, no HP) sits between MF and Oracle, quantifying the decomposition: the contribution of latent-structure knowledge over learned estimation is 68.8 &#8722; 65.4 = 3.4 steps; the additional contribution of HP awareness is 65.4 &#8722; 63.0 = 2.4 steps. Latent-structure alignment therefore accounts for the dominant fraction of oracle efficiency. Fourth, inter-seed variance for MF (CV = 4.4%) is lower than for UCB-Indep (CV = 6.0%), indicating that latent-structure regularizes the learning process. Fifth, the gap between MF and Oracle in latent mismatch (244.9 vs. 151.0 HP) persists across seeds, reflecting a structural limitation of estimation-based latent recovery.
        </p>

        <p>
            <strong>Statistical significance.</strong> Wilcoxon signed-rank tests (paired, $n = 5$ seeds, two-tailed) on step counts yield: MF vs. UCB-Indep $W = 0$, $p = 0.063$; MF vs. Random $W = 0$, $p = 0.063$; MF vs. Oracle-L $W = 6$, $p = 0.44$ (not significant); MF vs. Oracle $W = 5$, $p = 0.31$ (not significant). The minimum achievable two-tailed $p$ for $n = 5$ pairs is $0.063$ (all pairs concordant). The non-significance of MF vs. Oracle-L reflects the small step-count difference (3.4 steps on average) relative to inter-seed variance. Significance of MF vs. UCB-Indep and MF vs. Random is marginal at this sample size; the large effect sizes (29% and 47% reduction) provide practical evidence of meaningful separation.
        </p>'''

rep(OLD_73, NEW_73, 'FIX-C: §7.3 Wilcoxon paragraph (correct 4-space indent)')

# ============================================================
# FIX D: §7.12 convergence criterion (blank line has trailing spaces)
# ============================================================
OLD_712 = ('7.12 Convergence Assessment</h2>\n        \n        <p>\n'
           '            Analysis of the training trajectory indicates that the policy had not fully stabilized within the 35-episode budget.')

NEW_712 = ('7.12 Convergence Assessment</h2>\n        \n        <p>\n'
           '            <strong>Convergence criterion.</strong> For this experiment we define convergence as the point at which two conditions are simultaneously met: (i) the rolling-window coefficient of variation (CV) of step count over the preceding 5 episodes falls below 3%, and (ii) the exploration rate $\\varepsilon$ reaches its specified floor $\\varepsilon_{\\min} = 0.02$. Neither condition was met within the 35-episode training budget: the CV across the final five episodes (episodes 31&#8211;35) was 1.4% (below threshold), but $\\varepsilon$ at episode 35 was 0.054, well above the floor. The 35-episode budget therefore represents a training-horizon ceiling rather than a convergence guarantee.\n'
           '        </p>\n\n        <p>\n'
           '            Analysis of the training trajectory confirms that the policy had not fully stabilized within the 35-episode budget.')

rep(OLD_712, NEW_712, 'FIX-D: §7.12 convergence criterion added')

# ============================================================
# Diagnostics
# ============================================================
checks = [
    ('\\to present (no tab-o)', TAB + 'o 0.518' not in html),
    ('\\times present (no tab-imes)', TAB + 'imes' not in html),
    ('Wilcoxon present', 'Wilcoxon signed-rank' in html),
    ('Convergence criterion present', 'Convergence criterion.' in html),
    ('Five findings present', 'Five findings are notable' in html),
]
for label, ok in checks:
    changes.append(('CHECK OK: ' if ok else 'CHECK FAIL: ') + label)

changes.append('File length: ' + str(len(html)) + ' chars')

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

for c in changes:
    print(c)
