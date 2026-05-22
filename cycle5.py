"""
REVIEWER CYCLE 5: Final polish pass.
Issues:
  A) §6.3 still says "direct supervision mode...is not evaluated here"
     (we fixed it to reference §7.9, but check it reads correctly now)
  B) §9 conclusion: check the UCB-Indep fix reads naturally
  C) §4.5 closing paragraph: "four methods" now, verify
  D) Add match quality formula notation entry to §3.10 Table
  E) Fix the \\frac and LaTeX in the match quality formula
     (written in Python with double backslash, check it renders)
  F) Check for any remaining structural issues
"""
import re

with open('docs/index.html', encoding='utf-8') as f:
    html = f.read()

changes = []

# E) Verify / fix LaTeX in match quality formula
# The formula was inserted with escaped backslashes from Python - check it
mq_formula = r'$$\bar{q} = \frac{1}{E} \sum_{e=1}^{E} \frac{\mathbf{d}_e \cdot \mathbf{t}_e}{\|\mathbf{d}_e\| \|\mathbf{t}_e\|}$$'
if mq_formula in html:
    changes.append('OK: Match quality formula LaTeX correct')
else:
    # Check what was actually inserted
    idx = html.find('\\bar{q}')
    if idx > 0:
        changes.append(f'CHECK: formula at {idx}: {html[idx-5:idx+60]}')
    else:
        # Try to find it differently
        idx2 = html.find('average latent match quality')
        if idx2 > 0:
            snippet = html[idx2:idx2+300].replace('\n', ' ')
            changes.append(f'FORMULA REGION: {snippet[:200]}')

# D) Add match quality to §3.10 Notation Table
# Current notation table has $\bar{q}$ probably missing
if r'$\bar{q}$' in html:
    changes.append('OK: bar{q} already in notation table')
else:
    # Add it after the last row of the notation table in §3.10
    OLD_NOTROW = '<tr><td>$\\hat{r}_{kj}^{(a_i)}$</td><td>Predicted utility of drone $a_k$ engaging target $t_j$, as estimated by agent $a_i$</td></tr>'
    if OLD_NOTROW in html:
        NEW_NOTROW = (OLD_NOTROW +
                     '\n                <tr><td>$\\bar{q}$</td>'
                     '<td>Average latent match quality per episode (cosine similarity, averaged over all engagements)</td></tr>')
        html = html.replace(OLD_NOTROW, NEW_NOTROW, 1)
        changes.append('FIXED: Added bar{q} to notation table')
    else:
        # Try with slightly different formatting
        idx3 = html.find('Predicted utility of drone')
        if idx3 > 0:
            snippet3 = html[max(0,idx3-50):idx3+150].replace('\n', ' ')
            changes.append(f'NOTATION ROW FOUND AT {idx3}: {snippet3[:100]}')
        else:
            changes.append('MISS: notation table row for predicted utility not found')

# A) Check §6.3 supervision mode fix is clean
idx_63 = html.find('6.3 Supervision Mode')
if idx_63 > 0:
    section = html[idx_63:idx_63+800]
    if '7.9' in section:
        changes.append('OK: §6.3 references §7.9 correctly')
    else:
        changes.append('CHECK: §6.3 may still say "deferred"')

# C) Verify §4.5 four methods
if 'The four methods examined in this paper' in html:
    changes.append('OK: §4.5 says four methods')
else:
    changes.append('MISS: §4.5 four methods text not found')

# F) Check final structure
em_count = html.count('&mdash;') + html.count('—')
changes.append(f'Em-dashes: {em_count}')

# Count tables by caption
import re as RE
tables = RE.findall(r'Table (\d+)[.:]', html)
from collections import Counter
cnt = Counter(tables)
changes.append(f'Table counts: {dict(sorted(cnt.items(), key=lambda x: int(x[0])))}')

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

for c in changes:
    print(c)
print(f'File: {len(html)} chars')
