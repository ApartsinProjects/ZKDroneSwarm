"""
Cycle 5 fixes:
  1. Fix corrupted LaTeX formula in §3.9 match quality definition
  2. Fix §6.3 supervision mode text (still says 'not evaluated here')
  3. Add bar-q to notation table §3.10
"""
import re

with open('docs/index.html', encoding='utf-8') as f:
    html = f.read()

changes = []

# ============================================================
# FIX 1: Replace corrupted match quality formula block
# Find it by the surrounding unique text and replace completely
# ============================================================
# Find the block to replace
START_MARKER = '<p>The <strong>average latent match quality</strong> for a given episode is defined as:'
END_MARKER = '<p>These metrics constitute the set $M$ in the formal definition below.'

idx_start = html.find(START_MARKER)
idx_end   = html.find(END_MARKER)

if idx_start > 0 and idx_end > idx_start:
    old_block = html[idx_start:idx_end]
    # Replace with properly formatted LaTeX (using raw string to avoid escape issues)
    new_block = (
        '<p>The <strong>average latent match quality</strong> for a given episode is defined as:</p>\n\n'
        r'        $$\bar{q} = \frac{1}{E} \sum_{e=1}^{E} \frac{\mathbf{d}_e \cdot \mathbf{t}_e}{\|\mathbf{d}_e\| \|\mathbf{t}_e\|}$$'
        '\n\n'
        '        <p>where $E$ is the number of engagement events in the episode, $\\mathbf{d}_e$ is the latent vector of the drone engaged in event $e$, and $\\mathbf{t}_e$ is the latent vector of the target engaged in event $e$. This is the cosine similarity between the drone and target latent vectors, averaged over all engagements. It serves as a measure of how well the swarm\'s engagement pattern aligns with the hidden compatibility structure.</p>\n\n        '
    )
    html = html[:idx_start] + new_block + html[idx_end:]
    changes.append('FIXED 1: Replaced corrupted match quality formula')
else:
    changes.append(f'MISS 1: Block not found (start={idx_start}, end={idx_end})')

# ============================================================
# FIX 2: §6.3 supervision mode check and fix
# ============================================================
idx_63 = html.find('<h2>6.3 Supervision Mode</h2>')
if idx_63 > 0:
    section_63 = html[idx_63:idx_63+1000]
    if '7.9' in section_63:
        changes.append('OK 2: §6.3 correctly references §7.9')
    elif 'deferred' in section_63.lower():
        # Still has the old text - replace it
        OLD_63 = 'The direct supervision mode, in which the policy regresses directly against immediate rewards rather than accumulated integration estimates, is not evaluated here. Comparison of the two supervision modes is deferred to future work discussed in Section 8.'
        NEW_63 = 'The direct supervision mode, in which the policy regresses directly against immediate rewards rather than accumulated integration estimates, serves as an ablation condition. A systematic comparison of direct versus integration-matrix supervision across five scenario seeds is reported in §7.9 (Table&nbsp;7).'
        if OLD_63 in html:
            html = html.replace(OLD_63, NEW_63, 1)
            changes.append('FIXED 2: §6.3 deferred->§7.9')
        else:
            changes.append('MISS 2: §6.3 old text not found exactly')
    else:
        changes.append('OK 2: §6.3 no "deferred" found (OK)')
else:
    changes.append('MISS 2: §6.3 heading not found')

# ============================================================
# FIX 3: Add bar{q} to §3.10 notation table
# ============================================================
if r'$\bar{q}$' in html:
    changes.append('OK 3: bar{q} already in notation table')
else:
    # Find a suitable insertion point - the last row of the notation table
    # The notation table ends before §3.11
    # Look for the regularization lambda row (last row we know exists)
    OLD_NOTROW = '<tr><td>$\\lambda$</td><td>Regularization coefficient</td></tr>'
    if OLD_NOTROW in html:
        NEW_NOTROW = (OLD_NOTROW + '\n'
            r'                <tr><td>$\bar{q}$</td>'
            '<td>Average latent match quality: mean cosine similarity between drone and target latent vectors over all engagements in an episode</td></tr>')
        html = html.replace(OLD_NOTROW, NEW_NOTROW, 1)
        changes.append('FIXED 3: bar{q} added to notation table after lambda row')
    else:
        # Try another anchor row
        OLD_NOTROW2 = r'<tr><td>$\eta$</td><td>Learning rate</td></tr>'
        if OLD_NOTROW2 in html:
            NEW_NOTROW2 = (OLD_NOTROW2 + '\n'
                r'                <tr><td>$\lambda$</td><td>Regularization coefficient</td></tr>'
                '\n'
                r'                <tr><td>$\bar{q}$</td>'
                '<td>Average latent match quality: mean cosine similarity between drone and target latent vectors over all engagements in an episode</td></tr>')
            # Only add if lambda row doesn't already exist
            html = html.replace(OLD_NOTROW2, NEW_NOTROW2, 1)
            changes.append('FIXED 3: Added lambda and bar{q} rows after eta')
        else:
            # Find eta row with flexible matching
            idx_eta = html.find('Learning rate')
            if idx_eta > 0:
                snippet = html[max(0,idx_eta-50):idx_eta+80]
                changes.append(f'CHECK 3: eta region: {repr(snippet[:100])}')
            else:
                changes.append('MISS 3: Learning rate row not found')

# ============================================================
# Final checks
# ============================================================
em_count = html.count('&mdash;') + html.count('—')
changes.append(f'Em-dashes: {em_count}')
changes.append(f'File: {len(html)} chars')

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

for c in changes:
    print(c)
