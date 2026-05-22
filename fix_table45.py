"""Add UCB-Indep column to §4.5 Summary of Compared Methods table."""

with open('docs/index.html', encoding='utf-8') as f:
    html = f.read()

TD = 'style="border: 1px solid rgba(126, 104, 62, 0.2); padding: 0.75rem;"'

changes = []

def rep(old, new, label):
    global html
    if old in html:
        html = html.replace(old, new, 1)
        changes.append(f'FIXED: {label}')
    else:
        changes.append(f'MISS: {label}')

# Header row
rep(
    f'<th {TD} text-align: left;">Axis</th>\n'
    f'                    <th {TD} text-align: left;">Random</th>\n'
    f'                    <th {TD} text-align: left;">Oracle</th>\n'
    f'                    <th {TD} text-align: left;">Matrix-factorization</th>',
    f'<th {TD} text-align: left;">Axis</th>\n'
    f'                    <th {TD} text-align: left;">Random</th>\n'
    f'                    <th {TD} text-align: left;">UCB-Indep</th>\n'
    f'                    <th {TD} text-align: left;">Oracle</th>\n'
    f'                    <th {TD} text-align: left;">Matrix-factorization</th>',
    'Header: add UCB-Indep column'
)

# Row 1: Knowledge of latent vectors
rep(
    f'<td {TD}>Knowledge of latent drone/target vectors</td>\n'
    f'                    <td {TD}>None</td>\n'
    f'                    <td {TD}>Full (privileged)</td>\n'
    f'                    <td {TD}>None</td>',
    f'<td {TD}>Knowledge of latent drone/target vectors</td>\n'
    f'                    <td {TD}>None</td>\n'
    f'                    <td {TD}>None</td>\n'
    f'                    <td {TD}>Full (privileged)</td>\n'
    f'                    <td {TD}>None</td>',
    'Row 1: Knowledge'
)

# Row 2: Access to target HP
rep(
    f'<td {TD}>Access to target HP</td>\n'
    f'                    <td {TD}>None</td>\n'
    f'                    <td {TD}>Full (privileged)</td>\n'
    f'                    <td {TD}>None</td>',
    f'<td {TD}>Access to target HP</td>\n'
    f'                    <td {TD}>None</td>\n'
    f'                    <td {TD}>None</td>\n'
    f'                    <td {TD}>Full (privileged)</td>\n'
    f'                    <td {TD}>None</td>',
    'Row 2: HP access'
)

# Row 3: Information used at decision time
rep(
    f'<td {TD}>Information used at decision time</td>\n'
    f'                    <td {TD}>Active/inactive flags</td>\n'
    f'                    <td {TD}>Latent vectors + HP</td>\n'
    f'                    <td {TD}>Own learned latent row</td>',
    f'<td {TD}>Information used at decision time</td>\n'
    f'                    <td {TD}>Active/inactive flags</td>\n'
    f'                    <td {TD}>Per-arm reward statistics</td>\n'
    f'                    <td {TD}>Latent vectors + HP</td>\n'
    f'                    <td {TD}>Own learned latent row</td>',
    'Row 3: Information at decision time'
)

# Row 4: Learning
rep(
    f'<td {TD}>Learning</td>\n'
    f'                    <td {TD}>None</td>\n'
    f'                    <td {TD}>None (planner)</td>\n'
    f'                    <td {TD}>Online SGD on embeddings</td>',
    f'<td {TD}>Learning</td>\n'
    f'                    <td {TD}>None</td>\n'
    f'                    <td {TD}>Per-arm mean + UCB</td>\n'
    f'                    <td {TD}>None (planner)</td>\n'
    f'                    <td {TD}>Online SGD on embeddings</td>',
    'Row 4: Learning'
)

# Row 5: Coordination mechanism
rep(
    f'<td {TD}>Coordination mechanism</td>\n'
    f'                    <td {TD}>None</td>\n'
    f'                    <td {TD}>Centralized greedy planner</td>\n'
    f'                    <td {TD}>Implicit via shared observations</td>',
    f'<td {TD}>Coordination mechanism</td>\n'
    f'                    <td {TD}>None</td>\n'
    f'                    <td {TD}>None</td>\n'
    f'                    <td {TD}>Centralized greedy planner</td>\n'
    f'                    <td {TD}>Implicit via shared observations</td>',
    'Row 5: Coordination'
)

# Row 6: ZK-compliant
rep(
    f'<td {TD}>ZK-compliant</td>\n'
    f'                    <td {TD}>Yes</td>\n'
    f'                    <td {TD}>No</td>\n'
    f'                    <td {TD}>Yes</td>',
    f'<td {TD}>ZK-compliant</td>\n'
    f'                    <td {TD}>Yes</td>\n'
    f'                    <td {TD}>Yes</td>\n'
    f'                    <td {TD}>No</td>\n'
    f'                    <td {TD}>Yes</td>',
    'Row 6: ZK-compliant'
)

# Row 7: Memory across steps/episodes
rep(
    f'<td {TD}>Memory across steps/episodes</td>\n'
    f'                    <td {TD}>None</td>\n'
    f'                    <td {TD}>None</td>\n'
    f'                    <td {TD}>Persistent embeddings</td>',
    f'<td {TD}>Memory across steps/episodes</td>\n'
    f'                    <td {TD}>None</td>\n'
    f'                    <td {TD}>Persistent arm stats</td>\n'
    f'                    <td {TD}>None</td>\n'
    f'                    <td {TD}>Persistent embeddings</td>',
    'Row 7: Memory'
)

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

for c in changes:
    print(c)
print(f'File: {len(html)} chars')
