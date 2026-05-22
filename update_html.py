import pathlib, re

def md_inline(text):
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    text = re.sub(r'\$([^$\n]+?)\$', lambda m: r'\(' + m.group(1) + r'\)', text)
    return text

def md_section_to_html(md_text, indent='        '):
    lines = md_text.split('\n')
    html_parts = []
    para = []

    def flush():
        if para:
            body = ' '.join(l.strip() for l in para if l.strip())
            html_parts.append(f'{indent}<p>\n{indent}    {md_inline(body)}\n{indent}</p>')
            para.clear()

    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if re.match(r'^# [^#]', stripped): i += 1; continue
        if re.match(r'^## ', stripped):
            flush(); html_parts.append(f'{indent}<h2>{md_inline(stripped[3:])}</h2>'); i += 1; continue
        if re.match(r'^### ', stripped):
            flush(); html_parts.append(f'{indent}<h3>{md_inline(stripped[4:])}</h3>'); i += 1; continue
        if re.match(r'^#### ', stripped):
            flush(); html_parts.append(f'{indent}<h4>{md_inline(stripped[5:])}</h4>'); i += 1; continue
        if stripped.startswith('$$'):
            flush()
            math_lines = [stripped[2:]]
            i += 1
            while i < len(lines):
                s = lines[i].strip()
                if s.endswith('$$'): math_lines.append(s[:-2]); i += 1; break
                math_lines.append(s); i += 1
            html_parts.append(f'{indent}<div class="math display">\\[{"".join(math_lines)}\\]</div>')
            continue
        if re.match(r'^[-*] ', stripped):
            flush()
            items = []
            while i < len(lines) and re.match(r'^[-*] ', lines[i].strip()):
                items.append(f'{indent}    <li>{md_inline(lines[i].strip()[2:])}</li>'); i += 1
            html_parts.append(f'{indent}<ul>\n' + '\n'.join(items) + f'\n{indent}</ul>')
            continue
        if stripped.startswith('|'):
            flush()
            trows = []; header_done = False
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = lines[i].strip()
                if re.match(r'^\|[-:| ]+\|$', row): header_done = True; i += 1; continue
                cells = [c.strip() for c in row.strip('|').split('|')]
                tag = 'td' if header_done else 'th'
                trows.append('<tr>' + ''.join(f'<{tag}>{md_inline(c)}</{tag}>' for c in cells) + '</tr>'); i += 1
            html_parts.append(f'{indent}<table>\n' + '\n'.join(f'{indent}    {r}' for r in trows) + f'\n{indent}</table>')
            continue
        if stripped == '': flush(); i += 1; continue
        para.append(raw); i += 1

    flush()
    return '\n\n'.join(html_parts)

# Render introduction and lit review
intro_html  = md_section_to_html(pathlib.Path('docs/academic-paper/introduction.md').read_text(encoding='utf-8'))
litrev_html = md_section_to_html(pathlib.Path('docs/academic-paper/literature-review.md').read_text(encoding='utf-8'))

# New references HTML [33]-[45]
new_refs_entries = [
    ('[33]', 'F. A. Oliehoek and C. Amato, <em>A Concise Introduction to Decentralized POMDPs</em>, Springer Briefs in Intelligent Systems. Cham: Springer, 2016.'),
    ('[34]', 'D. S. Bernstein, R. Givan, N. Immerman, and S. Zilberstein, "The complexity of decentralized control of Markov decision processes," <em>Mathematics of Operations Research</em>, vol. 27, no. 4, pp. 819&ndash;840, 2002.'),
    ('[35]', 'Y. Yang, R. Luo, M. Li, M. Zhou, W. Zhang, and J. Wang, "Mean field multi-agent reinforcement learning," in <em>Proceedings of the 35th International Conference on Machine Learning</em>, PMLR, vol. 80, 2018, pp. 5571&ndash;5580.'),
    ('[36]', 'E. Hillel, Z. Karnin, T. Koren, R. Lempel, and O. Somekh, "Distributed exploration in multi-armed bandits," in <em>Advances in Neural Information Processing Systems</em>, vol. 26, 2013.'),
    ('[37]', 'P.-A. Wang, A. Proutiere, K. Ariu, Y. Jedra, and A. Russo, "Optimal algorithms for multiplayer multi-armed bandits," in <em>Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics</em>, PMLR, vol. 108, 2020.'),
    ('[38]', 'M. Agarwal, V. Aggarwal, and K. Azizzadenesheli, "Multi-agent multi-armed bandits with limited communication," <em>Journal of Machine Learning Research</em>, vol. 23, no. 212, pp. 1&ndash;24, 2022.'),
    ('[39]', 'D. Nagaraj, S. Kowshik, A. Agarwal, P. Netrapalli, and P. Jain, "Multi-user reinforcement learning with low rank rewards," in <em>Proceedings of the 40th International Conference on Machine Learning</em>, PMLR, vol. 202, 2023.'),
    ('[40]', 'S. Katariya, B. Kveton, C. Szepesvári, C. Vernade, and Z. Wen, "Stochastic rank-1 bandits," in <em>Proceedings of the 20th International Conference on Artificial Intelligence and Statistics</em>, PMLR, vol. 54, 2017.'),
    ('[41]', 'Y. Kang, C.-J. Hsieh, and T. C. M. Lee, "Efficient frameworks for generalized low-rank matrix bandit problems," in <em>Advances in Neural Information Processing Systems</em>, vol. 35, 2022.'),
    ('[42]', 'T. Wang, H. Dong, V. Lesser, and C. Zhang, "ROMA: Multi-agent reinforcement learning with emergent roles," in <em>Proceedings of the 37th International Conference on Machine Learning</em>, PMLR, vol. 119, 2020.'),
    ('[43]', 'T. Wang, T. Gupta, A. Mahajan, B. Peng, S. Whiteson, and C. Zhang, "RODE: Learning roles to decompose multi-agent tasks," in <em>Proceedings of the 9th International Conference on Learning Representations</em>, 2021.'),
    ('[44]', 'J. N. Hu, A. Lerer, A. Peysakhovich, and J. Foerster, "Other-play for zero-shot coordination," in <em>Proceedings of the 37th International Conference on Machine Learning</em>, PMLR, vol. 119, 2020.'),
    ('[45]', 'Y. Zhong, J. G. Kuba, X. Feng, S. Hu, J. Ji, and Y. Yang, "Heterogeneous-agent reinforcement learning," <em>Journal of Machine Learning Research</em>, vol. 25, no. 32, pp. 1&ndash;67, 2024.'),
]
new_refs_html = '\n'.join(
    f'        <p>{num} {body}</p>'
    for num, body in new_refs_entries
)

# Patterns that delimit the regions we want to replace
INTRO_START = re.compile(r'(<h1>1\. Introduction</h1>)')
INTRO_END   = re.compile(r'(?=<h1>2\. Literature Review</h1>)')
LIT_START   = re.compile(r'(<h1>2\. Literature Review</h1>)')
LIT_END     = re.compile(r'(?=<h1>3\. Model and Problem Definition</h1>)')

html_files = [
    'docs/index.html',
    'docs/paper.html',
    'tabula_viewer/public/html/tabula-drone-full-paper.html',
]

for fpath in html_files:
    p = pathlib.Path(fpath)
    html = p.read_text(encoding='utf-8')
    original_len = len(html)

    # 1. Replace introduction body
    m1 = INTRO_START.search(html)
    m2 = INTRO_END.search(html)
    if m1 and m2 and m2.start() > m1.end():
        html = html[:m1.end()] + '\n        \n' + intro_html + '\n\n' + html[m2.start():]
    else:
        print(f"  WARNING: could not find intro boundaries in {fpath}")

    # 2. Replace literature review body
    m3 = LIT_START.search(html)
    m4 = LIT_END.search(html)
    if m3 and m4 and m4.start() > m3.end():
        html = html[:m3.end()] + '\n        \n' + litrev_html + '\n\n' + html[m4.start():]
    else:
        print(f"  WARNING: could not find litrev boundaries in {fpath}")

    # 3. Fix [1] page range 9-19 -> 9-20 in references
    html = html.replace('pp. 9–19, 2008', 'pp. 9–20, 2008')
    html = html.replace('pp. 9-19, 2008', 'pp. 9-20, 2008')

    # 4. Fix [24] author order
    html = re.sub(
        r'A\.\s*Mnih and R\.\s*Salakhutdinov[^"]*"Probabilistic matrix factorization"',
        'R. Salakhutdinov and A. Mnih, "Probabilistic matrix factorization"',
        html
    )

    # 5. Replace [19] Prisadnikov with Kawale et al.
    html = re.sub(
        r'N\. Prisadnikov[^<]*ETH Z[^<]*2014\.',
        'J. Kawale, H. H. Bui, B. Kveton, L. Tran-Thanh, and S. Chawla, "Efficient Thompson sampling for online matrix-factorization recommendation," in <em>Advances in Neural Information Processing Systems</em>, vol. 28, 2015, pp. 1297&ndash;1305.',
        html, flags=re.DOTALL
    )

    # 6. Append new references [33]-[45] after [32]
    ref32_pat = re.compile(r'(<p>\[32\].*?</p>)', re.DOTALL)
    m32 = list(ref32_pat.finditer(html))
    if m32 and '[33]' not in html:
        pos = m32[-1].end()
        html = html[:pos] + '\n\n' + new_refs_html + html[pos:]
    elif '[33]' in html:
        print(f"  INFO: [33] already present in {fpath}, skipping ref append")

    p.write_text(html, encoding='utf-8')
    new_len = len(html)
    print(f"Updated {fpath}: {original_len} -> {new_len} chars (delta {new_len - original_len:+d})")

print("\nAll done.")
