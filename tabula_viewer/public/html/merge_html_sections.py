#!/usr/bin/env python3
"""Merge HTML paper sections into a single document with proper title/author header."""

import re
from pathlib import Path

# Define the order of sections
sections = [
    'abstract.html',
    'introduction.html',
    'literature-review.html',
    'model-problem-definition.html',
    'methods.html',
    'framework-description.html',
    'experiments.html',
    'results.html',
    'next-steps.html',
    'references.html'
]

html_dir = Path(__file__).parent
output_file = html_dir / 'tabula-drone-full-paper.html'

# Extract body content from each section
contents = []
for section_file in sections:
    path = html_dir / section_file
    with open(path, 'r', encoding='utf-8') as f:
        html = f.read()
        # Extract content between <div class="paper-section-container"> tags
        match = re.search(r'<div class="paper-section-container">(.*?)</div>\s*</body>', html, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Convert relative figure paths to absolute file:// URLs for Pandoc compatibility
            content = re.sub(
                r'src="figures/',
                f'src="file://{html_dir}/figures/',
                content
            )
            contents.append(content)

# Create merged HTML with proper title and author header
merged = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zero-Knowledge Multi-Robot Task Allocation: A Collaborative Filtering Approach</title>
    
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" integrity="sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV" crossorigin="anonymous">
    <link rel="stylesheet" href="paper-shared.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" integrity="sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzUOEleOLALmuqehneUG+vnGctmUb0ZY0l8" crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05" crossorigin="anonymous"
        onload="renderMathInElement(document.body, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\\\(', right: '\\\\)', display: false},
                {left: '\\\\[', right: '\\\\]', display: true}
            ]
        });"></script>
    
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            font-size: 0.95rem;
        }
        
        th, td {
            border: 1px solid rgba(126, 104, 62, 0.2);
            padding: 0.75rem;
            text-align: left;
        }
        
        th {
            background: rgba(255, 248, 233, 0.5);
            font-weight: 700;
            color: #1a1410;
        }
        
        td {
            vertical-align: top;
        }
        
        code {
            font-family: 'Courier New', monospace;
            background: rgba(255, 248, 233, 0.5);
            padding: 0.1rem 0.3rem;
            border-radius: 3px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="paper-section-container">
        <h1 style="text-align: center; margin-bottom: 0.5em;">Zero-Knowledge Multi-Robot Task Allocation: A Collaborative Filtering Approach</h1>
        
        <p style="text-align: center; margin-top: 0; margin-bottom: 0.3em;">
            <strong>Yigal Meshulam</strong>
        </p>
        <p style="text-align: center; margin-top: 0; margin-bottom: 2em; font-style: italic; color: #555;">
            Department of Computer Science
        </p>
        
''' + '\n\n'.join(contents) + '''
    </div>
</body>
</html>
'''

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(merged)

print(f'Merged HTML created: {output_file}')
print(f'Sections merged: {len(sections)}')
