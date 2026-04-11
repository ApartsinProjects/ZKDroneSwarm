# HTML to Academic DOCX Converter

Converts KaTeX-encoded HTML papers to Microsoft Word documents with native editable equations, scientific formatting, and proper typesetting.

## Overview

This tool transforms HTML academic papers (with KaTeX math) into polished Microsoft Word documents featuring:

- **Native Word equations** (OMML) - Fully editable in Word
- **Scientific paper formatting** - Times New Roman, proper spacing, headers
- **Professional tables** - Full-width, borders, formatted headers
- **Centered figures** - Images properly aligned
- **Justified text** - Body text alignment

## Prerequisites

### Required Tools

1. **Python 3.8+** - Core runtime
2. **Node.js** - For KaTeX CLI
3. **Pandoc executable** - Required for document conversion. `pypandoc` is the Python wrapper, but it does not guarantee Pandoc is already installed on the machine.

### Python Packages

```bash
pip install -r .codex/skills/html2doc/requirements.txt
```

### Node.js Packages

```bash
npm install
```

Install from the repository root so Node can resolve the shared `katex` dependency.

## Pipeline

The conversion happens in 3 stages:

```
HTML (KaTeX) → HTML (MathML) → DOCX (OMML) → Formatted DOCX
```

### Stage 1: KaTeX → MathML HTML

Converts LaTeX math (`$$...$$`, `$...$`) to MathML using KaTeX:

```bash
node .codex/skills/html2doc/scripts/katex_to_mathml.js --input paper.html --output paper_mathml.html
```

### Stage 2: MathML → Native Word DOCX

Converts MathML to Word's native OMML equations:

```bash
python3 .codex/skills/html2doc/scripts/convert_to_docx.py --input paper_mathml.html --output paper.docx
```

### Stage 3: Apply Academic Formatting

Applies scientific paper styling:

```bash
python3 .codex/skills/html2doc/scripts/apply_academic_style.py --input paper.docx --output paper_formatted.docx
```

## Quick Start

### One-Command Conversion

Run the full pipeline:

```bash
python3 .codex/skills/html2doc/html2doc.py --input paper_v5_final.html --output paper_final.docx
```

### Or Step by Step

```bash
# Stage 1: Convert math to MathML
node .codex/skills/html2doc/scripts/katex_to_mathml.js --input paper.html --output paper_mathml.html

# Stage 2: Convert to DOCX with native math
python3 .codex/skills/html2doc/scripts/convert_to_docx.py --input paper_mathml.html --output paper.docx

# Stage 3: Apply formatting
python3 .codex/skills/html2doc/scripts/apply_academic_style.py --input paper.docx --output paper_formatted.docx
```

## Configuration

### Custom Styles

Adjust the profile dictionaries in `scripts/create_reference_doc.py` and `scripts/apply_academic_style.py` to customize:

- Font family (default: Times New Roman)
- Font sizes (body: 11pt, headings: various)
- Line spacing (default: 1.5)
- Margins (default: 1 inch)
- Table styling
- Heading formats

### Equation Detection

The converter automatically detects:
- Display math: `$$...$$`
- Inline math: `$...$`

## Troubleshooting

### Equations Not Converting

1. Ensure KaTeX is properly installed: `npm list katex`
2. Check that math uses `$...$` or `$$...$$` format
3. Verify HTML has no syntax errors in math expressions
4. Re-run with `--keep-temp` so you can inspect the intermediate MathML HTML

### Images Not Showing

1. Ensure figures are in relative path from HTML
2. SVG images need to be PNG/JPG for Word compatibility

### Tables Not Full-Width

Tables inherit width from source. Use `--table-width 100` flag:

```bash
python3 .codex/skills/html2doc/scripts/apply_academic_style.py --input paper.docx --output paper.docx --table-width 100
```

## Output

The final DOCX includes:

| Feature | Status |
|---------|--------|
| Native equations | ✅ Editable in Word |
| Tables | ✅ Full-width, bordered |
| Images | ✅ Centered |
| Typography | ✅ Times New Roman |
| Formatting | ✅ Justified, proper spacing |

## Skill Usage

This directory now includes [SKILL.md](/Users/ymeshulam/PycharmProjects/TabulaDrone/.codex/skills/html2doc/SKILL.md), so Codex can use `html2doc` as a repo-local skill instead of treating it as an undocumented helper script bundle.

## Files

```
html2doc/
├── SKILL.md                  # Codex skill entrypoint
├── README.md                 # This file
├── html2doc.py               # Main entry point
└── scripts/
    ├── katex_to_mathml.js    # Stage 1: KaTeX → MathML
    ├── convert_to_docx.py    # Stage 2: MathML → DOCX
    ├── apply_academic_style.py  # Stage 3: Formatting
    └── create_reference_doc.py  # Reference style generator
```
