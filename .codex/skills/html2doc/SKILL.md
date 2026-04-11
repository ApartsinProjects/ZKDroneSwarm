---
name: html2doc
description: Convert HTML academic papers with KaTeX math into polished DOCX files with native Word equations and manuscript-style formatting.
---

# HTML to DOCX

Use this skill when the user wants an HTML paper or report turned into a Microsoft Word document without flattening equations into images.

## Goal

Produce a `.docx` file that preserves:
- editable equations via Word OMML
- academic/manuscript typography
- centered figures and formatted tables

## When to Use It

- The source document is already in HTML
- Math is written as KaTeX/LaTeX delimiters such as `$...$` or `$$...$$`
- The user wants a Word deliverable for editing, review, or submission

## Preconditions

Before running the converter, make sure these dependencies are available:
- Python packages from `requirements.txt` in the skill directory
- Node.js with `katex` installed from the repository root via `npm install`
- A Pandoc executable installed on the machine, or downloaded once through `pypandoc`

## Default Command

Run the main script from wherever you've placed the skill:

```bash
python3 html2doc.py --input path/to/paper.html --output path/to/paper.docx
```

Or with an absolute/relative path to the skill directory:

```bash
python3 /path/to/html2doc/html2doc.py --input paper.html --output paper.docx
```

## Profiles

- `camera-ready-generic`: tighter spacing for polished final documents
- `review-manuscript`: roomier spacing for reviewer-friendly drafts

## Useful Options

- `--profile review-manuscript`: switch formatting profile
- `--keep-temp`: keep intermediate MathML HTML and pre-style DOCX files for debugging

## Workflow

1. Convert KaTeX-style math in the HTML to MathML.
2. Convert the MathML HTML to DOCX through Pandoc so equations become native OMML.
3. Apply Word styling for front matter, captions, tables, references, and pagination.

## Constraints

- Math detection is optimized for `$...$` and `$$...$$`; unusual delimiter schemes may need preprocessing.
- Word handles raster images more reliably than SVG.
- If the user wants publication-specific styling, adjust the profile logic in the scripts before re-running.
