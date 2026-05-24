# Converting ras_paper.html to ras_paper.docx (RAS-compliant Word)

This documents how `docs/ras_paper.html` was converted into a well-formatted,
RAS (Elsevier Robotics and Autonomous Systems) compliant Microsoft Word file
`docs/ras_paper.docx` with native, editable Word equations, embedded figures,
and full-width tables, using the `html2doc` skill plus three small local helper
scripts in `docs/docx_build/`.

The output targets RAS "your paper your way": a single-column, justified,
Times New Roman manuscript (the skill's `camera-ready-generic` profile). No line
numbers and no double spacing (that is the `review-manuscript` profile, which RAS
does not require).

## Toolchain actually used (verified versions)

All paths are Git Bash style on this Windows 11 machine.

| Tool | Version | Location / notes |
| --- | --- | --- |
| pandoc | 3.1.13 | on PATH (used by pypandoc, Stage 2) |
| node | v24.13.1 | on PATH (Stage 1) |
| katex | 0.16.45 | inside the skill: `~/.claude/skills/html2doc/node_modules/katex` (reach via `NODE_PATH`) |
| Python | 3.14.3 | `/c/Python314/python` -> this is the `<PYTHON>` with the deps |
| pypandoc | 1.17 | in Python 3.14 (Stage 2) |
| python-docx | 1.2.0 | in Python 3.14 (Stage 3 + post-process) |
| PyMuPDF (fitz) | 1.27.2.2 | in Python 3.14 (rasterize PDF pages for inspection) |
| Microsoft Word | 16.0 | COM automation, used ONLY to render docx -> pdf for visual QA |

Resolution of `<PYTHON>`: Python 3.11 is also installed
(`C:\Users\apart\AppData\Local\Programs\Python\Python311\python.exe`) but it does
NOT have pypandoc, so it cannot run Stage 2. Use `/c/Python314/python` for all
three stages and the helper scripts.

### What is NOT installed (and the consequence)

- No LibreOffice (`soffice`), no `docx2pdf`, no standalone Office binary on PATH,
  no `pdftoppm`, no `wkhtmltopdf`, no LaTeX engine (pdflatex/xelatex/tectonic).
- Microsoft Word IS registered for COM (`Word.Application`, version 16.0), so the
  docx-to-pdf render for QA is done through Word via a small PowerShell script
  (`docx_build/docx2pdf.ps1`). This is the most faithful possible preview because
  it is the same engine that will open the file. If Word is ever uninstalled,
  fall back to installing LibreOffice and using
  `soffice --headless --convert-to pdf`.

## The exact commands (run from the docs/ directory)

CRITICAL: run everything with the current directory set to `docs/`. The figure
references in the HTML are relative (`src="figures/F*.png"`); pandoc resolves
relative image paths against the current working directory, so running from
`docs/` is what makes all 12 figures embed. (Stage 2 internally copies the HTML
to a temp file, but the CWD-based resource path still resolves `figures/...`
against `docs/`, so do not move the figures and do not run from elsewhere.)

```bash
cd /e/Projects/ColabDroneSwarm/docs
SKILL=/c/Users/apart/.claude/skills/html2doc
PY=/c/Python314/python
mkdir -p build_tmp

# Stage 0 (local helper): normalize the source into a build copy.
#   - removes the two fixed-position overlay <a> links ("Download .docx",
#     "Follow-up paper") so they do not land as stray text on page 1;
#   - strips one padding space inside each $$ ... $$ so the Stage-1 literal
#     replace matches (otherwise Eq. (1) and Eq. (2) silently fail to convert);
#   - removes the KaTeX <head> <link>/<script> (its delimiter config string
#     contains literal $$/$ that the Stage-1 regex mis-detects as math);
#   - turns the Algorithm 1-3 <div class='algo'> blocks' newlines into <br> and
#     their leading-space indentation into non-breaking spaces, so the listings
#     keep their line structure instead of collapsing into one prose blob.
$PY docx_build/preprocess.py ras_paper.html build_tmp/ras_paper_src.html

# Stage 1: KaTeX -> MathML. NODE_PATH lets `node` resolve the skill's katex.
NODE_PATH="$SKILL/node_modules" node "$SKILL/scripts/katex_to_mathml.js" \
    --input  build_tmp/ras_paper_src.html \
    --output build_tmp/build_mathml.html

# Stage 2: MathML -> DOCX with native OMML equations (single-column profile).
$PY "$SKILL/scripts/convert_to_docx.py" \
    --input  build_tmp/build_mathml.html \
    --output build_tmp/build_converted.docx \
    --profile camera-ready-generic

# Stage 3: academic styling (Times New Roman, justified, full-width tables,
# centered figures/captions, page numbers, 1.0in margins, 1.15 spacing).
$PY "$SKILL/scripts/apply_academic_style.py" \
    --input  build_tmp/build_converted.docx \
    --output ras_paper.docx \
    --profile camera-ready-generic

# Stage 4 (local helper): left-align the Algorithm listings. Stage 3 justifies
# every body paragraph, which stretches the (now multi-line) pseudocode; this
# switches the 3 algorithm captions + bodies to left alignment (ragged right).
$PY docx_build/postprocess.py --input ras_paper.docx --output ras_paper.docx
```

### Render to PDF and rasterize for visual QA (optional but recommended)

```bash
cd /e/Projects/ColabDroneSwarm/docs
# docx -> pdf via Word COM (wdFormatPDF = 17)
powershell.exe -NoProfile -ExecutionPolicy Bypass \
    -File docx_build/docx2pdf.ps1 -In ras_paper.docx -Out build_tmp/ras_paper_preview.pdf
# pdf -> page PNGs at 150 dpi for inspection
/c/Python314/python -c "import fitz,os; d=fitz.open('build_tmp/ras_paper_preview.pdf'); os.makedirs('build_tmp/pages',exist_ok=True); m=fitz.Matrix(150/72,150/72); [d[i].get_pixmap(matrix=m).save(f'build_tmp/pages/p{i+1:02d}.png') for i in range(d.page_count)]; print(d.page_count,'pages')"
```

The preview PDF and page PNGs are intermediate; delete `build_tmp/` when done.
Only `ras_paper.docx`, this notes file, and the reusable helpers in
`docx_build/` should remain.

## Profile chosen and why

`camera-ready-generic`: single-column, justified body, Times New Roman 11 pt,
1.0 in margins, 1.15 line spacing, full-width bordered tables, centered figures,
centered footer page numbers. This matches RAS single-column "your paper your
way" submission. Do NOT use `review-manuscript` (it adds continuous line numbers
and 1.5 spacing) unless a specific reviewer/journal stage asks for line numbers.

## Issues found and how they were fixed

1. Eq. (1) and Eq. (2) did not convert (most important). The Stage-1 script
   extracts `$$...$$` then does a LITERAL string replace using the trimmed
   LaTeX as the key. The source writes both tagged display equations with one
   padding space inside the delimiters (`...product$$ R_{ij} ... \tag{1} $$The
   low rank...`), so the trimmed key never matched the spaced document text;
   the equations stayed as raw `$$...$$`, and the later inline `$...$` pass then
   mismatched delimiters across those blocks and garbled the surrounding prose
   ("where for an offered set the oracle picks ..." turned into letter-by-letter
   MathML). Fix: `preprocess.py` strips the one padding space inside every
   `$$ ... $$` so the literal replace matches. Result: both equations now render
   as native centered OMML display equations with their `(1)`/`(2)` numbers, and
   the prose around them is intact.

2. KaTeX `<head>` script false-positive. The auto-render config string
   (`{left:'$$',right:'$$'},{left:'$',right:'$'}`) contains literal `$$`/`$`,
   which Stage 1 mis-detected as math, emitted a parse warning, and injected a
   stray MathML div into the dead script. pandoc drops `<head>`/`<script>` so it
   never reached the body, but to keep Stage-1 output clean `preprocess.py`
   removes the KaTeX `<link>` and `<script>` tags from the build copy.

3. Stray overlay links on page 1. The HTML pins two `position:fixed` anchors
   ("Download .docx", "Follow-up paper") to the screen corners. pandoc rendered
   them as plain text at the very top of page 1. Fix: `preprocess.py` deletes
   those two `<a>` anchors (the CSS rules can stay; pandoc ignores CSS).

4. Algorithm listings collapsed. The `<div class='algo'>` blocks rely on
   `white-space:pre-wrap` with literal newlines and leading-space indentation.
   pandoc collapses HTML whitespace, so Algorithms 1-3 flowed into single
   justified prose blobs with the steps and indentation lost. Two-part fix:
   (a) `preprocess.py` converts each algo block's newlines to `<br>` and its
   leading spaces to non-breaking spaces (pandoc keeps both), restoring the
   line-by-line structure with nested indentation; (b) `postprocess.py`
   left-aligns the algorithm captions and bodies (Stage 3 had justified them,
   which stretched the lines). Result: all three algorithms render as clean,
   left-aligned pseudocode with native inline math and correct nesting.

## Manual post-processing applied

Only `docx_build/postprocess.py` (Stage 4 above): left-aligns the 3 Algorithm
captions and their body blocks. No content, math, or wording was changed in any
step; the helpers only adjust whitespace, alignment, and remove web-only page
furniture.

## Known minor cosmetic notes (not fixed; acceptable for RAS)

- In Table 1 the narrow header cell "Prior knowledge" wraps as "Prior
  knowledg/e" because of autofit column widths. Legible; a human can widen that
  column in Word if desired.
- Wide tables (Table 2 has 12 rows, Table 3 has 9 rows) split across a page
  break and the header row does not repeat on the continuation page. If a
  repeating header is wanted, set it in Word (Table Tools > Layout > Repeat
  Header Rows) or via python-docx (`tr.trPr` + `w:tblHeader`).
- The source has 12 figures (Figures 1-12) and 12 `<img>` tags, not 13. All 12
  external PNGs were embedded.

## Verification of the result (all passed)

- Math: 480 native OMML `<m:oMath>` (4 `<m:oMathPara>` display blocks), 0 raw
  `$`/`\tag` left in the body. Eq. (1), Eq. (2), and the theorem inline math
  spot-checked in the rendered PDF.
- Figures: 12 `<w:drawing>` + 12 `word/media/*.png` embedded, all centered.
- Tables: all 4 present (Table 1 8x6, Table 2 12x7, Table 3 9x6, Table 4 4x3),
  full-width with borders and shaded headers, no overflow off the page.
- Front/back matter in order: Highlights, Abstract, Keywords, Sections 1-8
  (with 6.1-6.8), CRediT, Declaration of competing interest, Funding, Data
  availability, References (48 entries [1]-[48]), Appendices A-F, Algorithms 1-3.
- Layout: single-column, justified body, TNR 11 pt, 1.0 in margins, centered
  footer page numbers; 26 pages.

## Pre-flight checklist for re-converting after the HTML changes

(For example after the pending Figures 4/7 16-seed update, or any edit to
`ras_paper.html`. `docs/figures/F18*.png` and `F21*.png` may be rewritten by a
concurrent 16-seed re-run; if a figure looks truncated, just rebuild later.)

1. `cd /e/Projects/ColabDroneSwarm/docs` (mandatory for figure embedding).
2. Confirm tools: `pandoc --version` (3.1+), `node --version`,
   `/c/Python314/python -c "import pypandoc, docx"` (no error). If Python 3.14
   ever loses the deps, `pip install pypandoc python-docx pymupdf` into it.
3. Confirm every `src="figures/F*.png"` referenced by the HTML exists on disk
   (`grep -o 'src="figures/[^"]*"' ras_paper.html`).
4. Run Stages 0-4 exactly as above (do not skip the preprocess/postprocess
   helpers; without them Eq. (1)/(2) break and the algorithms collapse).
5. Sanity-check the new docx without opening Word:
   `/c/Python314/python -c "import zipfile,re; z=zipfile.ZipFile('ras_paper.docx'); x=z.read('word/document.xml').decode(); print('OMML',x.count('<m:oMath'),'imgs',x.count('<w:drawing'),'refs',x.count('ReferenceEntry'),'raw-$',len(re.findall(r'(?<![0-9])\\\\\$',x)))"`
   Expect OMML ~480, imgs = (number of figures), refs = 48, raw-$ = 0.
   (Counts shift if the HTML content changed; the point is OMML and imgs are
   non-zero and raw-$ is 0.)
6. Optional QA: render to PDF via `docx_build/docx2pdf.ps1` and rasterize with
   PyMuPDF (commands above); eyeball Eq. (1)/(2), the four tables, and the three
   algorithm listings.
7. Clean up: delete `build_tmp/`. Keep `ras_paper.docx`, this file, and
   `docx_build/`.

## Files

- `docs/ras_paper.docx` - the deliverable.
- `docs/DOCX_CONVERSION_NOTES.md` - this file.
- `docs/docx_build/preprocess.py` - Stage 0 source normalizer.
- `docs/docx_build/postprocess.py` - Stage 4 algorithm left-aligner.
- `docs/docx_build/docx2pdf.ps1` - Word COM docx-to-pdf for QA.
- `docs/build_tmp/` - throwaway intermediates (safe to delete).
