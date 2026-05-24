"""
Pre-process ras_paper.html into a build copy that the html2doc Stage-1 script
(katex_to_mathml.js) can fully convert.

Why this is needed
------------------
The Stage-1 script extracts display equations with /\$\$([^$]+)\$\$/ then does a
LITERAL string replace using the .trim()-ed LaTeX as the search key. The source
writes its two tagged display equations with one padding space inside the
delimiters, e.g.  `...product$$ R_{ij} ... \tag{1} $$The low rank...`.
Because the search key is trimmed but the document text still has the inner
spaces, the literal replace silently misses, leaving Eq.(1)/Eq.(2) as raw
`$$...$$`; the later inline `$...$` pass then mismatches delimiters across those
blocks and garbles the surrounding prose.

Fix: remove exactly one space immediately inside each `$$` display delimiter so
`$$ X $$` becomes `$$X$$`, which matches the trimmed search key. Nothing else is
altered. We do NOT touch the source file; we write a build copy.

This is a formatting normalization only (whitespace adjacent to math
delimiters); it does not change any content, math, or text.
"""
import re
import sys

src = sys.argv[1]
dst = sys.argv[2]

with open(src, "r", encoding="utf-8") as f:
    html = f.read()

# Remove the KaTeX <head> machinery (stylesheet link + the three <script> tags
# that load katex and call renderMathInElement). pandoc drops <head>/<script>
# anyway, but the auto-render config string contains literal $$ and $ delimiters
# (",{left:'$$',right:'$$'...}") which the Stage-1 regexes otherwise mis-detect
# as math, emitting spurious parse warnings and injecting a stray MathML div
# into the dead script. Stripping it makes Stage-1 output clean. This removes
# only browser-side rendering scaffolding, no paper content.
html = re.sub(
    r"<link[^>]*katex[^>]*>",
    "",
    html,
    flags=re.IGNORECASE,
)
html = re.sub(
    r"<script[^>]*katex[^>]*>.*?</script>",
    "",
    html,
    flags=re.IGNORECASE | re.DOTALL,
)
# The auto-render loader is a self-closing-ish <script ... onload="...">...</script>;
# also drop any remaining <script>...</script> that mentions renderMathInElement.
html = re.sub(
    r"<script[^>]*>(?:(?!</script>).)*renderMathInElement(?:(?!</script>).)*</script>",
    "",
    html,
    flags=re.IGNORECASE | re.DOTALL,
)

# Remove the two fixed-position overlay anchors ("Download .docx" and
# "Follow-up paper") that the HTML pins to the screen corners via position:fixed.
# They are page furniture for the web view, not manuscript content; in the DOCX
# they would otherwise land as stray text at the very top of page 1.
html = re.sub(
    r"<a\s+class=['\"](?:docxlink|followlink)['\"][^>]*>.*?</a>",
    "",
    html,
    flags=re.IGNORECASE | re.DOTALL,
)

# Collapse one optional run of spaces right after an opening $$ and right before
# a closing $$. We match a full $$...$$ block (no $ inside) and rebuild it with
# the inner content stripped, so the Stage-1 literal replace (which uses the
# trimmed LaTeX) matches the document text exactly.
def _strip_block(m):
    inner = m.group(1).strip()
    return "$$" + inner + "$$"

html, n = re.subn(r"\$\$([^$]+)\$\$", _strip_block, html)

# Preserve the line structure and indentation of the Algorithm listings.
# In the source these are <div class='algo'> blocks with white-space:pre-wrap
# and literal newlines plus leading-space indentation for nested steps. pandoc
# converts a <div> to a paragraph and collapses HTML whitespace, so without
# this the multi-line algorithm flows into one justified prose blob and the
# indentation is lost. We rewrite each algo block so each source line becomes a
# real line (<br>), and leading spaces become non-breaking spaces (so the
# indentation is not collapsed). The $...$ math inside is left untouched for the
# Stage-1 KaTeX->MathML pass. Content is unchanged; only line breaks and indent
# spacing are made explicit.
def _fix_algo(m):
    block = m.group(0)
    # Separate the caption div (keep it as-is) from the body that follows it.
    cap_match = re.match(
        r"(<div class=['\"]algo['\"]>\s*<div class=['\"]cap['\"]>.*?</div>)(.*)(</div>)$",
        block,
        flags=re.DOTALL,
    )
    if not cap_match:
        return block
    head, body, tail = cap_match.groups()
    lines = body.split("\n")
    out_lines = []
    for ln in lines:
        stripped = ln.lstrip(" ")
        n_lead = len(ln) - len(stripped)
        # Replace leading spaces with non-breaking spaces to keep indentation.
        out_lines.append((" " * n_lead) + stripped)
    # Join with <br> so pandoc emits a line break between algorithm steps.
    new_body = "<br>".join(out_lines)
    return head + new_body + tail

# An algo block is <div class='algo'><div class='cap'>CAP</div>BODY</div> with
# no further nested divs in BODY, i.e. exactly two closing </div>. Match the
# caption non-greedily, then the body non-greedily up to the block's own </div>.
html = re.sub(
    r"<div class=['\"]algo['\"]><div class=['\"]cap['\"]>.*?</div>.*?</div>",
    _fix_algo,
    html,
    flags=re.DOTALL,
)

with open(dst, "w", encoding="utf-8") as f:
    f.write(html)

print(f"display $$...$$ blocks normalized: {n}")
