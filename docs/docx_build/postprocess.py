"""
Post-process the styled DOCX to left-align the Algorithm listings.

Why this is needed
------------------
Stage 3 (apply_academic_style.py) justifies every body paragraph. The Algorithm
1-3 listings are multi-line monospace-style blocks (the preprocessor turned
their source newlines into <w:br> line breaks and their indentation into
non-breaking spaces). Justification stretches each algorithm line across the
full text width, opening large inter-word gaps and making the listing hard to
read. Algorithm pseudocode should be left-aligned (ragged right).

This script finds each "Algorithm N:" caption paragraph and the body paragraph
that follows it (the one carrying the <w:br> line breaks) and switches both to
LEFT alignment. It also gives the listing a touch of space-before so it reads as
a distinct block. No text, math, or content is changed; only paragraph
alignment/spacing.

Usage: python postprocess.py --input ras_paper.docx --output ras_paper.docx
"""
import argparse
import re

from docx import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.oxml.ns import qn
from docx.shared import Pt

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

CAP_RE = re.compile(r"^\s*Algorithm\s+\d+\s*:")


def _left_align(paragraph):
    paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    doc = Document(args.input)
    paras = doc.paragraphs

    n_caps = 0
    n_bodies = 0
    for i, p in enumerate(paras):
        text = p.text.strip()
        if CAP_RE.match(text):
            n_caps += 1
            _left_align(p)
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.keep_with_next = True
            # The algorithm body is the run of following paragraphs that carry
            # line breaks (<w:br>) - usually a single paragraph. Left-align each
            # such paragraph until we hit one without a line break.
            j = i + 1
            while j < len(paras):
                body = paras[j]
                has_br = body._element.find(".//" + W + "br") is not None
                # Stop once we leave the listing (a normal paragraph with no
                # internal line breaks, or another caption).
                if not has_br:
                    break
                _left_align(body)
                body.paragraph_format.space_after = Pt(6)
                n_bodies += 1
                j += 1

    doc.save(args.output)
    print(f"Left-aligned {n_caps} algorithm captions and {n_bodies} body blocks")


if __name__ == "__main__":
    main()
