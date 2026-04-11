#!/usr/bin/env python3
"""
HTML to Academic DOCX Converter - Main Entry Point

Converts KaTeX-encoded HTML papers to Microsoft Word documents
with native editable equations and scientific formatting.

Usage: python html2doc.py --input paper.html [--output paper.docx]

Requirements:
    pip install pypandoc python-docx
    npm install katex
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SKILL_DIR / "scripts"
REQUIREMENTS_FILE = SKILL_DIR / "requirements.txt"
VALID_PROFILES = ("camera-ready-generic", "review-manuscript")


def has_katex():
    """Return True when Node can resolve the katex package for this skill."""
    if shutil.which("node") is None:
        return False

    result = subprocess.run(
        ["node", "-e", "require.resolve('katex/package.json')"],
        cwd=SKILL_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import pypandoc
    except ImportError:
        missing.append(f"pypandoc (install with: pip install -r {REQUIREMENTS_FILE})")
    else:
        try:
            pypandoc.get_pandoc_path()
        except (OSError, RuntimeError):
            missing.append(
                "pandoc executable (install pandoc or run "
                "\"python -c 'import pypandoc; pypandoc.download_pandoc()'\" once)"
            )

    try:
        from docx import Document  # noqa: F401
    except ImportError:
        missing.append(f"python-docx (install with: pip install -r {REQUIREMENTS_FILE})")

    if shutil.which("node") is None:
        missing.append("node")
    elif not has_katex():
        missing.append("katex (run: npm install from the repository root)")

    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        return False

    return True


def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")

    result = subprocess.run(
        cmd,
        check=False,
    )

    if result.returncode != 0:
        print(f"Error: {description} failed")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert HTML with KaTeX math to academic DOCX'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input HTML file with KaTeX math'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output DOCX file (default: <input>_academic.docx)'
    )
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep intermediate files'
    )
    parser.add_argument(
        '--profile',
        default='camera-ready-generic',
        choices=VALID_PROFILES,
        help='Formatting profile for DOCX conversion'
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_academic.docx")
    )
    mathml_file = output_path.with_name(f"{output_path.stem}.mathml.html")
    docx_file = output_path.with_name(f"{output_path.stem}.converted.docx")

    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)

    print(f"\nConverting: {input_path}")
    print(f"Output: {output_path}")

    if not run_command(
        ["node", str(SCRIPTS_DIR / "katex_to_mathml.js"), "--input", str(input_path), "--output", str(mathml_file)],
        "Stage 1: Converting KaTeX to MathML"
    ):
        sys.exit(1)

    if not run_command(
        [
            sys.executable,
            str(SCRIPTS_DIR / "convert_to_docx.py"),
            "--input",
            str(mathml_file),
            "--output",
            str(docx_file),
            "--profile",
            args.profile,
        ],
        "Stage 2: Converting to DOCX with native equations"
    ):
        sys.exit(1)

    if not run_command(
        [
            sys.executable,
            str(SCRIPTS_DIR / "apply_academic_style.py"),
            "--input",
            str(docx_file),
            "--output",
            str(output_path),
            "--profile",
            args.profile,
        ],
        "Stage 3: Applying academic formatting"
    ):
        sys.exit(1)

    if not args.keep_temp:
        print("\nCleaning up intermediate files...")
        for f in [mathml_file, docx_file]:
            if f.exists():
                f.unlink()

    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print("\nFeatures:")
    print("  - Native Word equations (OMML) - fully editable")
    print("  - Academic formatting (Times New Roman, 1.5 spacing)")
    print("  - Full-width tables with borders")
    print("  - Centered images")
    print("  - Justified text")


if __name__ == '__main__':
    main()
