---
name: md2html
description: Convert academic-formatted Markdown documents into standalone, styled HTML files with KaTeX math rendering, using a shared CSS stylesheet and configurable output paths.
---

# Markdown to Academic HTML

Use this skill when converting Markdown documents (formatted per the `academic-paper-formatting` skill) into styled, self-contained HTML pages suitable for browser viewing or embedding in a web application.

## Goal

Produce clean, semantic HTML files that faithfully render academic Markdown content — including LaTeX math, section hierarchy, lists, tables, and cross-references — using a shared CSS stylesheet and client-side KaTeX rendering.

---

## Output Path Resolution

Resolve the output directory using the following priority:

1. **Explicit override**: If the user specifies an output path in the request, use it.
2. **Project config**: If `config.json` exists in the skill directory, read `outputDir` and `cssPath` from it.
3. **Convention default**: Place the output HTML file in an `html/` subdirectory relative to the source Markdown file. Create the subdirectory if it does not exist.

### Config File Format

`config.json` (optional, in the skill directory):

```json
{
  "outputDir": "relative/path/from/workspace/root",
  "cssPath": "paper-shared.css",
  "cssUrl": null
}
```

- **outputDir**: Directory for generated HTML files, relative to workspace root.
- **cssPath**: Filename of the shared CSS file, expected to live alongside the output HTML. When this is set, the generated HTML uses a relative `<link>` to this file.
- **cssUrl**: Absolute URL to an external CSS file. When set, takes precedence over `cssPath`.

If neither config nor override is provided, the skill copies `paper-shared.css` into the output directory alongside the HTML file.

---

## HTML Template Structure

Every generated HTML file must follow this skeleton:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{SECTION_TITLE}} - {{DOCUMENT_TITLE}}</title>

    <!-- KaTeX CSS -->
    <link rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"
          integrity="sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV"
          crossorigin="anonymous">

    <!-- Shared paper CSS -->
    <link rel="stylesheet" href="{{CSS_REFERENCE}}">

    <!-- KaTeX JS + auto-render -->
    <script defer
            src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"
            integrity="sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzUOEleOLALmuqehneUG+vnGctmUb0ZY0l8"
            crossorigin="anonymous"></script>
    <script defer
            src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
            integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05"
            crossorigin="anonymous"
            onload="renderMathInElement(document.body, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\(', right: '\\)', display: false},
                    {left: '\\[', right: '\\]', display: true}
                ]
            });"></script>
</head>
<body>
    <div class="paper-section-container">
        {{CONTENT}}
    </div>
</body>
</html>
```

### Placeholders

- **{{SECTION_TITLE}}**: The `<h1>` heading text of the section.
- **{{DOCUMENT_TITLE}}**: A short project or paper title. Use the top-level `#` heading from the source, or ask the user.
- **{{CSS_REFERENCE}}**: Resolved from config (`cssPath` or `cssUrl`) or default (`paper-shared.css`).
- **{{CONTENT}}**: The converted HTML body (see conversion rules below).

---

## Conversion Rules

### Headings

| Markdown | HTML |
|----------|------|
| `# Title` | `<h1>Title</h1>` |
| `## Section` | `<h2>Section</h2>` |
| `### Subsection` | `<h3>Subsection</h3>` |

### Paragraphs

- Each paragraph becomes a `<p>` element.
- Preserve blank-line separation between paragraphs.
- Apply `<strong>` for `**bold**` and `<em>` for `*italic*`.
- Apply `<code>` for inline backtick code.

### Math Notation

- **Preserve all LaTeX math delimiters verbatim.** KaTeX auto-render handles them client-side.
- `$...$` stays as literal text inside `<p>` elements.
- `$$...$$` stays as literal text on its own line, inside a `<p>` or directly within the container.
- Do not escape `$` signs. Do not wrap math in `<code>` or `<pre>`.

### Lists

- `- item` becomes `<ul><li>item</li></ul>`.
- `1. item` becomes `<ol><li>item</li></ol>`.
- Bold labels (`- **Label**: description`) become `<li><strong>Label</strong>: description</li>`.
- Nested lists become nested `<ul>` or `<ol>` inside the parent `<li>`.

### Tables

Convert Markdown tables to `<table>` with `<thead>` and `<tbody>`:

```html
<table>
  <thead>
    <tr><th>Symbol</th><th>Description</th></tr>
  </thead>
  <tbody>
    <tr><td>$\mathcal{A}$</td><td>Set of agents</td></tr>
  </tbody>
</table>
```

Preserve math notation inside table cells verbatim.

### Horizontal Rules

- Markdown `---` between `##` sections: omit from HTML output. The CSS handles section spacing via heading margins.

### Cross-References

- `§2` or `§8.1` references: wrap in a `<span>` or leave as plain text. Do not convert to hyperlinks unless the user requests linked navigation.

### Definition Labels

- `**Definition 1** (Title).` becomes `<p><strong>Definition 1</strong> (Title).</p>`.

---

## Shared CSS

When no project config exists and the output directory does not already contain a CSS file, generate a default `paper-shared.css` with these styles:

```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body {
    height: 100%;
    margin: 0;
}

body {
    font-family: 'Georgia', 'Times New Roman', serif;
    line-height: 1.8;
    color: #2c2416;
    background: white;
    padding: 2rem;
    box-sizing: border-box;
    overflow-y: auto;
}

.paper-section-container {
    max-width: 800px;
    margin: 0 auto;
    min-height: 100%;
    display: flex;
    flex-direction: column;
}

h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: #1a1410;
    margin-bottom: 0.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid rgba(126, 104, 62, 0.2);
    letter-spacing: 0.02em;
}

h2 {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1a1410;
    margin-top: 0rem;
    margin-bottom: 1rem;
    letter-spacing: 0.01em;
}

h2:first-of-type {
    margin-top: 0;
}

h3 {
    font-size: 1.1rem;
    font-weight: 700;
    color: #2c2416;
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
}

p {
    font-size: 1rem;
    margin-bottom: 1.25rem;
    text-align: justify;
    hyphens: auto;
}

p:last-child {
    margin-bottom: 0;
}

ul {
    margin-bottom: 1.25rem;
    padding-left: 2rem;
}

li {
    margin-bottom: 0.5rem;
    text-align: justify;
}

.katex {
    font-size: 1.05em;
}

.katex-display {
    margin: 1.5rem 0;
}

abbr {
    text-decoration: none;
    font-weight: 600;
    letter-spacing: 0.03em;
}

.intro-text {
    font-style: italic;
    color: #4a3f2f;
    margin-bottom: 1.5rem;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 1.25rem;
    font-size: 0.95rem;
}

th, td {
    border: 1px solid rgba(126, 104, 62, 0.25);
    padding: 0.5rem 0.75rem;
    text-align: left;
}

th {
    background: rgba(126, 104, 62, 0.08);
    font-weight: 700;
}
```

If the output directory already contains a `paper-shared.css`, do not overwrite it.

---

## Conversion Workflow

When asked to convert a Markdown document:

1. Read the source Markdown file fully.
2. Resolve the output path (override → config → convention).
3. Resolve the CSS reference (config → existing file → generate default).
4. Convert each Markdown element to its HTML equivalent using the rules above.
5. Preserve all LaTeX math delimiters verbatim — do not transform or escape them.
6. Wrap the converted content in the HTML template.
7. Write the output `.html` file. Derive the filename from the source (e.g., `methods.md` → `methods.html`).
8. If a default CSS was generated, report it to the user.

## Things to Avoid

- **Do not escape `$` signs** — KaTeX auto-render needs them as-is.
- **Do not wrap math in `<code>` or `<pre>`** — this prevents KaTeX rendering.
- **Do not add inline styles** — all styling comes from the shared CSS.
- **Do not invent content** — only convert what is in the source.
- **Do not embed CSS in the HTML `<head>`** — always use an external stylesheet reference.
- **Do not modify the source Markdown file.**
