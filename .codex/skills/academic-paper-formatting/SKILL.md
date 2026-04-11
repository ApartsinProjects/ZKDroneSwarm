---
name: academic-paper-formatting
description: Format technical and academic documents in a consistent, publication-ready Markdown style with proper LaTeX math notation, structured sections, and professional typography.
---

# Academic Paper Formatting

Use this skill when creating or reformatting technical documentation, academic paper sections, or any document that contains mathematical notation and formal definitions.

## Goal

Produce clean, consistent, publication-quality Markdown documents with proper structure, LaTeX math, and professional typography.

## Document Structure

### Title and Preamble

- Begin with a single `#` heading for the document title
- Follow with one or two introductory paragraphs in plain text that summarize the document's scope and purpose
- Add a **Notation conventions** paragraph (bolded label) that defines key symbols used throughout the document
- Place a horizontal rule (`---`) after the preamble before the first section

### Section Hierarchy

- `##` for numbered top-level sections (e.g., `## 1. Problem Setting`)
- `###` for numbered subsections (e.g., `### 1.1 Reward Signal`)
- Never skip heading levels
- Separate every `##` section with a horizontal rule (`---`) above it
- Use sentence-case for heading titles (capitalize first word and proper nouns only), except for acronyms

### Paragraphs and Prose

- One blank line between paragraphs
- Keep paragraphs focused on a single idea
- Avoid orphan single-sentence paragraphs — either expand or merge with an adjacent paragraph
- When introducing a formal concept, state it in prose first, then present the equation

## Math Notation

### Display Equations

Use `$$...$$` on their own lines for standalone equations:

```
$$a_t(i) \sim \text{Uniform}(\mathcal{A}_t(i))$$
```

- One blank line before and after the `$$` block
- No trailing punctuation inside display math unless it is part of a multi-clause expression

### Inline Math

Use `$...$` for symbols, variables, and short expressions embedded in prose:

```
each agent $a \in \mathcal{A}$ receives an observation $o_t(a) \in O$
```

### Conventions

- Use `\mathcal{}` for sets: $\mathcal{A}$, $\mathcal{T}$, $\mathcal{Z}$
- Use `\mathbf{}` for vectors and matrices: $\mathbf{z}$, $\mathbf{a}_t$, $P^{(a)}$
- Use `\hat{}` for estimates: $\hat{r}_{ij}$
- Use `\tilde{}` for noisy observations: $\tilde{r}_{ij}$
- Use `\text{}` for multi-letter operators or labels inside math: $\text{Uniform}$, $\text{HP}_j$
- Use `\mathbb{R}` for real number sets: $\mathbb{R}^d$
- Use standard LaTeX for Greek letters: $\varepsilon$, $\eta$, $\lambda$, $\tau$
- Superscripts for agent/type annotation: $\mathbf{z}_i^{(a)}$, $P^{(a)}$
- Subscripts for indices: $g_{ij}$, $r_{ij}$

### Broken Math Recovery

When source material contains math that has been corrupted into multi-line plain text (e.g., each symbol on its own line, duplicate rendered/source forms), reconstruct the intended LaTeX expression and present it as a single clean inline or display equation.

## Lists

### Bulleted Lists

- Use `- ` (hyphen) for unordered lists
- Bold the label when items have a title-description pattern:
  ```
  - **Direct mode**: The policy compares predicted utility directly to observed reward
  ```
- Use nested lists sparingly; prefer sub-items only when the parent item genuinely contains children

### Numbered Lists

- Use `1.`, `2.`, etc. for sequential or ordered items
- Reserve numbered lists for steps, procedures, or ranked items

### Definition Lists (via bold)

For formal definitions, use bold inline:

```
**Definition 1** (Zero-Knowledge Multi-Robot Task Allocation).
```

## Tables

Use standard Markdown tables with a header row and alignment:

```
| Symbol | Description |
|--------|-------------|
| $\mathcal{A}$ | Set of agents |
```

- Keep cell content concise
- Use math notation inside cells when appropriate
- Left-align text columns, left-align symbol columns

## Cross-References

Use `§` followed by the section number for internal references:

```
as defined in §2
the reward mode described in §8.1
```

## Horizontal Rules

- Place `---` on its own line between major `##` sections
- Do not use horizontal rules between subsections (`###`) within the same section
- Do not double up horizontal rules

## Things to Avoid

- **No code blocks** unless the document is a technical specification that explicitly includes implementation details
- **No emoji** in academic documents
- **No HTML** in Markdown academic documents
- **No raw Unicode math symbols** (e.g., `→`, `∈`, `⊤`) — always use LaTeX equivalents inside math delimiters
- **No trailing whitespace** or empty lines at end of file (single newline at EOF is acceptable)
- **Do not invent content** — only format and restructure what is provided

## Reformatting Workflow

When asked to reformat an existing document into this style:

1. Read the source document fully
2. Identify the logical section structure
3. Reconstruct all math notation into proper LaTeX
4. Apply heading hierarchy, horizontal rules, and list formatting
5. Preserve all original content — do not add, remove, or reword substantive text
6. Verify notation consistency across the entire document
