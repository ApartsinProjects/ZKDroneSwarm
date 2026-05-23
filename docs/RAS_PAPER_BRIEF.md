# RAS first-paper brief (saved guidance)

Extract a first, focused journal paper targeted at **Robotics and Autonomous Systems (RAS)** from the
full body of work (experiments, methods, baselines, theory, problem variants).

## Goal
Identify the strongest **GOLD** subset: the cleanest, most novel, best-supported contribution that can
stand as a complete, well-defined first paper. Reserve advanced ideas, complex variants, extra baselines,
and secondary results for follow-up papers.

## Task
- Review the full set of experiments, methods, baselines, results.
- Identify the coherent first-paper subset; select only clear wins, strong novelty, convincing support.
- Exclude interesting-but-complex / weakly-connected / later-work material.
- Exclude negative results/methods/sims/metrics unless necessary for the story or to pass review.
- Exclude advanced results if the paper is a 100% accept without them.
- Keep the main body within RAS limits (check guidelines: body, appendix/supplement, abstract length).
- Avoid a zoo of methods/figures/tables; keep the spine solid but not overcrowded; consolidate.
- Define the core contribution; introduce the problem; define setting/assumptions/metrics; propose the
  main method; compare against the most relevant baselines and prior art; emphasize novelty, practical
  value, autonomous-robotics benefits.

## Target to RAS
- Examine the journal scope, formatting expectations, contribution style.
- Review similar highly-cited RAS papers (multi-robot systems, task allocation, autonomous systems,
  decentralized learning).
- Shape framing, terminology, claims, evaluation to journal expectations.

## Plan before writing
- Main thesis; exact subset of methods/experiments; necessary theory; deferred material; section-by-section
  structure.

## Theory audit
- Include only necessary, high-value theory that directly supports empirical claims; remove/postpone
  distractions; ensure theory strengthens novelty. Deep-analyze theorems for (a) correctness, (b) novelty,
  (c) utility.

## Manuscript
- Complete HTML paper; KaTeX math; algorithms/models as clear pseudocode callout boxes; clean professional
  academic style; strong related-work section.

## Preserve follow-up
- Advanced methods, extra baselines, variants, deferred results into a "Future work" section, phrased as a
  planned follow-up ("In a follow-up paper, we plan to study..."). The first paper must feel complete.

## Names/symbols/terms
- Streamline names/symbols/terms now that the full picture is visible; good names for the problem,
  tasks, methods; align to field terminology.

## Use all artifacts
- HTML tutorial + paper2 drafts, text paper, markdown audits/registry/project-log, data and code.
- Rephrase/adapt/omit/rename for best value; do not let it read like a project report; reframe into a
  top-quality scientific paper.

## Title/abstract/intro
- Catchy, scientific, good-taste title; scientific abstract with a hook, importance, the closed gap.

## Appendices
- Move technical detail to appendices; keep the story flowing; do not overload the body.

## More experiments
- Feel free to schedule/run more experiments and modifications to strengthen the paper.

## Outcome + process
- A solid, clean, focused, RAS-ready first paper (HTML) with clear problem, novelty, strong evidence,
  necessary theory, well-chosen baselines, persuasive robotics framing.
- Run **5 autonomous cycles** of paper review + improvement; keep the main message crisp; move details to
  appendices; great visuals/figures/tables.
- Reinforce the main message throughout: a **domain-overlooked, most-restrictive setting** (zero prior,
  zero communication, partial + noisy broadcast with PRIVATE per-observer noise) with clear applications,
  occasional practical motivation (sensors, drone types, capabilities, ammo, target structure, sensing and
  communication limitations). Make the motivation super clear and fundamental; the gap in current
  theory/methods clear; language clear without losing rigor.
- Use a **RAS top-level reviewer role** to critique at the end of each cycle.
