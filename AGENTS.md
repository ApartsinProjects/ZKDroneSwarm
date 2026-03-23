# Repo Bootstrap

For every new thread in this repository, load the repo-specific governance context before doing substantial work.

## Startup Sequence

1. Resolve the governance root:
   - Prefer `src/dsg-windsurf`
   - Fallback to `.agent`
2. If neither path exists, continue normally and state that no governance root was found.
3. If a governance root exists:
   - Inventory `workflows/`, `rules/`, `skills/`, and `memories/`
   - Read only the files needed for the current request, but always read the applicable rule files before code edits
   - Treat workflow references to `.windsurf/...` as aliases for the detected governance root
4. Tell the user which governance root is active and which workflow or rules are being applied.

## Repo-Local Skill

Use [`.codex/skills/dsg-windsurf-bootstrap/SKILL.md`](/Users/ymeshulam/PycharmProjects/TabulaDrone/.codex/skills/dsg-windsurf-bootstrap/SKILL.md) as the startup guide for this repository.

# UNIVERSAL RULE - PRE-MANIFEST VALIDATION (APPLIES TO ALL VT WORKFLOWS)

If the active workflow defines a validator or completion-check skill, run it before producing any output manifest or declaring the workflow complete.

If validation returns `BLOCKED` or `NEEDS-APPROVAL`, stop there and wait for the user.

# APPROVAL GATE - OUTPUT MANIFEST (ALL VT WORKFLOWS)

Never produce an Output Manifest automatically.

Only continue to manifest generation after explicit user approval such as `proceed`.

# ARTIFACT CREATION RULE (GLOBAL)

Create workflow artifacts only when the active workflow calls for them or the user explicitly asks for them.

Do not invent missing artifacts, and do not silently skip required gates.
