# Issue tracker: GitHub

Issues and specs live in `BioMedAI-UCSC/terraforming` on GitHub.
Use `rtk proxy gh` with `--repo BioMedAI-UCSC/terraforming` for issue operations.

## Operations

- Read: `rtk proxy gh issue view <number> --repo BioMedAI-UCSC/terraforming --json number,title,body,labels,comments`.
- List: `rtk proxy gh issue list --repo BioMedAI-UCSC/terraforming --state open --json number,title,body,labels`.
- Create: `rtk proxy gh issue create --repo BioMedAI-UCSC/terraforming --title "..." --body-file <path>`.
- Comments also use `--body-file` to preserve multiline text. Follow the session's authorization rules before posting.
- Apply or remove labels with `gh issue edit --add-label` or `--remove-label`, using the same RTK prefix and repository argument.
- To close an issue, post the resolution first, then use `gh issue close` with the same RTK prefix and repository argument.

Use native GitHub sub-issues and issue dependencies when supported. If unavailable,
put `Part of #<parent>` and `Blocked by: #<number>` at the top of child issues.
An issue is ready to start when all its blockers are closed.

When a skill asks to publish to the tracker, create an issue here. When it asks
for a ticket, read the corresponding issue. Check the target repository before
creating an issue; this avoids posting into an upstream fork or the skills repo.

## Pull requests as a triage surface

**PRs as a request surface: no.**

Issues and PRs share a number space. Resolve an ambiguous number with
`gh pr view` and fall back to `gh issue view`, using the RTK prefix and repository argument.

## Wayfinding

A map is an issue labelled `wayfinder:map`. Its child tickets use
`wayfinder:research`, `wayfinder:prototype`, `wayfinder:grilling`, or `wayfinder:task`.
Link children as sub-issues, or list them in the map and use `Part of #<map>`.
Choose the first open, unassigned child with no open blockers, in map order.
Claim it by assigning yourself. Record the result, close it, and add a short
summary and link to the map's decisions.
