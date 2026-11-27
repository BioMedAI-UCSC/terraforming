# Domain docs

Use a single context for this repository:

- Root `GLOSSARY.md`: domain terms and their meanings.
- `docs/adr/`: architecture decision records, one decision per file.

Before exploring a domain, read the glossary and the ADRs relevant to that area.
If those files do not exist, continue without creating placeholders or requesting setup.
The domain-modeling skill creates them when terms or decisions are resolved.

Use glossary terms in code explanations, issue titles, tests and design proposals.
Record meaningful vocabulary gaps when working on the domain model.
If a proposal conflicts with an ADR, name the record and explain why the decision
should be reopened.
