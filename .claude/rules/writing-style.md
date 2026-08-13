---
description: >
  Governs the writing style of all generated prose. Write every output in
  ASD-STE100 Simplified Technical English by default. This applies to chat
  responses, documentation, plan files, docstrings, and code comments. Use a
  different style only when the user asks for one.
paths:
  - "**"
---

# Writing Style Rule — ASD-STE100

## When This Rule Applies

Write all prose in ASD-STE100 Simplified Technical English (STE) by default.
This rule applies to:

- Chat responses to the user.
- Documentation under `docs/`.
- Plan and idea files.
- Docstrings and code comments.
- Commit messages and pull-request text.

## When This Rule Does Not Apply

Do not change these items to STE:

- Words that the user gives you to keep.
- Code, identifiers, commands, and file paths.
- Direct quotations and proper names.
- Mathematical notation and equations.

**Override:** If the user asks for a different style, use that style. The user
request always wins over this rule. When the user asks, say which style you use.

## The STE Rules to Follow

1. **Use approved words.** Give each word one meaning. Give each word one part
   of speech. Do not use a noun as a verb.
2. **Use one term for one thing.** Do not use synonyms. If you call it a
   "tendency", always call it a "tendency".
3. **Write short sentences.** Keep a procedural sentence to 20 words or fewer.
   Keep a descriptive sentence to 25 words or fewer.
4. **Write one instruction per sentence.** If there are two steps, write two
   sentences.
5. **Use the active voice.** Write "The engine calls the function." Do not write
   "The function is called by the engine."
6. **Use the articles.** Write "the model" and "a step". Do not drop "the" or
   "a".
7. **Use simple verb tenses.** Use the present tense where you can. Do not use
   the future tense for a general fact.
8. **Do not use "-ing" verbs.** Write "to build the model". Do not write
   "building the model" as the main verb.
9. **Write positive sentences.** Tell the reader what to do. Use a negative only
   when it is necessary.
10. **Do not make long noun clusters.** Use three nouns together or fewer. Add a
    preposition to break a long cluster.
11. **Start a warning with a command.** Write the command first. Then give the
    reason.
12. **Do not use slang, idioms, or jargon.** You can use a correct technical
    name.

## Quick Check Before You Send

- Is each sentence short?
- Is there one instruction in each sentence?
- Is the voice active?
- Did you keep the articles?
- Did you use the same word for the same thing?

If the answer to any question is "no", correct the text.
