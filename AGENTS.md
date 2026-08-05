## Engineering principles

Prefer the simplest working solution for current needs. Skip speculative abstractions, config, or future-proofing.

Ship a minimal end-to-end version first, then layer features. Never abandon working code for unfinished complexity.

Keep modules focused with clear separation of concerns so changes stay local.

Reuse project code, stdlib, or proven libraries after checking docs. Avoid new packages or reimplementation without clear gain.

Drop obsolete paths cleanly. Skip compatibility layers and temporary hacks that force later rewrites.

## Agent skills

### Issue tracker

Issues live as local Markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Domain docs

This is a single-context repository with root `CONTEXT.md` and `docs/adr/`. See `docs/agents/domain.md`.
