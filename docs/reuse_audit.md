# Reuse Audit (`docs/reuse_audit.md`)

## Decision Summary
By explicit user and architectural decision during Phase 0 (2026-11-13), we elected a **100% clean-room implementation** in `hybrid-graphrag-system` rather than copying files from `https://github.com/shikharrsrivastava/graphrag-robustness-study` (`shikhar-rag-baseline`).

## Rationale
1. **Repository Isolation & Safety**: Zero risk of modifying or coupling to the historical research repository.
2. **Domain Independence**: Avoids carrying over dataset-specific assumptions or unvalidated prototype schemas into the reusable core engine.
3. **Strict Provenance & Type Safety**: Allows every stage (Stages 1–19) to be built cleanly around unified Pydantic v2 models and knowledge-base isolation (`kb_id`).
