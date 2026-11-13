# Architectural & Engineering Decisions (`docs/decisions.md`)

## ADR-001: Clean-Room Implementation
- **Date**: 2026-11-13
- **Decision**: Implement all 19 stages from scratch in `hybrid-graphrag-system` without copying code from `graphrag-robustness-study`.
- **Reason**: Ensures clean separation of Domain Configuration, Reusable GraphRAG Core, and Dataset Adapters.

## ADR-002: Verified Local Runtime Baseline
- **Date**: 2026-11-13
- **Environment**: macOS 27.2 (`arm64`, MacBook Air M2, 16 GB RAM, 90 GB free disk), Python 3.13.7, Ollama 0.40.16.
- **Models**: `gemma3:4b` (`68ceab7e468a`, 3.4 GB) and `embeddinggemma:latest` (`85462619ee72`, 621 MB, verified 768 dimensions).
