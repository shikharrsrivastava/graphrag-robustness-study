# Demo

Seed-7 fixed knowledge base (`experiments/synth_kg_fix/kb_dev/synth_company_kb`), 7 questions chosen by a fixed rule (printed at the top of the output).

```bash
# default: offline, reads experiments/demo/precomputed.json
python3 experiments/demo/demo.py

# live: runs the engine for real (Ollama must be running with gemma3:4b and embeddinggemma; about 6 minutes)
python3 experiments/demo/demo.py --live
```

`--live --record` also rewrites `precomputed.json`. `python3 experiments/demo/build_summary.py` regenerates `DEMO_SUMMARY.md`.
Note: opening a KB through the public API touches the modification time of its Chroma files.
