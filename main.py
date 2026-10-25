import argparse
import csv
import time
from pathlib import Path

import ollama

import compare
from chunker import chunk_records
from config import DATA_DIR, LLM_MODEL
from loaders import load_file
from rag import ask, chat
from store import add_chunks, delete_ids, embed, get_ids


def log_row(row):
    # Append one line per run to timings.csv, so you can build your comparison later.
    path = Path("timings.csv")
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(
                [
                    "label",
                    "mode",
                    "file",
                    "total_chunks",
                    "embedded",
                    "deleted",
                    "skipped",
                    "load_s",
                    "chunk_s",
                    "embed_store_s",
                    "total_s",
                ]
            )
        w.writerow(row)


def warm_up() -> None:
    print(f"[warm-up] loading '{LLM_MODEL}' and the embedding model into memory...")
    ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": "Ready."}])
    embed(["warm-up"])
    print("[warm-up] done.")


def ingest_file(path: Path, full: bool) -> dict:
    t0 = time.perf_counter()
    records = load_file(path)
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    chunks = chunk_records(records)
    chunk_s = time.perf_counter() - t0

    new_ids = {chunk["id"] for chunk in chunks}
    existing_ids = set(get_ids(path.name))

    if full:
        to_delete_ids = existing_ids
        to_add = chunks
        skipped_count = 0
    else:
        to_delete_ids = existing_ids - new_ids
        to_add = [chunk for chunk in chunks if chunk["id"] not in existing_ids]
        skipped_count = len(new_ids & existing_ids)

    delete_ids(list(to_delete_ids))

    t0 = time.perf_counter()
    embeddings = embed([chunk["text"] for chunk in to_add]) if to_add else []
    add_chunks(to_add, embeddings)
    embed_store_s = time.perf_counter() - t0

    return {
        "total_chunks": len(chunks),
        "embedded": len(to_add),
        "deleted": len(to_delete_ids),
        "skipped": skipped_count,
        "load_s": load_s,
        "chunk_s": chunk_s,
        "embed_store_s": embed_store_s,
        "total_s": load_s + chunk_s + embed_store_s,
    }


def cmd_ingest(args: argparse.Namespace) -> int:
    paths = (
        [Path(p) for p in args.files] if args.files else sorted(DATA_DIR.glob("*.pdf"))
    )
    paths = [p for p in paths if p.exists()]
    if not paths:
        print(f"No PDFs found. Pass file paths or add PDFs to {DATA_DIR}.")
        return 1
    DATA_DIR.mkdir(exist_ok=True)
    warm_up()
    mode = "full" if args.full else "incremental"
    for path in paths:
        label = args.label or f"{mode}-{time.strftime('%H%M%S')}"
        stats = ingest_file(path, full=args.full)
        print(
            f"{path.name} [{mode}]: {stats['total_chunks']} chunks | "
            f"embedded={stats['embedded']} deleted={stats['deleted']} "
            f"skipped={stats['skipped']} | total={stats['total_s']:.3f}s"
        )
        log_row(
            [
                label,
                mode,
                path.name,
                stats["total_chunks"],
                stats["embedded"],
                stats["deleted"],
                stats["skipped"],
                f"{stats['load_s']:.4f}",
                f"{stats['chunk_s']:.4f}",
                f"{stats['embed_store_s']:.4f}",
                f"{stats['total_s']:.4f}",
            ]
        )
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    answer, sources = ask(args.query)
    print(answer)
    if sources:
        print(f"\nsources: {'; '.join(sources)}")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    chat()
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    compare.run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local RAG pipeline with content-hashed incremental PDF ingestion."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser(
        "ingest", help="Ingest PDFs (incremental set-diff by default)."
    )
    p_ingest.add_argument(
        "files", nargs="*", help="PDF files (default: all PDFs in data/)."
    )
    p_ingest.add_argument(
        "--full", action="store_true", help="Re-embed every chunk from scratch."
    )
    p_ingest.add_argument(
        "--label", default=None, help="Label for the timings.csv row."
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a one-off question against the index.")
    p_ask.add_argument("query", help="The question to answer.")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="Start an interactive chat session.")
    p_chat.set_defaults(func=cmd_chat)

    p_compare = sub.add_parser(
        "compare", help="Render timing_comparison.png from timings.csv."
    )
    p_compare.set_defaults(func=cmd_compare)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
