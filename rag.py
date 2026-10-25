import ollama

from config import LLM_MODEL, TOP_K
from store import collection, embed

SYSTEM_PROMPT = (
    "You are a precise assistant. Answer strictly from the provided context. "
    "If the context does not contain the answer, say you don't know. "
    "Cite sources using their [file, page N] tags."
)


def ask(query: str, top_k: int = TOP_K) -> tuple[str, list[str]]:
    """Retrieve top-k passages for the query and answer them with the LLM."""
    if collection.count() == 0:
        return "The knowledge base is empty. Run 'python main.py ingest' first.", []
    query_embedding = embed([query], kind="query")
    n_results = min(top_k, collection.count())
    result = collection.query(query_embeddings=query_embedding, n_results=n_results)
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    if not documents:
        return "No relevant passages were found for that question.", []
    context = "\n\n".join(
        f"[{meta['source']}, page {meta['page']}]\n{document}"
        for document, meta in zip(documents, metadatas)
    )
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above:"
    )
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    answer = response["message"]["content"].strip()
    sources = sorted({f"{meta['source']}, page {meta['page']}" for meta in metadatas})
    return answer, sources


def chat() -> None:
    """Interactive terminal chat loop."""
    print(f"RAG chat with '{LLM_MODEL}'. Type 'exit' to quit.")
    while True:
        try:
            query = input("\nquestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        answer, sources = ask(query)
        print(f"\n{answer}")
        if sources:
            print(f"\nsources: {'; '.join(sources)}")
    print("bye.")
