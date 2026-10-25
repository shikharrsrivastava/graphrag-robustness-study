from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

LLM_MODEL = "gemma3:4b"

EMBED_MODEL = "gemma3:4b"
EMBED_FALLBACK = "embeddinggemma"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 0

DB_PATH = BASE_DIR / "chroma_db"
DATA_DIR = BASE_DIR / "data"
COLLECTION_NAME = "pdf_collection"

TIMINGS_CSV = BASE_DIR / "timings.csv"
TIMINGS_PNG = BASE_DIR / "timing_comparison.png"

TOP_K = 4

EMBED_DOC_PREFIX = "task: search result | document: "
EMBED_QUERY_PREFIX = "task: search result | query: "
