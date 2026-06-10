# ================================
# config.py  (v3 — production scale)
# ================================

# ── Ollama Vision Model ──────────────────────────────────────
OLLAMA_VISION_MODEL = "moondream"
OLLAMA_BASE_URL     = "http://localhost:11434"

# ── Groq API ─────────────────────────────────────────────────
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.1-8b-instant"
TEMPERATURE  = 0.1

# ── Qdrant ────────────────────────────────────────────────────
# Single node (current — works up to ~10M vectors on one machine)
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6MzM0NjdjZjEtNzU3YS00Mjc5LWE3MmQtNjUzYmVjMGMwZjVlIn0.ml0lrkM8aUUt8pbIZiao0Vio-HHPnEAmgVK3M1lP7rU"
QDRANT_URL     = "https://998f6d30-51d1-4174-92b0-e5954fd445b5.eu-west-1-0.aws.cloud.qdrant.io"

# ── Distributed mode (100M+ documents) ───────────────────────
# When DISTRIBUTED_MODE = True:
#   - Queries fan out to all QDRANT_SHARD_URLS in parallel
#   - Each shard returns SHARD_TOP_K results
#   - Coordinator merges all results and re-ranks globally
#   - Final FINAL_TOP_K results go to the LLM
#
# To enable: set DISTRIBUTED_MODE = True and add your shard URLs.
# Qdrant cluster mode handles sharding automatically — you only
# need to point to different node URLs here.
DISTRIBUTED_MODE  = False   # ← set True when scaling beyond one node
QDRANT_SHARD_URLS = [
    # Add shard node URLs here when scaling out, e.g.:
    # "https://shard-1.your-qdrant-cluster.com",
    # "https://shard-2.your-qdrant-cluster.com",
    # "https://shard-3.your-qdrant-cluster.com",
    # "https://shard-4.your-qdrant-cluster.com",
]
SHARD_TOP_K          = 20   # candidates fetched from EACH shard
SHARD_TIMEOUT_SEC    = 2.0  # max wait per shard before using partial results

# ── Single collection ─────────────────────────────────────────
COLLECTION_SPANISH = "rag_spanish"
COLLECTION_ENGLISH = "rag_spanish"

# ── Embedding ─────────────────────────────────────────────────
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v2-moe"
VECTOR_DIM      = 768

# ── Chunking ──────────────────────────────────────────────────
EN_CHUNK_SIZE    = 500
EN_CHUNK_OVERLAP = 100
ES_CHUNK_SIZE    = 500
ES_CHUNK_OVERLAP = 100

# ── Hybrid retrieval ──────────────────────────────────────────
VECTOR_TOP_K = 20
BM25_TOP_K   = 20
FINAL_TOP_K  = 15

# ── Incremental indexing ──────────────────────────────────────
# At 100M docs, full re-indexing takes days and is impractical.
# Strategy:
#   1. New documents go into a small DELTA collection first (fast writes)
#   2. A background job merges delta into the main collection off-peak
#   3. Every document has an ingestion_time timestamp (set in pdf_utils.py)
#   4. Deleted documents are SOFT-DELETED: flagged is_deleted=True in
#      metadata, then filtered out at query time — never removed immediately
#   5. Periodic compaction removes soft-deleted docs in bulk during off-peak
#
# DELTA_COLLECTION_NAME: small staging area for newly ingested documents.
# Queries search BOTH main and delta, then merge results.
INCREMENTAL_INDEXING  = False             # ← set True at scale
DELTA_COLLECTION_NAME = "rag_spanish_delta"
DELTA_MERGE_THRESHOLD = 10_000           # merge delta → main after this many chunks

# ── Caching ───────────────────────────────────────────────────
# Two-level cache to reduce latency for repeated queries:
#
#   Level 1 — QUERY CACHE:
#     If the exact same question + user_id was asked recently,
#     return the cached answer directly. Skip retrieval + LLM entirely.
#     TTL: 1 hour (hot queries change rarely within an hour).
#
#   Level 2 — EMBEDDING CACHE:
#     Cache the embedding vector for a query string.
#     If the same query text is embedded again (e.g. across retries),
#     return the cached vector. Saves ~50ms per embedding call.
#
# In production: use Redis for distributed caching across server instances.
# For single-server / prototype: use an in-process LRU dict (implemented
# in cache.py — falls back gracefully if Redis is not available).
CACHE_ENABLED         = True
CACHE_TTL_SECONDS     = 3600            # 1 hour TTL for query cache
CACHE_MAX_SIZE        = 1_000           # max entries in in-process LRU cache
REDIS_URL             = None            # set to "redis://localhost:6379" in production

# ── Query translation ─────────────────────────────────────────
TRANSLATE_EN_TO_ES = True

# ── Server ────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000

# ── Language config map ───────────────────────────────────────
LANGUAGE_CONFIG = {
    "en": {
        "chunk_size":    EN_CHUNK_SIZE,
        "chunk_overlap": EN_CHUNK_OVERLAP,
        "collection":    COLLECTION_SPANISH,
        "min_line_len":  30,
    },
    "es": {
        "chunk_size":    ES_CHUNK_SIZE,
        "chunk_overlap": ES_CHUNK_OVERLAP,
        "collection":    COLLECTION_SPANISH,
        "min_line_len":  30,
    },
}

# ── Intermediate element storage ─────────────────────────────
# When True, raw table HTML/markdown and image OCR text (+ base64) are
# stored as extra payload fields on every chunk that originates from a
# table or image element. This lets you:
#   1. Verify which exact table/image answered a query (via /element endpoint)
#   2. Track table_id / image_id in source attribution ("Table 3 from doc.pdf")
#   3. Re-render the original table client-side for human verification
#
# Set False to save Qdrant payload storage space in production.
STORE_INTERMEDIATES = True

# ── Poppler path (Windows only) ───────────────────────────────
POPPLER_PATH = r"C:\Users\prera\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"

# ── PaddleOCR ─────────────────────────────────────────────────
PADDLE_USE_GPU    = False
PADDLE_CONFIDENCE = 0.5
PADDLE_LANG_MAP   = {
    "en": "en",
    "es": "es",
}