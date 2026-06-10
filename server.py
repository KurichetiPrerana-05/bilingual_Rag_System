# ================================
# server.py  (v6 — multi-tenant + caching)
#
# CHANGES vs v5:
#   - Added cache.py integration (query cache + embedding cache)
#   - /ask   : checks cache FIRST before retrieval + LLM
#              stores answer in cache after LLM responds
#   - /ingest: invalidates cache for that user after new docs ingested
#   - GET /cache/stats        : monitor cache hit rate
#   - DELETE /cache/user/{id} : manually clear a user's cache
# ================================

import os
import re
import time
import tempfile
from typing import Optional, List

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import uvicorn

from config import (
    QDRANT_URL, QDRANT_API_KEY,
    COLLECTION_SPANISH,
    VECTOR_DIM, HOST, PORT,
    LANGUAGE_CONFIG, GROQ_MODEL
)
from embeddings import load_embedding_model
from rag_chain import load_llm, build_chain
from vector_store import (
    create_collection,
    ensure_payload_index,
    fetch_all_docs,
    index_documents,
    delete_user_documents,
    count_user_documents,
    fetch_element,
    list_user_elements,
)
from pdf_utils import chunk_pdf
from cache import cache                          # ← NEW: two-level cache

print(f"GPU Available: {torch.cuda.is_available()}")

# ── Load shared models once at startup ───────────────────────
print("\nLoading models...")
embedding_model = load_embedding_model()
llm             = load_llm()
qdrant_client   = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
print("Qdrant connected ✅")

# Ensure the payload index exists on startup — required for user_id filtering.
# This fixes "Index required but not found for metadata.user_id" errors
# on existing collections that were created before the index was added.
ensure_payload_index(qdrant_client, COLLECTION_SPANISH)

def _load_shared_vs() -> Optional[QdrantVectorStore]:
    try:
        qdrant_client.get_collection(COLLECTION_SPANISH)
        return QdrantVectorStore(
            client=qdrant_client,
            collection_name=COLLECTION_SPANISH,
            embedding=embedding_model
        )
    except Exception:
        return None

shared_vs = _load_shared_vs()
print(f"  Shared vector store: {'✅ Ready' if shared_vs else '⚠️  Not ingested yet'}")


# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="Multi-Tenant Bilingual RAG API",
    description=(
        "Multi-Tenant English + Spanish RAG System\n\n"
        "**Every request requires a `user_id`.**\n\n"
        "Each user's documents are isolated — queries only search that user's data.\n\n"
        "**Step 1:** Ingest PDFs via `POST /ingest?user_id=YOUR_ID`\n"
        "**Step 2:** Ask questions via `POST /ask` (include user_id in body)\n\n"
        "Retrieval: Hybrid BM25 + Vector + Cross-encoder re-ranking\n"
        "Caching:   Two-level (query cache + embedding cache)"
    ),
    version="6.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/", include_in_schema=False)
    def serve_ui():
        ui_path = os.path.join("static", "bilingual_rag_chat.html")
        if os.path.isfile(ui_path):
            return FileResponse(ui_path)
        return {"message": "Put bilingual_rag_chat.html inside the static/ folder."}


# ════════════════════════════════════════════════════════════
# MODELS
# ════════════════════════════════════════════════════════════

class AskRequest(BaseModel):
    question:     str
    user_id:      str
    language:     Optional[str]  = "auto"
    show_sources: Optional[bool] = False

class AskResponse(BaseModel):
    question:        str
    answer:          str
    user_id:         str
    language_used:   str
    collection_used: str
    cache_hit:       bool = False
    sources:         Optional[list] = []

class IngestResponse(BaseModel):
    filename:          str
    user_id:           str
    language:          str
    collection:        str
    pages_processed:   int
    chunks_added:      int
    total_user_chunks: int
    ocr_summary:       dict
    message:           str


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════

LANG_LABEL = {"en": "English", "es": "Spanish"}


def _validate_user_id(user_id: str):
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id cannot be empty.")
    if len(user_id) > 128:
        raise HTTPException(status_code=400, detail="user_id too long (max 128 chars).")
    if not re.match(r'^[\w.\-@]+$', user_id):
        raise HTTPException(
            status_code=400,
            detail="user_id contains invalid characters. Allowed: letters, digits, - _ . @"
        )


def detect_question_language(question: str) -> str:
    if not question.strip():
        return "es"
    if re.search(r'[ñÑ¿¡]', question):
        return "es"
    accented = len(re.findall(r'[áéíóúüÁÉÍÓÚÜ]', question))
    total    = len(question.strip())
    if accented / max(total, 1) > 0.03:
        return "es"
    spanish_words = len(re.findall(
        r'\b(qué|cómo|cuál|cuánto|dónde|cuándo|quién|'
        r'el|la|los|las|de|en|con|es|son|fue|una|uno|más|pero|'
        r'como|este|esta|para|que|sus|su|al|lo|ya|si)\b',
        question.lower()
    ))
    if spanish_words >= 2:
        return "es"
    return "en"


def build_source_entry(i: int, doc) -> dict:
    """
    Builds the source attribution dict for a single retrieved chunk.

    INTERMEDIATE ELEMENT FIELDS:
      element_id    — UUID to fetch the full stored element via
                      GET /element/{user_id}/{element_id}
      element_type  — "table" | "image" | "figure" | "text"
      element_label — human-readable label, e.g. "Table 2 from report.pdf, page 5"
      table_index   — 1-based table number within the page (tables only)
      image_index   — 1-based image number within the page (images only)

    These fields let you verify whether the table was correctly identified
    as the source of the answer, and retrieve its full HTML/markdown or image.
    """
    m = doc.metadata
    return {
        "rank":            i + 1,
        "pdf_name":        m.get("pdf_name",    m.get("source", "unknown")),
        "page_number":     str(m.get("page_number", m.get("page", "N/A"))),
        "total_pages":     str(m.get("total_pages", "N/A")),
        "chunk_index":     m.get("chunk_index", "N/A"),
        "chunk_total":     m.get("chunk_total",  "N/A"),
        "content_type":    m.get("content_type",    "text"),
        # ── Intermediate element attribution ──────────────
        "element_id":      m.get("element_id",    None),
        "element_type":    m.get("element_type",  m.get("content_type", "text")),
        "element_label":   m.get("element_label", ""),
        "table_index":     m.get("table_index",   None),
        "image_index":     m.get("image_index",   None),
        # ── Existing flags ────────────────────────────────
        "has_table":       m.get("has_table",        False),
        "has_image":       m.get("has_image",        False),
        "has_image_ocr":   m.get("has_image_ocr",   False),
        "has_page_ocr":    m.get("has_page_ocr",    False),
        "table_count":     m.get("table_count",      0),
        "image_count":     m.get("image_count",      0),
        "language":        m.get("language",         "unknown"),
        "section_heading": m.get("section_heading",  ""),
        "char_count":      m.get("char_count",        len(doc.page_content)),
        "ingestion_time":  m.get("ingestion_time",   "N/A"),
        "preview":         doc.page_content[:250],
    }


# ════════════════════════════════════════════════════════════
# ROUTE 1: GET /health
# ════════════════════════════════════════════════════════════
@app.get("/health", summary="Server health check")
def health():
    try:
        info = qdrant_client.get_collection(COLLECTION_SPANISH)
        coll_status = {"status": "ready", "total_vectors": info.points_count}
    except Exception:
        coll_status = {"status": "not_ingested", "total_vectors": 0}

    return {
        "server":          "ok",
        "gpu":             torch.cuda.is_available(),
        "embedding_model": "nomic-ai/nomic-embed-text-v2-moe",
        "llm":             f"groq/{GROQ_MODEL}",
        "retrieval":       "hybrid (BM25 + Vector + Cross-encoder re-rank)",
        "multi_tenant":    True,
        "cache":           cache.stats(),
        "collection":      coll_status,
    }


# ════════════════════════════════════════════════════════════
# ROUTE 2: POST /ask
# ════════════════════════════════════════════════════════════
@app.post("/ask", response_model=AskResponse, summary="Ask a question (user-scoped)")
def ask(request: AskRequest):
    """
    Ask a question in English or Spanish.

    CACHING:
      Step 1 — check cache. If this exact question from this user was
               asked recently, return the cached answer instantly.
               Saves full retrieval + LLM cost (~600ms to 1.6s).
      Step 2 — cache miss: run full retrieval + LLM pipeline.
      Step 3 — store new answer in cache for next time.
    """
    global shared_vs

    _validate_user_id(request.user_id)
    user_id  = request.user_id.strip()
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if request.language == "auto":
        lang = detect_question_language(question)
    elif request.language in ("en", "es"):
        lang = request.language
    else:
        raise HTTPException(status_code=400, detail="Use language: en | es | auto")

    # ── STEP 1: CACHE CHECK ──────────────────────────────────
    cached = cache.get_answer(user_id, question)
    if cached:
        return AskResponse(
            question        = question,
            answer          = cached["answer"],
            user_id         = user_id,
            language_used   = LANG_LABEL.get(lang, lang),
            collection_used = COLLECTION_SPANISH,
            cache_hit       = True,
            sources         = cached["sources"],
        )

    # ── STEP 2: FULL PIPELINE (cache miss) ───────────────────
    if shared_vs is None:
        shared_vs = _load_shared_vs()
    if shared_vs is None:
        raise HTTPException(
            status_code=404,
            detail=f"No documents ingested yet. Ingest first: POST /ingest?user_id={user_id}"
        )

    user_doc_count = count_user_documents(qdrant_client, COLLECTION_SPANISH, user_id)
    if user_doc_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No documents found for user_id='{user_id}'. Ingest first: POST /ingest?user_id={user_id}"
        )

    try:
        user_docs = fetch_all_docs(qdrant_client, COLLECTION_SPANISH, embedding_model, user_id=user_id)
        _, chain_fn = build_chain(shared_vs, llm, user_docs, user_id=user_id)
        answer, retrieved_docs = chain_fn(question, user_lang=lang)

        sources = []
        if request.show_sources and retrieved_docs:
            sources = [build_source_entry(i, doc) for i, doc in enumerate(retrieved_docs)]

        # ── STEP 3: STORE IN CACHE ───────────────────────────
        cache.set_answer(user_id, question, answer, sources)

        return AskResponse(
            question        = question,
            answer          = answer,
            user_id         = user_id,
            language_used   = LANG_LABEL.get(lang, lang),
            collection_used = COLLECTION_SPANISH,
            cache_hit       = False,
            sources         = sources,
        )

    except HTTPException:
        raise
    except Exception as e:
        err_str = str(e).lower()
        if "rate limit" in err_str or "429" in err_str:
            raise HTTPException(status_code=429, detail="Rate limit — try again in 30s.")
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════
# ROUTE 3: POST /ingest
# ════════════════════════════════════════════════════════════
@app.post("/ingest", response_model=IngestResponse, summary="Upload and ingest a PDF for a user")
async def ingest_pdf(
    file:     UploadFile = File(...),
    user_id:  str  = Query(...,            description="Unique user ID"),
    language: str  = Query(default="auto", description="Language: en | es | auto")
):
    global shared_vs

    _validate_user_id(user_id)
    user_id = user_id.strip()

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if language not in ("en", "es", "auto"):
        raise HTTPException(status_code=400, detail="Use language: en | es | auto")

    try:
        pdf_bytes = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        forced = None if language == "auto" else language
        docs, detected_lang = chunk_pdf(tmp_path, file.filename, forced_language=forced)
        os.remove(tmp_path)

        if not docs:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        create_collection(qdrant_client, COLLECTION_SPANISH)
        # elements list comes from chunk_pdf — see pdf_utils.py
        # chunk_pdf returns (docs, lang) in current version; when it is
        # updated to also return elements, unpack them here:
        #   docs, detected_lang, elements = chunk_pdf(...)
        # For now we pass elements=None until pdf_utils is updated.
        elements = getattr(docs, "_elements", None)  # forward-compat hook
        vs = index_documents(
            docs, embedding_model,
            user_id=user_id,
            collection_name=COLLECTION_SPANISH,
            elements=elements,
            qdrant_client=qdrant_client,
        )
        shared_vs = vs

        # Invalidate this user's cache — new docs mean old cached answers
        # may be missing content from the newly uploaded PDF.
        cache.invalidate_user(user_id)

        def get_page(d): return d.metadata.get("page_number", 0)

        total_pages         = len(set(get_page(d) for d in docs))
        pages_with_tables   = len(set(get_page(d) for d in docs if d.metadata.get("has_table")))
        pages_with_img_ocr  = len(set(get_page(d) for d in docs if d.metadata.get("has_image_ocr")))
        pages_with_page_ocr = len(set(get_page(d) for d in docs if d.metadata.get("has_page_ocr")))
        content_type_counts = {}
        for d in docs:
            ct = d.metadata.get("content_type", "text")
            content_type_counts[ct] = content_type_counts.get(ct, 0) + 1

        user_chunk_count = count_user_documents(qdrant_client, COLLECTION_SPANISH, user_id)
        final_lang       = language if language != "auto" else detected_lang

        return IngestResponse(
            filename          = file.filename,
            user_id           = user_id,
            language          = LANG_LABEL.get(final_lang, final_lang),
            collection        = COLLECTION_SPANISH,
            pages_processed   = total_pages,
            chunks_added      = len(docs),
            total_user_chunks = user_chunk_count,
            ocr_summary       = {
                "total_pages":          total_pages,
                "pages_with_tables":    pages_with_tables,
                "pages_with_image_ocr": pages_with_img_ocr,
                "pages_with_page_ocr":  pages_with_page_ocr,
                "content_type_counts":  content_type_counts,
                "retrieval_mode":       "hybrid (BM25 + Vector + Cross-encoder re-rank)",
            },
            message = (
                f"'{file.filename}' ingested for user='{user_id}' — "
                f"{len(docs)} chunks added. User now has {user_chunk_count} total chunks. "
                f"Cache invalidated for fresh retrieval."
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ════════════════════════════════════════════════════════════
# ROUTE 4: GET /user/{user_id}/status
# ════════════════════════════════════════════════════════════
@app.get("/user/{user_id}/status", summary="Check how many documents a user has")
def user_status(user_id: str):
    _validate_user_id(user_id)
    chunk_count = count_user_documents(qdrant_client, COLLECTION_SPANISH, user_id)
    return {
        "user_id":    user_id,
        "collection": COLLECTION_SPANISH,
        "chunks":     chunk_count,
        "status":     "ready" if chunk_count > 0 else "no_documents",
        "message": (
            f"User '{user_id}' has {chunk_count} chunks indexed."
            if chunk_count > 0
            else f"No documents found for user '{user_id}'. Ingest a PDF first."
        )
    }


# ════════════════════════════════════════════════════════════
# ROUTE 5: DELETE /user/{user_id}/documents
# ════════════════════════════════════════════════════════════
@app.delete("/user/{user_id}/documents", summary="Delete all documents for a user")
def delete_user_data(user_id: str):
    _validate_user_id(user_id)

    chunk_count_before = count_user_documents(qdrant_client, COLLECTION_SPANISH, user_id)
    if chunk_count_before == 0:
        return {
            "status":  "ok",
            "user_id": user_id,
            "message": f"No documents found for user '{user_id}' — nothing to delete."
        }

    delete_user_documents(qdrant_client, COLLECTION_SPANISH, user_id)
    cache.invalidate_user(user_id)

    return {
        "status":         "ok",
        "user_id":        user_id,
        "chunks_deleted": chunk_count_before,
        "message":        f"All {chunk_count_before} chunks for user '{user_id}' deleted. Cache cleared.",
        "next":           f"Re-ingest: POST /ingest?user_id={user_id}"
    }


# ════════════════════════════════════════════════════════════
# ROUTE 6: GET /collections  (admin)
# ════════════════════════════════════════════════════════════
@app.get("/collections", summary="Global collection status (admin)")
def collection_status():
    try:
        info = qdrant_client.get_collection(COLLECTION_SPANISH)
        return {"collections": [{
            "collection":    COLLECTION_SPANISH,
            "status":        "ready",
            "total_vectors": info.points_count,
            "note":          "Single collection for all users. Filtered per user at query time.",
        }]}
    except Exception:
        return {"collections": [{
            "collection":    COLLECTION_SPANISH,
            "status":        "not_ingested",
            "total_vectors": 0,
        }]}


# ════════════════════════════════════════════════════════════
# ROUTE 7: DELETE /collection  (admin — wipes ALL users)
# ════════════════════════════════════════════════════════════
@app.delete("/collection", summary="⚠️  ADMIN: Wipe the entire collection (all users)")
def clear_collection():
    global shared_vs
    try:
        qdrant_client.delete_collection(COLLECTION_SPANISH)
        qdrant_client.create_collection(
            collection_name=COLLECTION_SPANISH,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
        ensure_payload_index(qdrant_client, COLLECTION_SPANISH)
        shared_vs = None
        return {
            "status":  "ok",
            "message": f"Collection '{COLLECTION_SPANISH}' wiped (all users' data deleted).",
            "warning": "Use /user/{{user_id}}/documents for targeted single-user deletion.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════
# ROUTE 8: GET /element/{user_id}/{element_id}
#   Fetch a stored intermediate element (table HTML/markdown or image)
#   to verify it was correctly identified as the source of an answer.
# ════════════════════════════════════════════════════════════
@app.get("/element/{user_id}/{element_id}",
         summary="Fetch a stored table or image element by ID")
def get_element(user_id: str, element_id: str):
    """
    Retrieves the full stored intermediate element — table HTML/markdown
    or image base64 — identified by element_id from a source entry.

    Use this to verify that the table or image the system cited actually
    contains the answer it returned.

    Workflow:
      1. POST /ask with show_sources=true
      2. Look at sources[i].element_id and sources[i].element_label
         e.g.  element_label = "Table 2 from report.pdf, page 5"
      3. GET /element/{user_id}/{element_id} to fetch the raw table
      4. Inspect table_html or table_markdown to verify correctness

    Returns:
      element_type  : "table" | "image" | "figure"
      pdf_name      : source document
      page_number   : page it came from
      element_label : human-readable label
      table_html    : raw HTML (tables only)
      table_markdown: markdown representation (tables only)
      image_b64     : base64-encoded image (images only)
      image_ocr     : OCR text extracted from the image (images only)
      caption       : detected caption text
    """
    _validate_user_id(user_id)
    if not element_id or not element_id.strip():
        raise HTTPException(status_code=400, detail="element_id cannot be empty.")

    el = fetch_element(qdrant_client, COLLECTION_SPANISH, element_id.strip(), user_id.strip())
    if el is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Element '{element_id}' not found for user '{user_id}'. "
                "Either it doesn't exist, belongs to a different user, "
                "or STORE_INTERMEDIATES=False in config.py."
            )
        )
    return el


# ════════════════════════════════════════════════════════════
# ROUTE 9: GET /elements/{user_id}
#   List all stored elements for a user, with optional type filter.
# ════════════════════════════════════════════════════════════
@app.get("/elements/{user_id}",
         summary="List all stored table/image elements for a user")
def list_elements(
    user_id: str,
    element_type: Optional[str] = Query(
        default=None,
        description="Filter by type: 'table' | 'image' | 'figure' (omit for all)"
    )
):
    """
    Lists all intermediate elements stored during ingestion for a user.
    Heavy fields (image_b64, table_html) are truncated in the listing;
    use GET /element/{user_id}/{element_id} to fetch the full payload.
    """
    _validate_user_id(user_id)
    if element_type and element_type not in ("table", "image", "figure"):
        raise HTTPException(
            status_code=400,
            detail="element_type must be 'table', 'image', or 'figure'."
        )
    elements = list_user_elements(
        qdrant_client, COLLECTION_SPANISH, user_id.strip(), element_type
    )
    return {
        "user_id":       user_id,
        "element_type":  element_type or "all",
        "count":         len(elements),
        "elements":      elements,
    }


# ════════════════════════════════════════════════════════════
# ROUTE 10: GET /cache/stats  (was ROUTE 8)
# ════════════════════════════════════════════════════════════
@app.get("/cache/stats", summary="Cache hit rate and size stats")
def cache_stats():
    """
    Shows cache performance — hit rate, size, backend type.
    High hit rate = repeated queries served instantly without retrieval or LLM.
    """
    return cache.stats()


# ════════════════════════════════════════════════════════════
# ROUTE 9: DELETE /cache/user/{user_id}  (NEW)
# ════════════════════════════════════════════════════════════
@app.delete("/cache/user/{user_id}", summary="Manually clear cache for a user")
def clear_user_cache(user_id: str):
    """
    Manually clears cached answers for a user.
    Forces fresh retrieval on their next query without re-ingesting.
    """
    _validate_user_id(user_id)
    cache.invalidate_user(user_id)
    return {
        "status":  "ok",
        "user_id": user_id,
        "message": f"Cache cleared for user '{user_id}'. Next query will run fresh retrieval."
    }


# ── Run server ────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("STARTING MULTI-TENANT BILINGUAL RAG SERVER v6.0")
    print(f"  URL:          http://localhost:{PORT}")
    print(f"  Docs:         http://localhost:{PORT}/docs")
    print(f"  LLM:          groq/{GROQ_MODEL}")
    print(f"  Multi-tenant: ✅  (user_id isolation via Qdrant payload filter)")
    print(f"  Retrieval:    Hybrid (BM25 + Vector + Cross-encoder re-rank)")
    print(f"  Cache:        ✅  ({cache.stats()['backend']})")
    print(f"{'='*60}\n")
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False)