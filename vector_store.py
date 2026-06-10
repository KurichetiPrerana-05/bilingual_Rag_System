# ================================
# vector_store.py  (v2 — multi-tenant)
#
# MULTI-TENANT DESIGN:
#   - Single Qdrant collection for ALL users.
#   - Every chunk gets a `user_id` field in its metadata/payload.
#   - Queries always filter by `user_id` so users only see their own data.
#   - No separate collection per user — this scales to thousands of users.
# ================================

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from config import QDRANT_URL, QDRANT_API_KEY, VECTOR_DIM, COLLECTION_SPANISH, STORE_INTERMEDIATES

# ────────────────────────────────────────────────────────────
# INTERMEDIATE ELEMENT STORE
#
# WHY: Each chunk in Qdrant holds only the *text* used for retrieval.
#      But tables and images have richer structure (HTML, base64 image,
#      full OCR text) that is too large to embed and doesn't belong in
#      the vector payload.  We store these as a SEPARATE Qdrant collection
#      ("_elements" suffix) keyed by element_id.
#
#      At query time, when a chunk was sourced from a table or image,
#      the response includes element_id + element_type so the caller can
#      fetch the full element from GET /element/{user_id}/{element_id}
#      and display/verify it.
#
# SCHEMA per element point:
#   id          : UUID string (same as element_id in chunk metadata)
#   payload:
#     user_id       : str   — owner (for access control)
#     pdf_name      : str   — source document name
#     page_number   : int
#     element_type  : "table" | "image" | "figure"
#     table_index   : int   — 1-based table number within the page (tables only)
#     image_index   : int   — 1-based image number within the page (images only)
#     table_html    : str   — raw HTML from camelot/pdfplumber (tables only)
#     table_markdown: str   — markdown representation  (tables only)
#     image_b64     : str   — base64-encoded PNG/JPEG  (images only)
#     image_ocr     : str   — OCR text extracted from image (images only)
#     caption       : str   — detected caption text (figures)
#     chunk_ids     : list  — which chunk UUIDs reference this element
# ────────────────────────────────────────────────────────────

ELEMENTS_SUFFIX = "_elements"   # appended to collection name


def _elements_collection(collection_name: str) -> str:
    return collection_name + ELEMENTS_SUFFIX


def ensure_elements_collection(client: QdrantClient, collection_name: str):
    """
    Creates the elements store collection if it doesn't exist.
    Elements are stored as points with NO vector (payload-only) because
    we never do similarity search on them — only exact-key lookups.
    We use a dummy 1-dim vector to satisfy Qdrant's requirement.
    """
    if not STORE_INTERMEDIATES:
        return
    ename = _elements_collection(collection_name)
    try:
        client.get_collection(ename)
        print(f"  Elements collection '{ename}' already exists.")
    except Exception:
        client.create_collection(
            collection_name=ename,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )
        print(f"  Elements collection '{ename}' created ✅")
    # Index element_id and user_id for fast lookups
    from qdrant_client.models import PayloadSchemaType
    for field in ("user_id", "pdf_name", "element_id"):
        try:
            client.create_payload_index(
                collection_name=ename,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # already exists


def store_elements(
    client: QdrantClient,
    collection_name: str,
    elements: list,          # list of dicts — see schema above
    user_id: str,
):
    """
    Upserts intermediate elements (tables, images) into the elements store.

    Each element dict should have at minimum:
        element_id   : str (UUID)
        element_type : "table" | "image" | "figure"
        pdf_name     : str
        page_number  : int

    Optional rich fields (stored verbatim in payload):
        table_html, table_markdown, image_b64, image_ocr, caption, chunk_ids

    Returns the number of elements stored.
    """
    if not STORE_INTERMEDIATES or not elements:
        return 0

    import uuid as _uuid
    from qdrant_client.models import PointStruct

    ename  = _elements_collection(collection_name)
    points = []
    for el in elements:
        el_id = el.get("element_id") or str(_uuid.uuid4())
        payload = {
            "user_id":        user_id,
            "element_id":     el_id,
            "element_type":   el.get("element_type", "unknown"),
            "pdf_name":       el.get("pdf_name", ""),
            "page_number":    el.get("page_number", 0),
            "table_index":    el.get("table_index", 0),
            "image_index":    el.get("image_index", 0),
            "table_html":     el.get("table_html", ""),
            "table_markdown": el.get("table_markdown", ""),
            "image_b64":      el.get("image_b64", ""),
            "image_ocr":      el.get("image_ocr", ""),
            "caption":        el.get("caption", ""),
            "chunk_ids":      el.get("chunk_ids", []),
            "label":          el.get("label", ""),   # e.g. "Table 2 from report.pdf p.5"
        }
        points.append(PointStruct(
            id=str(_uuid.uuid4()),   # Qdrant point UUID (different from element_id)
            vector=[0.0],            # dummy vector — no similarity search needed
            payload=payload,
        ))

    BATCH = 50
    for i in range(0, len(points), BATCH):
        client.upsert(collection_name=ename, points=points[i:i+BATCH])

    print(f"  Stored {len(points)} intermediate elements in '{ename}' ✅")
    return len(points)


def fetch_element(
    client: QdrantClient,
    collection_name: str,
    element_id: str,
    user_id: str,
) -> dict | None:
    """
    Fetches a single stored element by element_id + user_id.
    Returns the payload dict or None if not found / access denied.
    """
    if not STORE_INTERMEDIATES:
        return None
    ename = _elements_collection(collection_name)
    try:
        results, _ = client.scroll(
            collection_name=ename,
            scroll_filter=Filter(must=[
                FieldCondition(key="element_id", match=MatchValue(value=element_id)),
                FieldCondition(key="user_id",    match=MatchValue(value=user_id)),
            ]),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if results:
            return results[0].payload
        return None
    except Exception as e:
        print(f"  fetch_element error: {e}")
        return None


def list_user_elements(
    client: QdrantClient,
    collection_name: str,
    user_id: str,
    element_type: str = None,   # filter by "table" | "image" | None (all)
) -> list:
    """
    Lists all intermediate elements for a user (optionally filtered by type).
    Returns list of payload dicts (without the heavy b64/html fields truncated).
    """
    if not STORE_INTERMEDIATES:
        return []
    ename = _elements_collection(collection_name)
    conditions = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    if element_type:
        conditions.append(FieldCondition(key="element_type", match=MatchValue(value=element_type)))

    all_els = []
    offset   = None
    try:
        while True:
            results, next_offset = client.scroll(
                collection_name=ename,
                scroll_filter=Filter(must=conditions),
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in results:
                p = dict(r.payload)
                # Truncate heavy fields in listing to keep response small
                if p.get("image_b64"):
                    p["image_b64"] = p["image_b64"][:40] + "...(truncated)"
                if p.get("table_html") and len(p["table_html"]) > 200:
                    p["table_html"] = p["table_html"][:200] + "...(truncated)"
                all_els.append(p)
            if next_offset is None:
                break
            offset = next_offset
    except Exception as e:
        print(f"  list_user_elements error: {e}")
    return all_els


def delete_user_elements(
    client: QdrantClient,
    collection_name: str,
    user_id: str,
):
    """Deletes all stored elements for a user. Called alongside delete_user_documents."""
    if not STORE_INTERMEDIATES:
        return
    from qdrant_client.models import FilterSelector
    ename = _elements_collection(collection_name)
    try:
        client.delete(
            collection_name=ename,
            points_selector=FilterSelector(filter=Filter(must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ])),
        )
        print(f"  Deleted all elements for user='{user_id}' from '{ename}' ✅")
    except Exception as e:
        print(f"  Warning: could not delete elements for user='{user_id}': {e}")


def ensure_payload_index(client: QdrantClient, collection_name: str = COLLECTION_SPANISH):
    """
    Creates a keyword payload index on 'metadata.user_id' if it doesn't exist.

    WHY THIS IS NEEDED:
      Qdrant requires an explicit index on any field used in a Filter.
      Without it, filtered queries (count, scroll, search) fail with:
        "Index required but not found for 'metadata.user_id'"
      Safe to call repeatedly — does nothing if index already exists.
    """
    try:
        from qdrant_client.models import PayloadSchemaType
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.user_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print(f"  Payload index created on 'metadata.user_id' ✅")
    except Exception as e:
        err = str(e).lower()
        # Qdrant raises an error if the index already exists — that's fine
        if "already exists" in err or "conflict" in err or "400" in err:
            print(f"  Payload index already exists — skipping.")
        else:
            print(f"  Warning: could not create payload index: {e}")


def get_qdrant_client() -> QdrantClient:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    print("Qdrant connected ✅")
    return client


def create_collection(client: QdrantClient, collection_name: str = COLLECTION_SPANISH):
    """
    Creates the collection only if it does not already exist.
    Safe to call before every ingest — never wipes existing data.
    """
    try:
        info = client.get_collection(collection_name)
        print(f"  Collection '{collection_name}' already exists "
              f"({info.points_count} vectors) — appending.")
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
        print(f"  Collection '{collection_name}' created ✅")

    # Always ensure the payload index exists — required for user_id filtering
    ensure_payload_index(client, collection_name)
    # Bootstrap intermediate element store (tables, images)
    ensure_elements_collection(client, collection_name)


def recreate_collection(client: QdrantClient, collection_name: str = COLLECTION_SPANISH):
    """
    DESTRUCTIVE: deletes and recreates the collection from scratch.
    Called by DELETE /collection endpoint only.
    WARNING: This wipes ALL users' data. Use delete_user_documents() to
    delete a single user's data instead.
    """
    try:
        client.delete_collection(collection_name)
        print(f"  Collection '{collection_name}' deleted.")
    except Exception:
        pass
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
    )
    print(f"  Collection '{collection_name}' recreated ✅")
    # Re-create the payload index after wipe
    ensure_payload_index(client, collection_name)
    # Re-create elements store after wipe
    ensure_elements_collection(client, collection_name)


def _stamp_user_id(docs: list, user_id: str) -> list:
    """
    Adds user_id to the metadata of every document chunk.

    This is the key step for multi-tenancy. Every chunk stored in Qdrant
    will carry a `user_id` payload field. Retrieval always filters on this
    field so users never see each other's data.

    Args:
        docs    : list of LangChain Document objects
        user_id : unique identifier for the user uploading the document

    Returns:
        Same list of documents, with `user_id` stamped into each metadata dict.
    """
    if not user_id:
        raise ValueError("user_id must not be empty — required for multi-tenant isolation.")

    for doc in docs:
        doc.metadata["user_id"] = user_id

    print(f"  Stamped user_id='{user_id}' on {len(docs)} chunks ✅")
    return docs


def index_documents(
    docs: list,
    embedding_model,
    user_id: str,                              # required for multi-tenancy
    collection_name: str = COLLECTION_SPANISH,
    elements: list = None,                     # ← NEW: intermediate table/image elements
    qdrant_client: QdrantClient = None,        # ← NEW: needed to store elements
) -> QdrantVectorStore:
    """
    Embeds and indexes documents in small manual batches to avoid
    WriteTimeout errors when uploading large PDFs (150+ chunks).

    MULTI-TENANT:
      Before indexing, stamps every chunk with `user_id` in metadata.
      This lets Qdrant filter by user at query time.

    INTERMEDIATE ELEMENTS:
      If `elements` is provided (list of table/image dicts from chunk_pdf),
      they are stored in the companion elements collection alongside the
      text chunks. Each chunk's metadata already contains:
        - element_id    : UUID linking back to the element
        - element_type  : "table" | "image" | "figure"
        - table_index   : 1-based table number within page
        - image_index   : 1-based image number within page
        - element_label : human-readable label e.g. "Table 2 from report.pdf p.5"
      This enables source attribution like "Answer taken from Table 2, report.pdf, page 5"
      and lets callers fetch the full HTML/image via GET /element/{user_id}/{element_id}.

    Args:
        docs            : chunked LangChain Documents from chunk_pdf()
        embedding_model : loaded HuggingFaceEmbeddings model
        user_id         : the uploader's unique ID (e.g. "user_abc123")
        collection_name : Qdrant collection (default: rag_spanish)
        elements        : optional list of intermediate element dicts from chunk_pdf()
        qdrant_client   : QdrantClient instance (required if elements provided)

    Returns:
        QdrantVectorStore pointing to the updated collection.
    """
    import time

    # Stamp user_id on every chunk BEFORE uploading
    docs = _stamp_user_id(docs, user_id)

    # Store intermediate elements (tables, images) if provided
    if elements and qdrant_client and STORE_INTERMEDIATES:
        store_elements(qdrant_client, collection_name, elements, user_id)

    BATCH = 25   # 25 chunks x 768 dims = ~75KB per request, well within timeout

    vs = None
    total = len(docs)
    for i in range(0, total, BATCH):
        batch = docs[i : i + BATCH]
        print(f"  Uploading batch {i//BATCH + 1}/{(total-1)//BATCH + 1} ({len(batch)} chunks)...", end=" ")
        for attempt in range(4):
            try:
                vs = QdrantVectorStore.from_documents(
                    documents=batch,
                    embedding=embedding_model,
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                    collection_name=collection_name,
                    force_recreate=False,
                )
                print("✓")
                break
            except Exception as e:
                if attempt < 3:
                    wait = 10 * (attempt + 1)
                    print(f"timeout, retrying in {wait}s...", end=" ")
                    time.sleep(wait)
                else:
                    print(f"FAILED after 4 attempts: {e}")
                    raise

    el_count = len(elements) if elements else 0
    print(f"  Indexed {total} chunks + {el_count} elements for user='{user_id}' into '{collection_name}' ✅")
    return vs


def load_vector_store(embedding_model, collection_name: str = COLLECTION_SPANISH) -> QdrantVectorStore:
    """Loads an existing collection as a LangChain vector store (unfiltered)."""
    return QdrantVectorStore(
        client=QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60),
        collection_name=collection_name,
        embedding=embedding_model
    )


def fetch_all_docs(
    client: QdrantClient,
    collection_name: str,
    embedding_model,
    user_id: str = None,                       # ← NEW: optional filter by user
) -> list:
    """
    Fetches stored points from Qdrant and reconstructs LangChain Documents.
    Used to build the BM25 index in-memory at query time.

    MULTI-TENANT:
      If `user_id` is provided, fetches only that user's chunks.
      This ensures the BM25 index built for a query only searches
      that user's documents — not all users' data.

      If `user_id` is None, fetches ALL documents (used by admin
      endpoints or collection-level health checks).

    Args:
        client          : QdrantClient instance
        collection_name : collection to scroll through
        embedding_model : not used here, kept for interface compatibility
        user_id         : if set, filters to only this user's chunks

    Returns:
        List of LangChain Document objects.
    """
    all_docs = []
    offset   = None

    # Build scroll filter if user_id is given
    scroll_filter = None
    if user_id:
        scroll_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.user_id",      # ← path inside Qdrant payload
                    match=MatchValue(value=user_id)
                )
            ]
        )
        print(f"  Fetching docs for user_id='{user_id}' from '{collection_name}'...")
    else:
        print(f"  Fetching ALL docs from '{collection_name}'...")

    try:
        while True:
            results, next_offset = client.scroll(
                collection_name=collection_name,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                scroll_filter=scroll_filter,    # ← apply user filter if set
            )
            for point in results:
                payload = point.payload or {}
                content = payload.get("page_content", "")
                if not content:
                    content = payload.get("text", "")
                if content:
                    meta = payload.get("metadata", {})
                    if not meta:
                        meta = {k: v for k, v in payload.items() if k != "page_content"}
                    all_docs.append(Document(page_content=content, metadata=meta))

            if next_offset is None:
                break
            offset = next_offset

        user_label = f"user='{user_id}'" if user_id else "ALL users"
        print(f"  Fetched {len(all_docs)} docs ({user_label}) for BM25 index ✅")

    except Exception as e:
        print(f"  Warning: could not fetch docs for BM25 ({e})")

    return all_docs


def delete_user_documents(
    client: QdrantClient,
    collection_name: str,
    user_id: str,
) -> int:
    """
    Deletes ALL chunks belonging to a specific user from the collection.
    Does NOT affect any other user's data.

    Called by DELETE /user/{user_id}/documents endpoint.

    Args:
        client          : QdrantClient instance
        collection_name : Qdrant collection name
        user_id         : the user whose data to delete

    Returns:
        Number of points deleted (approximate — Qdrant returns operation status).
    """
    from qdrant_client.models import FilterSelector

    delete_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.user_id",
                match=MatchValue(value=user_id)
            )
        ]
    )

    try:
        result = client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(filter=delete_filter),
        )
        print(f"  Deleted all documents for user='{user_id}' from '{collection_name}' ✅")
        print(f"  Operation status: {result.status}")
        # Also delete companion element store entries for this user
        delete_user_elements(client, collection_name, user_id)
        return 1  # Qdrant delete returns status, not exact count
    except Exception as e:
        print(f"  Failed to delete documents for user='{user_id}': {e}")
        raise


def count_user_documents(
    client: QdrantClient,
    collection_name: str,
    user_id: str,
) -> int:
    """
    Counts how many chunks are stored for a specific user.
    Used in /user/{user_id}/status endpoint.
    """
    count_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.user_id",
                match=MatchValue(value=user_id)
            )
        ]
    )
    try:
        result = client.count(
            collection_name=collection_name,
            count_filter=count_filter,
            exact=True,
        )
        return result.count
    except Exception as e:
        err_msg = str(e)
        print(f"  Could not count docs for user='{user_id}': {err_msg}")
        # Re-raise index errors so server.py can surface them clearly
        # instead of silently returning 0 (which causes a false 404)
        if "index required" in err_msg.lower() or "bad request" in err_msg.lower():
            raise RuntimeError(
                f"Qdrant payload index missing for 'metadata.user_id'. "
                f"Call ensure_payload_index() on startup to fix this. "
                f"Original error: {err_msg}"
            )
        return 0