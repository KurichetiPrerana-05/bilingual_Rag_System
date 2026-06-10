# ================================
# rag_chain.py  (v10 — multi-tenant user isolation)
# Hybrid retrieval: BM25 + Vector + Cross-encoder re-rank
# EN query translation → ES using googletrans (FREE, no API quota used)
# LLM answers via Groq API (text-only model — no vision quota burned)
#
# MULTI-TENANT CHANGES vs v9:
#
#   1. HybridRetriever.__init__() accepts `user_id` parameter.
#      All_docs passed in are ALREADY filtered to this user's documents
#      (done in server.py via fetch_all_docs(user_id=...)).
#      BM25 index is therefore user-scoped automatically.
#
#   2. HybridRetriever.invoke() passes a Qdrant payload filter on
#      `metadata.user_id` to every vector search call.
#      This ensures vector search only returns chunks for this user.
#
#   3. build_chain() now accepts `user_id` and passes it down to
#      HybridRetriever so the filter is applied at construction time.
#
#   4. run_chain() signature unchanged — callers don't need to change.
#      The user isolation happens inside the retriever, transparently.
#
# All other logic (query expansion, RRF, cross-encoder, prompts) is
# unchanged from v9. Multi-tenancy is purely a retrieval-layer concern.
# ================================

import re
import time
import unicodedata
from typing import List, Tuple, Optional

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import (
    GROQ_API_KEY, GROQ_MODEL, TEMPERATURE,
    VECTOR_TOP_K, BM25_TOP_K, FINAL_TOP_K,
    TRANSLATE_EN_TO_ES,
)

_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
_FINAL_K        = max(FINAL_TOP_K, 10)


# ════════════════════════════════════════════════════════════
# CONTEXT CLEANER
# ════════════════════════════════════════════════════════════

def clean_context_for_llm(context: str) -> str:
    context = re.sub(r'\[/?OCR\]\s*',           '', context)
    context = re.sub(r'\[/?TABLE\]\s*',          '', context)
    context = re.sub(r'\[/?TABLE_PROSE\]\s*',    '', context)
    context = re.sub(r'\[/?IMAGE_SUMMARY\]\s*',  '', context)
    context = re.sub(r'\[/?VISION\]\s*',         '', context)
    context = re.sub(r'\[/?DIAGRAM\]\s*',        '', context)
    context = re.sub(r'\b105\b', 'los', context)
    context = re.sub(r'\b106\b', 'lo',  context)
    context = context.replace('prop6sito',         'proposito')
    context = context.replace('prop6',             'propo')
    context = context.replace('6n',                'on')
    context = context.replace('Ifneas',            'lineas')
    context = context.replace('Uneas',             'lineas')
    context = context.replace('I[neas',            'lineas')
    context = context.replace('companfas',         'companias')
    context = context.replace('companlas',         'companias')
    context = context.replace('aprovisionam1ento', 'aprovisionamiento')
    context = context.replace('aprovisionam!ento', 'aprovisionamiento')
    context = context.replace('aprovlsionamiento', 'aprovisionamiento')
    context = context.replace('Iineas virtuales',  'lineas virtuales')
    context = context.replace('lineas virtuaIes',  'lineas virtuales')
    context = re.sub(r'\bfallo\b',  'falla',  context, flags=re.IGNORECASE)
    context = re.sub(r'\bfallos\b', 'fallas', context, flags=re.IGNORECASE)
    context = re.sub(r'\n{3,}',    '\n\n',   context)
    context = re.sub(r'[ \t]{2,}', ' ',      context)
    return context.strip()


# ════════════════════════════════════════════════════════════
# LLM
# ════════════════════════════════════════════════════════════

def load_llm() -> ChatGroq:
    llm = ChatGroq(
        api_key     = GROQ_API_KEY,
        model_name  = GROQ_MODEL,
        temperature = TEMPERATURE,
        max_tokens  = 1024,
        stop        = ["\n\n\n\n"],
    )
    print(f"Groq LLM ready  (model={GROQ_MODEL})")
    return llm


# ════════════════════════════════════════════════════════════
# QUERY TRANSLATION  (EN -> ES)
# ════════════════════════════════════════════════════════════

_ES_SIGNAL_WORDS = {
    'el', 'la', 'los', 'las', 'de', 'del', 'en', 'para', 'con',
    'que', 'por', 'una', 'un', 'al', 'se', 'es', 'son', 'fue',
    'cual', 'cuales', 'como', 'quien', 'donde', 'cuando',
}


def _looks_spanish(text: str) -> bool:
    if any(ch in text.lower() for ch in 'ñáéíóúü¿¡'):
        return True
    words = set(re.findall(r'\b\w+\b', text.lower()))
    return len(words & _ES_SIGNAL_WORDS) >= 1


def robust_translate_to_spanish(query: str) -> str:
    if not TRANSLATE_EN_TO_ES:
        return query

    for attempt in range(2):
        try:
            from googletrans import Translator
            result     = Translator().translate(query, src='en', dest='es')
            translated = (result.text or '').strip()

            if translated and len(translated) > 3 and _looks_spanish(translated):
                translated = re.sub(r'\bfallo\b',  'falla',  translated, flags=re.IGNORECASE)
                translated = re.sub(r'\bfallos\b', 'fallas', translated, flags=re.IGNORECASE)
                print(f"  Translated (attempt {attempt+1}): '{query}' -> '{translated}'")
                return translated

            print(f"  Translation attempt {attempt+1} non-Spanish: '{translated}' — retrying")

        except Exception as e:
            print(f"  googletrans error (attempt {attempt+1}): {e}")

    print(f"  WARNING: translation failed for '{query}'. English query will hit Spanish corpus.")
    return query


# ════════════════════════════════════════════════════════════
# UTILITY
# ════════════════════════════════════════════════════════════

def _normalize_accents(text: str) -> str:
    nfkd = unicodedata.normalize('NFD', text)
    return nfkd.encode('ascii', 'ignore').decode('utf-8').lower()


# ════════════════════════════════════════════════════════════
# QUERY EXPANSION  (v9 — unchanged in v10)
# ════════════════════════════════════════════════════════════

_QUESTION_PREFIXES = re.compile(
    r'^(¿|¡)?\s*'
    r'(qué|cuál|cuáles|quién|quiénes|cómo|cuándo|dónde|cuánto|cuánta|'
    r'what|which|who|how|when|where|why)\s+'
    r'(es|son|área|áreas|parte|partes|persona|tiempo|código|objetivo|'
    r'is|are|area|part|person|time|code|objective)?\s*',
    flags=re.IGNORECASE,
)

_LEVEL_QUALIFIER = re.compile(
    r'\s+(en el|en la|al|del|de)\s+(nivel|nivel\s+[IVX\d]+|primer nivel|'
    r'segundo nivel|tercer nivel|cuarto nivel|level\s*\d*)',
    flags=re.IGNORECASE,
)

_RESPONSIBLE_PATTERN = re.compile(
    r'(responsable\s+de|encargado\s+de|a cargo\s+de|'
    r'responsible\s+for|in charge\s+of)',
    flags=re.IGNORECASE,
)


def _expand_query(query: str) -> List[str]:
    """
    Returns a list of 2-4 sub-queries derived from the original query.
    The original query is always first in the list.
    (Unchanged from v9 — multi-tenancy doesn't affect query expansion.)
    """
    queries = [query]

    core_with_level = _QUESTION_PREFIXES.sub('', query).strip()
    core_with_level = core_with_level.strip('?¿').strip()
    if core_with_level and core_with_level.lower() != query.lower() and len(core_with_level) > 10:
        queries.append(core_with_level)

    core_no_level = _LEVEL_QUALIFIER.sub('', core_with_level).strip()
    if (
        core_no_level
        and core_no_level.lower() != core_with_level.lower()
        and len(core_no_level) > 10
        and core_no_level not in queries
    ):
        queries.append(core_no_level)

    if _RESPONSIBLE_PATTERN.search(query):
        after_resp = _RESPONSIBLE_PATTERN.split(query)
        if len(after_resp) >= 3:
            subject = after_resp[-1].strip().strip('?¿').strip()
            if subject and len(subject) > 8:
                responsible_query = f"área responsable de {subject}"
                if responsible_query not in queries:
                    queries.append(responsible_query)

    return queries


# ════════════════════════════════════════════════════════════
# HYBRID RETRIEVER  (v10 — multi-tenant user isolation)
# ════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    Two-stage retrieval scoped to a single user's documents.

    MULTI-TENANT:
      - `user_id` is set at construction time.
      - Every vector search call applies a Qdrant payload filter:
            metadata.user_id == user_id
        so only that user's chunks are candidates.
      - The BM25 index is built from `all_docs`, which the caller
        (server.py) has already fetched with user_id filtering via
        fetch_all_docs(user_id=user_id). So BM25 is also user-scoped.

    This means two users can ingest documents with the same filename
    and never see each other's content.
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        all_docs:     List[Document],
        user_id:      str,                         # ← NEW: required for isolation
        vector_k:     int = VECTOR_TOP_K,
        bm25_k:       int = BM25_TOP_K,
        final_k:      int = _FINAL_K,
    ):
        self.vector_store = vector_store
        self.vector_k     = vector_k
        self.bm25_k       = bm25_k
        self.final_k      = final_k
        self.all_docs     = all_docs
        self.user_id      = user_id                # ← stored for vector filter

        # Pre-build the Qdrant filter for this user.
        # Reused on every vector search call — built once here for efficiency.
        self._user_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.user_id",
                    match=MatchValue(value=user_id)
                )
            ]
        )

        self._bm25           = None
        self._bm25_originals = []

        if all_docs:
            # all_docs is already user-scoped (fetched with user_id filter).
            # Building BM25 over section_heading + page_content for better recall.
            normalised_docs = [
                Document(
                    page_content=_normalize_accents(
                        clean_context_for_llm(
                            (d.metadata.get("section_heading") or "") + " " + d.page_content
                        )
                    ),
                    metadata=d.metadata
                )
                for d in all_docs
            ]
            self._bm25           = BM25Retriever.from_documents(normalised_docs, k=bm25_k)
            self._bm25_originals = all_docs
            print(f"  BM25 index built over {len(all_docs)} docs for user='{user_id}' ✅")

        self._reranker = None
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(_RERANKER_MODEL, max_length=512)
            print(f"  Cross-encoder loaded ✅  (model={_RERANKER_MODEL})")
        except Exception as e:
            print(f"  Cross-encoder unavailable — using RRF scores only: {e}")

    def _bm25_lookup_key(self, doc: Document) -> str:
        page   = doc.metadata.get("page_number", "")
        prefix = _normalize_accents(doc.page_content[:80])
        return f"{page}_{prefix}"

    def _rrf_merge(self, list_a: List[Document], list_b: List[Document], k: int = 60) -> List[Document]:
        scores:  dict = {}
        doc_map: dict = {}
        for rank, doc in enumerate(list_a, start=1):
            key = doc.page_content[:120]
            scores[key]  = scores.get(key, 0) + 1.0 / (k + rank)
            doc_map[key] = doc
        for rank, doc in enumerate(list_b, start=1):
            key = doc.page_content[:120]
            scores[key]  = scores.get(key, 0) + 1.0 / (k + rank)
            doc_map[key] = doc
        return [doc_map[k] for k in sorted(scores, key=lambda x: scores[x], reverse=True)]

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if self._reranker is None or not docs:
            return docs
        try:
            pairs  = [(query, clean_context_for_llm(d.page_content)[:512]) for d in docs]
            scores = self._reranker.predict(pairs)
            return [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
        except Exception as e:
            print(f"  Re-ranking failed: {e} — using RRF order")
            return docs

    def _bm25_search(self, query: str) -> List[Document]:
        if self._bm25 is None:
            return []
        try:
            raw_results = self._bm25.invoke(_normalize_accents(query))
            print(f"  BM25 hits  : {len(raw_results)}")
        except Exception as e:
            print(f"  BM25 search failed: {e}")
            return []

        orig_lookup: dict = {}
        for orig_doc in self._bm25_originals:
            key = self._bm25_lookup_key(orig_doc)
            if key not in orig_lookup:
                orig_lookup[key] = orig_doc

        return [orig_lookup.get(self._bm25_lookup_key(r), r) for r in raw_results]

    def _vector_search(self, query: str, k: int) -> List[Document]:
        """
        MULTI-TENANT: Runs vector similarity search with a Qdrant payload
        filter so only this user's chunks are returned.

        The filter (metadata.user_id == self.user_id) is applied at the
        Qdrant server level — efficient even with millions of vectors.
        """
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k":      k,
                "filter": self._user_filter,   # ← user isolation filter
            }
        ).invoke(query)

    def invoke(self, query: str, user_lang: str = "es") -> List[Document]:
        """
        Full retrieval pipeline — scoped to self.user_id.

        Steps:
          1. Translate EN → ES (validated)
          2. Query expansion → 2-4 sub-queries
          3. Vector + BM25 search per sub-query (vector is user-filtered)
          4. RRF merge across all sub-query results
          5. Cross-encoder re-ranks merged candidates
          6. Pin top-1 BM25 hit (exact-match insurance)
          7. Fallback BM25 if < 3 docs returned
        """
        # Step 1 — translate
        search_query = query
        if user_lang == "en" and TRANSLATE_EN_TO_ES:
            search_query = robust_translate_to_spanish(query)

        # Step 2 — query expansion
        sub_queries = _expand_query(search_query)
        print(f"  Sub-queries : {len(sub_queries)}  [user='{self.user_id}']")
        for i, sq in enumerate(sub_queries, 1):
            print(f"    [{i}] {sq}")

        # Step 3 — vector + BM25 per sub-query
        sub_k = max(self.vector_k // max(len(sub_queries), 1), 8)

        def _dedup_key(doc: Document) -> str:
            return _normalize_accents(doc.page_content[:120])

        all_vector_docs: List[Document] = []
        all_bm25_docs:   List[Document] = []
        seen_vec_keys:   set = set()
        seen_bm25_keys:  set = set()

        for sq in sub_queries:
            # Vector search — user-filtered via _vector_search()
            try:
                v_docs = self._vector_search(sq, sub_k)
                for d in v_docs:
                    k = _dedup_key(d)
                    if k not in seen_vec_keys:
                        seen_vec_keys.add(k)
                        all_vector_docs.append(d)
            except Exception as e:
                print(f"  Vector search failed for sub-query '{sq[:40]}': {e}")

            # BM25 search — user-scoped because index was built from user's docs
            b_docs = self._bm25_search(sq)
            for d in b_docs:
                k = _dedup_key(d)
                if k not in seen_bm25_keys:
                    seen_bm25_keys.add(k)
                    all_bm25_docs.append(d)

        print(f"  Vector hits: {len(all_vector_docs)} (across {len(sub_queries)} sub-queries)")
        print(f"  BM25 hits  : {len(all_bm25_docs)} (across {len(sub_queries)} sub-queries)")

        # Step 4 — RRF merge
        if all_vector_docs and all_bm25_docs:
            merged = self._rrf_merge(all_vector_docs, all_bm25_docs)
            if len(all_bm25_docs) >= 3:
                merged = self._rrf_merge(merged, all_bm25_docs)
        elif all_vector_docs:
            merged = all_vector_docs
        else:
            merged = all_bm25_docs

        # Step 5 — cross-encoder re-rank
        reranked = self._rerank(search_query, merged)

        # Step 6 — pin top-1 BM25 hit (original query)
        first_bm25    = self._bm25_search(search_query)
        pinned_keys:  set  = set()
        pinned_docs:  list = []
        for doc in first_bm25[:1]:
            key = _dedup_key(doc)
            if key not in pinned_keys:
                pinned_keys.add(key)
                pinned_docs.append(doc)

        seen    = set(pinned_keys)
        results = list(pinned_docs)
        for doc in reranked:
            key = _dedup_key(doc)
            if key not in seen:
                seen.add(key)
                results.append(doc)
            if len(results) >= self.final_k:
                break

        # Step 7 — fallback BM25 on simplified query if < 3 docs
        if len(results) < 3 and self._bm25 is not None:
            key_words  = [w for w in search_query.split() if len(w) > 4]
            simplified = " ".join(key_words[:5])
            if simplified:
                print(f"  Fallback BM25 (simplified): '{simplified}'")
                fallback_docs = self._bm25_search(simplified)
                existing_keys = {_dedup_key(d) for d in results}
                added = 0
                for d in fallback_docs:
                    k = _dedup_key(d)
                    if k not in existing_keys:
                        results.append(d)
                        existing_keys.add(k)
                        added += 1
                    if len(results) >= self.final_k:
                        break
                if added:
                    print(f"  Fallback added {added} extra doc(s)")

        print(f"  Final docs  : {len(results)}")
        return results


# ════════════════════════════════════════════════════════════
# DYNAMIC PROMPT BUILDER  (v9 — unchanged in v10)
# ════════════════════════════════════════════════════════════

_PROMPT_BASE = """You are a precise bilingual document assistant (English and Spanish).

Context (each chunk is labelled with its exact source — PDF name, page, and for
tables/images the element label such as "Table 2 from report.pdf, page 5"):
{context}

Rules:
1. Answer ONLY from the context. Do not invent facts.
2. Read ALL chunks before concluding the answer is missing.
3. Copy exact values (names, codes, times, numbers) directly from the context.
4. {ocr_table_rule}
5. If the answer value is present anywhere in the context, state it directly
   and concisely. Do not hedge or qualify.
6. Be concise — one or two sentences maximum. Extract only the specific
   answer to the question asked.
6b. SOURCE CITATION RULE: After your answer, append a single line:
   Source: <exact [Source: ...] label from the chunk(s) that contained the answer>
   If the answer came from a table chunk, write the full table label, e.g.:
     Source: Table 2 from report.pdf, page 5
   If from an image chunk:
     Source: Image 1 from report.pdf, page 3
   If from a plain text chunk:
     Source: report.pdf, page 7
   If multiple chunks contributed, list all labels separated by " | ".
   This line is MANDATORY — always include it.
6c. IDENTIFIER RULE: If the question specifies a version number, row number,
   level, user type, product name, or any other discriminating identifier
   (e.g. "version 1.3", "Nivel III", "VPN Pura", "Q2"), you MUST locate
   EXACTLY that identifier in the context and answer ONLY from that row or
   section. Do NOT use data from a different row, version, or category even
   if it appears nearby or looks similar.
{synonym_rule}7. If the answer is genuinely absent from every chunk, say exactly:
   - English: "The information is not available in the provided context."
   - Spanish: "La informacion no esta disponible en el contexto proporcionado."
   (Do NOT append a Source line when the answer is absent.)
8. Answer in {user_lang_label} ONLY.

Question: {question}

Answer:"""


def _has_ocr_content(docs: List[Document]) -> bool:
    return any(
        d.metadata.get("has_page_ocr")
        or d.metadata.get("content_type") in ("page_ocr", "mixed")
        for d in docs
    )


def _has_table_content(docs: List[Document]) -> bool:
    return any(
        d.metadata.get("has_table")
        or d.metadata.get("content_type") in ("table", "mixed")
        for d in docs
    )


def _build_ocr_table_rule(docs: List[Document]) -> str:
    has_ocr   = _has_ocr_content(docs)
    has_table = _has_table_content(docs)

    if has_ocr and has_table:
        return (
            "These chunks come from OCR-scanned tables. A table row's columns "
            "often appear on separate lines — treat consecutive short lines as "
            "one table row (e.g. a responsible party on line 1, a category on "
            "line 2, and a time/value on line 3 all belong together as one data "
            "point). Do NOT say a value is missing if it appears on an adjacent "
            "line to the subject — that IS the explicit mention. "
            "IMPORTANT: When the question asks about a specific identifier "
            "(e.g. a version number, a level, a user type, a product name), "
            "locate EXACTLY that identifier in the table before answering. "
            "Do NOT use data from a different row even if it looks similar."
        )
    elif has_ocr:
        return (
            "These chunks come from OCR-scanned pages. Lines may be fragmented — "
            "read nearby lines together to understand the full meaning. Do NOT say "
            "a value is missing if it appears on an adjacent line."
        )
    elif has_table:
        return (
            "Some chunks contain table data. Column headers and cell values may "
            "appear on separate lines — read them together as one row. "
            "When the question asks about a specific identifier (e.g. a version, "
            "level, or category), match ONLY that row in the table."
        )
    else:
        return "Read each chunk carefully and extract the specific value asked for."


def _build_synonym_rule(docs: List[Document], question: str) -> str:
    abbreviations = re.findall(r'\b([A-ZÁÉÍÓÚ]{2,6})\b', question)
    if not abbreviations:
        return ""

    synonyms_found = []
    for abbr in abbreviations:
        for doc in docs:
            text = clean_context_for_llm(doc.page_content)

            m = re.search(
                rf'([A-ZÁÉÍÓÚa-záéíóúñ][A-Za-záéíóúñ\s]{{5,50}}?)\s*[\(]\s*{re.escape(abbr)}\s*[\)]',
                text
            )
            if m:
                full_name = m.group(1).strip().rstrip('/ ')
                if full_name and full_name.lower() != abbr.lower():
                    synonyms_found.append(f'"{abbr}" and "{full_name}" refer to the same entity')
                    break

            m2 = re.search(
                rf'{re.escape(abbr)}\s*[\(]\s*([A-ZÁÉÍÓÚa-záéíóúñ][A-Za-záéíóúñ\s]{{5,50}}?)\s*[\)]',
                text
            )
            if m2:
                full_name = m2.group(1).strip()
                if full_name and full_name.lower() != abbr.lower():
                    synonyms_found.append(f'"{abbr}" and "{full_name}" refer to the same entity')
                    break

            m3 = re.search(
                rf'([A-ZÁÉÍÓÚa-záéíóúñ][A-Za-záéíóúñ\s]{{5,50}}?)\s*/\s*{re.escape(abbr)}\b',
                text
            )
            if m3:
                full_name = m3.group(1).strip()
                if full_name and full_name.lower() != abbr.lower():
                    synonyms_found.append(f'"{abbr}" and "{full_name}" refer to the same entity')
                    break

    if not synonyms_found:
        return ""

    lines = "\n   ".join(synonyms_found)
    return f"6b. Treat these as equivalent when matching against context:\n   {lines}\n"


def _chunk_source_label(doc: Document) -> str:
    """
    Builds a rich, human-readable source label for each chunk injected into
    the LLM prompt.  Table and image chunks carry their element identity so
    the LLM can cite the exact element in its answer.

    Examples:
      Text  → [Source: report.pdf, page 5, chunk 3]
      Table → [Source: Table 2 from report.pdf, page 5, chunk 3]
      Image → [Source: Image 1 from report.pdf, page 5, chunk 3]
    """
    m      = doc.metadata
    name   = m.get("pdf_name", m.get("source", "?"))
    page   = m.get("page_number", "?")
    cidx   = m.get("chunk_index", "?")
    ctype  = m.get("content_type", "text")
    label  = m.get("element_label", "")

    if label:
        # element_label is set by chunk_pdf() for every table/image chunk
        return f"[Source: {label}, chunk {cidx}]"
    elif ctype in ("table", "table_prose"):
        tidx = m.get("table_index", "")
        return f"[Source: Table {tidx} from {name}, page {page}, chunk {cidx}]"
    elif ctype in ("image", "image_ocr", "figure"):
        iidx = m.get("image_index", "")
        return f"[Source: Image {iidx} from {name}, page {page}, chunk {cidx}]"
    else:
        return f"[Source: {name}, page {page}, chunk {cidx}]"


def build_prompt_for_docs(
    docs: List[Document],
    question: str,
    user_lang_label: str,
) -> str:
    context = "\n\n---\n\n".join(
        f"{_chunk_source_label(d)}\n{d.page_content}"
        for d in docs
    )
    context = clean_context_for_llm(context)

    ocr_table_rule = _build_ocr_table_rule(docs)
    synonym_rule   = _build_synonym_rule(docs, question)

    return _PROMPT_BASE.format(
        context         = context,
        ocr_table_rule  = ocr_table_rule,
        synonym_rule    = synonym_rule,
        user_lang_label = user_lang_label,
        question        = question,
    )


# ════════════════════════════════════════════════════════════
# BUILD CHAIN  (v10 — accepts user_id)
# ════════════════════════════════════════════════════════════

def build_chain(
    vector_store: QdrantVectorStore,
    llm: ChatGroq,
    all_docs: List[Document] = None,
    user_id: str = None,                        # ← NEW: required for multi-tenancy
):
    """
    Builds the hybrid RAG chain scoped to a single user.

    MULTI-TENANT:
      `user_id` is passed to HybridRetriever so both vector search
      (via Qdrant filter) and BM25 (via pre-filtered all_docs) are
      restricted to only this user's documents.

    Args:
        vector_store : Qdrant vector store (shared collection)
        llm          : Groq LLM instance (from load_llm())
        all_docs     : this user's documents for BM25 index
                       (already filtered — from fetch_all_docs(user_id=...))
        user_id      : the user whose chain we're building

    Returns:
        (hybrid_retriever, run_chain)
        run_chain(question, user_lang) -> (answer_str, source_docs)
    """
    if not user_id:
        raise ValueError("user_id is required to build a user-scoped chain.")

    retriever = HybridRetriever(
        vector_store = vector_store,
        all_docs     = all_docs or [],
        user_id      = user_id,                 # ← passed for isolation
    )

    parser = StrOutputParser()

    def run_chain(question: str, user_lang: str = "es") -> Tuple[str, List[Document]]:
        user_lang_label = "English" if user_lang == "en" else "Spanish"

        docs = retriever.invoke(question, user_lang=user_lang)

        if not docs:
            if user_lang == "en":
                return "The information is not available in the provided context.", []
            return "La informacion no esta disponible en el contexto proporcionado.", []

        filled = build_prompt_for_docs(docs, question, user_lang_label)

        for attempt in range(4):
            try:
                answer = parser.invoke(llm.invoke(filled))
                return answer.strip(), docs

            except Exception as e:
                err = str(e).lower()

                if "rate limit" in err or "429" in err:
                    wait = 15 * (attempt + 1)
                    print(f"  Groq rate limit — waiting {wait}s (attempt {attempt+1}/4)")
                    time.sleep(wait)

                elif "context" in err and "length" in err:
                    print("  Context too long — trimming to top-5 docs and retrying")
                    filled = build_prompt_for_docs(docs[:5], question, user_lang_label)

                elif "connection" in err or "timeout" in err:
                    wait = 5 * (attempt + 1)
                    print(f"  Groq connection error — waiting {wait}s")
                    time.sleep(wait)

                else:
                    raise

        return "Groq API unavailable after retries. Check your API key and rate limits.", docs

    print(f"Hybrid RAG chain built ✅  (v10 — multi-tenant, user='{user_id}')")
    return retriever, run_chain