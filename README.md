# Bilingual RAG System 🌐
**English + Spanish PDF Question Answering with Hybrid Retrieval**

> Upload PDFs in English or Spanish. Ask questions in either language. Get accurate answers powered by AI.

---

## Table of Contents
1. [What Is This?](#what-is-this)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [System Architecture](#system-architecture)
5. [Intermediate Storage](#intermediate-storage)
6. [Models Used](#models-used)
7. [Config Reference](#config-reference)
8. [API Endpoints](#api-endpoints)
9. [How to Run](#how-to-run)
10. [Key Design Decisions](#key-design-decisions)
11. [Limitations](#limitations)

---

## What Is This?

A production-ready **Bilingual RAG (Retrieval-Augmented Generation)** system. It:

- Accepts PDF documents in **English or Spanish**
- Extracts text, tables, and images from PDFs (including scanned ones)
- Saves extracted tables as **CSV files** and images as **PNG/JPG files** for manual inspection
- Supports **multi-tenant** usage — each user's documents are isolated in a single Qdrant collection
- Answers questions in **whichever language you ask** — even cross-language
- Uses **hybrid retrieval**: BM25 (keyword) + Vector (semantic) + Cross-encoder re-ranking

---

## Tech Stack

| Layer | Tool / Model |
|---|---|
| Embeddings | `nomic-ai/nomic-embed-text-v2-moe` (local, 768-dim, multilingual) |
| LLM (answers) | `llama-3.1-8b-instant` via Groq API (text-only) |
| Vector DB | Qdrant Cloud (single collection, multi-tenant via `user_id` filter) |
| OCR | PaddleOCR v3 (EN + ES, CPU, offline) |
| Vision (ingest) | moondream via Ollama (local, offline) |
| Re-ranker | `cross-encoder/mmarco-mMiniLMv2-L12` (local) |
| Translation | `googletrans` (free, no API key) |
| Framework | FastAPI + LangChain |

---

## Project Structure

```
├── config.py               # All API keys, model names, hyperparameters
├── embeddings.py           # Loads Nomic embedding model + patches simsimd bug
├── vector_store.py         # Qdrant CRUD — multi-tenant, user_id filtered
├── pdf_utils.py            # PDF ingestion — text + OCR + table/image extraction + chunking
├── rag_chain.py            # Hybrid retrieval — BM25 + Vector + Re-rank + LLM
├── cache.py                # Two-level cache — query cache + embedding cache (Redis or LRU)
├── server.py               # FastAPI server — REST API + intermediates inspection endpoints
├── ingest_docs.py          # CLI ingestion script
├── setup_ocr.py            # Dependency checker — run once before first use
├── bilingual_rag_chat.html # Browser chat UI (served at localhost:8000/)
├── pdfs/                   # Put your PDF files here
└── intermediates/          # Auto-created — extracted tables, images, OCR text per document
    └── <doc_slug>/
        ├── tables/         # .csv + .json per table  (native PDFs)
        ├── images/         # .jpg + .json per image
        └── ocr_pages/      # .txt + .json per scanned page (OCR + vision text)
```

---

## System Architecture

### Phase A — Ingestion Pipeline (`pdf_utils.py`)

```
PDF File
  ↓  Layer 1: pypdf           → native text extraction (digital PDFs)
  ↓  Layer 2: pdfplumber      → table extraction (runs on ALL pages, not just native)
  ↓  Layer 3: pdfplumber      → embedded images → moondream (local vision description)
  ↓  Layer 4: PaddleOCR v3   → full-page OCR for scanned/image-only pages
  ↓  Layer 5: moondream       → vision prose for pages needing extra context
  ↓  Intermediate save        → tables→CSV, images→JPG, scanned pages→TXT
  ↓  Language Detection       → custom regex (no external library)
  ↓  Text Cleaning            → fixes garbled Spanish chars (e.g. prop6sito → propósito)
  ↓  Table → Prose Conversion → [Version=1.3] Date: 01-Mar. Description: ...
  ↓  Chunking                 → RecursiveCharacterTextSplitter (500 chars, 100 overlap)
  ↓  Atomic blocks            → TABLE and level-structured OCR blocks never split
  ↓  Deduplication            → MD5 hash removes repeated headers/footers
  ↓  Upload to Qdrant         → batches of 25, user_id stamped on every chunk
  ↓
Collection: rag_spanish  ← ALL users, ALL languages — filtered by user_id at query time
```

#### Four-Layer PDF Extraction

| Layer | Method | Purpose |
|---|---|---|
| 1 — pypdf | Native text | Fast digital text + language detection |
| 2 — pdfplumber | Table extraction | Runs on every page regardless of scan status — reads PDF vector structure |
| 3 — pdfplumber + moondream | Image description | Extracts images, moondream generates prose description |
| 4 — pdf2image + PaddleOCR | Full-page OCR | For scanned pages: renders at 300 DPI, OCRs with PaddleOCR (EN+ES) |

> **Why pdfplumber runs on all pages (including scanned):**
> pdfplumber reads the PDF's vector/line structure to detect table grids — not the text content.
> So it can find native-embedded tables even on pages where `native_text < 80 chars`.

#### Table → Prose Conversion

```
Raw:       | Version | Date     | Description       |
           | 1.3     | 01-Mar   | Updated flowchart |

Converted: [Version=1.3] Date: 01-Mar. Description: Updated flowchart.
```

#### Atomic Chunking

- **TABLE blocks** → never split. Headers and values always stay together.
- **Level-structured OCR blocks** (e.g. Nivel I / Nivel II / Nivel III) → kept atomic.

---

### Phase B — Retrieval + Generation Pipeline (`rag_chain.py`)

```
User Question (EN or ES)
  ↓  Cache check              → return cached answer if same question asked recently
  ↓  Language Detection
  ↓  Query Expansion          → 4 sub-queries from different semantic angles
  ↓  EN→ES Translation        → googletrans (validated — must look Spanish)
  ↓  BM25 Retrieval           → keyword matching, top 20 per sub-query (user_id filtered)
  ↓  Vector Retrieval         → cosine similarity in Qdrant, top 20 per sub-query
  ↓  RRF Merge                → Reciprocal Rank Fusion — deduplicates + merges
  ↓  BM25 Top-1 Pinning       → exact-match result always included
  ↓  Cross-Encoder Re-ranking → (question, chunk) scored together — top 15 selected
  ↓  Dynamic Prompt           → built per-query with source attribution rules
  ↓  Groq LLM                 → generates answer, cites table/image sources
Answer returned in user's original language
```

---

## Intermediate Storage

Every ingestion automatically saves extracted content to disk under `intermediates/<doc_slug>/` so you can verify what was extracted without querying Qdrant.

### Folder structure

```
intermediates/
├── wla_mensajeria_corporativa_fi/     ← native PDF
│   ├── tables/
│   │   ├── table_001_page2.csv        ← open in Excel to verify rows
│   │   ├── table_001_page2.json       ← headers, row count, page, element_id
│   │   ├── table_001_page2.md         ← markdown version
│   │   └── ...
│   └── images/
│       ├── image_001_page4.jpg        ← actual extracted image
│       └── image_001_page4.json       ← page, OCR text, element_id
│
└── wla_rpt_pyme_mar2012/              ← scanned PDF
    └── ocr_pages/
        ├── page_001.txt               ← OCR text + moondream vision prose
        ├── page_001.json              ← char counts, element_id
        └── ...
```

### What each file contains

| File | Contents |
|---|---|
| `table_NNN_pageM.csv` | Raw table rows — open in Excel or Numbers |
| `table_NNN_pageM.json` | `headers`, `row_count`, `col_count`, `page_number`, `source_file`, `source` (pdfplumber/vision) |
| `table_NNN_pageM.md` | Markdown formatted table |
| `image_NNN_pageM.jpg` | Extracted image file |
| `image_NNN_pageM.json` | `page_number`, `element_type`, `ocr_text` (first 500 chars) |
| `page_NNN.txt` | `=== OCR (PaddleOCR) ===` section + `=== VISION (moondream) ===` section |
| `page_NNN.json` | `ocr_chars`, `vision_chars`, `page_number` |

### Inspection API endpoints

You can also browse intermediates through the API while the server is running:

```bash
# List all documents with stored intermediates
GET /intermediates

# List all tables for a document (with CSV preview)
GET /intermediates/{doc_id}/tables

# Download a specific table as CSV
GET /intermediates/{doc_id}/tables/table_001_page2.csv

# List all images
GET /intermediates/{doc_id}/images

# View a specific image in browser
GET /intermediates/{doc_id}/images/image_001_page4.jpg

# Full chunks manifest — filter by type
GET /intermediates/{doc_id}/chunks?chunk_type=table
```

> Set `STORE_INTERMEDIATES = False` in `pdf_utils.py` to disable disk writes.

---

## Multi-Tenant Design

All users share a **single Qdrant collection** (`rag_spanish`). Every chunk is tagged with `user_id` in its metadata payload. All queries filter by `user_id` so users never see each other's data.

```
# Ingest for different users
python ingest_docs.py --user_id alice --lang es --file pdfs/doc1.pdf
python ingest_docs.py --user_id bob   --lang en --file pdfs/doc2.pdf

# Delete all data for a specific user
curl -X DELETE http://localhost:8000/user/alice/documents

# Check how many chunks a user has
GET /user/{user_id}/status
```

---

## Models Used

| Type | Model | Notes |
|---|---|---|
| Embeddings | `nomic-ai/nomic-embed-text-v2-moe` | 768-dim MoE, multilingual EN+ES, runs locally |
| LLM | `llama-3.1-8b-instant` (Groq) | Text-only — fast, high free quota |
| OCR | PaddleOCR v3 | Offline, CPU, EN + ES models |
| Vision (ingest only) | moondream (Ollama) | Prose descriptions of images/diagrams. Fully offline. |
| Re-ranker | `cross-encoder/mmarco-mMiniLMv2-L12` | Multilingual, runs locally |
| Translation | googletrans | Free, no API key. Validated + retried automatically. |

> ⚠️ **Important:** moondream is used only during **ingestion** for image/diagram descriptions.
> It is **not** used for structured table extraction — it cannot reliably produce JSON/CSV.
> Tables from scanned pages are saved as OCR text in `ocr_pages/` for inspection.

> ⚠️ **Groq model warning:** Never use vision models on Groq for answers (2,000 token/min limit).
> Use text-only models (`llama-3.1-8b-instant` recommended — ~20,000 tokens/min free).

---

## Config Reference (`config.py`)

| Key | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Your free key from [console.groq.com](https://console.groq.com) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Text-only model |
| `TEMPERATURE` | `0.1` | Low = factual answers |
| `QDRANT_URL` | — | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | — | Qdrant Cloud API key |
| `COLLECTION_SPANISH` | `rag_spanish` | Single collection for all users + languages |
| `EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v2-moe` | 768-dim multilingual |
| `VECTOR_DIM` | `768` | Must match embedding model |
| `VECTOR_TOP_K` | `20` | Vector candidates per sub-query |
| `BM25_TOP_K` | `20` | BM25 candidates per sub-query |
| `FINAL_TOP_K` | `15` | Chunks sent to LLM after re-ranking |
| `EN_CHUNK_SIZE` | `500` | Chunk size in characters |
| `EN_CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TRANSLATE_EN_TO_ES` | `True` | Translate EN queries before retrieval |
| `CACHE_ENABLED` | `True` | Enable two-level cache |
| `CACHE_TTL_SECONDS` | `3600` | Query cache TTL (1 hour) |
| `REDIS_URL` | `None` | Set for distributed cache; falls back to in-process LRU |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_VISION_MODEL` | `moondream` | Local vision model |
| `POPPLER_PATH` | Windows path | Windows only — path to Poppler binaries |
| `STORE_INTERMEDIATES` | `True` | Save tables/images/OCR to disk during ingestion |

---

## API Endpoints

### Core

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server status + cache stats |
| `/ask` | POST | Ask a question — returns answer + full source attribution |
| `/upload` | POST | Upload a PDF — triggers extraction + intermediate storage |
| `/collection` | DELETE | Wipes collection and resets |
| `/` | GET | Browser chat UI |
| `/docs` | GET | Swagger interactive API docs |

### Multi-tenant

| Endpoint | Method | Description |
|---|---|---|
| `/user/{user_id}/status` | GET | Chunk count for a user |
| `/user/{user_id}/documents` | DELETE | Delete all data for a user |

### Intermediates inspection

| Endpoint | Method | Description |
|---|---|---|
| `/intermediates` | GET | List all doc_ids with stored intermediates |
| `/intermediates/{doc_id}/tables` | GET | All tables — metadata + CSV preview |
| `/intermediates/{doc_id}/tables/{filename}` | GET | Raw CSV download |
| `/intermediates/{doc_id}/images` | GET | All images — metadata + download URL |
| `/intermediates/{doc_id}/images/{filename}` | GET | Serve image as `image/png` |
| `/intermediates/{doc_id}/chunks` | GET | Full chunks manifest |
| `/intermediates/{doc_id}/chunks?chunk_type=table` | GET | Filter by chunk type |

### Example: Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the response time for VPN Pura?",
    "user_id": "alice",
    "language": "auto"
  }'
```

```json
{
  "answer": "According to Table 2 on page 4 of manual.pdf, the response time for VPN Pura is 7 to 30 days.",
  "sources": [
    {
      "label":       "manual.pdf · Page 4 · Table 2  [cols: Servicio, Tiempo, Nivel]",
      "chunk_type":  "table",
      "table_label": "Table 2",
      "page_number": 4,
      "stored_csv":  "manual/tables/table_002_page4.csv",
      "headers":     ["Servicio", "Tiempo", "Nivel"],
      "csv_preview": "Servicio,Tiempo,Nivel\nVPN Pura,7-30 días,III\n..."
    }
  ],
  "cached": false,
  "query_used": "¿Cuál es el tiempo de respuesta para VPN Pura?"
}
```

### Example: Ingest a PDF

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@pdfs/manual.pdf" \
  -F "user_id=alice" \
  -F "language=es"
```

```json
{
  "filename":       "manual.pdf",
  "doc_id":         "manual",
  "user_id":        "alice",
  "language":       "es",
  "chunk_counts":   {"text": 52, "table": 4, "image": 2},
  "total_chunks":   58,
  "intermediates":  "intermediates/manual/",
  "inspect_tables": "/intermediates/manual/tables",
  "inspect_images": "/intermediates/manual/images"
}
```

---

## How to Run

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Ollama (vision model for image descriptions)

```bash
# Install Ollama from https://ollama.com
ollama pull moondream
```

### 3. Fill in `config.py`

```python
GROQ_API_KEY   = "your_key_from_console.groq.com"
QDRANT_API_KEY = "your_qdrant_key"
QDRANT_URL     = "https://your-cluster.qdrant.io"

# Windows only:
POPPLER_PATH = r"C:\path\to\poppler\bin"
```

### 4. Check dependencies

```bash
python setup_ocr.py
```

### 5. Ingest your PDFs

```bash
# Ingest a single file
python ingest_docs.py --user_id alice --lang es --file pdfs/doc.pdf

# Auto-detect language
python ingest_docs.py --user_id alice --lang auto --file pdfs/doc.pdf

# Ingest all PDFs in pdfs/ folder
python ingest_docs.py --user_id alice

# Custom intermediates folder
python ingest_docs.py --user_id alice --file pdfs/doc.pdf --intermediates_dir /data/intermediates
```

After ingestion, inspect what was extracted:
```
intermediates/<doc_slug>/tables/    ← CSV files — open in Excel
intermediates/<doc_slug>/images/    ← JPG files — view directly
intermediates/<doc_slug>/ocr_pages/ ← TXT files — OCR + vision text
```

### 6. Start the server

```bash
python server.py
# Chat UI  : http://localhost:8000/
# API docs : http://localhost:8000/docs
# Tables   : http://localhost:8000/intermediates
```

---

## Key Design Decisions

### Why one collection for all users?
Multi-tenant via `user_id` payload filter — scales to thousands of users without managing separate collections. Every chunk is tagged with `user_id`; queries always filter by it.

### Why pdfplumber runs on ALL pages (including scanned)?
pdfplumber detects tables by reading PDF vector line/grid structure — not text content. So it finds native-embedded tables even on pages where PaddleOCR extracted the text. Gating it behind `is_scanned` was wrong and caused tables from scanned PDFs to be silently skipped.

### Why moondream for images but not tables?
moondream is excellent at prose descriptions ("this is a bar chart showing...") which embed well in vector space. But it cannot reliably produce structured output like JSON/CSV — it hallucinates columns and misformats rows. Tables are handled by pdfplumber (native PDFs) or saved as OCR text (scanned PDFs) for inspection.

### Why keep TABLE blocks atomic?
If a table is split mid-chunk, Chunk 1 gets headers with no values and Chunk 2 gets values with no headers. The LLM cannot answer correctly without both. Atomic table chunks fix this.

### Why 4 sub-queries?
Documents don't always use the same words as the question. Sub-query 3 (broad semantic, stripped level qualifiers) finds chunks where the exact phrasing differs. Sub-query 4 looks for accountability/ownership — useful for responsibility matrix tables.

### Why two-level cache?
- **Query cache**: same question + user_id → return cached answer, skip retrieval + LLM (~600ms–1.6s saved per hot query)
- **Embedding cache**: same query text → return cached vector, skip embedding model (~50ms saved)

---

## Limitations

| Issue | Notes |
|---|---|
| Scanned PDF tables | pdfplumber cannot find tables in scanned PDFs (they are flat images). Extracted as OCR text in `ocr_pages/` for inspection only — not as structured CSV. |
| moondream structure | moondream cannot reliably produce JSON/CSV. Used for prose descriptions only. |
| Language support | English and Spanish only. |
| Groq rate limits | Free tier may hit 429 under heavy load. Auto-retries with exponential backoff (4 attempts). |
| BM25 is in-memory | Rebuilt from Qdrant at query time. Large collections (10,000+ chunks) increase first-query latency. |
| googletrans | Occasional failures on unusual queries. Falls back to original query automatically. |

---

## Retrieval Pipeline Summary

```
Question → [Cache check] → [Expand × 4] → [Translate EN→ES] →
  BM25 (top 20 × 4 sub-queries) ──┐
                                   ├→ RRF Merge → BM25 Pin → Cross-Encoder Re-rank (top 15) → LLM → Answer
  Vector (top 20 × 4 sub-queries) ┘                                                              ↓
                                                                                        Cache answer for 1hr
```

---

*Built with LangChain · FastAPI · Qdrant · Groq · PaddleOCR · moondream · nomic-embed-text-v2-moe*