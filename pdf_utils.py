# ================================
# pdf_utils.py  (v2 — identifier-prefixed prose + atomic level-OCR chunks)
# Improved chunking + table-to-prose conversion
# All documents ingest into rag_spanish (single collection).
# EN docs are also stored here — queries are translated at retrieval time.
#
# Changes vs v1:
#
#   FIX 1 — table_to_prose(): identifier-prefixed row format
#     Problem: prose rows were generated as "Header1: Val1. Header2: Val2."
#     for every row — all rows had identical structure, so the LLM could not
#     reliably pinpoint a specific version/level/category when multiple similar
#     rows appeared in the same chunk. It would sometimes read the wrong row.
#     Fix: each prose row is now prefixed with [FirstCol=FirstCellValue], e.g.:
#       "[Version=1.3] Fecha: 01-Mar-12. Descripcion: Diagrama de flujo..."
#     The bracket prefix makes the row key identifier visually unambiguous,
#     so the LLM can directly scan for "[Version=1.3]" and answer from that
#     row only. Generic — the first column is always used as the identifier
#     regardless of domain, column name, or table content.
#
#   FIX 2 — chunk_pdf(): atomic treatment of level-structured OCR blocks
#     Problem: scanned pages containing multi-level responsibility tables
#     (Nivel I / Nivel II / Nivel III rows) were split by
#     RecursiveCharacterTextSplitter. The answer row (e.g. "TIAXA - Nivel III")
#     and the subject context ("aprovisionamiento de companias") ended up in
#     different chunks. Retrieval found the subject chunk but not the answer
#     chunk, so the LLM returned "not found".
#     Fix: _is_level_structured_ocr() detects OCR pages that contain level
#     tables (by counting level-keyword lines and short-line ratio) and marks
#     them as atomic — stored as one chunk, never split. Generic — no domain
#     terms hardcoded; detects structural patterns (level keywords + short
#     line density + compact size) that are universal to responsibility tables.
#     IMPORTANT: re-ingestion required after this change.
# ================================

import re
import uuid
import base64
import io
import csv
import json
import hashlib
import datetime
import numpy as np
import pypdf
import pdfplumber
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import LANGUAGE_CONFIG, POPPLER_PATH, COLLECTION_SPANISH, OLLAMA_BASE_URL, OLLAMA_VISION_MODEL

import requests   # used only for Vision LLM calls

# ════════════════════════════════════════════════════════════
# INTERMEDIATE STORAGE
# Saves tables (from pdfplumber OR vision) and images to disk.
#
#   intermediates/<doc_slug>/
#     tables/
#       table_001_page3.csv    <- native PDF table (pdfplumber)
#       table_002_page7.csv    <- scanned page table (vision LLM)
#       table_001_page3.json   <- headers, row count, page, element_id
#     images/
#       image_001_page2.jpg
#       image_001_page2.json
# ════════════════════════════════════════════════════════════

INTERMEDIATES_ROOT  = "intermediates"
STORE_INTERMEDIATES = True


def _doc_slug(filename: str) -> str:
    base = Path(filename).stem
    return re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")


def _save_intermediates(filename: str, elements: list) -> None:
    if not STORE_INTERMEDIATES or not elements:
        return

    slug     = _doc_slug(filename)
    base_dir = Path(INTERMEDIATES_ROOT) / slug
    t_dir    = base_dir / "tables"
    i_dir    = base_dir / "images"
    t_dir.mkdir(parents=True, exist_ok=True)
    i_dir.mkdir(parents=True, exist_ok=True)

    t_count = 0
    i_count = 0

    for el in elements:
        etype    = el.get("element_type", "")
        page_num = el.get("page_number", 0)

        if etype == "table":
            t_idx     = el.get("table_index", t_count + 1)
            slug_name = f"table_{t_idx:03d}_page{page_num}"
            raw_rows  = el.get("raw_rows", [])

            if raw_rows:
                csv_path = t_dir / f"{slug_name}.csv"
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for row in raw_rows:
                        writer.writerow([str(c or "").strip() for c in row])

            md_text = el.get("table_markdown", "")
            if md_text:
                (t_dir / f"{slug_name}.md").write_text(md_text, encoding="utf-8")

            headers = [str(c or "").strip() for c in raw_rows[0]] if raw_rows else []
            meta = {
                "label":       el.get("label", ""),
                "table_index": t_idx,
                "page_number": page_num,
                "source_file": filename,
                "source":      el.get("source", "pdfplumber"),
                "row_count":   max(0, len(raw_rows) - 1),
                "col_count":   len(headers),
                "headers":     headers,
                "element_id":  el.get("element_id", ""),
            }
            (t_dir / f"{slug_name}.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            t_count += 1
            src = el.get("source", "pdfplumber")
            print(f"    saved Table {t_idx} page {page_num} [{src}] -> {slug_name}.csv  ({max(0,len(raw_rows)-1)} rows)")

        elif etype in ("image", "figure"):
            i_idx     = el.get("image_index", i_count + 1)
            slug_name = f"image_{i_idx:03d}_page{page_num}"
            img_b64   = el.get("image_b64", "")

            if img_b64:
                try:
                    (i_dir / f"{slug_name}.jpg").write_bytes(base64.b64decode(img_b64))
                except Exception as exc:
                    print(f"    Warning: could not save image {i_idx}: {exc}")

            meta = {
                "label":        el.get("label", ""),
                "image_index":  i_idx,
                "page_number":  page_num,
                "source_file":  filename,
                "element_type": etype,
                "ocr_text":     el.get("image_ocr", "")[:500],
                "element_id":   el.get("element_id", ""),
            }
            (i_dir / f"{slug_name}.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            i_count += 1
            print(f"    saved Image {i_idx} page {page_num} -> {slug_name}.jpg")

    # ── Save scanned page OCR/vision text ──────────────────────────────────
    ocr_dir = base_dir / "ocr_pages"
    ocr_count = 0
    for el in elements:
        if el.get("element_type") == "scanned_page":
            page_num  = el.get("page_number", 0)
            slug_name = f"page_{page_num:03d}"
            ocr_text  = el.get("ocr_text", "")
            vision_text = el.get("vision_text", "")
            if ocr_text or vision_text:
                ocr_dir.mkdir(parents=True, exist_ok=True)
                combined = ""
                if ocr_text:
                    combined += "=== OCR (PaddleOCR) ===\n" + ocr_text + "\n"
                if vision_text:
                    combined += "\n=== VISION (moondream) ===\n" + vision_text + "\n"
                (ocr_dir / f"{slug_name}.txt").write_text(combined, encoding="utf-8")
                meta = {
                    "page_number":   page_num,
                    "source_file":   filename,
                    "ocr_chars":     len(ocr_text),
                    "vision_chars":  len(vision_text),
                    "element_id":    el.get("element_id", ""),
                }
                (ocr_dir / f"{slug_name}.json").write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                ocr_count += 1

    print(f"  Intermediates -> intermediates/{slug}/  ({t_count} tables, {i_count} images, {ocr_count} scanned pages)")


# ════════════════════════════════════════════════════════════
# PADDLEOCR
# ════════════════════════════════════════════════════════════

PADDLE_LANG_MAP = {"en": "en", "es": "es"}
CONF_THRESHOLD  = {"en": 0.60, "es": 0.60}
_ocr_models: dict = {}


def get_ocr_model(lang_code: str) -> PaddleOCR:
    if lang_code not in _ocr_models:
        paddle_lang = PADDLE_LANG_MAP.get(lang_code, "en")
        print(f"  Loading PaddleOCR (lang='{paddle_lang}')...")
        _ocr_models[lang_code] = PaddleOCR(
            lang=paddle_lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
        )
        print(f"  PaddleOCR ready (lang={paddle_lang}) ✅")
    return _ocr_models[lang_code]


# ════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    if not text or len(text.strip()) < 20:
        return "es"   # default to Spanish (our collection language)

    total         = len(text.strip())
    spanish_chars = len(re.findall(r'[áéíóúüàèñÁÉÍÓÚÜÀÈÑ¿¡]', text))
    n_tilde       = text.count('ñ') + text.count('Ñ')
    spanish_words = len(re.findall(
        r'\b(el|la|los|las|de|del|en|con|por|para|que|se|es|son|al|lo|le|'
        r'una|uno|más|pero|como|este|esta|sus|su|no|si|ya|fue|era)\b',
        text.lower()
    ))

    char_ratio   = spanish_chars / total
    word_density = spanish_words / max(len(text.split()), 1)

    if n_tilde >= 2 or char_ratio > 0.03 or word_density > 0.10:
        return "es"
    return "en"


# ════════════════════════════════════════════════════════════
# TEXT UTILITIES
# ════════════════════════════════════════════════════════════

def fix_pdf_encoding(text: str) -> str:
    """
    Fixes font-encoding errors common in older corporate PDFs where
    certain characters were mapped to wrong code points at creation time.
    Applied at ingest so stored chunks are clean from the start.
    """
    replacements = [
        ("105 ",   "los "),
        (" 105 ",  " los "),
        ("prop6sito", "propósito"),
        ("prop6",  "propó"),
        ("Ifneas", "líneas"),
        ("I[neas", "líneas"),
        ("Uneas",  "líneas"),
        ("companfas", "compañías"),
        ("companlas", "compañías"),
        ("Marfa",  "María"),
        ("Marla",  "María"),
        ("MENSAJERiA", "MENSAJERÍA"),
        ("MENSAJERfA", "MENSAJERÍA"),
        ("FunciOnalidad", "Funcionalidad"),
        ("activaci6n", "activación"),
        ("Activaci6n", "Activación"),
        ("resoluci6n", "resolución"),
        ("escalaci6n", "escalación"),
        ("configuraci6n", "configuración"),
        ("administraci6n", "administración"),
        ("gesti6n", "gestión"),
        ("secci6n", "sección"),
        ("informaci6n", "información"),
        ("comunicaci6n", "comunicación"),
    ]
    for bad, good in replacements:
        text = text.replace(bad, good)
    return text


def clean_text(text: str) -> str:
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    text = fix_pdf_encoding(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\n\s*){3,}', '\n\n', text)
    return text.strip()


def chunk_hash(text: str) -> str:
    """Stable content hash for deduplication."""
    return hashlib.md5(text.strip().encode()).hexdigest()


HEADING_PATTERNS = [
    r'^#{1,4}\s+(.+)',
    r'^\d+\.\s+[A-ZÁÉÍÓÚÑa-záéíóúñ].{2,}',
    r'^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{4,}$',
]


def extract_section_heading(text: str) -> str:
    for line in text.split('\n')[:5]:
        line = line.strip()
        for pattern in HEADING_PATTERNS:
            if re.match(pattern, line):
                return line[:120]
    return ""


def detect_content_type(has_table, has_image_ocr, has_page_ocr, has_text) -> str:
    flags = sum([has_table, has_image_ocr, has_page_ocr])
    if flags > 1:     return "mixed"
    if has_table:     return "table"
    if has_image_ocr: return "image_summary"
    if has_page_ocr:  return "page_ocr"
    return "text"


# ════════════════════════════════════════════════════════════
# TABLE → NATURAL LANGUAGE CONVERSION
# The key improvement: tables are converted to prose sentences
# so the embedding model can understand them semantically.
# Raw "cell1 | cell2 | cell3" rows embed poorly — LLMs answer
# table questions much better when each row is a sentence.
# ════════════════════════════════════════════════════════════

def table_to_prose(table: list[list]) -> str:
    """
    Converts a pdfplumber table (list of rows, each a list of cells)
    into two formats stored together:
      1. Original pipe-delimited rows (for exact keyword matching via BM25)
      2. Prose sentences (for semantic embedding / vector retrieval)

    v2 prose format: each row is prefixed with its key column value in
    square brackets so the LLM can unambiguously locate a specific row
    when the question asks about a particular identifier (version number,
    level, user type, product name, etc.).

    Example input (Control de Cambios table):
      header: ["Versión", "Fecha", "Descripción del Cambio", "Responsable"]
      row:    ["1.3",     "...",   "Diagrama de flujo...",   "J. García"]

    Old prose (ambiguous — all rows same format, LLM picks wrong one):
      "Versión: 1.3. Fecha: ... Descripción del Cambio: Diagrama de flujo."

    New prose (identifier-prefixed — LLM can pinpoint exact row):
      "[Versión=1.3] Fecha: ... Descripción del Cambio: Diagrama de flujo. Responsable: J. García."

    The bracketed prefix [ColName=Value] uses the FIRST non-empty cell of
    each row as the identifier, which is the natural row key in most tables
    (version numbers, service names, level labels, user categories, etc.).
    This is fully generic — no domain-specific column names are hardcoded.

    Returns combined string wrapped in [TABLE]...[/TABLE] tags.
    """
    if not table:
        return ""

    rows = []
    prose_lines = []
    header_row = None

    # Clean cells
    cleaned_table = []
    for row in table:
        cleaned_row = [str(c).strip() if c else "" for c in row]
        # Skip entirely empty rows
        if any(c for c in cleaned_row):
            cleaned_table.append(cleaned_row)

    if not cleaned_table:
        return ""

    # Detect header: first row where all cells are short and non-numeric
    first_row = cleaned_table[0]
    is_header = all(
        len(c) < 60 and not re.match(r'^\d+[\d.,\s%]*$', c)
        for c in first_row if c
    )
    if is_header and len(cleaned_table) > 1:
        header_row = first_row
        data_rows  = cleaned_table[1:]
    else:
        data_rows = cleaned_table

    # Build pipe rows (for BM25 keyword matching) — unchanged
    for row in cleaned_table:
        rows.append(" | ".join(c if c else "(empty)" for c in row))

    # Build prose sentences (for semantic embedding) — v2: identifier-prefixed rows
    #
    # Each prose line is prefixed with [FirstColumnHeader=FirstCellValue].
    # This lets the LLM immediately locate the row for a specific identifier
    # (e.g. "[Versión=1.3]", "[Nivel=III]", "[Tipo de usuario=VPN Pura]")
    # without having to compare all rows and risking picking the wrong one.
    #
    # Generic rule: the first column is used as the row identifier because
    # tables are almost always keyed on their leftmost column (version, level,
    # name, category). No column names are hardcoded here.
    if header_row and data_rows:
        key_header = header_row[0] if header_row else ""
        for row in data_rows:
            parts = []
            key_value = row[0] if row else ""
            # Build [KeyCol=KeyVal] prefix only when both are non-empty
            if key_header and key_value and key_value != "(empty)":
                prefix = f"[{key_header}={key_value}]"
            else:
                prefix = ""
            # Remaining columns: header: value pairs
            for header, value in zip(header_row, row):
                if header and value and value != "(empty)":
                    parts.append(f"{header}: {value}")
            if parts:
                line = ". ".join(parts) + "."
                prose_lines.append(f"{prefix} {line}".strip() if prefix else line)
    else:
        # No header detected — convert rows to flat statements (unchanged)
        for row in data_rows:
            values = [c for c in row if c and c != "(empty)"]
            if values:
                prose_lines.append(" — ".join(values) + ".")

    pipe_block  = "\n".join(rows)
    prose_block = "\n".join(prose_lines)

    result = f"[TABLE]\n{pipe_block}\n"
    if prose_block:
        result += f"\n[TABLE_PROSE]\n{prose_block}\n[/TABLE_PROSE]\n"
    result += "[/TABLE]"
    return result


# ════════════════════════════════════════════════════════════
# CONTENT HANDLERS
# ════════════════════════════════════════════════════════════

def handle_text_content(raw_text: str) -> str:
    text = clean_text(raw_text)
    if len(text) < 30:
        return ""
    return text


def handle_table_content(plumber_page, table_counter_start: int = 1) -> tuple[list[str], list[dict]]:
    """
    Extracts tables and converts each to prose + pipe format.

    Returns:
        table_strings : list of [TABLE]...[/TABLE] text blocks (unchanged)
        table_raws    : list of dicts with raw table data for element storage:
                          {
                            "table_index"   : int (1-based),
                            "raw_rows"      : list[list[str]],
                            "table_markdown": str,
                            "table_html"    : str,
                          }
    """
    table_strings = []
    table_raws    = []
    try:
        for _local_idx, table in enumerate(plumber_page.extract_tables() or []):
            t_idx = table_counter_start + _local_idx
            if not table:
                continue
            prose = table_to_prose(table)
            if prose and len(prose.strip()) > 10:
                table_strings.append(prose)

                # Build markdown + HTML for element storage
                try:
                    import pandas as pd
                    # Clean cells
                    cleaned = [[str(c).strip() if c else "" for c in row] for row in table if any(c for c in row)]
                    if cleaned:
                        # Use first row as header if it looks like one
                        first = cleaned[0]
                        is_hdr = all(len(c) < 60 and not re.match(r'^\d+[\d.,\s%]*$', c) for c in first if c)
                        if is_hdr and len(cleaned) > 1:
                            df = pd.DataFrame(cleaned[1:], columns=first)
                        else:
                            df = pd.DataFrame(cleaned)
                        table_raws.append({
                            "table_index":    t_idx,
                            "raw_rows":       cleaned,
                            "table_markdown": df.to_markdown(index=False),
                            "table_html":     df.to_html(index=False, border=1),
                        })
                    else:
                        table_raws.append({"table_index": t_idx, "raw_rows": table, "table_markdown": "", "table_html": ""})
                except Exception as e:
                    print(f"    Table serialization error: {e}")
                    table_raws.append({"table_index": t_idx, "raw_rows": table, "table_markdown": "", "table_html": ""})

    except Exception as e:
        print(f"    Table extraction error: {e}")
    return table_strings, table_raws


def _boxes_to_reading_order(rec_texts, rec_scores, rec_boxes, threshold: float) -> str:
    """
    Converts PaddleOCR output into clean readable text for embedding.

    Strategy: column-aware layout reconstruction.

    These PDFs have a 2-column layout. Naive Y-row grouping merges both
    columns into the same "row" producing giant unreadable blobs.

    Fix: detect page midpoint X, split boxes into LEFT and RIGHT columns,
    process each column top-to-bottom independently, then concatenate.
    Single-column pages (wide boxes spanning >60% of page width) are
    handled as one column automatically.

    Each text line becomes its own line in the output — this gives the
    splitter enough newline boundaries to produce proper chunks.
    """
    if not rec_texts:
        return ""

    # Filter by confidence and build (y_center, x_left, x_right, text) tuples
    items = []
    page_x_max = 0
    for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
        if score >= threshold and text.strip():
            try:
                pts     = np.array(box).reshape(-1, 2)
                y_cen   = float(pts[:, 1].mean())
                x_left  = float(pts[:, 0].min())
                x_right = float(pts[:, 0].max())
                page_x_max = max(page_x_max, x_right)
                items.append((y_cen, x_left, x_right, text.strip()))
            except Exception:
                pass

    if not items:
        return ""

    page_mid = page_x_max / 2.0

    # Detect if this is truly a 2-column layout:
    # if >30% of boxes span across the midpoint, treat as single column
    spanning = sum(1 for it in items if it[1] < page_mid * 0.8 and it[2] > page_mid * 1.2)
    two_col  = (spanning / len(items)) < 0.30

    if two_col:
        left_col  = [(it[0], it[1], it[3]) for it in items if it[1] < page_mid]
        right_col = [(it[0], it[1], it[3]) for it in items if it[1] >= page_mid]
        left_col.sort(key=lambda x: x[0])
        right_col.sort(key=lambda x: x[0])
        lines = [t for _, _, t in left_col] + [""] + [t for _, _, t in right_col]
    else:
        # Single column: sort purely by Y
        items.sort(key=lambda x: x[0])
        lines = [it[3] for it in items]

    # Collapse runs of empty lines to single blank line
    result_lines = []
    prev_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        result_lines.append(line)
        prev_blank = is_blank

    return "\n".join(result_lines).strip()


def ocr_with_paddle(pil_image: Image.Image, lang_code: str = "es") -> str:
    try:
        ocr       = get_ocr_model(lang_code)
        img_np    = np.array(pil_image.convert("RGB"))
        result    = ocr.predict(img_np)
        threshold = CONF_THRESHOLD.get(lang_code, 0.60)

        if not result or not result[0]:
            return ""

        page       = result[0]
        rec_texts  = page.get("rec_texts",  [])
        rec_scores = page.get("rec_scores", [])
        rec_boxes  = page.get("rec_boxes",  [])   # bounding box per text line

        # Use spatial reconstruction if boxes are available (always true for PaddleOCR v3)
        if len(rec_boxes) > 0 and len(rec_boxes) == len(rec_texts):
            return _boxes_to_reading_order(rec_texts, rec_scores, rec_boxes, threshold)

        # Fallback: plain join (old behaviour, boxes missing for some reason)
        lines = [t.strip() for t, s in zip(rec_texts, rec_scores)
                 if s >= threshold and t.strip()]
        return clean_text(" ".join(lines))

    except Exception as e:
        print(f"    PaddleOCR error: {e}")
        return ""


# ════════════════════════════════════════════════════════════
# VISION LLM — diagram / scanned-page description
# Uses Ollama local vision model (fully offline, no API key).
# Pull a vision model first: ollama pull llava
# Only two functions added here:
#   _is_diagram_image()         — heuristic to detect diagrams
#   describe_diagram_with_vision() — sends image → Ollama → prose
# Both are called from handle_image_content() and handle_scanned_page().
# Everything else in this file is unchanged from the original.
# ════════════════════════════════════════════════════════════

def _is_diagram_image(pil_img: Image.Image) -> bool:
    """
    Heuristic: returns True if the image looks like a flow diagram
    or process chart rather than a photo or small icon.

    Signals of a diagram:
      - Large enough to contain meaningful content (> 200x100 px)
      - Landscape or square orientation (not tall portrait)
      - Low colour variance — diagrams are mostly white/grey/black lines

    Only images passing this check are sent to the Vision LLM,
    keeping API costs minimal.
    """
    w, h = pil_img.size
    if w < 200 or h < 100:
        return False   # too small — logo or icon

    aspect = w / h
    if aspect < 0.8:
        return False   # tall portrait — more likely a photo or sidebar

    # Low colour variance = monochrome = likely a diagram
    try:
        arr         = np.array(pil_img.convert("RGB"), dtype=float)
        colour_std  = arr.std(axis=(0, 1)).mean()
        if colour_std > 80:
            return False   # high colour variance — likely a photo
    except Exception:
        pass   # if numpy fails, assume it might be a diagram

    return True


def _needs_vision(ocr_text: str) -> bool:
    """
    Smart gate: returns True only when the page likely has tables/diagrams
    that PaddleOCR cannot represent structurally.

    Fires vision when:
      A) OCR produced < 300 chars — page is mostly a diagram/table OCR failed on
      B) > 55% of lines are short (< 35 chars) — table cells / diagram text boxes
      C) Structural keywords present — nivel, hdi, tiaxa, tiempo de respuesta...
      D) Pipe characters — table structure detected

    Skips vision on plain paragraph pages (~60% of pages), conserving quota.
    """
    if not ocr_text or not ocr_text.strip():
        return True   # no text at all → definitely needs vision

    text  = ocr_text.strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # A) Very little text → table/diagram OCR failed
    if len(text) < 300:
        return True

    # B) Many short lines → table cells or diagram boxes
    short = sum(1 for l in lines if len(l) < 35)
    if len(lines) > 4 and (short / len(lines)) > 0.55:
        return True

    # C) Structural keywords
    kws = [
        "nivel", "flujo", "proceso", "diagrama",
        "primer nivel", "segundo nivel", "tercer nivel",
        "help desk", "escalamiento", "aprovisionamiento",
        "hdi", "tiaxa", "tiempo de respuesta",
    ]
    tl = text.lower()
    if any(k in tl for k in kws):
        return True

    # D) Pipe characters → table structure
    if sum(1 for l in lines if "|" in l) >= 2:
        return True

    return False   # plain text page — OCR is sufficient


def describe_diagram_with_vision(
    pil_img: Image.Image,
    lang_code: str = "es"
) -> str:
    """
    Sends a page image to a local Ollama vision model (e.g. llava) and returns
    a structured natural-language description of ALL content visible on it.

    Fully offline — no API key, no quota, no internet required.
    Pull a vision model first: ollama pull llava

    Other options (set OLLAMA_VISION_MODEL in config.py):
      llava:13b    — higher quality, needs ~8GB VRAM or ~16GB RAM
      moondream    — very fast, lower quality
      minicpm-v    — good multilingual support
    """
    try:
        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        if lang_code == "es":
            prompt = (
                "Esta es una página de un documento PDF escaneado en español. "
                "Extrae y describe TODO el contenido visible con máximo detalle:\n"
                "- Si hay TABLAS: lista cada fila con sus encabezados de columna. "
                "  Formato: 'Columna1: Valor1. Columna2: Valor2.' para cada fila.\n"
                "- Si hay DIAGRAMAS DE FLUJO: describe cada nivel, los actores "
                "  involucrados y las conexiones entre ellos.\n"
                "- Si hay TEXTO NORMAL: transcríbelo fielmente.\n"
                "- Si hay LISTAS: enuméralas completas.\n"
                "Sé EXHAUSTIVO — incluye todos los nombres, códigos, tiempos, "
                "niveles y valores numéricos que aparezcan. "
                "Responde SOLO con el contenido extraído, sin introducción."
            )
        else:
            prompt = (
                "This is a scanned PDF page. "
                "Extract and describe ALL visible content in maximum detail:\n"
                "- TABLES: list each row with column headers. "
                "  Format: 'Column1: Value1. Column2: Value2.' per row.\n"
                "- FLOW DIAGRAMS: describe each level, actors and connections.\n"
                "- NORMAL TEXT: transcribe faithfully.\n"
                "- LISTS: enumerate completely.\n"
                "Be EXHAUSTIVE — include all names, codes, times, levels, numbers. "
                "Respond ONLY with extracted content, no introduction."
            )

        payload = {
            "model":  OLLAMA_VISION_MODEL,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 1024},
        }

        url  = f"{OLLAMA_BASE_URL}/api/generate"
        resp = requests.post(url, json=payload, timeout=120)  # vision is slower than text

        if resp.status_code != 200:
            print(f"[vision HTTP {resp.status_code}: {resp.text[:80]}]", end=" ")
            return ""

        description = resp.json().get("response", "").strip()
        if description and len(description) > 30:
            print(f"[vision✓ {len(description)}c]", end=" ")
            return description

    except requests.exceptions.Timeout:
        print(f"[vision timeout — try a smaller model e.g. moondream]", end=" ")
    except Exception as e:
        print(f"[vision error: {str(e)[:80]}]", end=" ")

    return ""


def extract_tables_with_vision(
    pil_img,
    lang_code: str = "es",
    table_counter_start: int = 1,
) -> list[dict]:
    """
    Sends a scanned page image to the Vision LLM and asks it to extract
    any tables as structured JSON rows.

    WHY A SEPARATE CALL:
      describe_diagram_with_vision() uses a prose prompt — good for
      embedding but hard to parse back into CSV rows reliably.
      This function uses a strict JSON-only prompt so we get clean
      structured data we can save directly as CSV.

    Returns:
      List of table dicts in the same format as handle_table_content():
        {
          "table_index":    int,   # global 1-based counter
          "raw_rows":       list[list[str]],
          "table_markdown": str,
          "table_html":     str,
        }
      Empty list if no tables found or vision call fails.
    """
    try:
        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        if lang_code == "es":
            prompt = (
                "Analiza esta imagen de página PDF. "
                "Si contiene UNA O MÁS TABLAS, extrae CADA tabla como JSON. "
                "Responde ÚNICAMENTE con JSON válido, sin texto adicional, "
                "sin bloques de código, sin explicaciones. "
                "Formato exacto:\n"
                '[{"headers": ["Col1","Col2","Col3"], '
                '"rows": [["val1","val2","val3"], ["val4","val5","val6"]]}, ...]\n'
                "Si NO hay tablas, responde exactamente: []"
            )
        else:
            prompt = (
                "Analyze this PDF page image. "
                "If it contains ONE OR MORE TABLES, extract EACH table as JSON. "
                "Respond ONLY with valid JSON, no extra text, "
                "no code blocks, no explanation. "
                "Exact format:\n"
                '[{"headers": ["Col1","Col2","Col3"], '
                '"rows": [["val1","val2","val3"], ["val4","val5","val6"]]}, ...]\n'
                "If there are NO tables, respond exactly: []"
            )

        payload = {
            "model":  OLLAMA_VISION_MODEL,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 2048},
        }

        url  = f"{OLLAMA_BASE_URL}/api/generate"
        resp = requests.post(url, json=payload, timeout=120)
        if resp.status_code != 200:
            return []

        raw = resp.json().get("response", "").strip()

        # Strip markdown code fences if model wrapped in ```json ... ```
        raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
        raw = re.sub(r"```$", "", raw.strip()).strip()

        if not raw or raw == "[]":
            return []

        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return []

        table_raws = []
        for local_idx, tbl in enumerate(parsed):
            if not isinstance(tbl, dict):
                continue
            headers = [str(h).strip() for h in tbl.get("headers", [])]
            rows    = tbl.get("rows", [])
            if not headers and not rows:
                continue

            # Build raw_rows: header row first, then data rows
            cleaned_data = [[str(c).strip() for c in row] for row in rows]
            raw_rows     = [headers] + cleaned_data if headers else cleaned_data

            t_idx = table_counter_start + local_idx

            # Build markdown
            md = ""
            try:
                import pandas as pd
                if headers and cleaned_data:
                    df = pd.DataFrame(cleaned_data, columns=headers)
                    md = df.to_markdown(index=False)
            except Exception:
                pass

            table_raws.append({
                "table_index":    t_idx,
                "raw_rows":       raw_rows,
                "table_markdown": md,
                "table_html":     "",
                "source":         "vision",
            })
            print(f"[vision-table {t_idx}: {len(cleaned_data)} rows × {len(headers)} cols]", end=" ")

        return table_raws

    except json.JSONDecodeError:
        # Model returned non-JSON — not a table page, ignore quietly
        return []
    except Exception as e:
        print(f"[vision-table error: {str(e)[:60]}]", end=" ")
        return []


def handle_image_content(plumber_page, lang_code: str = "es") -> tuple[list[str], list[dict]]:
    """
    Extracts embedded images from a native-text PDF page.

    For each image:
      - If it looks like a flow diagram (_is_diagram_image) AND Ollama is
        available → send to Vision LLM for a prose description.
      - Otherwise → run PaddleOCR to extract visible text.

    Returns:
        image_strings : list of [IMAGE_SUMMARY]...[/IMAGE_SUMMARY] text blocks
        image_raws    : list of dicts with raw image data for element storage:
                          {
                            "image_index": int (1-based),
                            "image_b64"  : str (base64 PNG),
                            "image_ocr"  : str (OCR or vision text),
                            "caption"    : str,
                          }
    """
    image_strings = []
    image_raws    = []
    try:
        for img_idx, img_info in enumerate(plumber_page.images, start=1):
            try:
                page_height = plumber_page.height
                x0 = img_info.get("x0", 0)
                y0 = img_info.get("y0", 0)
                x1 = img_info.get("x1", 0)
                y1 = img_info.get("y1", 0)

                bbox = (x0, page_height - y1, x1, page_height - y0)
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue

                cropped = plumber_page.within_bbox(bbox).to_image(resolution=150)
                pil_img = cropped.original

                if pil_img.width < 30 or pil_img.height < 30:
                    continue

                # Capture base64 for element store (PNG, quality-limited to ~200KB)
                buf = io.BytesIO()
                pil_img.convert("RGB").save(buf, format="JPEG", quality=70)
                img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                ocr_text_for_store = ""

                # ── Vision path: flow diagrams ────────────────
                if _is_diagram_image(pil_img):
                    print(f"[diagram → vision LLM]", end=" ")
                    description = describe_diagram_with_vision(pil_img, lang_code)
                    if description:
                        image_strings.append(
                            f"[IMAGE_SUMMARY]\n[DIAGRAM]\n{description}\n[/DIAGRAM]\n[/IMAGE_SUMMARY]"
                        )
                        ocr_text_for_store = description
                        image_raws.append({
                            "image_index": img_idx,
                            "image_b64":   img_b64,
                            "image_ocr":   ocr_text_for_store,
                            "caption":     "",
                            "element_type": "figure",
                        })
                        continue   # skip PaddleOCR for this image

                # ── PaddleOCR path: all other images ──────────
                ocr_text = ocr_with_paddle(pil_img, lang_code)
                if ocr_text and len(ocr_text) > 10:
                    image_strings.append(f"[IMAGE_SUMMARY]\n{ocr_text}\n[/IMAGE_SUMMARY]")
                    ocr_text_for_store = ocr_text
                    image_raws.append({
                        "image_index":  img_idx,
                        "image_b64":    img_b64,
                        "image_ocr":    ocr_text_for_store,
                        "caption":      "",
                        "element_type": "image",
                    })

            except Exception:
                continue
    except Exception as e:
        print(f"    Image OCR error: {e}")
    return image_strings, image_raws


def handle_scanned_page(pdf_path: str, page_num_1indexed: int,
                        lang_code: str = "es",
                        table_counter_start: int = 1) -> tuple[str, list]:
    """
    Renders a scanned PDF page and extracts its content.

    Strategy (two stages, both always run):

    Stage 1 — PaddleOCR:
      Always runs. Extracts raw character-level text from the page.
      Fast, works offline, good for plain text paragraphs.

    Stage 2 — Vision LLM (Gemini Flash 2.0, if API key available):
      Sends the full page image to the Vision LLM which returns a
      structured prose description of ALL content — including tables,
      nested structures, and flow diagrams that PaddleOCR cannot
      understand structurally.

      The Vision prose is appended to the OCR text inside the [OCR]
      block, wrapped in [VISION]...[/VISION] tags.  This means:
        - BM25 search matches exact OCR keywords (raw text)
        - Vector search matches natural-language questions (vision prose)
      Both retrieval methods benefit from a single page.

    Falls back cleanly if Vision LLM fails — OCR text is used alone.

    DPI: 150 (not 300) to stay within PaddleOCR CPU memory limits.
    """
    try:
        kwargs = {
            "first_page": page_num_1indexed,
            "last_page":  page_num_1indexed,
            "dpi":        150,   # 150 DPI = ~1240x1754px — safe for CPU PaddleOCR
        }
        if POPPLER_PATH:
            kwargs["poppler_path"] = POPPLER_PATH

        images = convert_from_path(pdf_path, **kwargs)
        if not images:
            return ""

        img = images[0]

        # Safety cap: if image is still too large, resize to max 1600px width
        MAX_W = 1600
        if img.width > MAX_W:
            ratio = MAX_W / img.width
            img   = img.resize((MAX_W, int(img.height * ratio)), Image.LANCZOS)

        # ── Stage 1: PaddleOCR ────────────────────────────────
        ocr_text = ocr_with_paddle(img, lang_code)

        # ── Stage 2: Vision LLM (Ollama) — prose + structured tables ──
        # _needs_vision() skips plain-text pages (~60%), keeping
        # vision calls fast and focused on pages that need them.
        vision_prose  = ""
        scanned_table_raws = []
        if _needs_vision(ocr_text):
            print(f"[vision→]", end=" ")
            # 2a. Prose description (for embedding / BM25)
            vision_prose = describe_diagram_with_vision(img, lang_code)
            # 2b. moondream cannot reliably produce structured JSON for tables.
            # Tables from scanned pages are saved via the prose text in ocr_pages/.
            scanned_table_raws = []
        else:
            print(f"[ocr-only]", end=" ")

        # Nothing extracted at all — skip this page
        if not ocr_text and not vision_prose:
            return "", [], None

        # Combine: OCR text first (for BM25), Vision prose second (for vectors)
        parts = []
        if ocr_text:
            parts.append(ocr_text)
        if vision_prose:
            parts.append(f"[VISION]\n{vision_prose}\n[/VISION]")

        combined = "\n\n".join(parts)
        # Return the raw OCR and vision text separately so the call site
        # can save them as intermediate files for inspection
        page_data = {
            "ocr_text":    ocr_text,
            "vision_text": vision_prose,
        }
        return f"[OCR]\n{combined}\n[/OCR]", scanned_table_raws, page_data

    except Exception as e:
        print(f"    Full-page OCR failed (page {page_num_1indexed}): {e}")
    return "", [], None


# ════════════════════════════════════════════════════════════
# CHUNK SEPARATORS
# ════════════════════════════════════════════════════════════

CHUNK_SEPARATORS = {
    "en": ["\n\n", "\n", ". ",  " ", ""],
    "es": ["\n\n", "\n", ". ", "¡", "¿", " ", ""],
}


# ════════════════════════════════════════════════════════════
# MAIN EXTRACTION
# ════════════════════════════════════════════════════════════

def extract_all_content(pdf_path: str, forced_language: str = None) -> tuple:
    pages_data  = []
    reader      = pypdf.PdfReader(pdf_path)
    total_pages = len(reader.pages)

    native_texts = {
        i: clean_text(p.extract_text() or "")
        for i, p in enumerate(reader.pages)
    }

    all_native   = " ".join(native_texts.values())
    doc_language = forced_language if forced_language else detect_language(all_native)
    print(f"  Language: {doc_language}  |  Pages: {total_pages}")

    _doc_table_counter = 1  # global table index across all pages
    with pdfplumber.open(pdf_path) as pdf:
        for idx, plumber_page in enumerate(pdf.pages):
            page_num    = idx + 1
            native_text = native_texts.get(idx, "")

            print(f"  Processing page {page_num}/{total_pages}...", end=" ")

            text_content = handle_text_content(native_text)
            is_scanned   = len(native_text.strip()) < 80

            # Skip image-crop OCR on fully scanned pages — pdfplumber bbox coords
            # are unreliable for full-page image objects; use full-page OCR instead.
            # Always run pdfplumber table detection regardless of is_scanned.
            # pdfplumber reads PDF vector structure — finds tables even on scanned pages.
            table_contents, table_raws = handle_table_content(plumber_page, _doc_table_counter)
            _doc_table_counter += len(table_raws)
            image_contents, image_raws = handle_image_content(plumber_page, doc_language) if not is_scanned else ([], [])

            ocr_content            = ""
            scanned_pages_this_page = []
            if is_scanned:
                print(f"scanned → full-page OCR...", end=" ")
                ocr_content, scanned_table_raws, scanned_page_data = handle_scanned_page(
                    pdf_path, page_num, doc_language,
                    table_counter_start=_doc_table_counter,
                )
                # Merge any tables found by vision into table_raws
                if scanned_table_raws:
                    table_raws.extend(scanned_table_raws)
                    _doc_table_counter += len(scanned_table_raws)
                # Collect OCR+vision text for intermediate saving
                if scanned_page_data and (scanned_page_data.get("ocr_text") or scanned_page_data.get("vision_text")):
                    scanned_page_data.update({
                        "element_type": "scanned_page",
                        "element_id":   str(uuid.uuid4()),
                        "page_number":  page_num,
                    })
                    scanned_pages_this_page.append(scanned_page_data)
                if ocr_content:
                    print(f"{len(ocr_content)} chars", end=" ")

            print("✓")

            parts = []
            if text_content:
                parts.append(text_content)
            parts.extend(table_contents)
            parts.extend(image_contents)
            if ocr_content:
                parts.append(ocr_content)

            combined = "\n\n".join(p for p in parts if p.strip())

            if len(combined.strip()) > 10:
                heading = extract_section_heading(native_text)
                pages_data.append({
                    "page":         page_num,
                    "combined":     combined,
                    "text_content": text_content,
                    "table_count":  len(table_contents),
                    "image_count":  len(plumber_page.images),
                    "has_table":    bool(table_contents),
                    "has_imgocr":   bool(image_contents),
                    "has_pgocr":    bool(ocr_content),
                    "language":     doc_language,
                    "heading":      heading,
                    "table_raws":    table_raws,    # raw table data for element store
                    "image_raws":    image_raws,    # raw image data for element store
                    "scanned_pages": scanned_pages_this_page,  # OCR+vision text for scanned pages
                })

    print(f"  Extracted {len(pages_data)} pages with content")
    return pages_data, doc_language, total_pages


# ════════════════════════════════════════════════════════════
# CHUNKING + DEDUPLICATION + METADATA
# ════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════
# LEVEL-STRUCTURE DETECTOR FOR OCR BLOCKS
#
# Purpose: identify OCR pages that contain responsibility/escalation tables
# so they can be stored as atomic chunks instead of being split.
#
# Problem: scanned pages with multi-level tables look like this after OCR:
#
#   Nivel I   Help Desk          24-48 hrs
#   Nivel II  Soporte Sistemas   48-72 hrs
#   Nivel III TIAXA              5 días
#
# After RecursiveCharacterTextSplitter, the first 2 rows might land in
# chunk A and "Nivel III / TIAXA" in chunk B. If the user asks about
# the Nivel III area, retrieval finds the context chunk (A) but not the
# answer chunk (B), so the LLM returns "not found".
#
# Detection criteria (all purely structural, no domain hardcoding):
#   1. At least 2 lines contain a level-like keyword
#      (Nivel, Level, Tier, Etapa, Fase, Capa — multilingual)
#   2. More than 40% of lines are short (< 50 chars) — typical of table cells
#   3. The block as a whole is under 3000 chars — confirms it is a compact table,
#      not a long narrative that merely mentions levels
#
# Criterion 3 prevents large narrative OCR pages from being treated as atomic
# just because they mention the word "nivel" somewhere.
# ════════════════════════════════════════════════════════════

_LEVEL_KEYWORD_PATTERN = re.compile(
    r'\b(nivel|level|tier|etapa|fase|capa)\s*[IVX\d]',
    flags=re.IGNORECASE,
)


def _is_level_structured_ocr(text: str) -> bool:
    """
    Returns True if the OCR block appears to contain a level/responsibility
    table that would be broken by normal text splitting.

    Purely structural detection — no domain-specific terms hardcoded.
    Works for any language or corporate domain.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return False

    # Criterion 1: at least 2 lines with a level-like keyword + number/roman
    level_lines = sum(1 for l in lines if _LEVEL_KEYWORD_PATTERN.search(l))
    if level_lines < 2:
        return False

    # Criterion 2: majority of lines are short (< 50 chars) — table cell pattern
    short_lines = sum(1 for l in lines if len(l) < 50)
    if len(lines) == 0 or (short_lines / len(lines)) < 0.40:
        return False

    # Criterion 3: block is compact (not a long narrative)
    if len(text) > 3000:
        return False

    return True


def chunk_pdf(pdf_path: str, filename: str,
              forced_language: str = None) -> tuple:
    """
    Returns (list[Document], detected_language, list[element_dicts]).

    The third return value is the intermediate elements list — one entry per
    table or image found during extraction. Each entry carries:
        element_id    : UUID (also stamped on every chunk derived from this element)
        element_type  : "table" | "image" | "figure"
        element_label : human-readable label, e.g. "Table 2 from doc.pdf, page 5"
        table_html    : full HTML (tables only)
        table_markdown: markdown (tables only)
        image_b64     : base64 JPEG (images only)
        image_ocr     : OCR / vision text (images only)

    Every chunk that comes from a table or image gets these fields stamped
    into its metadata:
        element_id, element_type, element_label, table_index / image_index

    This enables:
      - Source attribution: "Answer taken from Table 2 from doc.pdf, page 5"
      - Verification: fetch the raw table HTML via GET /element/{user_id}/{element_id}

    KEY CHUNKING STRATEGY:
    - Table blocks ([TABLE]...[/TABLE]) are stored as ATOMIC documents —
      never split by the text splitter. This keeps every header+data row
      together so the LLM can always match "VPN Pura → 7 a 30 días".
    - Plain text and OCR blocks are split normally with RecursiveCharacterTextSplitter.
    - Each page's content is separated into table vs non-table parts before splitting.
    """
    print(f"\n  Processing: {filename}")
    pages_data, language, total_pages = extract_all_content(pdf_path, forced_language)

    if not pages_data:
        print(f"  No content extracted from '{filename}'")
        return [], "es", []

    cfg        = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["es"])
    separators = CHUNK_SEPARATORS.get(language, CHUNK_SEPARATORS["es"])
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = cfg["chunk_size"],
        chunk_overlap = cfg["chunk_overlap"],
        separators    = separators,
    )

    ingestion_time = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    docs           = []
    elements       = []   # ← intermediate element store entries
    seen_hashes    = set()

    for page in pages_data:
        combined    = page["combined"]
        page_num    = page["page"]

        # ── Build element store entries for tables on this page ──────────────
        # Map table_index → element_id so we can stamp matching chunks below.
        # We do this BEFORE splitting so the UUID is ready when we process chunks.
        table_element_map = {}   # table_index → element_id
        for traw in page.get("table_raws", []):
            el_id = str(uuid.uuid4())
            t_idx = traw["table_index"]
            label = f"Table {t_idx} from {filename}, page {page_num}"
            elements.append({
                "element_id":     el_id,
                "element_type":   "table",
                "pdf_name":       filename,
                "page_number":    page_num,
                "table_index":    t_idx,
                "table_html":     traw.get("table_html", ""),
                "table_markdown": traw.get("table_markdown", ""),
                "raw_rows":       traw.get("raw_rows", []),
                "source":         traw.get("source", "pdfplumber"),
                "image_b64":      "",
                "image_ocr":      "",
                "caption":        "",
                "label":          label,
                "chunk_ids":      [],
            })
            table_element_map[t_idx] = el_id

        # ── Build element store entries for images on this page ──────────────
        image_element_map = {}   # image_index → element_id
        for iraw in page.get("image_raws", []):
            el_id  = str(uuid.uuid4())
            i_idx  = iraw["image_index"]
            etype  = iraw.get("element_type", "image")
            label  = f"{'Figure' if etype == 'figure' else 'Image'} {i_idx} from {filename}, page {page_num}"
            elements.append({
                "element_id":     el_id,
                "element_type":   etype,
                "pdf_name":       filename,
                "page_number":    page_num,
                "image_index":    i_idx,
                "table_index":    0,
                "table_html":     "",
                "table_markdown": "",
                "image_b64":      iraw.get("image_b64", ""),
                "image_ocr":      iraw.get("image_ocr", ""),
                "caption":        iraw.get("caption", ""),
                "label":          label,
                "chunk_ids":      [],
            })
            image_element_map[i_idx] = el_id

        # ── Separate TABLE blocks and OCR blocks ─────────────
        table_blocks = re.findall(r'\[TABLE\].*?\[/TABLE\]', combined, flags=re.DOTALL)
        ocr_blocks   = re.findall(r'\[OCR\].*?\[/OCR\]',   combined, flags=re.DOTALL)

        non_special = re.sub(r'\[TABLE\].*?\[/TABLE\]', '', combined, flags=re.DOTALL)
        non_special = re.sub(r'\[OCR\].*?\[/OCR\]',   '', non_special, flags=re.DOTALL).strip()

        HEADER_PATTERNS = [
            r'P[aá]gina\s+\d+\s+de\s+\d+',
            r'Versi[oó]n:\s*\[?[\d.]+\]?[^\n]*\n',
            r'Fecha de Elaboraci[oó]n[^\n]*\n',
            r'\[\d{1,2}-[A-Za-z]{3}-\d{4}\][^\n]*\n',
        ]
        cleaned_ocr_blocks = []
        for ocr_block in ocr_blocks:
            inner = re.sub(r'\[/?OCR\]', '', ocr_block).strip()
            for pat in HEADER_PATTERNS:
                inner = re.sub(pat, '', inner, flags=re.IGNORECASE)
            inner = re.sub(r'\n{3,}', '\n\n', inner).strip()
            if len(inner) >= 40:
                cleaned_ocr_blocks.append(inner)
        ocr_blocks_to_split = cleaned_ocr_blocks

        atomic_table_chunks = [b.strip() for b in table_blocks if len(b.strip()) >= 40]

        atomic_ocr_chunks = []
        splittable_ocr_blocks = []
        for inner in ocr_blocks_to_split:
            if _is_level_structured_ocr(inner):
                if len(inner) >= 40:
                    atomic_ocr_chunks.append(f"[OCR]\n{inner}\n[/OCR]")
            else:
                splittable_ocr_blocks.append(inner)

        ocr_chunks = []
        for inner in splittable_ocr_blocks:
            for c in splitter.split_text(inner):
                c = c.strip()
                if len(c) >= 40:
                    ocr_chunks.append(f"[OCR]\n{c}\n[/OCR]")

        text_chunks = [c.strip() for c in splitter.split_text(non_special)
                       if len(c.strip()) >= 40] if non_special else []

        raw_chunks  = atomic_table_chunks + atomic_ocr_chunks + ocr_chunks + text_chunks
        chunk_total = len(raw_chunks)

        # Track which table_index we're on as we process table chunks in order.
        # atomic_table_chunks come first in raw_chunks, in the same order as
        # table_raws, so we can match them by position.
        table_chunk_counter = 0

        for chunk_idx, chunk in enumerate(raw_chunks):
            h = chunk_hash(chunk)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            has_tbl  = "[TABLE]"         in chunk
            has_img  = "[IMAGE_SUMMARY]" in chunk
            has_ocr  = "[OCR]"           in chunk
            has_text = bool(re.sub(r'\[.*?\]', '', chunk).strip())

            content_type = detect_content_type(has_tbl, has_img, has_ocr, has_text)

            # ── Stamp element_id + label on table chunks ─────
            el_id    = None
            el_type  = None
            el_label = ""
            t_idx    = None
            i_idx    = None

            if has_tbl and table_element_map:
                # Match by position: atomic_table_chunks are in table_raws order
                t_idx   = table_chunk_counter + 1
                el_id   = table_element_map.get(t_idx)
                el_type = "table"
                el_label = f"Table {t_idx} from {filename}, page {page_num}"
                table_chunk_counter += 1
                # Register chunk with the element entry for back-reference
                for el in elements:
                    if el.get("element_id") == el_id:
                        el["chunk_ids"].append(chunk_hash(chunk))
                        break

            elif has_img and image_element_map:
                # IMAGE_SUMMARY blocks — match first available image on this page
                i_idx   = min(image_element_map.keys()) if image_element_map else 1
                el_id   = image_element_map.get(i_idx)
                el_type = "image"
                el_label = f"Image {i_idx} from {filename}, page {page_num}"
                for el in elements:
                    if el.get("element_id") == el_id:
                        el["chunk_ids"].append(chunk_hash(chunk))
                        break

            docs.append(Document(
                page_content=chunk,
                metadata={
                    "pdf_name":        filename,
                    "page_number":     page_num,
                    "total_pages":     total_pages,
                    "chunk_index":     chunk_idx,
                    "chunk_total":     chunk_total,
                    "content_type":    content_type,
                    "has_table":       has_tbl,
                    "has_image":       page["image_count"] > 0,
                    "has_image_ocr":   has_img,
                    "has_page_ocr":    has_ocr,
                    "table_count":     page["table_count"],
                    "image_count":     page["image_count"],
                    "language":        language,
                    "section_heading": page["heading"],
                    "char_count":      len(chunk),
                    "ingestion_time":  ingestion_time,
                    "collection":      COLLECTION_SPANISH,
                    # ── Element attribution (populated for table/image chunks) ──
                    "element_id":      el_id,       # UUID → fetch via /element endpoint
                    "element_type":    el_type,     # "table" | "image" | None
                    "element_label":   el_label,    # "Table 2 from doc.pdf, page 5"
                    "table_index":     t_idx,       # 1-based table number on the page
                    "image_index":     i_idx,       # 1-based image number on the page
                }
            ))

    # Summary
    w_tbl = sum(1 for p in pages_data if p["has_table"])
    w_img = sum(1 for p in pages_data if p["has_imgocr"])
    w_ocr = sum(1 for p in pages_data if p["has_pgocr"])
    types = {}
    for d in docs:
        ct = d.metadata["content_type"]
        types[ct] = types.get(ct, 0) + 1

    n_table_els = sum(1 for e in elements if e["element_type"] == "table")
    n_image_els = sum(1 for e in elements if e["element_type"] in ("image", "figure"))

    print(f"\n  ✅ {filename}")
    print(f"  ┌──────────────────────────────────────────┐")
    print(f"  │ Pages        : {total_pages:<27}│")
    print(f"  │ Chunks       : {len(docs):<27}│")
    print(f"  │ Elements     : {len(elements)} ({n_table_els} tables, {n_image_els} images){'':>5}│")
    print(f"  │ Language     : {language:<27}│")
    print(f"  │ Collection   : {COLLECTION_SPANISH:<27}│")
    print(f"  │ Tables       : {w_tbl} page(s){'':<20}│")
    print(f"  │ Image OCR    : {w_img} page(s){'':<20}│")
    print(f"  │ Full-pg OCR  : {w_ocr} page(s){'':<20}│")
    for ct, n in types.items():
        print(f"  │   {ct:<14}: {n:<23}│")
    print(f"  └──────────────────────────────────────────┘")

    # Flatten scanned page data from all pages into elements for saving
    for page in pages_data:
        elements.extend(page.get("scanned_pages", []))
    _save_intermediates(filename, elements)

    return docs, language, elements


def extract_pdf_pages(pdf_path: str) -> list:
    """Fast native-text extraction for language detection only."""
    results = []
    try:
        for i, page in enumerate(pypdf.PdfReader(pdf_path).pages):
            text = clean_text(page.extract_text() or "")
            results.append((i + 1, text, detect_language(text)))
    except Exception as e:
        print(f"  extract_pdf_pages error: {e}")
    return results