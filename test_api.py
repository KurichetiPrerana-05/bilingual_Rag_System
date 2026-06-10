# ================================
# test_api.py  (English + Spanish RAG)
#
# Test suite for the bilingual RAG system.
# Tests are structured around the 3 types of content
# your Spanish PDFs contain: tables, images, and paragraphs.
#
# Populate the expected_keywords lists with actual values
# from your Spanish PDFs once you know the content.
#
# Usage:
#   python test_api.py              # run all tests
#   python test_api.py --lang en    # English only
#   python test_api.py --lang es    # Spanish only
#   python test_api.py --lang auto  # cross-language auto-detect tests
#   python test_api.py --ingest     # ingest PDFs then run all tests
# ================================

import argparse
import json
import re
import requests
from collections import Counter

BASE_URL = "http://localhost:8000"

# ── Result tracking ───────────────────────────────────────────
results = {"PASS": 0, "WARN": 0, "FAIL": 0, "ERROR": 0}


# ════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ════════════════════════════════════════════════════════════

def sep(c="═", w=70):
    print(c * w)


def hdr(title, emoji="▶"):
    print("\n" + "█" * 70)
    print(f"  {emoji}  {title}")
    print("█" * 70)


def sec(title):
    print(f"\n▶  {title}")
    sep("─")


def print_sources(sources: list):
    """Display source chunks with rich metadata from pdf_utils.py."""
    if not sources:
        return
    print(f"\n  {'─'*66}")
    print(f"  Sources ({len(sources)} chunks retrieved):")
    for s in sources:
        pdf_name     = s.get("pdf_name",        s.get("source",   "unknown"))
        page_num     = s.get("page_number",      s.get("page",     "N/A"))
        total_pages  = s.get("total_pages",      "N/A")
        content_type = s.get("content_type",     "text")
        heading      = s.get("section_heading",  "")
        language     = s.get("language",         "?")
        has_table    = s.get("has_table",         False)
        has_img_ocr  = s.get("has_image_ocr",    False)
        chunk_idx    = s.get("chunk_index",       "?")
        chunk_total  = s.get("chunk_total",       "?")
        ingest_time  = s.get("ingestion_time",    "N/A")
        char_count   = s.get("char_count",        "?")
        preview      = s.get("preview", "")[:180]

        print(f"\n  [{s.get('rank', '')}] {'─'*40}")
        print(f"      📄 File         : {pdf_name}")
        print(f"      📖 Page         : {page_num} / {total_pages}")
        print(f"      🏷  Content type : {content_type}  |  lang={language}")
        print(f"      📊 Has table    : {has_table}  |  Has image OCR: {has_img_ocr}")
        print(f"      🔢 Chunk        : {chunk_idx} of {chunk_total}  |  {char_count} chars")
        if heading:
            print(f"      📌 Section      : {heading}")
        print(f"      🕐 Ingested     : {ingest_time}")
        print(f"      💬 Preview      : …{preview}…")


# ════════════════════════════════════════════════════════════
# QUALITY ASSESSMENT
# ════════════════════════════════════════════════════════════

WARN_PHRASES = [
    "i don't have enough information",
    "cannot be found",
    "not available in",
    "i cannot determine",
    "not explicitly mentioned",
    "not provided in",
    "not mentioned in",
    "i'm unable to find",
    "no information",
    # Spanish equivalents
    "no está disponible",
    "no se encuentra",
    "no tengo información",
    "no se menciona",
    "no puedo determinar",
    "no se proporciona",
]

FAIL_PHRASES = [
    "let me try again",
    "i need to search",
    "voy a intentar de nuevo",
]

MIN_ANSWER_LEN      = 5
LOOP_REPEAT_THRESHOLD = 6
LOOP_WINDOW         = 20


def _has_char_loop(text: str) -> bool:
    if len(text) < LOOP_WINDOW * LOOP_REPEAT_THRESHOLD:
        return False
    for start in range(0, min(len(text) - LOOP_WINDOW, 200)):
        needle = text[start : start + LOOP_WINDOW]
        if text.count(needle) > LOOP_REPEAT_THRESHOLD:
            return True
    return False


def quality_flag(answer: str) -> str:
    if not answer or len(answer.strip()) < MIN_ANSWER_LEN:
        return "FAIL"
    if _has_char_loop(answer):
        return "FAIL"
    lines = [l.strip() for l in answer.splitlines() if l.strip()]
    if lines:
        freq = Counter(lines)
        if freq.most_common(1)[0][1] > 5:
            return "FAIL"
    lower = answer.lower()
    for phrase in FAIL_PHRASES:
        if phrase in lower:
            return "FAIL"
    for phrase in WARN_PHRASES:
        if phrase in lower:
            return "WARN"
    return "PASS"


# ════════════════════════════════════════════════════════════
# CORE TEST FUNCTION
# ════════════════════════════════════════════════════════════

def ask_and_check(
    question: str,
    language: str = "auto",
    expected_keywords: list = None,
    show_sources: bool = False,
    label: str = None,
):
    """
    Post one question to /ask, display the answer, and evaluate.

    Keyword evaluation:
      ALL keywords found  → PASS
      SOME keywords found → WARN
      NO keywords found   → FAIL
      No keywords given   → quality_flag() on answer text
    """
    global results
    display_q = (label or question)[:65]

    try:
        r = requests.post(
            f"{BASE_URL}/ask",
            json={
                "question":     question,
                "language":     language,
                "show_sources": show_sources,
            },
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()

        answer     = data.get("answer", "")
        lang_used  = data.get("language_used", "?")
        collection = data.get("collection_used", "?")
        sources    = data.get("sources", [])

        # Evaluate
        if expected_keywords:
            ans_lower   = answer.lower()
            found       = [kw for kw in expected_keywords if kw.lower() in ans_lower]
            missing     = [kw for kw in expected_keywords if kw.lower() not in ans_lower]
            if len(found) == len(expected_keywords):
                flag = "PASS"
            elif found:
                flag = "WARN"
            else:
                flag = "FAIL"
        else:
            found   = []
            missing = []
            flag    = quality_flag(answer)

        results[flag] += 1

        # Display
        icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌", "ERROR": "💥"}[flag]
        print(f"\n  {icon} [{flag}]  {display_q}")
        print(f"       Lang: {lang_used}  |  Collection: {collection}")
        print(f"       Answer: {answer[:200]}")
        if expected_keywords:
            if found:
                print(f"       ✓ Found   : {found}")
            if missing:
                print(f"       ✗ Missing : {missing}")

        if show_sources:
            print_sources(sources)

    except requests.exceptions.ConnectionError:
        print(f"\n  💥 [ERROR] Cannot connect to {BASE_URL}")
        print(f"       Make sure the server is running: python server.py")
        results["ERROR"] += 1
    except Exception as e:
        print(f"\n  💥 [ERROR] {display_q}")
        print(f"       {e}")
        results["ERROR"] += 1


# ════════════════════════════════════════════════════════════
# UTILITY: health + collections
# ════════════════════════════════════════════════════════════

def health():
    hdr("HEALTH CHECK", "🔍")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        r.raise_for_status()
        data = r.json()
        print(f"\n  Server     : {data.get('server', '?')}")
        print(f"  GPU        : {data.get('gpu', False)}")
        print(f"  Embed model: {data.get('embedding_model', '?')}")
        for lang, info in data.get("collections", {}).items():
            icon = "✅" if info["status"] == "ready" else "⚠️ "
            print(f"  [{lang}] {icon} {info['language']} — {info['vectors']} vectors")
    except Exception as e:
        print(f"  ❌ Health check failed: {e}")


def show_collections():
    sec("Collection Status")
    try:
        r = requests.get(f"{BASE_URL}/collections", timeout=10)
        r.raise_for_status()
        for c in r.json().get("collections", []):
            icon = "✅" if c["status"] == "ready" else "⚠️ "
            print(f"  {icon} [{c['lang_code']}] {c['language']} — "
                  f"{c['collection']} — {c['vector_count']} vectors")
    except Exception as e:
        print(f"  ❌ {e}")


def ingest_all():
    """Ingest sample PDFs from pdfs/ folder."""
    sec("Ingesting PDFs")
    import os
    pdf_dir = "pdfs"
    if not os.path.exists(pdf_dir):
        print(f"  ⚠️  pdfs/ folder not found. Create it and add your PDFs.")
        return
    for filename in os.listdir(pdf_dir):
        if not filename.lower().endswith(".pdf"):
            continue
        filepath = os.path.join(pdf_dir, filename)
        name     = filename.lower()
        if any(k in name for k in ["english", "_en", "-en"]):
            lang = "en"
        elif any(k in name for k in ["spanish", "espanol", "_es", "-es"]):
            lang = "es"
        else:
            lang = "auto"
        try:
            with open(filepath, "rb") as f:
                r = requests.post(
                    f"{BASE_URL}/ingest",
                    files={"file": (filename, f, "application/pdf")},
                    params={"language": lang},
                    timeout=300,
                )
                r.raise_for_status()
                data = r.json()
                print(f"  ✅ {filename} → {data['language']} — "
                      f"{data['chunks_added']} chunks  "
                      f"[tables:{data['ocr_summary']['pages_with_tables']} "
                      f"img_ocr:{data['ocr_summary']['pages_with_image_ocr']} "
                      f"pg_ocr:{data['ocr_summary']['pages_with_page_ocr']}]")
        except Exception as e:
            print(f"  ❌ {filename}: {e}")


# ════════════════════════════════════════════════════════════
# ENGLISH TESTS
# Update expected_keywords with actual values from your PDFs.
# ════════════════════════════════════════════════════════════

def test_english():
    hdr("ENGLISH TESTS", "🔵")

    sec("Text — general facts")
    ask_and_check(
        "What is the main topic of this document?",
        "en",
        show_sources=True
        # No strict keyword — quality_flag() evaluates
    )
    ask_and_check(
        "What are the key findings or conclusions?",
        "en"
    )

    sec("Table — structured data")
    ask_and_check(
        "What values are shown in the table?",
        "en",
        show_sources=True
        # Add specific keywords once you know your PDF content:
        # expected_keywords=["column_header", "specific_value"]
    )
    ask_and_check(
        "Which row has the highest value in the table?",
        "en"
    )

    sec("Image / Chart content")
    ask_and_check(
        "What does Figure 1 show?",
        "en",
        show_sources=True
    )
    ask_and_check(
        "Describe the chart or diagram in the document.",
        "en"
    )

    sec("Paragraph / Summary")
    ask_and_check(
        "Summarize the introduction section.",
        "en"
    )
    ask_and_check(
        "What recommendations are made in this document?",
        "en"
    )


# ════════════════════════════════════════════════════════════
# SPANISH TESTS
# Questions are in Spanish → answers must be in Spanish.
# Update expected_keywords with actual values from your PDFs.
# ════════════════════════════════════════════════════════════

def test_spanish():
    hdr("SPANISH TESTS — Preguntas en español", "🟠")

    sec("Texto — información general")
    ask_and_check(
        "¿Cuál es el tema principal de este documento?",
        "es",
        show_sources=True
        # Add keywords once you know your PDF content:
        # expected_keywords=["palabra_clave", "tema"]
    )
    ask_and_check(
        "¿Cuáles son las conclusiones principales?",
        "es"
    )

    sec("Tabla — datos estructurados")
    ask_and_check(
        "¿Qué valores se muestran en la tabla?",
        "es",
        show_sources=True
        # Add specific keywords from your table:
        # expected_keywords=["encabezado", "valor_específico"]
    )
    ask_and_check(
        "¿Qué fila tiene el valor más alto en la tabla?",
        "es"
    )
    ask_and_check(
        "¿Cuál es el total o la suma de los valores de la tabla?",
        "es"
    )

    sec("Imagen / Gráfico — contenido visual")
    ask_and_check(
        "¿Qué muestra la Figura 1?",
        "es",
        show_sources=True
    )
    ask_and_check(
        "Describe el gráfico o diagrama del documento.",
        "es"
    )

    sec("Párrafo — resumen y análisis")
    ask_and_check(
        "Resume la sección de introducción.",
        "es"
    )
    ask_and_check(
        "¿Qué recomendaciones se hacen en este documento?",
        "es"
    )
    ask_and_check(
        "¿Cuáles son los objetivos descritos en el documento?",
        "es"
    )


# ════════════════════════════════════════════════════════════
# CROSS-LANGUAGE AUTO-DETECT TESTS
# Both languages in one session — auto-detect must route correctly.
# ════════════════════════════════════════════════════════════

def test_cross():
    hdr("CROSS-LANGUAGE AUTO-DETECT TESTS", "🔄")

    # English question → must answer in English
    ask_and_check(
        "What is this document about?",
        "auto"
    )

    # Spanish question → must answer in Spanish
    ask_and_check(
        "¿De qué trata este documento?",
        "auto"
    )

    # Spanish with ñ → must detect Spanish
    ask_and_check(
        "¿Cuántos años tiene el informe?",
        "auto"
    )

    # English specific fact
    ask_and_check(
        "What is the total value mentioned?",
        "auto"
    )

    # Spanish with accented chars
    ask_and_check(
        "¿Cuál es el porcentaje más alto registrado?",
        "auto"
    )


# ════════════════════════════════════════════════════════════
# LANGUAGE ISOLATION TESTS
# Verify that English questions never return Spanish answers
# and vice versa, regardless of which PDF was ingested.
# ════════════════════════════════════════════════════════════

def test_language_isolation():
    hdr("LANGUAGE ISOLATION TESTS", "🛡️")

    sec("English questions must return English answers")

    def check_english_answer(question):
        """Verify answer does not contain Spanish-only chars."""
        try:
            r = requests.post(
                f"{BASE_URL}/ask",
                json={"question": question, "language": "en"},
                timeout=60
            )
            r.raise_for_status()
            answer    = r.json().get("answer", "")
            has_ñ     = 'ñ' in answer or 'Ñ' in answer
            has_inverted = '¿' in answer or '¡' in answer
            flag = "FAIL" if (has_ñ or has_inverted) else quality_flag(answer)
            results[flag] += 1
            icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}.get(flag, "?")
            print(f"\n  {icon} [{flag}]  EN→EN: {question[:50]}")
            print(f"       Answer: {answer[:150]}")
            if has_ñ or has_inverted:
                print(f"       ❌ Answer contains Spanish-only characters!")
        except Exception as e:
            print(f"\n  💥 [ERROR] {e}")
            results["ERROR"] += 1

    check_english_answer("What is the main subject of this document?")
    check_english_answer("How many items are listed in the table?")

    sec("Spanish questions must return Spanish answers")

    def check_spanish_answer(question):
        """Verify answer contains Spanish characteristics."""
        try:
            r = requests.post(
                f"{BASE_URL}/ask",
                json={"question": question, "language": "es"},
                timeout=60
            )
            r.raise_for_status()
            answer      = r.json().get("answer", "")
            answer_lower = answer.lower()
            # Spanish answer should contain at least one Spanish function word
            spanish_signal = bool(re.search(
                r'\b(el|la|los|las|de|en|es|son|fue|una|uno|más|pero|para|que|'
                r'con|por|no|si|ya|al|se|le|su|sus|este|esta|del)\b',
                answer_lower
            ))
            flag = quality_flag(answer)
            if flag == "PASS" and not spanish_signal:
                flag = "WARN"  # Answer present but no Spanish words detected
            results[flag] += 1
            icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}.get(flag, "?")
            print(f"\n  {icon} [{flag}]  ES→ES: {question[:50]}")
            print(f"       Answer: {answer[:150]}")
            if not spanish_signal and answer:
                print(f"       ⚠️  No Spanish words detected in answer — check LLM response")
        except Exception as e:
            print(f"\n  💥 [ERROR] {e}")
            results["ERROR"] += 1

    check_spanish_answer("¿Cuál es el tema principal?")
    check_spanish_answer("¿Qué datos muestra la tabla?")


# ════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ════════════════════════════════════════════════════════════

def print_summary():
    total  = sum(results.values())
    passed = results["PASS"]
    warned = results["WARN"]
    failed = results["FAIL"]
    errors = results["ERROR"]
    pct    = int((passed / total) * 100) if total else 0

    bar_len   = 50
    p_filled  = int(bar_len * passed / total) if total else 0
    w_filled  = int(bar_len * warned / total) if total else 0
    remainder = bar_len - p_filled - w_filled
    bar       = "█" * p_filled + "▒" * w_filled + "░" * remainder

    print("\n" + "═" * 70)
    print("  TEST SUMMARY")
    print("─" * 70)
    print(
        f"  PASS : {passed:<5}  "
        f"WARN : {warned:<5}  "
        f"FAIL : {failed:<5}  "
        f"ERROR: {errors:<5}  "
        f"TOTAL: {total}"
    )
    print(f"\n  [{bar}]  {pct}% passing")
    print()

    if warned > 0:
        print(
            f"  ⚠  {warned} WARN = partial keyword match or answer quality issue.\n"
            f"     Check those answers manually.\n"
            f"     Common cause: LLM rephrases numbers or uses synonyms."
        )
    if failed > 0:
        print(
            f"  ❌  {failed} FAIL = no keywords found or answer quality too low.\n"
            f"     Likely causes:\n"
            f"     1. PDF not ingested yet — run ingest_docs.py first\n"
            f"     2. Wrong collection — check language detection\n"
            f"     3. Retrieval miss — raise TOP_K in config.py\n"
            f"     4. Split table — raise CHUNK_OVERLAP in config.py\n"
            f"     5. LLM hallucination — check max_tokens in rag_chain.py"
        )
    print("═" * 70 + "\n")


# ════════════════════════════════════════════════════════════
# RUN ALL
# ════════════════════════════════════════════════════════════

def run_all(do_ingest: bool = False):
    print("\n" + "═" * 70)
    print("  🌐  BILINGUAL RAG — FULL TEST SUITE  (English + Spanish)")
    print("═" * 70)

    health()
    show_collections()

    if do_ingest:
        ingest_all()

    test_english()
    test_spanish()
    test_cross()
    test_language_isolation()

    print_summary()


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bilingual RAG API — Test Suite (English + Spanish)"
    )
    parser.add_argument(
        "--lang",
        choices=["en", "es", "auto", "isolation", "all"],
        default="all",
        help="Tests to run (default: all)",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest PDFs before running tests",
    )
    args = parser.parse_args()

    if args.lang == "all":
        run_all(do_ingest=args.ingest)
    else:
        health()
        show_collections()
        if args.ingest:
            ingest_all()
        {
            "en":        test_english,
            "es":        test_spanish,
            "auto":      test_cross,
            "isolation": test_language_isolation,
        }[args.lang]()
        print_summary()