# ================================
# setup_ocr.py  (English + Spanish)
# Run this BEFORE ingest_docs.py
# ================================

import os
import numpy as np
from PIL import Image, ImageDraw

try:
    from config import POPPLER_PATH
except ImportError:
    POPPLER_PATH = None

print("\n" + "="*62)
print("  OCR + PIPELINE DEPENDENCY CHECK  (English + Spanish)")
print("="*62)
print(f"  POPPLER_PATH : {POPPLER_PATH or '(not set)'}")
print("="*62)


def check(label, fn):
    try:
        result = fn()
        print(f"  [OK]   {label}")
        if result:
            for line in str(result).splitlines():
                print(f"         {line}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}")
        for line in str(e).splitlines()[:5]:
            print(f"         {line}")
        return False


all_ok = True


def check_pypdf():
    import pypdf
    return f"pypdf v{pypdf.__version__}"
all_ok &= check("pypdf  (Layer 1 — native text)", check_pypdf)


def check_pdfplumber():
    import pdfplumber
    return f"pdfplumber v{pdfplumber.__version__}"
all_ok &= check("pdfplumber  (Layer 2/3 — tables + embedded images)", check_pdfplumber)


def check_pdf2image():
    from pdf2image import convert_from_bytes
    blank = (
        b'%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj '
        b'2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj '
        b'3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\n'
        b'xref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n'
        b'0000000058 00000 n\n0000000115 00000 n\n'
        b'trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF'
    )
    kwargs = {"dpi": 300}
    if POPPLER_PATH:
        kwargs["poppler_path"] = POPPLER_PATH
    imgs = convert_from_bytes(blank, **kwargs)
    return f"Rendered {len(imgs)} page(s) at 300 DPI"
all_ok &= check("pdf2image + Poppler  (Layer 4 — full-page 300 DPI)", check_pdf2image)


def check_pillow():
    import PIL
    img = Image.new("RGB", (200, 50), color="white")
    return f"Pillow v{PIL.__version__}"
all_ok &= check("Pillow", check_pillow)


def check_numpy():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    return f"numpy v{np.__version__}  shape={arr.shape}"
all_ok &= check("numpy", check_numpy)


def check_paddle_en():
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="en", use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False, device="cpu")
    img = Image.new("RGB", (300, 50), color="white")
    ImageDraw.Draw(img).text((10, 15), "OCR Test 123", fill="black")
    result = ocr.predict(np.array(img.convert("RGB")))
    texts = []
    if result and result[0]:
        page = result[0]
        texts = [t for t, s in zip(page.get("rec_texts",[]), page.get("rec_scores",[]))
                 if s >= 0.60 and t.strip()]
    if not texts:
        raise Exception("PaddleOCR returned empty text.")
    return f"lang='en' → '{' '.join(texts)}'"
all_ok &= check("PaddleOCR v3  lang='en'  (English)", check_paddle_en)


def check_paddle_es():
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="es", use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False, device="cpu")
    img = Image.new("RGB", (400, 50), color="white")
    ImageDraw.Draw(img).text((10, 15), "español niño café", fill="black")
    result = ocr.predict(np.array(img.convert("RGB")))
    texts = []
    if result and result[0]:
        page = result[0]
        texts = [t for t, s in zip(page.get("rec_texts",[]), page.get("rec_scores",[]))
                 if s >= 0.60 and t.strip()]
    return f"lang='es' → Spanish model loaded ✅  Recognised: '{' '.join(texts) if texts else '(ok)'}'"
all_ok &= check("PaddleOCR v3  lang='es'  (Spanish)", check_paddle_es)


def check_lang_detect():
    from pdf_utils import detect_language
    tests = [
        ("Hello, this is an English document about technology.", "en"),
        ("El informe presenta los resultados del año fiscal.", "es"),
        ("¿Cuántos pacientes fueron atendidos en el hospital?", "es"),
        ("The quarterly revenue grew by 18% year over year.", "en"),
    ]
    results = []
    for text, expected in tests:
        detected = detect_language(text)
        status   = "✅" if detected == expected else "❌"
        results.append(f"{status} '{text[:40]}...' → {detected}")
    return "\n".join(results)
all_ok &= check("Language detection", check_lang_detect)


def check_bm25():
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    docs = [Document(page_content="El tiempo de respuesta es 7 a 30 días para VPN Pura.")]
    r = BM25Retriever.from_documents(docs, k=1)
    results = r.invoke("tiempo respuesta")
    if not results:
        raise Exception("BM25 returned no results")
    return f"BM25 OK — returned {len(results)} doc(s)"
all_ok &= check("BM25Retriever (hybrid keyword search)", check_bm25)


def check_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        score = ce.predict([("query", "document text")])
        return f"Cross-encoder OK — score={score[0]:.4f}"
    except ImportError:
        return "(optional) sentence-transformers not installed — pip install sentence-transformers"
all_ok &= check("Cross-encoder re-ranker (optional)", check_cross_encoder)


def check_config():
    from config import LANGUAGE_CONFIG, COLLECTION_SPANISH, GROQ_MODEL
    return (
        f"Language keys : {list(LANGUAGE_CONFIG.keys())}\n"
        f"Collection    : {COLLECTION_SPANISH}  (single, all languages)\n"
        f"Groq model    : {GROQ_MODEL}\n"
        f"POPPLER_PATH  : {POPPLER_PATH or '(None)'}"
    )
all_ok &= check("config.py", check_config)


print("\n" + "="*62)
if all_ok:
    print("  ✅  All checks passed — ready to ingest!")
    print()
    print("  Architecture:")
    print("  ┌──────────────────────────────────────────────────────────┐")
    print("  │ Collection   │ rag_spanish (ALL languages)               │")
    print("  │ EN queries   │ translated → ES before embedding          │")
    print("  │ Retrieval    │ BM25 + Vector → RRF merge → Re-rank       │")
    print("  │ LLM          │ llama-3.3-70b-versatile (Groq)            │")
    print("  │ Answer lang  │ matches user's question language          │")
    print("  └──────────────────────────────────────────────────────────┘")
    print()
    print("  Run next:")
    print("    python ingest_docs.py --lang es --file pdfs/your_doc.pdf")
    print("    python ingest_docs.py --lang en --file pdfs/english_doc.pdf")
    print("    python server.py")
else:
    print("  ❌  Fix the [FAIL] items above, then run setup_ocr.py again.")
    print()
    print("  Most common fix:")
    print("    pip install sentence-transformers rank-bm25 langchain-community")
print("="*62 + "\n")