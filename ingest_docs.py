# ================================
# ingest_docs.py  (v2 — multi-tenant)
#
# MULTI-TENANT CHANGE:
#   Added --user_id argument. Every chunk ingested is tagged with this ID
#   so queries from this user only search their own documents.
#
# Usage:
#   python ingest_docs.py --user_id alice --lang es --file pdfs/doc.pdf
#   python ingest_docs.py --user_id bob   --lang en --file pdfs/doc.pdf
#   python ingest_docs.py --user_id alice --lang auto --file pdfs/doc.pdf
#   python ingest_docs.py --user_id alice             # auto-ingest all PDFs in pdfs/
#
# If you don't have real user IDs yet, just pick any string:
#   --user_id alice
#   --user_id bob
#   --user_id test_user
#   --user_id user_123
# ================================

import os
import sys
import argparse
import torch

print(f"GPU Available: {torch.cuda.is_available()}")

from embeddings import load_embedding_model
from pdf_utils import chunk_pdf, detect_language, extract_pdf_pages
from vector_store import get_qdrant_client, create_collection, index_documents
from config import COLLECTION_SPANISH

LANG_LABEL = {"en": "English", "es": "Spanish"}


def ingest_single(pdf_path: str, language: str, user_id: str, embedding_model, client):
    if not os.path.exists(pdf_path):
        print(f"  ❌ File not found: {pdf_path}")
        return False

    if language not in ("en", "es"):
        print(f"  ❌ Unknown language: '{language}'. Use: en | es")
        return False

    filename = os.path.basename(pdf_path)

    print(f"\n{'='*55}")
    print(f"  Ingesting : {filename}")
    print(f"  User ID   : {user_id}")
    print(f"  Language  : {LANG_LABEL[language]}")
    print(f"  Collection: {COLLECTION_SPANISH}  (single collection, all users)")
    print(f"{'='*55}")

    docs, detected_lang, _elements = chunk_pdf(pdf_path, filename, forced_language=language)

    if not docs:
        print(f"  ❌ Could not extract text from '{filename}'.")
        return False

    create_collection(client, COLLECTION_SPANISH)

    # Pass user_id so every chunk is tagged with the owner
    index_documents(docs, embedding_model, user_id=user_id, collection_name=COLLECTION_SPANISH)

    print(f"  ✅ Done — {len(docs)} chunks stored for user='{user_id}' in '{COLLECTION_SPANISH}'")
    return True


def ingest_all(pdf_dir: str, user_id: str, embedding_model, client):
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"  No PDFs found in '{pdf_dir}/'")
        return

    print(f"\n  Found {len(pdf_files)} PDF(s): {pdf_files}")

    for filename in pdf_files:
        pdf_path   = os.path.join(pdf_dir, filename)
        name_lower = filename.lower()

        if any(k in name_lower for k in ["english", "_en", "-en"]):
            language = "en"
        elif any(k in name_lower for k in ["spanish", "espanol", "español", "_es", "-es"]):
            language = "es"
        else:
            pages = extract_pdf_pages(pdf_path)
            if pages:
                lang_votes = [lang for _, _, lang in pages[:3]]
                language   = max(set(lang_votes), key=lang_votes.count)
                print(f"  Auto-detected language for '{filename}': {LANG_LABEL.get(language, language)}")
            else:
                language = "es"

        ingest_single(pdf_path, language, user_id, embedding_model, client)


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into multi-tenant RAG system")
    parser.add_argument("--user_id", type=str, required=True,
                        help="Who owns these documents. Can be any string e.g. alice, bob, user_123")
    parser.add_argument("--lang",    type=str, choices=["en", "es", "auto"])
    parser.add_argument("--file",    type=str)
    args = parser.parse_args()

    user_id = args.user_id.strip()
    if not user_id:
        print("❌ --user_id cannot be empty.")
        sys.exit(1)

    print("\n" + "="*55)
    print(f"LOADING EMBEDDING MODEL  (user='{user_id}')")
    print("="*55)
    embedding_model = load_embedding_model()
    client          = get_qdrant_client()

    if args.file and args.lang and args.lang != "auto":
        ingest_single(args.file, args.lang, user_id, embedding_model, client)

    elif args.file:
        pages = extract_pdf_pages(args.file)
        if pages:
            lang_votes = [lang for _, _, lang in pages[:3]]
            language   = max(set(lang_votes), key=lang_votes.count)
            print(f"Auto-detected language: {LANG_LABEL.get(language, language)}")
            ingest_single(args.file, language, user_id, embedding_model, client)
        else:
            print("Could not read PDF.")

    else:
        print("\n" + "="*55)
        print(f"INGESTING ALL PDFs FROM pdfs/  (user='{user_id}')")
        print("="*55)
        ingest_all("pdfs", user_id, embedding_model, client)

    print("\n" + "="*55)
    print(f"✅ Ingestion complete for user='{user_id}'!")
    print("   Now run: python server.py")
    print("="*55)


if __name__ == "__main__":
    main()