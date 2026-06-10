# ================================
# embeddings.py
# Loads Nomic embedding model
# and patches simsimd bug
# ================================

import os
import logging
import numpy as np
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

from langchain_huggingface import HuggingFaceEmbeddings
import langchain_community.utils.math as lc_math
from config import EMBEDDING_MODEL, VECTOR_DIM


# ── Patch simsimd bug ─────────────────────────────────────────
def _patched_cosine_similarity(X, Y):
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
    return np.dot(X_norm, Y_norm.T)

lc_math.cosine_similarity = _patched_cosine_similarity
print("simsimd patch applied ✅")


# ── Load model ────────────────────────────────────────────────
def load_embedding_model() -> HuggingFaceEmbeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on {device}...")
    model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        # nomic-embed-text-v2-moe is a multilingual MoE model that
        # natively supports English and Spanish with the same 768-dim
        # embedding space — no changes needed for Spanish support.
        model_kwargs={"trust_remote_code": True, "device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32}
    )
    print(f"Embedding model ready ✅  ({EMBEDDING_MODEL}, dim={VECTOR_DIM})")
    return model