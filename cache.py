# ================================
# cache.py  — Two-level cache for production scale
#
# WHY CACHING MATTERS AT 100M DOCUMENTS:
#
#   At large scale, retrieval + LLM inference is expensive:
#     - Vector search across shards: ~200–500ms
#     - Cross-encoder re-ranking:    ~100–300ms
#     - LLM inference (Groq):        ~300–800ms
#     Total per query:               ~600ms–1.6s
#
#   Real workloads have HOT QUERIES — the same question is asked
#   repeatedly by many users (e.g. "What is the SLA?", "How to reset?").
#   Caching these saves the full retrieval + LLM cost.
#
# TWO LEVELS:
#
#   Level 1 — QUERY CACHE:
#     Key:   (user_id, question_normalized)
#     Value: (answer_str, source_doc_previews)
#     TTL:   1 hour  (answers don't change within an hour)
#     Hit:   Skip retrieval + LLM entirely → ~0ms response
#
#   Level 2 — EMBEDDING CACHE:
#     Key:   query_text (normalized)
#     Value: embedding vector (768-dim list)
#     TTL:   24 hours (embeddings are deterministic — same text = same vector)
#     Hit:   Skip embedding model call → saves ~50ms
#
# STORAGE BACKENDS (in priority order):
#   1. Redis  — distributed, shared across multiple server instances (production)
#   2. In-process LRU dict — single server, lost on restart (prototype/dev)
#
# The code tries Redis first and falls back to in-process silently,
# so your server starts fine even without Redis installed.
# ================================

import time
import json
import hashlib
import unicodedata
import re
from collections import OrderedDict
from typing import Optional, Tuple, List, Any

from config import CACHE_ENABLED, CACHE_TTL_SECONDS, CACHE_MAX_SIZE, REDIS_URL


# ════════════════════════════════════════════════════════════
# KEY NORMALISATION
# Normalise query text before using as cache key so that
# "What is the SLA?" and "what is the sla" hit the same entry.
# ════════════════════════════════════════════════════════════

def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'[^\w\s]', '', text)   # remove punctuation
    text = re.sub(r'\s+', ' ', text)
    return text


def _make_key(prefix: str, *parts: str) -> str:
    """
    Creates a short cache key from prefix + query parts.
    Uses SHA-256 so keys are fixed-length regardless of query size.
    """
    raw  = prefix + "|" + "|".join(_normalise(p) for p in parts)
    return prefix + ":" + hashlib.sha256(raw.encode()).hexdigest()[:32]


# ════════════════════════════════════════════════════════════
# IN-PROCESS LRU CACHE  (fallback when Redis is unavailable)
# A simple OrderedDict-based LRU with TTL.
# NOT shared across multiple server processes/instances.
# Fine for single-server / prototype use.
# ════════════════════════════════════════════════════════════

class _LRUCache:
    """
    Thread-UNSAFE in-process LRU cache with TTL.
    For production multi-process deployments, use Redis instead.
    """

    def __init__(self, max_size: int = CACHE_MAX_SIZE, ttl: int = CACHE_TTL_SECONDS):
        self._store:    OrderedDict = OrderedDict()
        self._max_size: int        = max_size
        self._ttl:      int        = ttl
        self.hits   = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        if key not in self._store:
            self.misses += 1
            return None
        value, expires_at = self._store[key]
        if time.time() > expires_at:
            del self._store[key]
            self.misses += 1
            return None
        # Move to end (most recently used)
        self._store.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: str, value: Any, ttl: int = None):
        ttl        = ttl or self._ttl
        expires_at = time.time() + ttl
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, expires_at)
        # Evict oldest if over capacity
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def delete(self, key: str):
        self._store.pop(key, None)

    def clear(self):
        self._store.clear()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size":      len(self._store),
            "max_size":  self._max_size,
            "hits":      self.hits,
            "misses":    self.misses,
            "hit_rate":  f"{self.hits / total * 100:.1f}%" if total else "0%",
        }


# ════════════════════════════════════════════════════════════
# REDIS BACKEND  (production — shared across server instances)
# ════════════════════════════════════════════════════════════

class _RedisCache:
    """
    Redis-backed cache. Shared across all server processes/instances.
    Falls back to None if Redis is unavailable — caller handles fallback.
    """

    def __init__(self, redis_url: str, ttl: int = CACHE_TTL_SECONDS):
        import redis
        self._client = redis.from_url(redis_url, decode_responses=True)
        self._ttl    = ttl
        self._client.ping()   # will raise if Redis is down
        print(f"  Redis cache connected ✅  ({redis_url})")

    def get(self, key: str) -> Optional[Any]:
        raw = self._client.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def set(self, key: str, value: Any, ttl: int = None):
        self._client.setex(key, ttl or self._ttl, json.dumps(value))

    def delete(self, key: str):
        self._client.delete(key)

    def clear_pattern(self, pattern: str):
        """Delete all keys matching a pattern, e.g. 'query:user_abc*'"""
        for key in self._client.scan_iter(pattern):
            self._client.delete(key)


# ════════════════════════════════════════════════════════════
# CACHE MANAGER  — public interface used by server.py / rag_chain.py
# ════════════════════════════════════════════════════════════

class CacheManager:
    """
    Unified cache interface.
    Tries Redis first; falls back to in-process LRU automatically.
    Your server code never needs to know which backend is active.
    """

    def __init__(self):
        self._query_cache:     Any = None   # Level 1: query → answer
        self._embedding_cache: Any = None   # Level 2: text  → vector
        self._backend:         str = "none"

        if not CACHE_ENABLED:
            print("  Cache disabled (CACHE_ENABLED=False)")
            return

        # Try Redis first
        if REDIS_URL:
            try:
                self._query_cache     = _RedisCache(REDIS_URL, ttl=CACHE_TTL_SECONDS)
                self._embedding_cache = _RedisCache(REDIS_URL, ttl=86400)  # 24h for embeddings
                self._backend         = "redis"
                return
            except Exception as e:
                print(f"  Redis unavailable ({e}) — falling back to in-process LRU cache")

        # Fall back to in-process LRU
        self._query_cache     = _LRUCache(max_size=CACHE_MAX_SIZE,   ttl=CACHE_TTL_SECONDS)
        self._embedding_cache = _LRUCache(max_size=CACHE_MAX_SIZE*2, ttl=86400)
        self._backend         = "lru"
        print(f"  In-process LRU cache active ✅  (max={CACHE_MAX_SIZE} entries, TTL={CACHE_TTL_SECONDS}s)")
        print(f"  NOTE: LRU cache is NOT shared across multiple server processes.")
        print(f"        For production multi-instance deployments, set REDIS_URL in config.py.")

    # ── Level 1: Query Cache ─────────────────────────────────

    def get_answer(self, user_id: str, question: str) -> Optional[Tuple[str, list]]:
        """
        Returns (answer, sources) if this exact question from this user
        was recently answered and is still within TTL. Otherwise None.

        At 100M documents scale, a cache HIT saves:
          ~200–500ms  vector search across shards
          ~100–300ms  cross-encoder re-ranking
          ~300–800ms  LLM inference
        Total savings: ~600ms–1.6s per hot query.
        """
        if self._query_cache is None:
            return None
        key    = _make_key("query", user_id, question)
        cached = self._query_cache.get(key)
        if cached:
            print(f"  Cache HIT (query) — skipping retrieval + LLM ✅")
        return cached

    def set_answer(self, user_id: str, question: str, answer: str, sources: list):
        """Stores an answer in the query cache."""
        if self._query_cache is None:
            return
        key   = _make_key("query", user_id, question)
        value = {"answer": answer, "sources": sources}
        self._query_cache.set(key, value)

    def invalidate_user(self, user_id: str):
        """
        Clears all cached answers for a specific user.
        Call this after a user ingests new documents — their cached
        answers may now be stale since new content was added.
        """
        if self._query_cache is None:
            return
        if self._backend == "redis":
            # Redis supports pattern-based deletion
            pattern = f"query:{hashlib.sha256(('query|' + user_id).encode()).hexdigest()[:8]}*"
            self._query_cache.clear_pattern(pattern)
        else:
            # In-process LRU: scan and delete matching keys
            prefix = _make_key("query", user_id, "")[:20]
            keys_to_delete = [k for k in self._query_cache._store if k.startswith("query:")]
            for k in keys_to_delete:
                del self._query_cache._store[k]
        print(f"  Cache invalidated for user='{user_id}' ✅")

    # ── Level 2: Embedding Cache ─────────────────────────────

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Returns cached embedding vector for this text, or None.
        Embeddings are deterministic — same text always produces same vector —
        so TTL can be very long (24 hours).

        Saves ~50ms per embedding call (significant at high query volume).
        """
        if self._embedding_cache is None:
            return None
        key = _make_key("emb", text)
        return self._embedding_cache.get(key)

    def set_embedding(self, text: str, vector: List[float]):
        """Stores an embedding vector in the cache."""
        if self._embedding_cache is None:
            return
        key = _make_key("emb", text)
        self._embedding_cache.set(key, vector, ttl=86400)

    # ── Stats ────────────────────────────────────────────────

    def stats(self) -> dict:
        result = {"backend": self._backend, "enabled": CACHE_ENABLED}
        if self._backend == "lru" and self._query_cache:
            result["query_cache"]     = self._query_cache.stats()
            result["embedding_cache"] = self._embedding_cache.stats()
        return result


# Singleton — one CacheManager shared across the whole server process
cache = CacheManager()