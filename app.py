"""
Trademarkia AI/ML Engineer Assignment - FastAPI Semantic Cache Service
=======================================================================

Phase 1 produced search_index.pkl with these exact keys:
    - embeddings  : np.array (19997, 384)  - raw document vectors
    - texts       : list[str]  19997        - cleaned document strings
    - centroids   : np.array (10, 384)     - GMM cluster means
    - ivf_index   : dict {0..9: np.array}  - cluster_id -> vectors in that cluster
    - cluster_ids : np.array (19997,)      - dominant cluster per document (int)

Design decisions:
    1. ivf_index is a plain Python dict — NO FAISS. Searching it with np.dot
       is correct and intentional. FAISS would be overkill for 20k docs.
    2. Semantic cache is bucketed by cluster_id (mirrors IVF partitioning).
       Lookup cost is O(cache_size / k) instead of O(cache_size). As cache
       grows this scales far better than a flat list.
    3. SIMILARITY_THRESHOLD = 0.85 is the single tunable knob. Comments
       below explain what different values reveal about system behaviour.
    4. All vectors are L2-normalised at startup once. After that, cosine
       similarity = plain dot product — no repeated norm computation.
    5. No Redis, no caching library. Every data structure is hand-rolled.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# ══════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════
app = FastAPI(title="Semantic Search + Cache API")


# ══════════════════════════════════════════════════════════════
# Load embedding model
# Must be the SAME model used in Phase 1 (all-MiniLM-L6-v2).
# Changing the model would make query vectors incomparable to
# the stored document vectors.
# ══════════════════════════════════════════════════════════════
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")


# ══════════════════════════════════════════════════════════════
# Load Phase 1 index
# ══════════════════════════════════════════════════════════════
print("Loading search index...")
with open("search_index.pkl", "rb") as f:
    data = pickle.load(f)

texts       = data["texts"]        # list[str]         19997 cleaned documents
embeddings  = data["embeddings"]   # np.array (N, 384) raw document vectors
centroids   = data["centroids"]    # np.array (10,384) GMM cluster centres
ivf_index   = data["ivf_index"]    # dict {cid: np.array of vectors}
cluster_ids = data["cluster_ids"]  # np.array (N,)     dominant cluster per doc


# ── Pre-normalise everything once at startup ──────────────────
# After normalisation, cosine_similarity(a, b) == np.dot(a, b)
# This avoids computing norms on every single query.

def _normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Safe against zero vectors."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


centroids_n = _normalise(centroids)   # shape (10, 384)

# Normalise each cluster's vector matrix
ivf_normed: dict = {}
for cid, vecs in ivf_index.items():
    ivf_normed[cid] = _normalise(np.array(vecs, dtype=np.float32))

# Build cluster_id -> [global_doc_index, ...] lookup once at startup.
# Used to map a local best-match position back to its original text.
cluster_to_global: dict = defaultdict(list)
for global_idx, cid in enumerate(cluster_ids):
    cluster_to_global[int(cid)].append(global_idx)

print(f"Index loaded: {len(texts)} documents, {len(ivf_index)} clusters.")


# ══════════════════════════════════════════════════════════════
# Semantic Cache
#
# Structure:
#   semantic_cache = {
#       cluster_id: [
#           { "query": str, "embedding": np.array(384,), "result": str },
#           ...
#       ],
#       ...
#   }
#
# Why bucket by cluster?
#   Flat cache lookup is O(total_entries).
#   Bucketing by cluster reduces it to O(entries_in_cluster) ≈ O(N/k).
#   For k=10 that is a ~10x speedup at no quality cost — a query in
#   cluster 3 will never semantically match a cached result from cluster 7.
#
# SIMILARITY_THRESHOLD — the one tunable decision:
#   0.95+ : only near-exact rewrites hit; behaves almost like no cache
#   0.85  : catches genuine paraphrases without false positives (default)
#   0.70  : more hits but wrong results for loosely related queries
#   0.50  : almost everything hits; cache becomes unreliable
#
#   The key insight: lowering the threshold does NOT improve recall quality
#   — it just makes the cache return stale results more aggressively.
# ══════════════════════════════════════════════════════════════
semantic_cache: dict = defaultdict(list)

SIMILARITY_THRESHOLD = 0.85   # tune this to explore system behaviour

_hit_count  = 0
_miss_count = 0


# ══════════════════════════════════════════════════════════════
# Request schema
# ══════════════════════════════════════════════════════════════
class QueryRequest(BaseModel):
    query: str


# ══════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════

def _embed_query(text: str) -> np.ndarray:
    """
    Embed a query string and L2-normalise the result.
    Returns a (384,) float32 vector.
    Normalising here means downstream dot products equal cosine similarity.
    """
    vec  = model.encode(text, convert_to_numpy=True).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _find_dominant_cluster(query_vec: np.ndarray) -> int:
    """
    Coarse IVF search: dot query against all 10 centroids.
    Returns the cluster_id of the nearest centroid.

    This mirrors the quantiser step in a real IVF index:
    instead of scanning 19,997 vectors we scan 10 centroids, then
    restrict the fine search to ~2,000 vectors in that cluster.
    """
    scores = np.dot(centroids_n, query_vec)   # (10,)
    return int(np.argmax(scores))


def _fine_search(query_vec: np.ndarray, cluster_id: int) -> tuple:
    """
    Fine search inside a single cluster's normalised vector matrix.
    Returns (matched_text: str, similarity_score: float).

    Cost: O(cluster_size) ≈ O(N/k) instead of O(N).
    For k=10 and N=19997 that is ~2000 comparisons instead of 19997.
    """
    cluster_vecs = ivf_normed.get(cluster_id)

    if cluster_vecs is None or len(cluster_vecs) == 0:
        return texts[0], 0.0

    scores     = np.dot(cluster_vecs, query_vec)   # (cluster_size,)
    local_best = int(np.argmax(scores))
    best_score = float(scores[local_best])

    global_idx   = cluster_to_global[cluster_id][local_best]
    matched_text = texts[global_idx]

    return matched_text, best_score


def _check_cache(query_vec: np.ndarray, cluster_id: int) -> tuple:
    """
    Check the semantic cache bucket for this cluster only.

    Steps:
      1. Retrieve entries for this cluster (skips all other clusters).
      2. Dot product between query and each cached embedding.
         Both are normalised so dot product == cosine similarity.
      3. If best score >= SIMILARITY_THRESHOLD → cache hit.

    Returns (hit: bool, entry: dict | None, best_score: float)
    """
    bucket     = semantic_cache.get(cluster_id, [])
    best_entry = None
    best_score = -1.0

    for entry in bucket:
        score = float(np.dot(query_vec, entry["embedding"]))
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_score >= SIMILARITY_THRESHOLD:
        return True, best_entry, best_score

    return False, None, best_score


# ══════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════

@app.post("/query")
def query_endpoint(request: QueryRequest):
    """
    Main search + cache endpoint.

    Pipeline:
      1. Embed query → normalised (384,) vector.
      2. Coarse search → find dominant cluster (10 dot products).
      3. Cache check  → scan only that cluster's cache bucket.
           HIT  → return cached result, skip document search entirely.
           MISS → proceed to fine search.
      4. Fine search  → scan cluster's ~2000 document vectors.
      5. Store result in cache under this cluster's bucket.
      6. Return JSON.

    The cluster structure does real work in steps 2, 3, and 4:
      step 2: reduces centroid scan from N to k  (10 vs 19997)
      step 3: reduces cache scan from total_entries to entries_in_cluster
      step 4: reduces doc scan from N to cluster_size (~2000 vs 19997)
    """
    global _hit_count, _miss_count

    query_text = request.query

    # 1 — embed
    query_vec = _embed_query(query_text)

    # 2 — coarse: find cluster
    cluster_id = _find_dominant_cluster(query_vec)

    # 3 — cache check
    hit, cached_entry, sim_score = _check_cache(query_vec, cluster_id)

    if hit:
        _hit_count += 1
        return {
            "query":            query_text,
            "cache_hit":        True,
            "matched_query":    cached_entry["query"],
            "similarity_score": round(float(sim_score), 6),
            "result":           cached_entry["result"],
            "dominant_cluster": cluster_id,
        }

    # 4 — cache miss → fine search
    _miss_count += 1
    matched_text, search_score = _fine_search(query_vec, cluster_id)

    result_snippet = matched_text[:600].strip()

    # 5 — store in cache (normalised embedding for fast future lookups)
    semantic_cache[cluster_id].append({
        "query":     query_text,
        "embedding": query_vec,
        "result":    result_snippet,
    })

    return {
        "query":            query_text,
        "cache_hit":        False,
        "matched_query":    None,
        "similarity_score": None,
        "result":           result_snippet,
        "dominant_cluster": cluster_id,
    }


@app.get("/cache/stats")
def cache_stats():
    """
    Return current cache state.
    total_entries counts cached queries across all cluster buckets.
    hit_rate = hit_count / (hit_count + miss_count).
    """
    total_entries = sum(len(v) for v in semantic_cache.values())
    total_queries = _hit_count + _miss_count
    hit_rate      = round(_hit_count / total_queries, 6) if total_queries > 0 else 0.0

    return {
        "total_entries": total_entries,
        "hit_count":     _hit_count,
        "miss_count":    _miss_count,
        "hit_rate":      hit_rate,
    }


@app.delete("/cache")
def clear_cache():
    """
    Flush the semantic cache entirely and reset all stats.
    """
    global semantic_cache, _hit_count, _miss_count

    semantic_cache = defaultdict(list)
    _hit_count     = 0
    _miss_count    = 0

    return {"message": "Cache cleared and stats reset."}
