"""
The Unofficial Guide - Milestone 4: Embedding + Vector Store + Retrieval

Pipeline stage (per diagram.png):
    Chunking -> [Embedding: all-MiniLM-L6-v2] -> [Storage: ChromaDB] -> Retrieval

What this does:
  1. Loads the chunks produced by chunk_pipeline.py (documents/chunks.json).
  2. Embeds each chunk's text locally with sentence-transformers all-MiniLM-L6-v2
     (no API key, no rate limits).
  3. Stores the vectors in a persistent ChromaDB collection, with metadata for
     each chunk: source document name + position in that document (plus doc_type,
     section, author) so we can attribute answers later.
  4. Exposes retrieve(query, k) -> top-k chunks with their source + distance.

Distance note: the collection uses cosine distance (hnsw:space="cosine"), so
scores run 0 (identical) -> 1 (unrelated) -> 2 (opposite). Chroma's default is
squared-L2, whose numbers would NOT line up with the "below 0.5" guidance in the
milestone. Lower = more relevant.

Run:  python3 embed_store.py            # build the store, then run eval queries
"""
import os
import json

import chromadb
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH = os.path.join(HERE, "documents", "chunks.json")
CHROMA_DIR = os.path.join(HERE, "chroma_db")
COLLECTION_NAME = "unofficial_guide"
MODEL_NAME = "all-MiniLM-L6-v2"

# A few of the evaluation-plan queries (planning.md) used to sanity-check retrieval.
EVAL_QUERIES = [
    "What resources can alumni use on campus?",
    "Is there an alumni I can connect with for career coaching?",
    "Is there a networking website for alumni?",
    "What membership length periods are available?",
    "Can I use the library late at night after graduation?",
]

# Cache the model so repeated retrieve() calls don't reload it.
_model = None


def get_model():
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME} ...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_collection(reset=True):
    """Embed every chunk and (re)load it into ChromaDB. Returns the collection."""
    chunks = load_chunks()
    model = get_model()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # Start clean so re-runs don't stack duplicate/stale vectors.
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    # normalize_embeddings keeps vectors unit-length, which pairs cleanly with
    # cosine distance and is the standard setup for all-MiniLM-L6-v2.
    embeddings = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True
    ).tolist()

    # Position of each chunk within its own source document (0-based).
    pos_by_source = {}
    ids, metadatas = [], []
    for i, c in enumerate(chunks):
        src = c["source"]
        pos = pos_by_source.get(src, 0)
        pos_by_source[src] = pos + 1
        ids.append(f"chunk_{i}")
        metadatas.append({
            "source": src,
            "chunk_index": pos,          # position in the source document
            "doc_type": c.get("doc_type", ""),
            "section": c.get("section", ""),
            "author": c.get("author", ""),
        })

    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    print(f"Stored {collection.count()} chunks in collection '{COLLECTION_NAME}'.")
    return collection


def get_collection():
    """Open the already-built collection (no re-embedding)."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


def retrieve(query, k=5, collection=None):
    """Return the top-k most relevant chunks for `query`.

    Each result is a dict: {text, source, chunk_index, section, doc_type,
    author, distance}. Lower distance = more relevant (cosine).
    """
    collection = collection or get_collection()
    q_emb = get_model().encode([query], normalize_embeddings=True).tolist()
    res = collection.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append({"text": doc, "distance": dist, **meta})
    return out


def run_eval(queries=EVAL_QUERIES, k=5, collection=None):
    collection = collection or get_collection()
    for q in queries:
        print("\n" + "=" * 78)
        print(f"QUERY: {q}")
        print("=" * 78)
        for rank, r in enumerate(retrieve(q, k=k, collection=collection), 1):
            print(f"\n  #{rank}  distance={r['distance']:.3f}  "
                  f"[{r['doc_type']} | {r['section']} | pos {r['chunk_index']}]")
            print(f"      source: {r['source'][:60]}")
            text = r["text"].replace("\n", " ")
            print(f"      {text[:300]}")


def main():
    collection = build_collection(reset=True)
    print("\nTesting retrieval against evaluation-plan queries (top-5 each):")
    run_eval(k=5, collection=collection)


if __name__ == "__main__":
    main()
