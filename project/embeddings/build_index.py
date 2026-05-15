"""
FAISS index builder for SHL catalog embeddings.

Run this script once before starting the server:
    python embeddings/build_index.py

It will:
  1. Load all assessments from data/catalog.json
  2. Build a rich text representation for each
  3. Embed them using sentence-transformers (all-MiniLM-L6-v2)
  4. Save the FAISS index to embeddings/faiss_index/index.faiss
  5. Save metadata (entity_id mapping) to embeddings/faiss_index/meta.json

The index uses flat L2 search for correctness (no approximation errors).
For a catalog of ~500 items, this is fast enough (< 1ms per query).
"""

import json
import sys
from pathlib import Path

# Ensure project root is in path when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import (
    CATALOG_PATH,
    FAISS_INDEX_DIR,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    build_catalog_text,
    get_logger,
    load_catalog,
)

_log = get_logger("build_index")


def build_index() -> None:
    """Build and save the FAISS index for the SHL catalog."""

    # Check dependencies
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError as e:
        _log.error(
            "Missing dependency: %s\n"
            "Install with: pip install faiss-cpu sentence-transformers",
            e,
        )
        sys.exit(1)

    # Load catalog
    _log.info("Loading catalog from %s", CATALOG_PATH)
    catalog = load_catalog()
    _log.info("Loaded %d assessments.", len(catalog))

    # Build text representations
    _log.info("Building text representations...")
    texts = []
    meta = []
    for item in catalog:
        text = build_catalog_text(item)
        texts.append(text)
        meta.append({"entity_id": item["entity_id"], "name": item["name"]})

    # Embed
    _log.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    _log.info("Encoding %d assessments...", len(texts))
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )
    embeddings = embeddings.astype("float32")
    _log.info("Embeddings shape: %s", embeddings.shape)

    # Build FAISS index (flat L2 — exact search, best for small catalogs)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    _log.info("Added %d vectors to FAISS index (dim=%d).", index.ntotal, dim)

    # Save index and metadata
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    _log.info("Saved FAISS index to %s", FAISS_INDEX_PATH)

    with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    _log.info("Saved metadata to %s", FAISS_META_PATH)

    _log.info("Index build complete. %d vectors indexed.", index.ntotal)


if __name__ == "__main__":
    build_index()
