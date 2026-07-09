# Advanced RAG on MongoDB Atlas

Three notebooks that build a movie-discovery assistant one retrieval technique at a time. By the
end you'll have used every major retrieval pattern MongoDB Atlas supports on a single free cluster,
with no separate vector database, search engine, or graph store.

## The notebooks

Run them in order; each builds on indexes and concepts from the previous one.

| # | Notebook | Techniques | Time |
|---|----------|-----------|------|
| 1 | [Setup & Vector Search](./01_atlas_setup_and_vector_search.ipynb) | Atlas connection, embeddings, vector search, metadata filtering, a first RAG chain | ~30 min |
| 2 | [Full-Text, Hybrid & Reranking](./02_fulltext_hybrid_and_reranking.ipynb) | Full-text (BM25) search, hybrid `$rankFusion`, reranking, side-by-side comparison | ~30 min |
| 3 | [GraphLookup & Auto-Embedding](./03_graphlookup_and_auto_embedding.ipynb) | `$graphLookup` traversal, Atlas auto-embedding | ~30 min |

## What you need

- A free Atlas **M0** cluster with the **sample dataset** loaded (see the [root README](../README.md)).
- An **Atlas API key**. The notebooks call MongoDB's hosted embedding and rerank services at
  `ai.mongodb.com`.
- A **Claude API key** for the RAG generation step. An Anthropic-compatible gateway works too; set
  `ANTHROPIC_BASE_URL` (see [`.env.example`](../.env.example)).

Copy `../.env.example` to `../.env` and fill it in, or let the notebooks prompt you for each value.

## How the data is set up

The `sample_mflix.embedded_movies` collection ships with pre-computed plot embeddings
(`plot_embedding_voyage_3_large`, 2048 dimensions from Voyage AI's `voyage-3-large`). The notebooks
use those directly, so there's no bulk embedding step. You create an index and start searching.
Query embeddings use the same model so they share the vector space.

## A note on the free tier

M0 clusters allow up to **3 search indexes**. These notebooks create two on `embedded_movies`
(vector and full-text) and one short-lived index for the auto-embedding demo, which notebook 3
drops when it finishes. If you hit a "maximum number of indexes" error, drop the `auto_embed_demo`
collection or any leftover demo indexes from a previous run.

## Shared helpers

`utils.py` holds the setup boilerplate the notebooks share: reading secrets, connecting to Atlas,
calling the embedding and rerank endpoints, and configuring the LLM. The teaching code stays in the
notebooks. `test_utils.py` covers the helpers with fast, offline unit tests:

```bash
cd advanced-rag
python -m pytest test_utils.py -v
```
