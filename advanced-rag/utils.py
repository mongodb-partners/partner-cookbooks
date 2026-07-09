"""Shared setup helpers for the Advanced RAG workshop notebooks.

Teaching code (retrievers, comparisons, chains) stays inline in the notebooks.
This module holds only the boilerplate every notebook repeats: reading secrets,
connecting to Atlas, calling the Atlas-hosted embedding and rerank endpoints, and
configuring ChatAnthropic against either the Anthropic API or a compatible gateway.

Nothing here talks to the network at import time, so it is safe to `import utils`
in any cell. The one HTTP dependency (`requests`) is injected as a `poster` seam so
the unit tests in test_utils.py run without a network or API keys.
"""
from __future__ import annotations

import getpass
import os
from typing import Any, Callable, List, Optional
from urllib.parse import urlparse, urlunparse

from langchain_core.embeddings import Embeddings

# Endpoints for MongoDB's Atlas-hosted AI services.
ATLAS_EMBEDDINGS_URL = "https://ai.mongodb.com/v1/embeddings"
ATLAS_RERANK_URL = "https://ai.mongodb.com/v1/rerank"
ANTHROPIC_VERSION = "2023-06-01"

# Telemetry tag so MongoDB can attribute this workload (GenAI-Showcase convention).
APPNAME = "devrel.partner.advanced_rag.python"


class EmbeddingDimensionMismatch(RuntimeError):
    """Raised when the embedding service returns a vector of unexpected length."""


# --------------------------------------------------------------------------
# Secrets: environment first, interactive prompt as a fallback.
# --------------------------------------------------------------------------
def get_secret(name: str, prompt: Optional[str] = None) -> str:
    """Return a secret from the environment, or prompt for it if unset/blank.

    Reading the environment first lets a local `.env` (loaded via python-dotenv)
    drive the notebooks unattended, while the getpass fallback keeps them working
    in Google Colab where no `.env` exists.
    """
    value = os.environ.get(name)
    if value and value.strip():
        return value
    return getpass.getpass(prompt or f"Enter {name}: ")


# --------------------------------------------------------------------------
# LLM configuration: direct Anthropic API vs. an Anthropic-compatible gateway.
# --------------------------------------------------------------------------
def _strip_messages_path(base_url: str) -> str:
    """Drop a trailing /v1/messages from a gateway URL.

    The Anthropic SDK appends `/v1/messages` itself, so `base_url` must be the API
    root. Users often paste the full endpoint from a curl example, so we normalise.
    """
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    for suffix in ("/v1/messages", "/messages"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunparse(parsed._replace(path=path))


def build_llm_kwargs(
    *,
    api_key: str,
    model: str,
    base_url: Optional[str],
    max_tokens: int = 1024,
) -> dict:
    """Build kwargs for ChatAnthropic for either a direct or gateway connection.

    Direct Anthropic API: pass the key through as usual.
    Gateway (e.g. Azure API Management fronting Grove): the SDK's default
    `x-api-key` header is rejected, so we send an `api-key` header instead and
    normalise the base URL to the API root.
    """
    kwargs: dict[str, Any] = {"model": model, "max_tokens": max_tokens}
    if base_url:
        kwargs["base_url"] = _strip_messages_path(base_url)
        # A placeholder keeps the SDK happy; real auth rides in default_headers.
        kwargs["api_key"] = "placeholder"
        kwargs["default_headers"] = {
            "api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
    else:
        kwargs["api_key"] = api_key
    return kwargs


# --------------------------------------------------------------------------
# Embeddings: LangChain Embeddings backed by the Atlas-hosted endpoint.
# --------------------------------------------------------------------------
def _default_poster(url, headers, json, timeout):  # pragma: no cover - thin wrapper
    import requests

    return requests.post(url, headers=headers, json=json, timeout=timeout)


class AtlasEmbeddings(Embeddings):
    """Voyage embeddings via MongoDB's Atlas-hosted embedding endpoint.

    By default one model embeds both documents and queries -- required when querying
    a collection whose stored vectors come from a single model (e.g. sample_mflix's
    `voyage-3-large` plot vectors), since query and document vectors must share a
    space. Pass `query_model` to use an asymmetric pair instead. The returned
    dimension is validated on every call so a misconfiguration fails fast.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "voyage-3-large",
        query_model: Optional[str] = None,
        dimensions: int = 2048,
        url: str = ATLAS_EMBEDDINGS_URL,
        timeout: float = 60.0,
        poster: Callable = _default_poster,
    ):
        self.api_key = api_key
        self.doc_model = model
        self.query_model = query_model or model
        self.dimensions = dimensions
        self.url = url
        self.timeout = timeout
        self._poster = poster

    def _embed(self, texts: List[str], *, model: str, input_type: str) -> List[List[float]]:
        resp = self._poster(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": texts,
                "input_type": input_type,
                "output_dimension": self.dimensions,
            },
            timeout=self.timeout,
        )
        if not resp.ok:
            raise RuntimeError(
                f"Atlas embeddings request failed ({resp.status_code}): {resp.text[:300]}"
            )
        vectors = [item["embedding"] for item in resp.json()["data"]]
        for vec in vectors:
            if len(vec) != self.dimensions:
                raise EmbeddingDimensionMismatch(
                    f"Expected {self.dimensions}-d vectors, got {len(vec)}"
                )
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(list(texts), model=self.doc_model, input_type="document")

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], model=self.query_model, input_type="query")[0]


# --------------------------------------------------------------------------
# Reranking: reorder candidate documents via the Atlas-hosted rerank endpoint.
# --------------------------------------------------------------------------
def atlas_rerank(
    query: str,
    documents: List[str],
    *,
    api_key: str,
    top_n: Optional[int] = None,
    model: str = "rerank-2.5",
    url: str = ATLAS_RERANK_URL,
    timeout: float = 60.0,
    poster: Callable = _default_poster,
) -> List[str]:
    """Return `documents` reordered most-relevant-first, truncated to `top_n`."""
    resp = poster(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "query": query,
            "documents": documents,
            "top_k": top_n or len(documents),
        },
        timeout=timeout,
    )
    if not resp.ok:
        raise RuntimeError(
            f"Atlas rerank request failed ({resp.status_code}): {resp.text[:300]}"
        )
    results = sorted(resp.json()["data"], key=lambda r: r["relevance_score"], reverse=True)
    ordered = [documents[r["index"]] for r in results]
    return ordered[: top_n] if top_n else ordered


# --------------------------------------------------------------------------
# Atlas connection.
# --------------------------------------------------------------------------
def connect(uri: str, *, appname: str = APPNAME):  # pragma: no cover - needs live Atlas
    """Return a MongoClient, verifying the connection with a ping."""
    from pymongo import MongoClient

    client = MongoClient(uri, appname=appname, serverSelectionTimeoutMS=20000)
    client.admin.command("ping")
    return client
