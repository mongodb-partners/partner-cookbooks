"""Unit tests for advanced-rag/utils.py.

Run from the advanced-rag/ directory:  python -m pytest test_utils.py -v
These use in-memory fakes only -- no live Atlas, no network, no keys required.
"""
import pytest

import utils


# --------------------------------------------------------------------------
# get_secret: environment-first, getpass fallback (so Colab works)
# --------------------------------------------------------------------------
def test_get_secret_reads_from_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "from-env")
    assert utils.get_secret("MY_KEY") == "from-env"


def test_get_secret_prompts_when_missing(monkeypatch):
    monkeypatch.delenv("MY_KEY", raising=False)
    monkeypatch.setattr(utils.getpass, "getpass", lambda prompt="": "typed-in")
    assert utils.get_secret("MY_KEY") == "typed-in"


def test_get_secret_treats_blank_env_as_missing(monkeypatch):
    monkeypatch.setenv("MY_KEY", "   ")
    monkeypatch.setattr(utils.getpass, "getpass", lambda prompt="": "typed-in")
    assert utils.get_secret("MY_KEY") == "typed-in"


# --------------------------------------------------------------------------
# build_llm_kwargs: gateway vs direct Anthropic auth
# --------------------------------------------------------------------------
def test_build_llm_kwargs_direct_anthropic():
    kw = utils.build_llm_kwargs(api_key="sk-ant-xxx", model="claude-haiku-4-5",
                                base_url=None, max_tokens=256)
    assert kw["model"] == "claude-haiku-4-5"
    assert kw["api_key"] == "sk-ant-xxx"
    assert kw["max_tokens"] == 256
    assert "base_url" not in kw
    assert "default_headers" not in kw


def test_build_llm_kwargs_gateway_strips_messages_path_and_sets_header():
    kw = utils.build_llm_kwargs(
        api_key="gw-key",
        model="claude-haiku-4-5",
        base_url="https://gw.example.net/foundry/anthropic/v1/messages",
        max_tokens=128,
    )
    # SDK appends /v1/messages itself, so the trailing path must be stripped.
    assert kw["base_url"] == "https://gw.example.net/foundry/anthropic"
    # Gateway authenticates via the "api-key" header, not x-api-key.
    assert kw["default_headers"]["api-key"] == "gw-key"
    assert kw["default_headers"]["anthropic-version"]


def test_build_llm_kwargs_gateway_root_url_untouched():
    kw = utils.build_llm_kwargs(api_key="gw-key", model="m",
                                base_url="https://gw.example.net/anthropic", max_tokens=1)
    assert kw["base_url"] == "https://gw.example.net/anthropic"


# --------------------------------------------------------------------------
# AtlasEmbeddings: LangChain Embeddings over ai.mongodb.com, with dim guard
# --------------------------------------------------------------------------
class FakePoster:
    """Stand-in for the HTTP POST seam. Records calls, returns canned vectors."""

    def __init__(self, dims=1024, status=200):
        self.dims = dims
        self.status = status
        self.calls = []

    def __call__(self, url, headers, json, timeout):
        self.calls.append({"url": url, "headers": headers, "json": json})
        n = len(json["input"])
        vecs = [[0.1] * self.dims for _ in range(n)]
        return _FakeResp(self.status, {"data": [{"embedding": v} for v in vecs]})


class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.ok = 200 <= status_code < 300
        self.text = str(payload)

    def json(self):
        return self._payload


def test_embed_uses_single_model_for_both_by_default():
    # sample_mflix ships one model (voyage-3-large); queries must match it.
    poster = FakePoster(dims=2048)
    emb = utils.AtlasEmbeddings(api_key="k", model="voyage-3-large", dimensions=2048,
                                poster=poster)
    emb.embed_documents(["a"])
    emb.embed_query("q")
    assert poster.calls[0]["json"]["model"] == "voyage-3-large"
    assert poster.calls[1]["json"]["model"] == "voyage-3-large"


def test_embed_documents_returns_vectors_and_uses_document_input_type():
    poster = FakePoster(dims=2048)
    emb = utils.AtlasEmbeddings(api_key="k", model="voyage-3-large", dimensions=2048,
                                poster=poster)
    out = emb.embed_documents(["a", "b"])
    assert len(out) == 2 and len(out[0]) == 2048
    assert poster.calls[0]["json"]["input_type"] == "document"
    assert poster.calls[0]["json"]["output_dimension"] == 2048
    assert poster.calls[0]["headers"]["Authorization"] == "Bearer k"


def test_embed_query_uses_query_input_type():
    poster = FakePoster(dims=2048)
    emb = utils.AtlasEmbeddings(api_key="k", model="voyage-3-large", dimensions=2048,
                                poster=poster)
    v = emb.embed_query("hello")
    assert len(v) == 2048
    assert poster.calls[0]["json"]["input_type"] == "query"


def test_embed_optional_asymmetric_query_model():
    poster = FakePoster(dims=1024)
    emb = utils.AtlasEmbeddings(api_key="k", model="voyage-4", query_model="voyage-4-lite",
                                dimensions=1024, poster=poster)
    emb.embed_documents(["a"])
    emb.embed_query("q")
    assert poster.calls[0]["json"]["model"] == "voyage-4"
    assert poster.calls[1]["json"]["model"] == "voyage-4-lite"


def test_embed_dimension_guard_raises_on_mismatch():
    poster = FakePoster(dims=512)  # server returns wrong size
    emb = utils.AtlasEmbeddings(api_key="k", model="voyage-3-large", dimensions=2048,
                                poster=poster)
    with pytest.raises(utils.EmbeddingDimensionMismatch):
        emb.embed_query("hello")


def test_embed_raises_on_http_error():
    poster = FakePoster(status=401)
    emb = utils.AtlasEmbeddings(api_key="bad", model="voyage-3-large", dimensions=2048,
                                poster=poster)
    with pytest.raises(RuntimeError):
        emb.embed_query("hello")


def test_embeddings_is_langchain_embeddings():
    from langchain_core.embeddings import Embeddings
    emb = utils.AtlasEmbeddings(api_key="k", model="voyage-3-large", poster=FakePoster())
    assert isinstance(emb, Embeddings)


# --------------------------------------------------------------------------
# atlas_rerank: reorders documents by relevance
# --------------------------------------------------------------------------
class FakeRerankPoster:
    def __init__(self, order):
        self.order = order  # list of indices in ranked order
        self.calls = []

    def __call__(self, url, headers, json, timeout):
        self.calls.append({"url": url, "json": json})
        results = [{"index": i, "relevance_score": 1.0 - n * 0.1}
                   for n, i in enumerate(self.order)]
        return _FakeResp(200, {"data": results})


def test_atlas_rerank_reorders_by_relevance():
    docs = ["doc0", "doc1", "doc2"]
    poster = FakeRerankPoster(order=[2, 0, 1])
    ranked = utils.atlas_rerank("query", docs, api_key="k", top_n=2, poster=poster)
    assert ranked == ["doc2", "doc0"]
    assert poster.calls[0]["json"]["query"] == "query"


def test_atlas_rerank_default_top_n_returns_all():
    docs = ["doc0", "doc1", "doc2"]
    poster = FakeRerankPoster(order=[2, 0, 1])
    ranked = utils.atlas_rerank("query", docs, api_key="k", poster=poster)
    assert ranked == ["doc2", "doc0", "doc1"]
    # top_k defaults to the full document count when top_n is unset.
    assert poster.calls[0]["json"]["top_k"] == 3


def test_atlas_rerank_top_n_zero_is_not_treated_as_all():
    # A falsy-but-explicit top_n=0 must mean "zero", not "all documents".
    docs = ["doc0", "doc1", "doc2"]
    poster = FakeRerankPoster(order=[2, 0, 1])
    ranked = utils.atlas_rerank("query", docs, api_key="k", top_n=0, poster=poster)
    assert ranked == []
    assert poster.calls[0]["json"]["top_k"] == 0
