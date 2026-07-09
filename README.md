# Partner Cookbooks

Hands-on Jupyter notebooks for building AI applications on MongoDB Atlas. Each set is
self-contained, runs on a free Atlas cluster, and is written for developers new to the topic.

## Notebook sets

| Set | What you'll build | Notebooks |
|-----|-------------------|-----------|
| [Advanced RAG](./advanced-rag/) | A movie-discovery assistant that retrieves with vector, full-text, hybrid, graph, and auto-embedding techniques | 3 |
| [Claude Code & MCP](./claude-code-mcp/) | An AI agent that queries MongoDB through the MCP Server, plus the Claude Code plugin | 1 |

Both sets use the Atlas `sample_mflix` dataset, so you load data once and reuse it throughout.

## Getting started

You need a free MongoDB Atlas cluster and a couple of API keys.

1. **Create an Atlas cluster.** Sign up at [mongodb.com/atlas](https://www.mongodb.com/atlas) and
   create a free **M0** cluster. Everything in these notebooks runs on the free tier.
2. **Load the sample data.** In the Atlas UI, open your cluster's menu and choose
   **Load Sample Dataset**. This adds `sample_mflix`, including movie plot embeddings.
3. **Get your keys.** You'll need an Atlas API key (for the hosted embedding and rerank services)
   and a Claude API key. The Advanced RAG notebooks list exactly what each one is for.
4. **Open a notebook.** Click the **Open in Colab** badge at the top of any notebook to run it in
   the browser, or run locally (see below).

### Running locally

```bash
git clone https://github.com/mongodb-partners/partner-cookbooks.git
cd partner-cookbooks
pip install -r requirements.txt
cp .env.example .env      # then fill in your connection string and keys
jupyter lab
```

The notebooks read credentials from a `.env` file when present and prompt for anything missing, so
the same notebook works in Colab or on your machine.

## Repository layout

```
partner-cookbooks/
├── advanced-rag/          # Advanced RAG (3 notebooks + shared helpers)
├── claude-code-mcp/       # MongoDB MCP Server + AI agents (1 notebook)
├── requirements.txt       # Pinned dependency versions
└── .env.example           # Every setting the notebooks read
```

## Contributing

Issues and pull requests are welcome. New notebook sets should follow the conventions here: one
topic per set, runnable on a free M0 cluster, credentials via environment-or-prompt, and an
**Open in Colab** badge on every notebook.

## License

Apache 2.0. See [LICENSE](./LICENSE).
