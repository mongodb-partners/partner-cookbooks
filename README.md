# Partner Cookbooks

Hands-on notebooks for building AI applications on MongoDB Atlas, organized by partner
framework. Each set is self-contained, runs on a free Atlas cluster, and is written for
developers new to the topic.

## Partners

| Partner | Sets | Language | Runs in Colab |
|---------|------|----------|---------------|
| [LangChain](./partners/langchain/) | [Advanced RAG](./partners/langchain/advanced-rag/) (3 notebooks), [Claude Code & MCP](./partners/langchain/claude-code-mcp/) (1 notebook) | Python | Yes |
| [Mastra](./partners/mastra/) | Retrieval, BYO collections, hybrid search, agents with memory (4 notebooks) | TypeScript | No, needs a local clone |

Every set uses the Atlas `sample_mflix` dataset, so you load data once and reuse it
throughout. The Mastra notebooks reuse the two search indexes the LangChain set creates,
which matters on the free tier: M0 allows three, and eight notebooks share them.

All notebook outputs are committed. You can read what any of this returns before running
anything yourself.

## Getting started

You need a free MongoDB Atlas cluster and a couple of API keys.

1. **Create an Atlas cluster.** Sign up at [mongodb.com/atlas](https://www.mongodb.com/atlas)
   and create a free **M0** cluster. Everything here runs on the free tier. Choose MongoDB
   **8.0 or later** if you plan to run the Mastra set, which uses `$rankFusion`.
2. **Load the sample data.** In the Atlas UI, open your cluster's menu and choose
   **Load Sample Dataset**. This adds `sample_mflix`, including movie plot embeddings.
3. **Get your keys.** You need an Atlas API key, for the hosted embedding and rerank
   services, and a Claude API key. [`.env.example`](./.env.example) documents every setting
   and what it is for.
4. **Pick a partner** from the table above and follow its README. Setup differs: the
   LangChain sets are `pip install`, the Mastra set is `npm install` plus a kernel
   registration.

```bash
git clone https://github.com/mongodb-partners/partner-cookbooks.git
cd partner-cookbooks
cp .env.example .env      # then fill in your connection string and keys
```

Credentials live in one `.env` at the repository root, shared by every set. The Python
notebooks read it and prompt for anything missing, so they work in Colab without a file at
all. The TypeScript notebooks read it and fail with the name of the variable they wanted.

## Repository layout

```
partner-cookbooks/
├── partners/
│   ├── langchain/
│   │   ├── advanced-rag/        # 3 notebooks + shared helpers + tests
│   │   ├── claude-code-mcp/     # 1 notebook
│   │   └── requirements.txt     # pinned Python dependencies
│   └── mastra/
│       ├── 01..04_*.ipynb       # 4 notebooks
│       ├── src/                 # shared helpers + unit tests
│       ├── scripts/             # live smoke test, output credential scan
│       └── package.json         # pinned npm dependencies
├── .env.example                 # every setting the notebooks read
└── LICENSE
```

## Contributing

Issues and pull requests are welcome. A new notebook set should sit under
`partners/<partner>/`, cover one topic, run on a free M0 cluster, and read credentials from
the repository `.env` rather than hard-coding them.

Two more conventions, both easy to get wrong:

- **Commit your outputs, and scrub them first.** These notebooks are read as often as they
  are run, so one with empty cells is a worse artifact. The cost of committing outputs is
  that a connection string printed in a cell becomes a published connection string.
  `partners/mastra/scripts/check-notebook-outputs.ts` scans every notebook in the
  repository, not just the Mastra ones, and `npm test` in `partners/mastra/` runs it.
- **Add an Open in Colab badge only if the kernel supports it.** The Python sets have one
  on every notebook. The Mastra set has none, because tslab needs a local `npm install`
  that Colab cannot provide. If your set cannot offer Colab, say so in its README instead
  of linking a badge that will not work.

## License

Apache 2.0. See [LICENSE](./LICENSE).
