# Mastra + MongoDB Atlas

Four notebooks that build a movie-discovery agent with [Mastra](https://mastra.ai), the
TypeScript agent framework, backed entirely by a free Atlas M0 cluster. Retrieval, hybrid
search, reranking, and conversation memory all land in the same database.

The code is TypeScript, run by the [tslab](https://github.com/yunabe/tslab) kernel rather
than IPython. That is a deliberate trade. Mastra is a TypeScript framework, and a cookbook
that taught it through a Python wrapper would be teaching the wrapper. The cost is real:
these notebooks do not open in Colab. You need a local clone and `npm install`.

## The notebooks

Run them in order. Each one builds on indexes and ideas from the last.

| # | Notebook | Covers | Runtime |
|---|----------|--------|---------|
| 1 | [Setup, Voyage Embeddings & Vector Search](./01_setup_voyage_and_vector_search.ipynb) | `MongoDBVector`, Voyage embeddings three ways, `$vectorSearch`, filter pushdown | 31s |
| 2 | [Bring Your Own Operational Collections](./02_byo_operational_collections.ipynb) | Querying a collection Mastra did not create, the read-only guard, `allowWrites`, the index registry | 65s |
| 3 | [Full-Text, Hybrid Search & Reranking](./03_hybrid_search_and_reranking.ipynb) | `textQuery` (BM25), `hybridQuery` (`$rankFusion`), weights, `rerank-2.5` | 18s |
| 4 | [Agents, Tools & Persistent Memory](./04_agent_with_tools_and_memory.ipynb) | `createTool`, tool selection, `Memory` on `MongoDBStore`, proving persistence | 55s |

Runtime is wall-clock for a full `nbconvert --execute` run against a live M0 cluster, which
is a floor rather than an estimate of your time. Notebook 02 is the slowest because it
creates an Atlas Search index and waits for it to become queryable.

Every notebook here has its outputs committed, so you can read what these actually return
before running anything yourself.

## What you need

- A MongoDB Atlas cluster with the **`sample_mflix` sample dataset** loaded. M0 works, with
  one caveat about the index budget below.
- **MongoDB 8.0 or later.** `$rankFusion` in notebook 03 needs it.
- **Node.js 20+.**
- An **Atlas API key**, used for the hosted Voyage embedding and rerank endpoints at
  `ai.mongodb.com`.
- A **Claude API key**, or an Anthropic-compatible gateway via `ANTHROPIC_BASE_URL`. Only
  notebook 04 calls a model, but `loadNotebookConfig` validates all three secrets up front,
  so the key has to be present for any of them to start.

Copy [`.env.example`](../../.env.example) at the repository root to `.env` and fill it in.

## Setup

There is no `%pip install` cell here. The kernel resolves imports from
`partners/mastra/node_modules`, so dependencies have to be installed before Jupyter starts:

```bash
cd partners/mastra
npm install
npx tslab install --python="$(which python3)"   # registers the kernel with Jupyter
jupyter lab
```

Then pick the **TypeScript** kernel when you open a notebook.

## A note on the version pin

`package.json` pins `@mastra/mongodb` to **1.15.0**. Three of the things these notebooks
teach, bring-your-own collections, `textQuery`, and `hybridQuery`, do not exist in 1.14.0.
All four notebooks depend on at least one of them: even notebook 01 reaches `sample_mflix`
through a bring-your-own registration. Downgrade below 1.15.0 and they fail on a missing
method.

This set was originally written against `1.15.0-alpha.1`, which was the only version with
those three features at the time. 1.15.0 shipped stable on 2026-07-28 and the pin moved off
the prerelease.

## Two dependency advisories we do not fix

`npm install` reports advisories that cannot be cleared without breaking something. Both are
left in place deliberately.

**`brace-expansion` (high, GHSA-mh99-v99m-4gvg).** Patched only in 5.0.8, and 5.x replaced the
default export with a named `{ expand }`. The copy here is 1.1.16, reached through
`tslab -> @rollup/plugin-commonjs -> glob -> minimatch`, and `minimatch@3` calls the bare
export. Overriding it to 5.0.8 makes `minimatch` throw `expand is not a function`, which takes
out the Jupyter kernel and with it all four notebooks. The chain is `devDependencies`-only and
is reached by tslab's build tooling, not by notebook code, so no untrusted input flows into
it here. It clears when tslab updates its rollup plugin.

**`@ai-sdk/provider-utils` (low, GHSA-866g-f22w-33x8).** No patched version exists: the
advisory covers `<= 3.0.97` and 3.0.30 is the newest 3.x published. `@mastra/core` pins it by
alias as `@ai-sdk/provider-utils-v5: npm:@ai-sdk/provider-utils@3.0.30`, so an override on the
plain name does not reach it. Waiting on upstream.

Two others are fixed by the `overrides` block in `package.json` rather than by upgrading the
parent: `sharp` to 0.35.3 and `adm-zip` to 0.6.0. `@huggingface/transformers` and
`onnxruntime-node` still declare the vulnerable ranges, and both packages load and run
correctly against the overridden versions.

## The M0 index budget

A free cluster allows **three** Atlas Search indexes, and this set spends all three.

Two come from `sample_mflix` and are shared with the LangChain notebooks:
`movies_vector_index` and `movies_fulltext_index`. Notebook 02 creates the third on its own
`movie_reviews` collection and drops it during cleanup.

Two consequences worth knowing before you deviate from the happy path:

- **A managed `createIndex` costs two slots**, not one, because Mastra provisions a
  companion full-text index alongside the vector one. There is no room for that here, which
  is why every notebook registers existing indexes instead of provisioning new ones.
- **`deleteIndex` on a bring-your-own registration drops the underlying Atlas index.** The
  collection survives; the index does not. Notebook 01 explains this and deliberately does
  not call it, because doing so would destroy `movies_vector_index` for every other notebook
  in the repository.

If `createSearchIndex` reports an index limit, drop the `movie_reviews` collection left
behind by an interrupted notebook 02 run.

## Shared helpers

Two modules hold the setup boilerplate the notebooks share. Teaching code stays in the
notebooks where you can read it.

- **`src/config.ts`** reads the repository `.env` and validates what it finds. One
  deliberate difference from the Python set: a tunable the file omits falls back to its
  documented default rather than to the surrounding shell. A stray `ANTHROPIC_MODEL` left in
  a shell by unrelated work is otherwise silently the model your notebook uses, and the
  resulting error points nowhere near the cause.
- **`src/utils.ts`** wraps the Atlas embedding and rerank endpoints, builds the Claude
  model, and opens the MongoDB connection.

## Verifying it

```bash
npm test            # 63 offline unit tests, plus a credential scan of every notebook output
npm run typecheck   # tsc --noEmit over src/ and scripts/
npm run smoke       # 8 steps against real Atlas, real Voyage, real Claude
```

`npm test` needs no cluster and no keys. It also runs
`scripts/check-notebook-outputs.ts`, which scans all eight notebooks in the repository for
connection strings and API keys in committed output, including the live values in your
`.env`. That check is why these notebooks can ship their results.

`npm run smoke` is the opposite tier: it needs a populated `.env` and crosses every
boundary the notebooks cross in a single run. The failures it exists to catch are the ones
a fake cannot reproduce, an index that reports READY but is not queryable, a 2048-d query
vector against a 1024-d index, an agent whose tool never fires. It cleans up what it
creates, and it reports skipped steps as SKIP rather than passing them silently.

## What this set leaves out

Mastra's semantic recall, working memory, and workflows are all absent, and not because
they are uninteresting. Each needs either vector-index budget or model calls beyond what a
free M0 cluster can spare. They are the natural next step on a paid cluster.
