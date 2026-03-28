# Gilded Age RAG — Telegram Agent

A retrieval-augmented generation (RAG) agent built to support food styling research for HBO's *The Gilded Age*. Deployed as a Telegram bot, it answers questions about Gilded Age culinary history by drawing directly from a curated knowledge base of period-accurate sources.

---

## What It Is

A domain-specific RAG pipeline that ingests 50+ culinary and historical documents into a vector database and exposes them through a Telegram bot powered by GPT-4o. The agent is prompted as a Gilded Age culinary historian — it answers questions about period ingredients, menus, cooking techniques, yields, and food costing by citing the actual source documents.

## Why It Was Built

Built as a pre-production research tool and portfolio piece in preparation for a food styling assistant role on *The Gilded Age*. The goal: instant, source-grounded answers to on-set questions like "What would a Gilded Age oyster course look like?" or "What's the trimmed yield on a whole beef tenderloin?"

---

## Knowledge Base

50+ documents covering:

- **Period cookbooks** — Ranhofer's *The Epicurean* (1894), and other late 19th-century culinary references
- **Menus** — Delmonico's restaurant menus, Grover Cleveland 1891 state dinner records, Newport mansion event records
- **Production materials** — HBO *The Gilded Age* show wiki, IMDB episode data, and production notes on the show's food styling approach
- **Food costing reference** — F.T. Lynch's *The Book of Yields* (full extraction: ~1,000 rows of yield percentages and weights for produce, meats, poultry, seafood, dairy, grains, and more)
- **RI Food Code** — relevant health and food safety standards
- **Supplementary web content** — Downton Abbey Cooks period recipes, Pamela Holt's brown Windsor soup history

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI `text-embedding-3-small` |
| RAG framework | LlamaIndex 0.14 |
| Vector store | ChromaDB (persistent) |
| PDF extraction | pdfplumber + pytesseract (OCR fallback) |
| Bot interface | python-telegram-bot v22 |
| Deployment | Railway (Nixpacks) |

---

## Key Technical Decisions

**ChromaDB over LlamaIndex's default JSON store** — The default store serialized to a 1.3GB JSON file that hung on load. Switching to ChromaDB's persistent client resolved this entirely; cold start time dropped from unusable to seconds.

**Batch ingestion for Railway's 512MB RAM limit** — Loading all 50+ documents at once silently OOM-killed the process on Railway. Rewrote `ingest.py` to process 5 files per batch with explicit `gc.collect()` between batches, keeping peak RSS well under the limit.

**pdfplumber word-position extraction for Book of Yields** — pdfplumber's table extraction mode saw the multi-column yield tables as two visual columns, merging all numeric data into one string. Fixed by using `extract_words()` with x-coordinate column bucketing, confirmed against actual word positions (yield % consistently at x0 ≈ 460 for produce tables, x0 ≈ 413 for meat tables).

**Base64 blob cleanup** — 9 markdown files contained embedded base64 image data totaling ~23MB. Stripped with a single regex pass; transform time on those files dropped from 10 minutes to 17 seconds.

**System prompt conflict fix** — Initial system prompt contained "if a question falls outside your sources, say so," which caused the LLM to ignore retrieved context and fall back to base model knowledge. Removed that line; the agent now always answers from the provided source excerpts.

---

## Project Structure

```
├── KB/                          # Knowledge base documents
│   ├── Book_of_Yields_clean.md  # Extracted yield tables (~1,000 rows)
│   └── ...                      # PDFs, OCR outputs, scraped markdown
├── chroma_storage/              # Persistent vector index (gitignored)
├── ingest.py                    # Builds ChromaDB index from KB/
├── preprocess_yields.py         # Extracts Book of Yields PDF → markdown
├── query_engine.py              # LlamaIndex query engine with historian persona
├── telegram_bot.py              # Telegram bot wrapper
├── start.py                     # Railway entrypoint: ingest → bot
├── nixpacks.toml                # Railway build config (python312, tesseract, poppler)
└── railway.toml                 # Railway deploy config
```

---

## Running Locally

```bash
pip install -r requirements.txt
cp .env.example .env           # add OPENAI_API_KEY and TELEGRAM_BOT_TOKEN

python ingest.py               # build the index (run once, or to rebuild)
python telegram_bot.py         # start the bot
```

## Deployment

Deployed on [Railway](https://railway.app). On each deploy, `start.py` runs `ingest.py` to rebuild the index from the committed KB files, then hands off to `telegram_bot.py`.

Environment variables required: `OPENAI_API_KEY`, `TELEGRAM_BOT_TOKEN`
