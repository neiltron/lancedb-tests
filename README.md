# LanceDB Dual-Embedding Image Search (Local Prototype)

## Setup

- Node 22+
- pnpm

## Install

```bash
pnpm install
```

## Monorepo layout

```
apps/
  api/
  web/
packages/
  ingest/
  shared/
```

## Data layout (local)

```
data/
  wikiart/
  images/
    original/
    thumb/
  lancedb/
    wikiart.lancedb/
```

## Ingest

```bash
pnpm --filter @lancedb/ingest dev -- download --out data/wikiart
pnpm --filter @lancedb/ingest dev -- build --dataset data/wikiart --db data/lancedb/wikiart.lancedb --images data/images --limit 100
```

### Dataset metadata format

Provide one of the following in `data/wikiart`:

- `metadata.jsonl` (one JSON object per line)
- `metadata.json` (array of objects)
- `metadata.csv` (header row)

Required fields per row:

- `file` (relative path to image under dataset root)
- `artist`
- `style`
- `genre`

Optional fields:

- `id` (if absent, a hash of the path is used)
- `title`
- `year`

Example `metadata.jsonl` entry:

```json
{ "file": "artist/painting.jpg", "artist": "Artist", "style": "Impressionism", "genre": "Landscape", "year": 1890 }
```

### Manual validation

- Run `ingest build` with `--limit 10` and confirm `data/images` + `data/lancedb/wikiart.lancedb` are created.
- Check `data/ingest/manifest.jsonl` for `status: "ok"` entries.

## Run API

```bash
pnpm --filter @lancedb/api dev
```

### API environment variables

```bash
# Optional overrides
export LANCEDB_DIR=/path/to/wikiart.lancedb
export IMAGES_DIR=/path/to/images
export MODEL_CACHE_DIR=/path/to/model-cache
export HOST=127.0.0.1
export PORT=8787
export HF_REMOTE_HOST=https://huggingface.co/
```

## Run Web

```bash
pnpm --filter @lancedb/web dev
```

## Notes

- LanceDB integration and embedding extraction are stubbed. Next step is wiring `lancedb` and `@huggingface/transformers`.
- Use a local dataset (Kaggle WikiArt or Hugging Face huggan/wikiart).
