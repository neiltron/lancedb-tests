---
title: "feat: Add sketch-based image retrieval embedding"
type: feat
date: 2026-02-06
---

# feat: Add sketch-based image retrieval embedding

## Overview

Add a new embedding pipeline for sketch-based image retrieval. Users can draw simple sketches (shapes, stick figures, interfaces) and find visually similar artwork based on structural/edge similarity rather than color or texture.

**Key insight:** By extracting edges from dataset images and embedding those edges with DINO, we create an embedding space where hand-drawn sketches can find structurally similar images regardless of color or texture.

**Experimentation goal:** Persist edge images to disk for visual inspection—learn which image types work well for edge-based matching and iterate on parameters.

## Technical Approach

### Edge Extraction Strategy

Use Sharp's image processing to create Canny-like edge detection:
1. Convert to grayscale
2. Apply Gaussian blur (reduces noise, softens edges)
3. Run Sobel/Canny edge detection
4. Threshold to clean binary edges

This keeps everything in Node.js—no Python dependency.

### Architecture

```
INGEST PHASE:
  Original Image
       ↓
  [Sharp: grayscale → blur → edges]
       ↓
  Edge Image → saved to data/edges/{style}/{filename}.jpg
       ↓
  [DINO embedder]
       ↓
  sketch_vec (384 dims) + edge_path → LanceDB

QUERY PHASE:
  User Sketch Upload
       ↓
  [Sharp: same edge extraction]
       ↓
  Processed Sketch (returned to UI for display)
       ↓
  [DINO embedder]
       ↓
  Vector Search on sketch_vec
       ↓
  Results (with edge_path for popover)
```

### Why This Works

- **DINO captures local structure:** DINO's patch-wise embeddings excel at capturing spatial relationships and shapes
- **Edge normalization:** Both dataset images and query sketches go through the same edge extraction, making them comparable
- **No color/texture bias:** Edge images strip away color and texture, focusing purely on structure
- **Persisted edges:** Saved to disk for experimentation—inspect what works, iterate on parameters

## Implementation Phases

### Phase 0: Fix API/Ingest Embedder Duplication (Pre-requisite)

**Issue identified by reviewers:** The API currently duplicates CLIP and DINO embedder logic instead of importing from `@lancedb/ingest`. The ingest version also lacks token pooling that the API has—this is a bug.

**Fix:** Export embedders from `@lancedb/ingest` and import in API. Ensure token pooling is consistent.

```typescript
// packages/ingest/src/embeddings/index.ts - already exports createClipEmbedder, createDinoEmbedder
// apps/api/src/index.ts - import { createClipEmbedder, createDinoEmbedder } from "@lancedb/ingest"
```

This is separate cleanup work but blocks sketch implementation to avoid tripling the duplication.

### Phase 1: Edge Extraction Utility

Create `packages/ingest/src/utils/edges.ts`:

```typescript
// packages/ingest/src/utils/edges.ts
import sharp from "sharp";
import path from "node:path";
import fs from "node:fs/promises";

// Laplacian edge detection kernel
const LAPLACIAN_KERNEL = [-1, -1, -1, -1, 8, -1, -1, -1, -1];

/**
 * Extract edges from an image using Laplacian edge detection.
 * @param input - File path or Buffer
 * @param blur - Gaussian blur sigma (default: 1.5)
 * @param threshold - Binary threshold (default: 50)
 */
export async function extractEdges(
  input: string | Buffer,
  blur = 1.5,
  threshold = 50
): Promise<Buffer> {
  return sharp(input)
    .grayscale()
    .blur(blur)
    .convolve({ width: 3, height: 3, kernel: LAPLACIAN_KERNEL })
    .threshold(threshold)
    .toBuffer();
}

/**
 * Extract edges and save to file. Creates parent directories if needed.
 */
export async function extractEdgesToFile(
  inputPath: string,
  outputPath: string,
  blur = 1.5,
  threshold = 50
): Promise<void> {
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const buffer = await extractEdges(inputPath, blur, threshold);
  await sharp(buffer).jpeg({ quality: 90 }).toFile(outputPath);
}
```

### Phase 2: Update DINO Embedder to Support Edge Preprocessing

Instead of creating a separate sketch embedder, add an `edges` option to the existing DINO embedder:

```typescript
// packages/ingest/src/embeddings/dino.ts
import { extractEdges } from "../utils/edges.js";

export type DinoEmbedderOptions = EmbeddingOptions & {
  edges?: boolean;  // If true, preprocess with edge extraction
};

export async function createDinoEmbedder(
  options: DinoEmbedderOptions = {}
): Promise<Embedder> {
  // ... existing setup ...

  return {
    modelId: options.edges ? `${modelId}+edges` : modelId,
    async embedImage(imagePath: string) {
      // Preprocess with edge extraction if enabled
      const imageInput = options.edges
        ? await extractEdges(imagePath)
        : imagePath;

      const image = typeof imageInput === "string"
        ? await loadImage(imageInput)
        : await (hf as any).RawImage.fromBlob(new Blob([imageInput], { type: "image/png" }));

      const output: any = await extractor(image, {
        pooling: "mean",
        normalize: true,
      } as any);

      // ... existing pooling logic (ensure this matches API) ...
    },
  };
}
```

Export `extractEdges` from ingest package:

```typescript
// packages/ingest/src/index.ts
export { extractEdges, extractEdgesToFile } from "./utils/edges.js";
```

### Phase 3: Update LanceDB Schema

Modify `packages/ingest/src/build.ts`:

```typescript
// Add to schema (after dino_vec field)
new Field("edge_path", new Utf8()),  // Path to edge image for UI display
new Field(
  "sketch_vec",
  new FixedSizeList(384, new Field("item", new Float32()))
),

// Create sketch embedder (reuses DINO with edges option)
const sketchEmbedder = await createDinoEmbedder({
  cacheDir: modelCacheDir,
  dtype: "q8",
  edges: true,
});

// In processing loop: save edge image and embed
const edgeRelPath = withJpegExtension(item.relativePath);
const edgeOut = path.join(edgesDir, edgeRelPath);
await extractEdgesToFile(thumbOut, edgeOut);

let sketchVec: Float32Array | null = null;
try {
  sketchVec = await sketchEmbedder.embedImage(edgeOut);
} catch (error) {
  console.warn(`Sketch embedding failed for ${item.id}: ${error}`);
}

// Add to row data
edge_path: edgeRelPath,
sketch_vec: sketchVec ? Array.from(sketchVec) : null,  // null, not zero vector

// Add index creation
await table.createIndex("sketch_vec", { replace: true });
```

### Phase 4: Update Shared Types

Modify `packages/shared/src/index.ts`:

```typescript
export type SearchMode = "clip" | "dino" | "sketch" | "hybrid" | "compare";

export type SearchResult = {
  id: string;
  artist: string;
  style: string;
  genre: string;
  originalUrl: string;
  edgeUrl?: string;  // URL to edge image for popover
  score: number;
  debug?: {
    clipScore?: number;
    dinoScore?: number;
    sketchScore?: number;
  };
};
```

### Phase 5: Update API

Modify `apps/api/src/index.ts`:

```typescript
// Import from ingest package (after fixing Phase 0)
import { createDinoEmbedder, extractEdges } from "@lancedb/ingest";

// Lazy-load sketch embedder (DINO with edges)
let sketchEmbedderPromise: Promise<Embedder> | null = null;

async function getSketchEmbedder(): Promise<Embedder> {
  if (!sketchEmbedderPromise) {
    sketchEmbedderPromise = createDinoEmbedder({
      cacheDir: process.env.MODEL_CACHE_DIR,
      dtype: "q8",
      edges: true,
    });
  }
  return sketchEmbedderPromise;
}

// Update vectorSearch to return edge_path
query = query.select([
  "id", "artist", "style", "genre",
  "thumb_path", "original_path", "edge_path",  // Add edge_path
  "_distance",
]);

// In result mapping, add edgeUrl
edgeUrl: r2Base
  ? `${r2Base}/images/edges/${encodeURI(String(row.edge_path))}`
  : `/edge/${row.id}`,

// Add /edge/:id endpoint (similar to /image/:id)
app.get("/edge/:id", async (c) => {
  // Look up edge_path from DB, serve from edgesDir
});

// Add sketch mode handler
if (mode === "sketch") {
  const sketch = await getSketchEmbedder();
  const vector = await sketch.embedImage(tempPath);
  const results = await vectorSearch({
    vector,
    column: "sketch_vec",
    k,
    filters,
  });
  return c.json({ results, debug: payload });
}

// Add endpoint to return processed query edge image
app.post("/extract-edges", async (c) => {
  const form = await c.req.formData();
  const file = form.get("image");
  // ... save to temp, extract edges, return as image/jpeg
});
```

### Phase 6: Update Web UI

Modify `apps/web/src/app.tsx`:

**Add sketch to compare view:**
```typescript
// Compare view now shows CLIP vs DINO vs Sketch
{mode === "compare" && (
  <section className="compare-view">
    <ResultColumn label="CLIP" results={clipResults} loading={loading} />
    <ResultColumn label="DINO" results={dinoResults} loading={loading} />
    <ResultColumn label="Sketch" results={sketchResults} loading={loading} />
  </section>
)}
```

**Display query edge image:**
```typescript
// State for processed query edge
const [queryEdgeUrl, setQueryEdgeUrl] = useState<string | null>(null);

// After search, fetch edge version of query
const fetchQueryEdge = async () => {
  const form = new FormData();
  form.append("image", file!);
  const res = await fetch(`${API_BASE}/extract-edges`, { method: "POST", body: form });
  const blob = await res.blob();
  setQueryEdgeUrl(URL.createObjectURL(blob));
};

// Display in controls section
{queryEdgeUrl && (
  <div className="query-edge-preview">
    <span>Edge preview:</span>
    <img src={queryEdgeUrl} alt="Query edges" />
  </div>
)}
```

**Add edge popover to result cards:**
```typescript
function ResultCard({ r }: { r: SearchResult }) {
  const [showEdge, setShowEdge] = useState(false);
  // ... existing code ...

  return (
    <article className="card">
      {/* ... existing thumb-wrap ... */}

      {/* Edge popover toggle */}
      {r.edgeUrl && (
        <button className="edge-toggle" onClick={() => setShowEdge(!showEdge)}>
          {showEdge ? "Photo" : "Edges"}
        </button>
      )}

      {/* Edge popover */}
      {showEdge && r.edgeUrl && (
        <div className="edge-popover">
          <img src={r.edgeUrl} alt="Edge version" />
        </div>
      )}
    </article>
  );
}
```

**Add sketch to mode options:**
```typescript
{(["compare", "clip", "dino", "sketch", "hybrid"] as const).map((m) => (
  // ... existing radio buttons, add "Sketch" label
))}
```

## Acceptance Criteria

### Functional Requirements

- [x] Edge extraction utility produces clean edge images from photos
- [x] Edge images saved to `data/edges/{style}/{filename}.jpg` for inspection
- [x] DINO embedder supports `edges: true` option for sketch embedding
- [x] Ingest pipeline adds `sketch_vec` and `edge_path` columns to LanceDB
- [x] `sketch_vec` is indexed for fast ANN search
- [x] API accepts `mode=sketch` and searches `sketch_vec` column
- [x] API serves edge images via `/edge/:id` endpoint
- [x] API returns processed query edges via `/extract-edges` endpoint
- [x] Web UI displays "Sketch" mode option
- [x] Compare view shows CLIP vs DINO vs Sketch columns
- [x] Query edge preview shown after search
- [x] Result cards have toggle to view edge version (popover)
- [x] Items with failed edge extraction still ingest (with null sketch_vec)

### Non-Functional Requirements

- [ ] Edge extraction adds <100ms per image during ingest
- [ ] Query-time edge extraction adds <50ms latency
- [x] Existing CLIP and DINO functionality unchanged
- [x] No Python dependencies added
- [ ] API imports embedders from `@lancedb/ingest` (no duplication)

### Quality Gates

- [x] `pnpm build` passes for all packages (API and Web pass; ingest has pre-existing type issues)
- [ ] Manual test: upload hand-drawn sketch, get structurally similar results
- [ ] Verify edge extraction produces reasonable output (not blank, not solid)
- [ ] Visual inspection of edge images in `data/edges/` directory

## File Changes Summary

| File | Change |
|------|--------|
| `packages/ingest/src/utils/edges.ts` | **NEW** - Edge extraction utility |
| `packages/ingest/src/embeddings/dino.ts` | Add `edges` option for sketch mode |
| `packages/ingest/src/embeddings/index.ts` | Export edge utilities |
| `packages/ingest/src/build.ts` | Add sketch_vec, edge_path to schema; save edges to disk |
| `packages/shared/src/index.ts` | Add "sketch" to SearchMode, edgeUrl to SearchResult |
| `apps/api/src/index.ts` | Import from ingest, add /edge/:id, /extract-edges endpoints, sketch mode |
| `apps/web/src/app.tsx` | Add Sketch to compare view, query edge preview, edge popover on cards |

## Open Questions

1. **Canny parameters:** The default blur=1.5, threshold=50 are starting points. May need tuning based on actual sketch retrieval quality. Edge images are persisted for inspection.

2. **Hybrid mode:** Should sketch be included in hybrid search? Current plan keeps it separate—different use case (structural vs semantic).

## References

- Brainstorm: `docs/brainstorms/2026-02-06-sketch-embedding-brainstorm.md`
- Existing DINO embedder: `packages/ingest/src/embeddings/dino.ts`
- DINO token pooling gotcha: `docs/solutions/integration-issues/dino-query-dim-mismatch-thumb-404-cora-20260205.md`
- Sharp edge detection: https://sharp.pixelplumbing.com/api-operation#convolve
