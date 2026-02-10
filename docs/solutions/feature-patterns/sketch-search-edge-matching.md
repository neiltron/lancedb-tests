---
title: "Sketch-Based Image Search with Edge Matching"
date: 2026-02-06
category: feature-patterns
module: embedding_pipeline
tags: [lancedb, dino, sharp, edge-detection, laplacian, sketch-search, embeddings, transformers-js]
problem_type: feature_implementation
resolution_type: new_feature
severity: enhancement
symptoms:
  - "Need sketch-based image search by structural similarity"
  - "Find images matching hand-drawn shapes regardless of color/texture"
  - "Existing CLIP/DINO embeddings capture semantic/color but not structural layout"
---

# Sketch-Based Image Search with Edge Matching

## Problem

Standard image embeddings (CLIP, DINO) capture semantic content and visual appearance but have a domain gap when matching hand-drawn sketches to photographs. Users drawing simple shapes on an iPad canvas need to find structurally similar artwork regardless of color or texture.

## Solution

Create a parallel embedding space where both database images and query sketches are preprocessed through edge extraction before embedding with DINO. This normalizes both inputs to structural features only.

### Architecture

```
INGEST PHASE:
  Original Image
       ↓
  [Sharp: grayscale → blur → Laplacian → threshold]
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

### Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Edge detection | Laplacian kernel (Sharp) | Node.js native, no Python dependency |
| Embedding model | DINO (not CLIP) | Better structural features without text bias |
| Architecture | `edges` option on existing DINO embedder | Composition over duplication |
| Edge storage | Persist to disk | Enable experimentation and visual inspection |
| Schema design | Nullable `sketch_vec` and `edge_path` | Graceful degradation for failed extractions |

## Implementation

### 1. Edge Extraction Utility

```typescript
// packages/ingest/src/utils/edges.ts
import sharp from "sharp";

const LAPLACIAN_KERNEL = [-1, -1, -1, -1, 8, -1, -1, -1, -1];

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
```

### 2. DINO Embedder with Edges Option

```typescript
// packages/ingest/src/embeddings/dino.ts
export type DinoEmbedderOptions = EmbeddingOptions & {
  edges?: boolean; // If true, preprocess with edge extraction
};

// In embedImage():
if (options.edges) {
  const edgeBuffer = await extractEdges(imagePath);
  image = await (hf as any).RawImage.fromBlob(
    new Blob([new Uint8Array(edgeBuffer)], { type: "image/png" })
  );
}
```

### 3. LanceDB Schema Extension

```typescript
// packages/ingest/src/build.ts
new Field("edge_path", new Utf8(), true),  // nullable
new Field(
  "sketch_vec",
  new FixedSizeList(384, new Field("item", new Float32())),
  true  // nullable
),

// Create index
await table.createIndex("sketch_vec", { replace: true });
```

### 4. API Endpoints

```typescript
// apps/api/src/index.ts

// Serve edge images
app.get("/edge/:id", async (c) => { /* ... */ });

// Extract edges from query image
app.post("/extract-edges", async (c) => { /* ... */ });

// Search with mode=sketch
if (mode === "sketch") {
  const sketch = await getSketchEmbedder();
  const vector = await sketch.embedImage(tempPath);
  return vectorSearch({ vector, column: "sketch_vec", k, filters });
}
```

### 5. Web UI Updates

- Added "Sketch" to mode selector
- Compare view shows 3 columns: CLIP | DINO | Sketch
- Query edge preview displayed after search
- Result cards have "Photo/Edges" toggle button

## Gotchas

### Buffer to Blob Type Error

**Problem:** TypeScript complains `Buffer` doesn't satisfy `BlobPart`.

```typescript
// WRONG
new Blob([buffer], { type: "image/png" });

// CORRECT
new Blob([new Uint8Array(buffer)], { type: "image/png" });
```

### Hono c.body() Type Error

**Problem:** Hono's `c.body()` doesn't accept `Buffer` directly.

```typescript
// WRONG
c.body(buffer, 200, { "Content-Type": "image/jpeg" });

// CORRECT
c.body(new Uint8Array(buffer), 200, { "Content-Type": "image/jpeg" });
```

### RawImage Loading from Buffer

**Problem:** HuggingFace's `RawImage` has no `fromBuffer()` method.

**Solution:** Wrap in Blob and use `fromBlob()`:

```typescript
const image = await RawImage.fromBlob(
  new Blob([new Uint8Array(buffer)], { type: "image/png" })
);
```

### Embedding Space Consistency

**Critical:** The same edge extraction parameters must be used at both index time and query time. Changing blur or threshold invalidates existing embeddings.

## Best Practices Established

### 1. Composition Over Creation

Add options to existing embedders rather than creating new embedder classes:

```typescript
// Good: extends existing embedder
createDinoEmbedder({ edges: true })

// Avoid: separate class with duplicated logic
createSketchEmbedder()
```

### 2. Persist Intermediate Results

Save edge images to disk during ingest for:
- Visual inspection of edge quality
- Parameter tuning experiments
- Fast serving without recomputation

### 3. Nullable Schema Fields

Make new columns nullable to allow graceful degradation:

```typescript
new Field("sketch_vec", vectorType, true)  // true = nullable
```

Items with failed edge extraction still ingest successfully.

### 4. Export from Package Root

Re-export utilities so API can import from single source:

```typescript
// packages/ingest/src/index.ts
export { extractEdges } from "./utils/edges.js";
export { createDinoEmbedder } from "./embeddings/dino.js";
```

## Testing Checklist

- [ ] Edge extraction produces clean output (not blank, not solid)
- [ ] Edge images saved to `data/edges/` directory
- [ ] `sketch_vec` indexed for fast ANN search
- [ ] API accepts `mode=sketch` and returns results
- [ ] `/edge/:id` serves edge images
- [ ] `/extract-edges` returns processed query
- [ ] Compare view shows 3 columns
- [ ] Edge toggle works on result cards
- [ ] Items with failed extraction still ingest (null sketch_vec)

## Files Changed

| File | Change |
|------|--------|
| `packages/ingest/src/utils/edges.ts` | **NEW** - Edge extraction utility |
| `packages/ingest/src/embeddings/dino.ts` | Add `edges` option |
| `packages/ingest/src/embeddings/index.ts` | Export edge utilities |
| `packages/ingest/src/build.ts` | Add sketch_vec, edge_path; save edges |
| `packages/ingest/src/index.ts` | **NEW** - Package exports |
| `packages/shared/src/index.ts` | Add "sketch" mode, edgeUrl field |
| `apps/api/src/index.ts` | Add endpoints, sketch mode handler |
| `apps/web/src/app.tsx` | Add sketch to compare view, edge toggle |

## Related Documents

- Brainstorm: `docs/brainstorms/2026-02-06-sketch-embedding-brainstorm.md`
- Plan: `docs/plans/2026-02-06-feat-sketch-embedding-plan.md`
- DINO dim mismatch fix: `docs/solutions/integration-issues/dino-query-dim-mismatch-thumb-404-cora-20260205.md`
- Hybrid sort fix: `docs/solutions/logic-errors/hybrid-sort-distance-vs-score.md`

## Future Considerations

1. **Canny parameters:** Default blur=1.5, threshold=50 are starting points. May need tuning based on actual sketch retrieval quality.

2. **Hybrid mode integration:** Currently sketch is separate from hybrid search. Could add weighted sketch contribution for multi-modal queries.

3. **Alternative edge detectors:** If Laplacian proves insufficient, could explore Sobel, Scharr, or even lightweight neural edge detectors that run in ONNX.
