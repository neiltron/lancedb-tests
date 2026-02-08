import path from "node:path";
import os from "node:os";
import fs from "node:fs/promises";
import fsSync from "node:fs";
import { Hono } from "hono";
import { cors } from "hono/cors";
import { serve } from "@hono/node-server";
import * as lancedb from "@lancedb/lancedb";
import * as hf from "@huggingface/transformers";
import sharp from "sharp";
import type { SearchFilters, SearchRequest, SearchResult } from "@lancedb/shared";

const app = new Hono();
app.use("*", cors({ origin: "*" }));

app.get("/health", (c) => c.json({ ok: true }));

const DEFAULT_DB_DIR = "data/lancedb/wikiart.lancedb";
const DEFAULT_IMAGES_DIR = "data/images";
const DEFAULT_THUMB_SIZE = 256;
const THUMB_QUALITY = 82;
const HYBRID_POOL = 200;

// Path to skeletonize binary
const SKELETONIZE_PATH = "/tmp/skeletonize/target/release/examples/skeletonize";

type EdgeMethod = "sobel" | "skeleton";

/**
 * Extract edges using the specified method.
 * - sobel: Gradient-based edge detection (thicker lines)
 * - skeleton: Sobel + Zhang-Suen thinning (thin lines)
 */
async function extractEdges(
  input: string | Buffer,
  method: EdgeMethod = "sobel",
  blur = 2,
  threshold = 0.35
): Promise<Buffer> {
  // Check if skeletonize exists
  const hasSkeletonize = fsSync.existsSync(SKELETONIZE_PATH);

  if (method === "skeleton" && hasSkeletonize) {
    return extractEdgesWithSkeletonize(input, blur, threshold, true);
  }

  if (method === "sobel" && hasSkeletonize) {
    return extractEdgesWithSkeletonize(input, blur, threshold, false);
  }

  // Fallback to manual Sobel for both methods if skeletonize not available
  return extractEdgesFallback(input, blur, threshold);
}

async function extractEdgesWithSkeletonize(
  input: string | Buffer,
  blur: number,
  threshold: number,
  thin: boolean = true
): Promise<Buffer> {
  const { execFile } = await import("node:child_process");
  const { promisify } = await import("node:util");
  const execFileAsync = promisify(execFile);

  // Create temp directory for processing
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "edges-"));
  const inputPng = path.join(tempDir, "input.png");
  const outputPng = path.join(tempDir, "output.png");

  try {
    // Pre-blur and convert to PNG
    await sharp(input)
      .blur(blur)
      .png()
      .toFile(inputPng);

    // Run skeletonize
    const args = [
      "-i", inputPng,
      "-o", outputPng,
      "-e", "sobel",
      "-t", String(threshold),
      "-f", "white",
    ];
    if (!thin) {
      args.push("--no-thin");
    }
    await execFileAsync(SKELETONIZE_PATH, args);

    // Read result and convert to JPEG
    return sharp(outputPng)
      .jpeg({ quality: 90 })
      .toBuffer();
  } finally {
    // Cleanup temp files
    await fs.rm(tempDir, { recursive: true, force: true });
  }
}

// Fallback Sobel implementation
const SOBEL_X = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
const SOBEL_Y = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

async function extractEdgesFallback(
  input: string | Buffer,
  blur: number,
  threshold: number
): Promise<Buffer> {
  // Convert threshold from 0-1 to 0-255 range
  const thresh = Math.round(threshold * 255);

  const base = await sharp(input)
    .grayscale()
    .blur(blur)
    .raw()
    .toBuffer({ resolveWithObject: true });

  const { data, info } = base;
  const { width, height } = info;

  const edges = Buffer.alloc(width * height);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let gx = 0, gy = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const pixel = data[(y + ky) * width + (x + kx)];
          const ki = (ky + 1) * 3 + (kx + 1);
          gx += pixel * SOBEL_X[ki];
          gy += pixel * SOBEL_Y[ki];
        }
      }
      const magnitude = Math.min(255, Math.sqrt(gx * gx + gy * gy));
      edges[y * width + x] = magnitude > thresh ? 255 : 0;
    }
  }

  return sharp(edges, { raw: { width, height, channels: 1 } })
    .jpeg({ quality: 90 })
    .toBuffer();
}

type Embedder = {
  modelId: string;
  embedImage: (imagePath: string) => Promise<Float32Array>;
};

function resolveRepoPath(inputPath: string): string {
  const absolute = path.resolve(inputPath);
  if (fsSync.existsSync(absolute)) return absolute;

  const cwdResolved = path.resolve(process.cwd(), inputPath);
  if (fsSync.existsSync(cwdResolved)) return cwdResolved;

  const repoRoot = path.resolve(process.cwd(), "..", "..");
  return path.resolve(repoRoot, inputPath);
}

function configureModelCache() {
  const env = (hf as any).env;
  const cacheDir = process.env.MODEL_CACHE_DIR;
  if (env && cacheDir) {
    env.cacheDir = resolveRepoPath(cacheDir);
  }
}

function l2Normalize(values: Float32Array): Float32Array {
  let sum = 0;
  for (const v of values) sum += v * v;
  const norm = Math.sqrt(sum) || 1;
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = values[i] / norm;
  return out;
}

function meanPoolTokens(
  data: Float32Array,
  seq: number,
  dim: number,
  startToken = 0
): Float32Array {
  const out = new Float32Array(dim);
  const count = Math.max(0, seq - startToken);
  if (count === 0) return out;
  for (let t = startToken; t < seq; t++) {
    const base = t * dim;
    for (let i = 0; i < dim; i++) out[i] += data[base + i];
  }
  for (let i = 0; i < dim; i++) out[i] /= count;
  return out;
}

function escapeSqlString(value: string): string {
  return value.replace(/'/g, "''");
}

function buildFilter(filters?: SearchFilters): string | undefined {
  if (!filters) return undefined;
  const clauses: string[] = [];
  if (filters.artist) clauses.push(`artist = '${escapeSqlString(filters.artist)}'`);
  if (filters.style) clauses.push(`style = '${escapeSqlString(filters.style)}'`);
  if (filters.genre) clauses.push(`genre = '${escapeSqlString(filters.genre)}'`);
  if (!clauses.length) return undefined;
  return clauses.join(" AND ");
}

function resolveUnder(baseDir: string, relativePath: string): string {
  const base = path.resolve(baseDir);
  const full = path.resolve(base, relativePath);
  if (full !== base && !full.startsWith(base + path.sep)) {
    throw new Error("Invalid path");
  }
  return full;
}

async function ensureDirForFile(filePath: string) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
}

let clipEmbedderPromise: Promise<Embedder> | null = null;
let dinoEmbedderPromise: Promise<Embedder> | null = null;
let sketchSobelEmbedderPromise: Promise<Embedder> | null = null;
let sketchSkeletonEmbedderPromise: Promise<Embedder> | null = null;

async function createClipEmbedder(): Promise<Embedder> {
  configureModelCache();
  const dtype = "q8";
  const modelId = "Xenova/clip-vit-base-patch32";
  const extractor = await hf.pipeline(
    "image-feature-extraction",
    modelId,
    { dtype, subfolder: "onnx" } as any
  );
  const loadImage =
    (hf as any).load_image ||
    (async (imgPath: string) => (hf as any).RawImage.fromFile(imgPath));
  return {
    modelId,
    async embedImage(imagePath: string) {
      const image = await loadImage(imagePath);
      const output: any = await extractor(
        image,
        { pooling: "mean", normalize: true } as any
      );
      const data = output.data as Float32Array | number[];
      return l2Normalize(Float32Array.from(data as Iterable<number>));
    },
  };
}

async function createDinoEmbedder(): Promise<Embedder> {
  configureModelCache();
  const dtype = "q8";
  const modelId = "Xenova/dinov2-small";
  const extractor = await hf.pipeline(
    "image-feature-extraction",
    modelId,
    { dtype, subfolder: "onnx" } as any
  );
  const loadImage =
    (hf as any).load_image ||
    (async (imgPath: string) => (hf as any).RawImage.fromFile(imgPath));
  return {
    modelId,
    async embedImage(imagePath: string) {
      const image = await loadImage(imagePath);
      const output: any = await extractor(
        image,
        { pooling: "mean", normalize: true } as any
      );
      const data = Float32Array.from(
        (output.data as Float32Array | number[]) as Iterable<number>
      );
      const dims = output.dims as number[] | undefined;
      if (dims && dims.length >= 2) {
        const dim = dims[dims.length - 1];
        const seq = data.length / dim;
        if (Number.isFinite(seq) && Number.isInteger(seq) && seq > 1) {
          const pooled = meanPoolTokens(data, seq, dim, 1);
          return l2Normalize(pooled);
        }
      }
      return l2Normalize(data);
    },
  };
}

async function getClipEmbedder(): Promise<Embedder> {
  if (!clipEmbedderPromise) clipEmbedderPromise = createClipEmbedder();
  return clipEmbedderPromise;
}

async function getDinoEmbedder(): Promise<Embedder> {
  if (!dinoEmbedderPromise) dinoEmbedderPromise = createDinoEmbedder();
  return dinoEmbedderPromise;
}

async function createSketchEmbedder(method: EdgeMethod): Promise<Embedder> {
  configureModelCache();
  const dtype = "q8";
  const modelId = "Xenova/dinov2-small";
  const extractor = await hf.pipeline(
    "image-feature-extraction",
    modelId,
    { dtype, subfolder: "onnx" } as any
  );
  return {
    modelId: `${modelId}+edges-${method}`,
    async embedImage(imagePath: string) {
      // Resize to thumbnail size first (matching ingest), then extract edges
      const resizedBuffer = await sharp(imagePath)
        .resize(256, 256, { fit: "inside" })
        .toBuffer();
      const edgeBuffer = await extractEdges(resizedBuffer, method);
      const image = await (hf as any).RawImage.fromBlob(
        new Blob([new Uint8Array(edgeBuffer)], { type: "image/png" })
      );
      const output: any = await extractor(
        image,
        { pooling: "mean", normalize: true } as any
      );
      const data = Float32Array.from(
        (output.data as Float32Array | number[]) as Iterable<number>
      );
      const dims = output.dims as number[] | undefined;
      if (dims && dims.length >= 2) {
        const dim = dims[dims.length - 1];
        const seq = data.length / dim;
        if (Number.isFinite(seq) && Number.isInteger(seq) && seq > 1) {
          const pooled = meanPoolTokens(data, seq, dim, 1);
          return l2Normalize(pooled);
        }
      }
      return l2Normalize(data);
    },
  };
}

async function getSketchSobelEmbedder(): Promise<Embedder> {
  if (!sketchSobelEmbedderPromise) sketchSobelEmbedderPromise = createSketchEmbedder("sobel");
  return sketchSobelEmbedderPromise;
}

async function getSketchSkeletonEmbedder(): Promise<Embedder> {
  if (!sketchSkeletonEmbedderPromise) sketchSkeletonEmbedderPromise = createSketchEmbedder("skeleton");
  return sketchSkeletonEmbedderPromise;
}

let tablePromise: Promise<any> | null = null;
let dbDirResolved: string | null = null;
async function getTable() {
  if (!tablePromise) {
    const dbDir = resolveRepoPath(process.env.LANCEDB_DIR || DEFAULT_DB_DIR);
    dbDirResolved = dbDir;
    tablePromise = (async () => {
      const db = await lancedb.connect(dbDir);
      return db.openTable("artworks");
    })();
  }
  return tablePromise;
}

function getImagesDir(): string {
  if (process.env.IMAGES_DIR) {
    return resolveRepoPath(process.env.IMAGES_DIR);
  }
  if (dbDirResolved) {
    const dataDir = path.resolve(dbDirResolved, "..", "..");
    return path.join(dataDir, "images");
  }
  return resolveRepoPath(DEFAULT_IMAGES_DIR);
}

function getThumbSize(): number {
  const value = Number(process.env.THUMB_SIZE || DEFAULT_THUMB_SIZE);
  return Number.isFinite(value) && value > 0 ? value : DEFAULT_THUMB_SIZE;
}

async function writeTempImage(file: File): Promise<{ dir: string; path: string }> {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "lancedb-query-"));
  const tempPath = path.join(tempDir, "query.jpg");
  const buffer = Buffer.from(await file.arrayBuffer());
  await fs.writeFile(tempPath, buffer);
  return { dir: tempDir, path: tempPath };
}

async function vectorSearch({
  vector,
  column,
  k,
  filters,
}: {
  vector: Float32Array;
  column: "clip_vec" | "dino_vec" | "sketch_sobel_vec" | "sketch_skeleton_vec";
  k: number;
  filters?: SearchFilters;
}) {
  const table = await getTable();
  let query = table.search(vector).column(column).limit(k);
  const filter = buildFilter(filters);
  if (filter) query = query.where(filter);
  query = query.select([
    "id",
    "artist",
    "style",
    "genre",
    "thumb_path",
    "original_path",
    "edge_sobel_path",
    "edge_skeleton_path",
    "_distance",
  ]);
  const rows = (await query.toArray()) as Array<Record<string, unknown>>;
  return rows.map((row) => ({
    id: String(row.id),
    artist: String(row.artist),
    style: String(row.style),
    genre: String(row.genre),
    // Return direct paths for Vite to serve
    thumbUrl: `/images/original/${row.original_path || ""}`,
    edgeSobelUrl: row.edge_sobel_path ? `/images/edges-sobel/${row.edge_sobel_path}` : undefined,
    edgeSkeletonUrl: row.edge_skeleton_path ? `/images/edges-skeleton/${row.edge_skeleton_path}` : undefined,
    score: Number(
      row._distance ?? row.score ?? row._score ?? row.distance ?? 0
    ),
  }));
}

app.get("/thumb/:id", async (c) => {
  const id = c.req.param("id");
  try {
    const table = await getTable();
    const filter = `id = '${escapeSqlString(id)}'`;
    const rows = (await table
      .query()
      .where(filter)
      .limit(1)
      .select(["thumb_path", "original_path"])
      .toArray()) as Array<Record<string, unknown>>;

    if (!rows.length) return c.json({ error: "not found" }, 404);

    const imagesDir = getImagesDir();
    const thumbsDir = path.join(imagesDir, "thumb");
    const originalsDir = path.join(imagesDir, "original");

    const thumbPath = String(rows[0].thumb_path || "");
    const originalPath = String(rows[0].original_path || "");
    if (!thumbPath && !originalPath) return c.json({ error: "not found" }, 404);

    const thumbFull = thumbPath
      ? resolveUnder(thumbsDir, thumbPath)
      : undefined;
    const originalFull = originalPath
      ? resolveUnder(originalsDir, originalPath)
      : undefined;

    const thumbExists =
      thumbFull &&
      (await fs
        .access(thumbFull)
        .then(() => true)
        .catch(() => false));

    if (!thumbExists) {
      if (!originalFull) return c.json({ error: "not found" }, 404);
      const originalExists = await fs
        .access(originalFull)
        .then(() => true)
        .catch(() => false);
      if (!originalExists) return c.json({ error: "not found" }, 404);

      if (!thumbFull) return c.json({ error: "not found" }, 404);
      await ensureDirForFile(thumbFull);
      await sharp(originalFull)
        .rotate()
        .resize(getThumbSize(), getThumbSize(), { fit: "inside" })
        .jpeg({ quality: THUMB_QUALITY })
        .toFile(thumbFull);
    }

    if (!thumbFull) return c.json({ error: "not found" }, 404);
    const buffer = await fs.readFile(thumbFull);
    return c.body(buffer, 200, { "Content-Type": "image/jpeg" });
  } catch (error) {
    return c.json(
      { error: "thumb_failed", detail: error instanceof Error ? error.message : String(error) },
      500
    );
  }
});

app.get("/image/:id", async (c) => {
  const id = c.req.param("id");
  try {
    const table = await getTable();
    const filter = `id = '${escapeSqlString(id)}'`;
    const rows = (await table
      .query()
      .where(filter)
      .limit(1)
      .select(["original_path"])
      .toArray()) as Array<Record<string, unknown>>;

    if (!rows.length) return c.json({ error: "not found" }, 404);

    const imagesDir = getImagesDir();
    const originalsDir = path.join(imagesDir, "original");
    const originalPath = String(rows[0].original_path || "");
    if (!originalPath) return c.json({ error: "not found" }, 404);

    const originalFull = resolveUnder(originalsDir, originalPath);
    if (!originalFull) return c.json({ error: "not found" }, 404);

    const buffer = await fs.readFile(originalFull);
    const ext = path.extname(originalPath).toLowerCase();
    const mime =
      ext === ".png" ? "image/png" :
      ext === ".webp" ? "image/webp" :
      "image/jpeg";
    return c.body(buffer, 200, { "Content-Type": mime });
  } catch (error) {
    return c.json(
      { error: "image_failed", detail: error instanceof Error ? error.message : String(error) },
      500
    );
  }
});

app.get("/edge/:id", async (c) => {
  const id = c.req.param("id");
  try {
    const table = await getTable();
    const filter = `id = '${escapeSqlString(id)}'`;
    const rows = (await table
      .query()
      .where(filter)
      .limit(1)
      .select(["edge_path"])
      .toArray()) as Array<Record<string, unknown>>;

    if (!rows.length) return c.json({ error: "not found" }, 404);

    const imagesDir = getImagesDir();
    const edgesDir = path.join(imagesDir, "edges");
    const edgePath = String(rows[0].edge_path || "");
    if (!edgePath) return c.json({ error: "not found" }, 404);

    const edgeFull = resolveUnder(edgesDir, edgePath);
    if (!edgeFull) return c.json({ error: "not found" }, 404);

    const buffer = await fs.readFile(edgeFull);
    return c.body(buffer, 200, { "Content-Type": "image/jpeg" });
  } catch (error) {
    return c.json(
      { error: "edge_failed", detail: error instanceof Error ? error.message : String(error) },
      500
    );
  }
});

app.post("/extract-edges", async (c) => {
  const form = await c.req.formData();
  const file = form.get("image");
  if (!file || typeof (file as File).arrayBuffer !== "function") {
    return c.json({ error: "missing_image" }, 400);
  }

  try {
    const buffer = Buffer.from(await (file as File).arrayBuffer());
    // Resize to thumbnail size first (matching ingest/sketch search)
    const resizedBuffer = await sharp(buffer)
      .resize(256, 256, { fit: "inside" })
      .toBuffer();
    const edgeBuffer = await extractEdges(resizedBuffer);
    return c.body(new Uint8Array(edgeBuffer), 200, { "Content-Type": "image/jpeg" });
  } catch (error) {
    return c.json(
      { error: "edge_extraction_failed", detail: error instanceof Error ? error.message : String(error) },
      500
    );
  }
});

app.post("/search", async (c) => {
  const form = await c.req.formData();
  const file = form.get("image");
  if (!file || typeof (file as File).arrayBuffer !== "function") {
    return c.json({ error: "missing_image" }, 400);
  }

  const mode = (form.get("mode") as SearchRequest["mode"]) || "clip";
  const kRaw = Number(form.get("k") || 30);
  const k = Number.isFinite(kRaw) ? Math.max(1, Math.min(200, kRaw)) : 30;
  const wRaw = Number(form.get("w") || 0.5);
  const w = Number.isFinite(wRaw) ? Math.min(1, Math.max(0, wRaw)) : 0.5;
  const filtersRaw = form.get("filters");
  let filters: SearchFilters | undefined;
  if (filtersRaw) {
    try {
      filters = JSON.parse(String(filtersRaw)) as SearchFilters;
    } catch {
      return c.json({ error: "invalid_filters" }, 400);
    }
  }

  const payload: SearchRequest = { mode, k, w, filters };

  let tempDir: string | null = null;
  try {
    const temp = await writeTempImage(file as File);
    tempDir = temp.dir;
    const tempPath = temp.path;

    if (mode === "clip") {
      const clip = await getClipEmbedder();
      const vector = await clip.embedImage(tempPath);
      const results = await vectorSearch({
        vector,
        column: "clip_vec",
        k,
        filters,
      });
      return c.json({ results, debug: payload });
    }

    if (mode === "dino") {
      const dino = await getDinoEmbedder();
      const vector = await dino.embedImage(tempPath);
      const results = await vectorSearch({
        vector,
        column: "dino_vec",
        k,
        filters,
      });
      return c.json({ results, debug: payload });
    }

    if (mode === "sketch-sobel") {
      const sketch = await getSketchSobelEmbedder();
      const vector = await sketch.embedImage(tempPath);
      const results = await vectorSearch({
        vector,
        column: "sketch_sobel_vec",
        k,
        filters,
      });
      return c.json({ results, debug: payload });
    }

    if (mode === "sketch-skeleton") {
      const sketch = await getSketchSkeletonEmbedder();
      const vector = await sketch.embedImage(tempPath);
      const results = await vectorSearch({
        vector,
        column: "sketch_skeleton_vec",
        k,
        filters,
      });
      return c.json({ results, debug: payload });
    }

    const [clipEmbedder, dinoEmbedder] = await Promise.all([
      getClipEmbedder(),
      getDinoEmbedder(),
    ]);
    const [clipVec, dinoVec] = await Promise.all([
      clipEmbedder.embedImage(tempPath),
      dinoEmbedder.embedImage(tempPath),
    ]);

    const pool = Math.max(HYBRID_POOL, k);
    const [clipResults, dinoResults] = await Promise.all([
      vectorSearch({ vector: clipVec, column: "clip_vec", k: pool, filters }),
      vectorSearch({ vector: dinoVec, column: "dino_vec", k: pool, filters }),
    ]);

    const merged = new Map<string, SearchResult>();

    for (const row of clipResults) {
      merged.set(row.id, {
        ...row,
        debug: { clipScore: row.score },
      });
    }

    for (const row of dinoResults) {
      const existing = merged.get(row.id);
      if (existing) {
        merged.set(row.id, {
          ...existing,
          debug: {
            clipScore: existing.debug?.clipScore ?? existing.score,
            dinoScore: row.score,
          },
        });
      } else {
        merged.set(row.id, {
          ...row,
          debug: { dinoScore: row.score },
        });
      }
    }

    const scored = Array.from(merged.values()).map((row) => {
      const clipScore = row.debug?.clipScore ?? 0;
      const dinoScore = row.debug?.dinoScore ?? 0;
      const score = w * clipScore + (1 - w) * dinoScore;
      return { ...row, score };
    });

    scored.sort((a, b) => a.score - b.score);
    return c.json({ results: scored.slice(0, k), debug: payload });
  } catch (error) {
	  console.log(error);
    return c.json(
      { error: "search_failed", detail: error instanceof Error ? error.message : String(error) },
      500
    );
  } finally {
    if (tempDir) await fs.rm(tempDir, { recursive: true, force: true });
  }
});

const port = Number(process.env.PORT || 8787);
const hostname = process.env.HOST || "127.0.0.1";
serve({ fetch: app.fetch, port, hostname });
console.log(`API listening on http://${hostname}:${port}`);
