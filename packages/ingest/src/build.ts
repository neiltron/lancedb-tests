import path from "node:path";
import fs from "fs-extra";
import pLimit from "p-limit";
import sharp from "sharp";
import * as lancedb from "@lancedb/lancedb";
import { Field, FixedSizeList, Float32, Int32, Schema, Utf8 } from "apache-arrow";

import { loadDataset } from "./datasets/index.js";
import { createClipEmbedder, createDinoEmbedder } from "./embeddings/index.js";
import { batches } from "./utils/batches.js";
import { hashId } from "./utils/hash.js";
import { ensureDirForFile, withJpegExtension } from "./utils/paths.js";

export type BuildOptions = {
  datasetDir: string;
  dbDir: string;
  imagesDir: string;
  concurrency: number;
  batchSize: number;
  limit?: number;
  thumbSize?: number;
  modelCacheDir?: string;
  resume?: boolean;
  resetManifest?: boolean;
};

type ManifestEntry = {
  id: string;
  status: "ok" | "skipped" | "error";
  error?: string;
  clip?: string;
  dino?: string;
  imagePath?: string;
};

const DEFAULT_THUMB_SIZE = 256;

function resolveInputPath(inputPath: string): string {
  const absolute = path.resolve(inputPath);
  if (fs.pathExistsSync(absolute)) return absolute;

  const repoRoot = path.resolve(process.cwd(), "..", "..");
  const repoResolved = path.resolve(repoRoot, inputPath);
  if (fs.pathExistsSync(repoResolved)) return repoResolved;

  return absolute;
}

async function loadManifestIds(manifestPath: string): Promise<Set<string>> {
  const ids = new Set<string>();
  if (!(await fs.pathExists(manifestPath))) return ids;
  const content = await fs.readFile(manifestPath, "utf8");
  for (const line of content.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const entry = JSON.parse(trimmed) as ManifestEntry;
      if (entry.id) ids.add(entry.id);
    } catch {
      continue;
    }
  }
  return ids;
}

async function appendManifest(manifestPath: string, entry: ManifestEntry) {
  await ensureDirForFile(manifestPath);
  await fs.appendFile(manifestPath, `${JSON.stringify(entry)}\n`);
}

function resolveRelativePath(inputPath: string): string {
  const normalized = path.normalize(inputPath);
  if (normalized.startsWith("..")) {
    throw new Error(`Invalid relative path: ${inputPath}`);
  }
  return normalized.replace(/^\/+/, "");
}

export async function buildIndex(opts: BuildOptions) {
  const {
    datasetDir: datasetDirInput,
    dbDir: dbDirInput,
    imagesDir: imagesDirInput,
    concurrency,
    batchSize,
    limit,
    thumbSize = DEFAULT_THUMB_SIZE,
    modelCacheDir,
    resume = true,
    resetManifest = false
  } = opts;

  const effectiveLimit = limit && limit > 0 ? limit : undefined;
  const datasetDir = resolveInputPath(datasetDirInput);
  const dbDir = resolveInputPath(dbDirInput);
  const imagesDir = resolveInputPath(imagesDirInput);
  const manifestPath = resolveInputPath(path.join("data", "ingest", "manifest.jsonl"));
  if (resetManifest && (await fs.pathExists(manifestPath))) {
    await fs.remove(manifestPath);
  }
  const processedIds = resume ? await loadManifestIds(manifestPath) : new Set<string>();

  const items = await loadDataset({ datasetDir, limit: effectiveLimit });
  if (items.length === 0) {
    console.warn("No dataset items found.");
    return;
  }

  const originalsDir = path.join(imagesDir, "original");
  const thumbsDir = path.join(imagesDir, "thumb");
  await fs.ensureDir(originalsDir);
  await fs.ensureDir(thumbsDir);

  const clip = await createClipEmbedder({ cacheDir: modelCacheDir, dtype: "q8" });
  const dino = await createDinoEmbedder({ cacheDir: modelCacheDir, dtype: "q8" });

  const db = await lancedb.connect(dbDir);
  let table: any;
  const schema = new Schema([
    new Field("id", new Utf8()),
    new Field("original_path", new Utf8()),
    new Field("thumb_path", new Utf8()),
    new Field("artist", new Utf8()),
    new Field("style", new Utf8()),
    new Field("genre", new Utf8()),
    new Field("title", new Utf8(), true),
    new Field("year", new Int32(), true),
    new Field(
      "clip_vec",
      new FixedSizeList(512, new Field("item", new Float32()))
    ),
    new Field(
      "dino_vec",
      new FixedSizeList(384, new Field("item", new Float32()))
    )
  ]);

  try {
    table = await db.openTable("artworks");
  } catch {
    table = await db.createEmptyTable("artworks", schema, { mode: "overwrite" });
  }

  const limitConcurrency = pLimit(concurrency);
  let processed = 0;

  for (const batch of batches(items, batchSize)) {
    const tasks = batch.map((item) =>
      limitConcurrency(async () => {
        if (effectiveLimit && processed >= effectiveLimit) return null;
        const relativePath = resolveRelativePath(item.imagePath);
        const sourcePath = path.join(datasetDir, relativePath);
        const id = hashId(relativePath);
        if (processedIds.has(id)) {
          await appendManifest(manifestPath, { id, status: "skipped", imagePath: relativePath });
          return null;
        }

        const originalOut = path.join(originalsDir, relativePath);
        const thumbRelative = withJpegExtension(relativePath);
        const thumbOut = path.join(thumbsDir, thumbRelative);

        try {
          await ensureDirForFile(originalOut);
          await ensureDirForFile(thumbOut);
          await fs.copyFile(sourcePath, originalOut);

          await sharp(sourcePath)
            .rotate()
            .resize(thumbSize, thumbSize, { fit: "inside" })
            .jpeg({ quality: 82 })
            .toFile(thumbOut);

          const clipVec = await clip.embedImage(thumbOut);
          const dinoVec = await dino.embedImage(thumbOut);

          processed++;
          await appendManifest(manifestPath, {
            id,
            status: "ok",
            clip: clip.modelId,
            dino: dino.modelId,
            imagePath: relativePath
          });

          return {
            id,
            original_path: path.relative(originalsDir, originalOut),
            thumb_path: path.relative(thumbsDir, thumbOut),
            artist: item.artist,
            style: item.style,
            genre: item.genre,
            title: item.title ?? null,
            year: item.year ?? null,
            clip_vec: Array.from(clipVec),
            dino_vec: Array.from(dinoVec)
          };
        } catch (error) {
          await appendManifest(manifestPath, {
            id,
            status: "error",
            error: error instanceof Error ? error.message : String(error),
            imagePath: relativePath
          });
          console.warn("Skipping image:", relativePath, error);
          return null;
        }
      })
    );

    const rows = (await Promise.all(tasks)).filter(Boolean) as Record<string, unknown>[];
    if (rows.length) {
      await table
        .mergeInsert("id")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute(rows);
      console.log(`Upserted batch: ${rows.length}, total processed: ${processed}`);
    }
  }

  await table.createIndex("clip_vec", { replace: true });
  await table.createIndex("dino_vec", { replace: true });

  console.log("Ingest complete.");
}
