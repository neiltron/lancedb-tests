#!/usr/bin/env node
import { Command } from "commander";
import { downloadDataset } from "./download.js";
import { buildIndex } from "./build.js";

process.on("unhandledRejection", (reason) => {
  console.error("Unhandled rejection:", reason);
});
process.on("uncaughtException", (err) => {
  console.error("Uncaught exception:", err);
});

if (process.argv[2] === "--") {
  process.argv.splice(2, 1);
}

const program = new Command();

program
  .name("ingest")
  .description("WikiArt ingest pipeline")
  .version("0.0.0");

program
  .command("download")
  .description("Download dataset to data/wikiart")
  .option("-o, --out <dir>", "Output directory", "data/wikiart")
  .option("--source <type>", "Source: local|kaggle|hf", "local")
  .option("--from <dir>", "Local source directory")
  .action(async (opts) => {
    await downloadDataset({ outDir: opts.out, source: opts.source, from: opts.from });
  });

program
  .command("build")
  .description("Build thumbnails, embeddings, and LanceDB table")
  .option("-d, --dataset <dir>", "Dataset root", "data/wikiart")
  .option("--db <dir>", "LanceDB directory", "data/lancedb/wikiart.lancedb")
  .option("--images <dir>", "Images output", "data/images")
  .option("--concurrency <n>", "Concurrency", "2")
  .option("--batch <n>", "Batch size", "4")
  .option("--limit <n>", "Limit number of items")
  .option("--thumb-size <n>", "Thumbnail size (px)", "256")
  .option("--model-cache <dir>", "Model cache directory", "data/models")
  .option("--no-resume", "Ignore manifest and reprocess all items")
  .option("--reset-manifest", "Delete manifest before running")
  .action(async (opts) => {
    await buildIndex({
      datasetDir: opts.dataset,
      dbDir: opts.db,
      imagesDir: opts.images,
      concurrency: Number(opts.concurrency),
      batchSize: Number(opts.batch),
      limit: opts.limit ? Number(opts.limit) : undefined,
      thumbSize: Number(opts.thumbSize),
      modelCacheDir: opts.modelCache,
      resume: opts.resume,
      resetManifest: opts.resetManifest
    });
  });

program.parseAsync(process.argv);
