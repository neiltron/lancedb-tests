import path from "node:path";
import fs from "fs-extra";
import { parse } from "csv-parse/sync";
import type { DatasetItem, DatasetLoadOptions } from "./types.js";

function toNumber(value: unknown): number | undefined {
  if (value === undefined || value === null || value === "") return undefined;
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
}

function normalizeRow(row: Record<string, unknown>): Omit<DatasetItem, "id"> & { id?: string } {
  const imagePath =
    (row.file as string) ||
    (row.path as string) ||
    (row.image as string) ||
    (row.image_path as string) ||
    "";

  return {
    id: (row.id as string) || undefined,
    imagePath,
    artist: String(row.artist || "unknown"),
    style: String(row.style || "unknown"),
    genre: String(row.genre || "unknown"),
    title: row.title ? String(row.title) : undefined,
    year: toNumber(row.year)
  };
}

async function readJsonl(filePath: string): Promise<Record<string, unknown>[]> {
  const content = await fs.readFile(filePath, "utf8");
  const rows: Record<string, unknown>[] = [];
  for (const line of content.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    rows.push(JSON.parse(trimmed));
  }
  return rows;
}

async function readJson(filePath: string): Promise<Record<string, unknown>[]> {
  const content = await fs.readFile(filePath, "utf8");
  const parsed = JSON.parse(content);
  if (Array.isArray(parsed)) return parsed;
  if (typeof parsed === "object" && parsed) return [parsed as Record<string, unknown>];
  throw new Error("Unsupported JSON metadata format");
}

async function readCsv(filePath: string): Promise<Record<string, unknown>[]> {
  const content = await fs.readFile(filePath, "utf8");
  return parse(content, { columns: true, skip_empty_lines: true });
}

export async function loadLocalDataset(options: DatasetLoadOptions): Promise<DatasetItem[]> {
  const { datasetDir, limit } = options;
  const jsonl = path.join(datasetDir, "metadata.jsonl");
  const json = path.join(datasetDir, "metadata.json");
  const csv = path.join(datasetDir, "metadata.csv");

  let rows: Record<string, unknown>[] = [];
  if (await fs.pathExists(jsonl)) {
    rows = await readJsonl(jsonl);
  } else if (await fs.pathExists(json)) {
    rows = await readJson(json);
  } else if (await fs.pathExists(csv)) {
    rows = await readCsv(csv);
  } else {
    throw new Error(
      "No metadata found. Provide metadata.jsonl, metadata.json, or metadata.csv in dataset root."
    );
  }

  const items: DatasetItem[] = [];
  for (const row of rows) {
    const normalized = normalizeRow(row);
    if (!normalized.imagePath) continue;
    const id = normalized.id || normalized.imagePath;
    items.push({
      id,
      imagePath: normalized.imagePath,
      artist: normalized.artist,
      style: normalized.style,
      genre: normalized.genre,
      title: normalized.title,
      year: normalized.year
    });
    if (limit && items.length >= limit) break;
  }

  return items;
}
