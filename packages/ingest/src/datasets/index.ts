import type { DatasetLoadOptions, DatasetItem } from "./types.js";
import { loadLocalDataset } from "./local.js";

export async function loadDataset(options: DatasetLoadOptions): Promise<DatasetItem[]> {
  return loadLocalDataset(options);
}

export type { DatasetLoadOptions, DatasetItem };
