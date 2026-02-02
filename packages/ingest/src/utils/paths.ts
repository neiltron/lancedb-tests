import path from "node:path";
import fs from "fs-extra";

export async function ensureDirForFile(filePath: string) {
  await fs.ensureDir(path.dirname(filePath));
}

export function withJpegExtension(relativePath: string): string {
  const ext = path.extname(relativePath);
  if (!ext) return `${relativePath}.jpg`;
  return relativePath.replace(ext, ".jpg");
}
