import sharp from "sharp";
import path from "node:path";
import fs from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import os from "node:os";

const execFileAsync = promisify(execFile);

/**
 * Edge detection methods available
 */
export type EdgeMethod = "sobel" | "laplacian" | "canny" | "skeleton";

/**
 * Edge detection options
 */
export interface EdgeOptions {
  method?: EdgeMethod;
  blur?: number;
  threshold?: number;
}

// Default options per method
const METHOD_DEFAULTS: Record<EdgeMethod, { blur: number; threshold: number }> = {
  sobel: { blur: 2, threshold: 0.35 },
  laplacian: { blur: 1.5, threshold: 0.12 },
  canny: { blur: 1.5, threshold: 0.25 },
  skeleton: { blur: 2, threshold: 0.35 },
};

// Path to skeletonize binary - check common locations
const SKELETONIZE_PATHS = [
  "/tmp/skeletonize/target/release/examples/skeletonize",
  "/usr/local/bin/skeletonize",
  path.join(os.homedir(), ".cargo/bin/skeletonize"),
];

let cachedSkeletonizePath: string | null | undefined;

async function findSkeletonize(): Promise<string | null> {
  if (cachedSkeletonizePath !== undefined) return cachedSkeletonizePath;

  for (const p of SKELETONIZE_PATHS) {
    try {
      await fs.access(p);
      cachedSkeletonizePath = p;
      return p;
    } catch {
      // continue
    }
  }
  cachedSkeletonizePath = null;
  return null;
}

/**
 * Extract edges from an image using the specified method.
 * @param input - File path or Buffer
 * @param options - Edge detection options
 */
export async function extractEdges(
  input: string | Buffer,
  options: EdgeOptions = {}
): Promise<Buffer> {
  const method = options.method ?? "sobel";
  const defaults = METHOD_DEFAULTS[method];
  const blur = options.blur ?? defaults.blur;
  const threshold = options.threshold ?? defaults.threshold;

  switch (method) {
    case "skeleton":
      return extractSkeleton(input, blur, threshold);
    case "canny":
      return extractCanny(input, blur, threshold);
    case "laplacian":
      return extractLaplacian(input, blur, threshold);
    case "sobel":
    default:
      return extractSobel(input, blur, threshold);
  }
}

/**
 * Skeleton: Sobel + Zhang-Suen thinning via skeletonize binary
 */
async function extractSkeleton(
  input: string | Buffer,
  blur: number,
  threshold: number
): Promise<Buffer> {
  const skeletonizePath = await findSkeletonize();

  if (!skeletonizePath) {
    console.warn("skeletonize binary not found, falling back to sobel");
    return extractSobel(input, blur, threshold);
  }

  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "edges-"));
  const inputPng = path.join(tempDir, "input.png");
  const outputPng = path.join(tempDir, "output.png");

  try {
    await sharp(input)
      .blur(blur)
      .png()
      .toFile(inputPng);

    await execFileAsync(skeletonizePath, [
      "-i", inputPng,
      "-o", outputPng,
      "-e", "sobel",
      "-t", String(threshold),
      "-f", "white",
    ]);

    const result = await sharp(outputPng)
      .jpeg({ quality: 90 })
      .toBuffer();
    return result;
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true });
  }
}

/**
 * Canny: Sobel edges via skeletonize without thinning
 */
async function extractCanny(
  input: string | Buffer,
  blur: number,
  threshold: number
): Promise<Buffer> {
  const skeletonizePath = await findSkeletonize();

  if (!skeletonizePath) {
    console.warn("skeletonize binary not found, falling back to sobel");
    return extractSobel(input, blur, threshold);
  }

  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "edges-"));
  const inputPng = path.join(tempDir, "input.png");
  const outputPng = path.join(tempDir, "output.png");

  try {
    await sharp(input)
      .blur(blur)
      .png()
      .toFile(inputPng);

    await execFileAsync(skeletonizePath, [
      "-i", inputPng,
      "-o", outputPng,
      "-e", "sobel",
      "-t", String(threshold),
      "-f", "white",
      "--no-thin",  // Skip thinning for canny-style output
    ]);

    const result = await sharp(outputPng)
      .jpeg({ quality: 90 })
      .toBuffer();
    return result;
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true });
  }
}

/**
 * Laplacian: Second derivative edge detection (sketch-like, thinner lines)
 */
async function extractLaplacian(
  input: string | Buffer,
  blur: number,
  threshold: number
): Promise<Buffer> {
  const LAPLACIAN = [-1, -1, -1, -1, 8, -1, -1, -1, -1];
  const thresh = Math.round(threshold * 255);

  // Need to use PNG between steps to preserve data
  const gray = await sharp(input)
    .grayscale()
    .blur(blur)
    .png()
    .toBuffer();

  const convolved = await sharp(gray)
    .convolve({ width: 3, height: 3, kernel: LAPLACIAN })
    .png()
    .toBuffer();

  return sharp(convolved)
    .threshold(thresh)
    .jpeg({ quality: 90 })
    .toBuffer();
}

/**
 * Sobel: Gradient-based edge detection (thicker lines)
 */
const SOBEL_X = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
const SOBEL_Y = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

async function extractSobel(
  input: string | Buffer,
  blur: number,
  threshold: number
): Promise<Buffer> {
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

/**
 * Extract edges and save to file. Creates parent directories if needed.
 */
export async function extractEdgesToFile(
  inputPath: string,
  outputPath: string,
  options: EdgeOptions = {}
): Promise<void> {
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const buffer = await extractEdges(inputPath, options);
  await sharp(buffer).jpeg({ quality: 90 }).toFile(outputPath);
}

// Legacy export for backwards compatibility
export { extractEdges as default };
