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
