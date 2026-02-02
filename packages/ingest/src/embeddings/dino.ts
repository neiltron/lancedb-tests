import * as hf from "@huggingface/transformers";
import { l2Normalize } from "../utils/normalize.js";
import type { Embedder, EmbeddingOptions } from "./types.js";

export async function createDinoEmbedder(
  options: EmbeddingOptions = {},
): Promise<Embedder> {
  if (options.cacheDir && (hf as any).env) (hf as any).env.cacheDir = options.cacheDir;
  const dtype = options.dtype || "q8";
  const modelId = "Xenova/dinov2-small";
  const extractor = await hf.pipeline("image-feature-extraction", modelId, {
    dtype,
  });
  const loadImage =
    (hf as any).load_image ||
    (async (path: string) => (hf as any).RawImage.fromFile(path));

  return {
    modelId,
    async embedImage(imagePath) {
      const image = await loadImage(imagePath);
      const output: any = await extractor(image, {
        pooling: "mean",
        normalize: true,
      });
      const data = output.data as Float32Array | number[];
      const vec = Float32Array.from(data as Iterable<number>);
      return l2Normalize(vec);
    },
  };
}
