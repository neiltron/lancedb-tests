import * as hf from "@huggingface/transformers";
import { l2Normalize } from "../utils/normalize.js";
import { extractEdges } from "../utils/edges.js";
import type { Embedder, EmbeddingOptions } from "./types.js";

export type DinoEmbedderOptions = EmbeddingOptions & {
  edges?: boolean; // If true, preprocess with edge extraction
};

export async function createDinoEmbedder(
  options: DinoEmbedderOptions = {},
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
    modelId: options.edges ? `${modelId}+edges` : modelId,
    async embedImage(imagePath) {
      // Preprocess with edge extraction if enabled
      let image;
      if (options.edges) {
        const edgeBuffer = await extractEdges(imagePath);
        image = await (hf as any).RawImage.fromBlob(
          new Blob([new Uint8Array(edgeBuffer)], { type: "image/png" })
        );
      } else {
        image = await loadImage(imagePath);
      }

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
