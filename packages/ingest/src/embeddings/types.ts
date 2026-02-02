export type EmbeddingOptions = {
  cacheDir?: string;
  dtype?: "fp32" | "fp16" | "q8" | "q4" | "q4f16";
};

export type Embedder = {
  modelId: string;
  embedImage: (imagePath: string) => Promise<Float32Array>;
};
