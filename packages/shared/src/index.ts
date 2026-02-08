export type SearchMode = "clip" | "dino" | "sketch-sobel" | "sketch-skeleton" | "hybrid" | "compare";

export type SearchFilters = {
  artist?: string;
  style?: string;
  genre?: string;
};

export type SearchRequest = {
  mode: SearchMode;
  k?: number;
  w?: number;
  filters?: SearchFilters;
};

export type SearchResult = {
  id: string;
  artist: string;
  style: string;
  genre: string;
  thumbUrl: string;
  edgeUrl?: string;
  score: number;
  edgeSobelUrl?: string;
  edgeSkeletonUrl?: string;
  debug?: {
    clipScore?: number;
    dinoScore?: number;
    sketchSobelScore?: number;
    sketchSkeletonScore?: number;
  };
};
