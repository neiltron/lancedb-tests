export type SearchMode = "clip" | "dino" | "hybrid";

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
  score: number;
  debug?: {
    clipScore?: number;
    dinoScore?: number;
  };
};
