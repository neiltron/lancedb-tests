export type DatasetItem = {
  id: string;
  imagePath: string;
  artist: string;
  style: string;
  genre: string;
  title?: string;
  year?: number;
};

export type DatasetLoadOptions = {
  datasetDir: string;
  limit?: number;
};
