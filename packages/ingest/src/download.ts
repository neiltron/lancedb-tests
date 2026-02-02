import path from "node:path";
import fs from "fs-extra";

export type DownloadOptions = {
  outDir: string;
  source?: "local" | "kaggle" | "hf";
  from?: string;
};

export async function downloadDataset({ outDir, source = "local", from }: DownloadOptions) {
  await fs.ensureDir(outDir);

  if (source === "local") {
    if (!from) {
      console.log("Local source selected. Provide --from <path> to stage files.");
      console.log(`Output directory: ${outDir}`);
      return;
    }
    const resolved = path.resolve(from);
    await fs.copy(resolved, outDir, { overwrite: false, errorOnExist: false });
    console.log(`Staged dataset from ${resolved} -> ${outDir}`);
    return;
  }

  console.log("Download is not automated for this source yet.");
  console.log("Place dataset files manually, then rerun build.");
  console.log("Hints:");
  console.log("- Kaggle WikiArt: download zip, extract to data/wikiart");
  console.log("- Hugging Face huggan/wikiart: export images + metadata to data/wikiart");
  console.log(`Expected output directory: ${outDir}`);
}
