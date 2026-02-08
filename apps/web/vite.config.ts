import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import fs from "fs";

export default defineConfig({
  plugins: [
    react(),
    // Serve images from packages/ingest/data/images
    {
      name: "serve-images",
      configureServer(server) {
        const imagesDir = path.resolve(__dirname, "../../packages/ingest/data/images");
        server.middlewares.use("/images", (req, res, next) => {
          const filePath = path.join(imagesDir, req.url || "");
          if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
            res.setHeader("Content-Type", "image/jpeg");
            fs.createReadStream(filePath).pipe(res);
          } else {
            next();
          }
        });
      },
    },
  ],
  server: {
    port: 5173,
  },
});
