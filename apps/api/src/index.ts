import { Hono } from "hono";
import { serve } from "@hono/node-server";
import type { SearchRequest } from "@lancedb/shared";

const app = new Hono();

app.get("/health", (c) => c.json({ ok: true }));

app.get("/thumb/:id", (c) => {
  const id = c.req.param("id");
  return c.json({ error: "not implemented", id }, 501);
});

app.post("/search", async (c) => {
  const form = await c.req.formData();
  const mode = form.get("mode");
  const k = Number(form.get("k") || 30);
  const w = Number(form.get("w") || 0.5);
  const filtersRaw = form.get("filters");

  const payload: SearchRequest = {
    mode: (mode as SearchRequest["mode"]) || "clip",
    k,
    w,
    filters: filtersRaw ? JSON.parse(String(filtersRaw)) : undefined
  };

  return c.json({ results: [], debug: payload });
});

serve({ fetch: app.fetch, port: 8787 });
console.log("API listening on http://localhost:8787");
