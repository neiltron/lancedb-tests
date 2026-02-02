import React, { useMemo, useState } from "react";
import type { SearchMode, SearchResult } from "@lancedb/shared";

export function App() {
  const [mode, setMode] = useState<SearchMode>("clip");
  const [weight, setWeight] = useState(0.5);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);

  const isHybrid = mode === "hybrid";

  const preview = useMemo(() => {
    if (!file) return null;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return url;
  }, [file]);

  const onFile = (f: File | null) => {
    setFile(f);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  };

  const onDrop: React.DragEventHandler<HTMLDivElement> = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f);
  };

  const onBrowse: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    const f = e.target.files?.[0] || null;
    onFile(f);
  };

  const onSearch = async () => {
    if (!file) return;
    const form = new FormData();
    form.append("image", file);
    form.append("mode", mode);
    form.append("k", "30");
    if (isHybrid) form.append("w", String(weight));

    const res = await fetch("http://localhost:8787/search", {
      method: "POST",
      body: form
    });
    const data = await res.json();
    setResults(data.results || []);
  };

  return (
    <div className="page">
      <header className="hero">
        <h1>LanceDB Dual-Embedding Search</h1>
        <p>Compare CLIP vs DINO neighborhood similarity side-by-side.</p>
      </header>

      <section className="controls">
        <div
          className="drop"
          onDragOver={(e) => e.preventDefault()}
          onDrop={onDrop}
        >
          <input type="file" accept="image/*" onChange={onBrowse} />
          <div className="drop-label">
            <span>Drag & drop</span>
            <span>or click to choose</span>
          </div>
          {preview && <img className="preview" src={preview} alt="query" />}
        </div>

        <div className="mode">
          <label>
            <input
              type="radio"
              checked={mode === "clip"}
              onChange={() => setMode("clip")}
            />
            CLIP
          </label>
          <label>
            <input
              type="radio"
              checked={mode === "dino"}
              onChange={() => setMode("dino")}
            />
            DINO
          </label>
          <label>
            <input
              type="radio"
              checked={mode === "hybrid"}
              onChange={() => setMode("hybrid")}
            />
            Hybrid
          </label>
        </div>

        {isHybrid && (
          <div className="slider">
            <label>Hybrid weight: {weight.toFixed(2)}</label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={weight}
              onChange={(e) => setWeight(Number(e.target.value))}
            />
          </div>
        )}

        <button className="search" onClick={onSearch} disabled={!file}>
          Search
        </button>
      </section>

      <section className={isHybrid ? "results" : "results two-col"}>
        {results.map((r) => (
          <article key={r.id} className="card">
            <div className="thumb" />
            <div className="meta">
              <div className="title">{r.artist}</div>
              <div className="tags">{r.style} · {r.genre}</div>
              <div className="score">{r.score.toFixed(4)}</div>
            </div>
          </article>
        ))}
      </section>
    </div>
  );
}
