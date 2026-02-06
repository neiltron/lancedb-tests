import React, { useMemo, useState } from "react";
import type { SearchMode, SearchResult } from "@lancedb/shared";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8787";

function ResultCard({ r }: { r: SearchResult }) {
  const [imgError, setImgError] = useState(false);
  const imgSrc = r.originalUrl.startsWith("http")
    ? r.originalUrl
    : `${API_BASE}${r.originalUrl}`;
  const hasImage = !imgError;

  return (
    <article className="card">
      <div className="thumb-wrap">
        {hasImage ? (
          <img
            className="thumb"
            src={imgSrc}
            alt={`${r.artist} – ${r.style}`}
            loading="lazy"
            onError={() => setImgError(true)}
          />
        ) : (
          <div className="thumb thumb-fallback" />
        )}
        {hasImage && (
          <div className="thumb-popover">
            <img src={imgSrc} alt={`${r.artist} – ${r.style}`} />
          </div>
        )}
      </div>
      <div className="meta">
        <div className="title">{r.artist}</div>
        <div className="tags">
          {r.style}
        </div>
        <div className="score">{r.score.toFixed(4)}</div>
      </div>
    </article>
  );
}

function ResultColumn({
  label,
  results,
  loading,
}: {
  label: string;
  results: SearchResult[];
  loading: boolean;
}) {
  return (
    <div className="result-column">
      <h3 className="column-header">{label}</h3>
      {loading ? (
        <div className="placeholder">Searching…</div>
      ) : results.length === 0 ? (
        <div className="placeholder">No results</div>
      ) : (
        <div className="column-cards">
          {results.map((r) => (
            <ResultCard key={r.id} r={r} />
          ))}
        </div>
      )}
    </div>
  );
}

export function App() {
  const [mode, setMode] = useState<SearchMode>("compare");
  const [weight, setWeight] = useState(0.5);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [clipResults, setClipResults] = useState<SearchResult[]>([]);
  const [dinoResults, setDinoResults] = useState<SearchResult[]>([]);
  const [hybridResults, setHybridResults] = useState<SearchResult[]>([]);
  const [singleResults, setSingleResults] = useState<SearchResult[]>([]);

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

  const fetchResults = async (
    searchMode: "clip" | "dino" | "hybrid",
    k = 30,
  ): Promise<SearchResult[]> => {
    const form = new FormData();
    form.append("image", file!);
    form.append("mode", searchMode);
    form.append("k", String(k));
    if (searchMode === "hybrid") form.append("w", String(weight));

    const res = await fetch(`${API_BASE}/search`, {
      method: "POST",
      body: form,
    });
    const data = await res.json();
    return data.results || [];
  };

  const onSearch = async () => {
    if (!file) return;
    setLoading(true);

    try {
      if (mode === "compare") {
        const [clip, dino] = await Promise.all([
          fetchResults("clip"),
          fetchResults("dino"),
        ]);
        setClipResults(clip);
        setDinoResults(dino);
      } else if (mode === "hybrid") {
        const results = await fetchResults("hybrid");
        setHybridResults(results);
      } else {
        const results = await fetchResults(mode);
        setSingleResults(results);
      }
    } catch {
      // clear on error
      if (mode === "compare") {
        setClipResults([]);
        setDinoResults([]);
      } else if (mode === "hybrid") {
        setHybridResults([]);
      } else {
        setSingleResults([]);
      }
    } finally {
      setLoading(false);
    }
  };

  const hasSearched =
    mode === "compare"
      ? clipResults.length > 0 || dinoResults.length > 0
      : mode === "hybrid"
        ? hybridResults.length > 0
        : singleResults.length > 0;

  const currentResults =
    mode === "hybrid"
      ? hybridResults
      : mode === "compare"
        ? []
        : singleResults;

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
          {(["compare", "clip", "dino", "hybrid"] as const).map((m) => (
            <label key={m}>
              <input
                type="radio"
                checked={mode === m}
                onChange={() => setMode(m)}
              />
              {m === "compare"
                ? "Compare"
                : m === "clip"
                  ? "CLIP"
                  : m === "dino"
                    ? "DINO"
                    : "Hybrid"}
            </label>
          ))}
        </div>

        {mode === "hybrid" && (
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

        <button className="search" onClick={onSearch} disabled={!file || loading}>
          {loading ? "Searching…" : "Search"}
        </button>
      </section>

      {mode === "compare" ? (
        <section className="compare-view">
          <ResultColumn label="CLIP" results={clipResults} loading={loading} />
          <ResultColumn label="DINO" results={dinoResults} loading={loading} />
        </section>
      ) : (
        <section className="results">
          {loading ? (
            <div className="placeholder full-width">Searching…</div>
          ) : !hasSearched ? null : currentResults.length === 0 ? (
            <div className="placeholder full-width">No results</div>
          ) : (
            <>
              {mode === "hybrid" && (
                <h3 className="column-header full-width">Hybrid</h3>
              )}
              {mode === "clip" && (
                <h3 className="column-header full-width">CLIP</h3>
              )}
              {mode === "dino" && (
                <h3 className="column-header full-width">DINO</h3>
              )}
              {currentResults.map((r) => (
                <ResultCard key={r.id} r={r} />
              ))}
            </>
          )}
        </section>
      )}
    </div>
  );
}
