---
title: "feat: Flesh out web UI results view"
type: feat
date: 2026-02-02
---

# ✨ Flesh out web UI results view (compare + hybrid)

## Overview
Implement a results UI that supports side-by-side comparison of CLIP vs DINO outputs and a dedicated hybrid list view. The UI’s goal is to visually validate that DINO captures style/texture rather than semantic similarity.

## Problem Statement / Motivation
The current web UI renders a single flat list with placeholder thumbnails, which does not enable side-by-side comparison. We need a clear compare view that shows CLIP vs DINO results concurrently, plus a hybrid view that ranks combined results.

## Proposed Solution
- Add a **Compare** results layout with two columns:
  - Left column: CLIP results
  - Right column: DINO results
- Add a **Hybrid** results layout with a single ranked list.
- Update search logic to fetch results for **both** modes when in compare view (two API calls or one combined response if API evolves).
- Render real thumbnails via `thumbUrl` and show key metadata (artist/style/genre/score).
- Add UI affordances that make the comparison obvious (column headers, counts, score labels, consistent card sizes).

## Technical Considerations
- **State shape:** separate `clipResults`, `dinoResults`, and `hybridResults` to avoid mixing modes.
- **Network:** compare view likely needs two requests (`mode=clip` and `mode=dino`) unless API adds a multi-mode response.
- **Loading/empty states:** show per-column empty states and a global loading indicator.
- **Layout:** keep card sizes aligned across columns for easy visual comparison; handle responsive stacking on narrow screens.
- **Accessibility:** include alt text for thumbnails and readable contrast for score labels.

## Acceptance Criteria
- [x] Compare view shows **two columns** labeled "CLIP" and "DINO" with results in each.
- [x] Hybrid mode shows a **single list** labeled "Hybrid" with results ordered by hybrid score.
- [x] Thumbnails render via `img` using `thumbUrl` and have visible fallback if missing.
- [x] Each result card displays `artist`, `style`, `genre`, and `score`.
- [x] Empty results show a friendly placeholder instead of blank columns.
- [x] Loading state appears while search is in progress.
- [x] UI remains usable on mobile (columns stack vertically).

## Success Metrics
- A developer can visually compare CLIP vs DINO outputs in one screen without switching modes.
- The UI makes it easy to validate whether DINO emphasizes style/texture.

## Dependencies & Risks
- **Dependencies:** `apps/api` must return `thumbUrl` + `score` values; thumbnail server must be working.
- **Risks:** Double-request for compare could feel slow; inconsistent result lengths across columns may reduce comparability.

## SpecFlow Analysis (Summary)
### User Flow Overview
1. **Compare flow:** user uploads image → UI runs CLIP + DINO searches → results populate two columns.
2. **Hybrid flow:** user selects Hybrid → UI runs hybrid search → results populate single list.
3. **Empty flow:** API returns no results → UI shows “No results” placeholder for affected view.
4. **Error flow:** API errors → UI shows error message + retry.

### Missing Elements & Gaps
- **Mode semantics:** clarify whether compare is the default or tied to a new UI toggle.
- **Result limits:** confirm if both columns should use identical `k` for fair comparison.
- **Performance:** decide if compare results should be fetched in parallel or sequentially.

### Critical Questions
1. **Critical:** Should compare mode always run **both** searches even when user selects CLIP or DINO radio buttons?
   - Why: affects UX expectations and API load.
   - Default: add a dedicated “Compare” toggle that triggers both searches.
2. **Important:** Should hybrid list include per-result debug scores (clip/dino) in the UI?
   - Why: helpful for validating the hybrid weight’s effect.
   - Default: show only overall score unless a debug toggle is added.
3. **Important:** Do we need a dedicated empty/error message per column or a single global message?
   - Why: CLIP may return results while DINO doesn’t.
   - Default: per-column placeholders.

### Recommended Next Steps
- Decide compare toggle UX and whether to deprecate single-mode list views.
- Define score display format (distance vs similarity label).
- Confirm result `k` for compare vs hybrid.

## References & Research
- Spec: `_docs/spec.md`
- Current UI: `apps/web/src/app.tsx`
- Current styles: `apps/web/src/styles.css`
- Shared types: `packages/shared/src/index.ts`
- Note: no `docs/solutions/` directory found for institutional learnings.

## AI-Era Considerations
- Prompt used: “Flesh out the web UI results view with side-by-side compare and hybrid list.”
- AI-generated UI changes should be reviewed for accessibility and layout correctness.
- Emphasize manual testing with real images and non-empty results.
