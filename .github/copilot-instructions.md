# Project Conventions

- Keep reusable Python modules under `src/`, executable entrypoints under `script/`, and source datasets/assets under root `data/`.
- All lap-running vehicles travel clockwise. The canonical Serpentine starts at its northwest corner and initially travels east (+x), preserving the clockwise full-lap direction. Reverse direction only when explicitly requested.
- Obstacle plots must preserve the established paper/reference appearance and make obstacles clearly visible and proportionate. Do not render obstacles as tiny gray dots. A controller's circular obstacle approximation is not automatically the physical obstacle geometry for visualization; use the track's physical geometry when available and do not silently replace it.
- After changing track geometry, regenerate and open the track preview, then report the geometry checks.
- Keep `docs/paper_copy/` synchronized with the latest manuscript when a change materially affects the paper's method, assumptions, experiments, results, figures, or conclusions. Update the manuscript and any affected copied figures in the same work step; do not present incomplete or failed runs as successful results.