# Synchronized Manuscript Copy

- Baseline source: `/Users/nevinxue/Desktop/tongji_phd_paper/submission/20260613/manuscript_20260929.tex`
- Baseline revision date: 2026-09-30
- Copied contents: manuscript source, `sn-jnl.cls`, and all five referenced figure assets.
- Synchronization rule: update this copy in the same work step whenever project changes materially affect the paper's method, assumptions, experiments, results, figures, or conclusions. Keep the manuscript and affected figures consistent with verified outcomes.
- Status note from the baseline manuscript: its abstract says the reported simulation results were generated under the previous safety formulation and are to be replaced after rerunning under the revised condition. The 2026-11-13 zero-disturbance OSQP diagnostic with 13 physical obstacles completed one lap for the robust controller (1427.05 m, no collision); the non-robust controller stopped at 664.93 m (46.6%) after 12 stalled steps. The run did not crash, but recorded 253 OSQP solve failures and 622 half-space fallbacks in the robust controller. This single diagnostic is not a validated paired result and is not included in the manuscript's reported results.
- Diagnostic trajectory plot: [robust vs. non-robust, 2026-11-13](figures/diagnostic_heading_fix_full_lap_20261113.png). It is retained for troubleshooting and is not a manuscript result figure.
- Repository organization update: reusable modules now live under `src/` and executable entrypoints under `script/`. The manuscript source and referenced baseline figures remain intact; the diagnostic update above is kept separate from formal paper results.
