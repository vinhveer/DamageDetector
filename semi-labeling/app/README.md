# Semi-labeling Reviewer

Standalone Electron + React + Redux app for the semi-labeling review loop
(pipeline steps 4–8). Reads pipeline output straight from SQLite — no Python
process, no workflow runner.

## Tabs

| Tab | Step | Source |
|---|---|---|
| Dedup | Step 4 | `step4_class_aware_dedup/dedup.sqlite3` |
| Cluster | Step 5 | `step5_clustering/clusters.sqlite3` |
| Classifier | Step 6 | `step6_classifier/` |
| Review | Step 7 (default tab) | `step7_label_review/suspect_clusters.sqlite3` |
| Final | Step 8 | `step7_label_review/final_labels_*.csv` |

Default paths live in `electron/defaults.js` (`SEMI_LABELING_DEFAULTS`) and resolve
under `infer_results/semi-labeling/`. Each tab lets you override its paths.

## Develop

```bash
npm install
npm run dev      # vite + electron
npm run build    # renderer build
npm start        # run electron against built renderer
npm run lint
```

> Electron 42 bundles Node 22+, which provides the built-in `node:sqlite` used by
> the handlers. The `EBADENGINE` warning during `npm install` on Node 20 is
> harmless — it only affects tooling, not the running app.
