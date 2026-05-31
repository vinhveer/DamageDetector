# CLAUDE.md — semi-labeling/app (Electron + React)

Guidance for working in `semi-labeling/app/`.

## Vai trò

Standalone desktop app review nhãn cho vòng lặp bán nhãn (pipeline steps 4–8).
Đọc trực tiếp output SQLite/CSV của pipeline — **không** spawn Python, **không**
workflow runner, **không** SAM/GDINO. App gốc `DamageDetector/app` được copy rồi
strip xuống chỉ còn phần review.

## Stack

| Layer | Technology |
|-------|-----------|
| Desktop shell | Electron 42 (bundle Node 22+, có `node:sqlite`) |
| Frontend | React 19 + Redux Toolkit (chỉ `theme`) + Tailwind CSS |
| Bundler | Vite + @vitejs/plugin-react |
| Icons | @tabler/icons-react |
| IPC | Electron contextBridge (preload.cjs) |
| SQLite | `node:sqlite` (built-in), read-only |

## Dev commands

```bash
npm install
npm run dev      # Vite (5173) + Electron
npm run dev:react
npm run build
npm run lint
npm start
```

## Cấu trúc source

```
src/
├── App.jsx                 # Nav phẳng 5 step + Settings; activeTab = useState('labelReview')
├── app/store.js            # Redux store: { theme } (đã strip workflows/segment/inspection)
├── components/{ui,overlays}/
├── utils/                  # imageCache, tiffSrc, decode worker
└── features/
    ├── dedupGroups/        # Step 4
    ├── clusterLabeling/    # Step 5
    ├── classifierResults/  # Step 6
    ├── labelReview/        # Step 7 (tab mặc định, core)
    ├── finalReview/        # Step 8
    ├── settings/           # Save dir + dark mode
    ├── shared/ theme/

electron/
├── main.js                 # IPC handlers (SQLite/CSV only, no subprocess)
├── preload.cjs             # contextBridge → window.electronAPI
├── defaults.js             # SEMI_LABELING_DEFAULTS — default path tập trung
├── dedup_groups/ cluster_labeling/ classifier_results/ label_review/ final_review/
    ├── db.js               # node:sqlite read-only connection (dedup_groups/db.js dùng chung)
    ├── queries.js          # SELECT
    └── sessions.js         # JSON session files (cluster, label_review)
```

## Default paths

Tất cả default path nằm ở `electron/defaults.js` export `SEMI_LABELING_DEFAULTS`.
`LAB_ROOT` resolve từ `electron/` lên 4 cấp: `electron → app → semi-labeling →
DamageDetector → Lab`. Mỗi handler `getXxxDefaults()` đọc từ object này. Đổi root
hay layout chỉ cần sửa 1 file.

Image root mặc định: `<Lab>/data/HinhAnh`.
DB output: `<Lab>/infer_results/semi-labeling/step*/`.

## Lưu ý khi sửa code

- `preload.cjs` là CommonJS (Electron preload không support ESM). File khác trong `electron/` dùng ESM.
- Dùng `node:sqlite` (Electron Node 22+), không phải `better-sqlite3`. Mở read-only.
- `contextIsolation: true` — renderer không có `require`. Node API phải expose qua `preload.cjs`.
- Vite base `'./'` cần cho Electron `file://`.
- Active tab dùng local `useState`, không persist qua session.
- Khi thêm IPC: thêm handler ở `main.js` + expose ở `preload.cjs` + (nếu có path) thêm default ở `defaults.js`.
