# CLAUDE.md — app (Electron + React)

This file provides guidance to Claude Code when working in `app/`.

## Vai trò của module

`app/` là **desktop application** dạng Electron + React. Nó phục vụ 2 mục đích hoàn toàn khác nhau:

1. **Workflows Tab**: Chạy Python scripts tùy chỉnh (training, evaluation, data processing) từ GUI
2. **Result Viewer**: Xem và đánh giá kết quả data splitting từ SQLite database của `data_split`

**Lưu ý:** App này **không** phải là interface cho real-time inference. Inference được xử lý bởi PySide6 editor app (`ui/`).

---

## Stack

| Layer | Technology |
|-------|-----------|
| Desktop shell | Electron 42 |
| Frontend | React 19 + Redux Toolkit + Tailwind CSS |
| Bundler | Vite + @vitejs/plugin-react |
| Icons | @tabler/icons-react |
| IPC | Electron contextBridge (preload.cjs) |

---

## Dev commands

```bash
cd app
npm install

npm run dev          # Vite dev server (5173) + Electron, cả hai cùng lúc
npm run dev:react    # Chỉ Vite (không có Electron, cho UI dev nhanh)
npm run build        # Build production → dist/
npm run lint         # ESLint
npm start            # Chạy Electron load từ dist/ (cần build trước)
```

---

## Cấu trúc source

```
app/src/
├── App.jsx                      # Root: tab navigation (Workflows | Results | Prototypes | Settings)
├── app/store.js                 # Redux store: {workflows, resultViewer}
├── components/ui/               # Reusable dumb components (Button, Field, Badge, ...)
└── features/
    ├── workflows/               # Tab 1: chạy Python workflows
    │   ├── workflowsSlice.js    # Redux state
    │   └── components/
    │       ├── WorkflowsTab.jsx # List workflows bên trái
    │       ├── WorkflowForm.jsx # Form nhập params
    │       └── WorkflowTerminal.jsx  # stdout/stderr stream
    ├── resultViewer/            # Tab 2: xem kết quả data split
    │   ├── resultViewerSlice.js # Redux state
    │   ├── hooks/useResultViewer.js
    │   └── components/
    │       ├── ConnectView.jsx   # Nhập paths → Load
    │       ├── ResultsView.jsx   # Danh sách runs + cluster list
    │       ├── ClusterDetailView.jsx  # Image grid cho 1 cluster
    │       └── ...
    ├── prototypeReview/         # Tab 3: review DINOv2 prototypes
    └── settings/                # Tab 4: lưu preferences vào localStorage

app/electron/
├── main.js          # Main process: IPC handlers + workflow subprocess management
├── preload.cjs      # contextBridge → window.electronAPI
└── result_viewer/   # SQLite query helpers (Node.js)
    ├── db.js        # better-sqlite3 connection
    ├── queries.js   # SELECT queries
    └── mutations.js # UPDATE queries (flag management)
```

---

## Redux State

### workflowsSlice

```javascript
{
  items: [],           // available workflows từ workflows/ folder
  sessions: [],        // running/done sessions
  // session: {id, workflowId, status: "running"|"done"|"error", log: []}
  selectedWorkflowId: null,
  formValues: {},      // input values cho form
  sidebarOpen: true,
}
```

### resultViewerSlice

```javascript
{
  paths: {featureDbPath, sourceDbPath, imageRootPath},
  screen: "connect" | "results" | "detail",
  runs: [],
  selectedRunId: null,
  selectedLabel: "crack",       // "crack" | "mold" | "spall"
  clustersByLabel: {},          // {label: [{cluster_id, size, ...}]}
  selectedCluster: null,
  assignments: [],              // images trong cluster đang chọn
  mode: "grid" | "table",
  imageSize: 160,
  settingsOpen: false,
}
```

---

## IPC bridge (window.electronAPI)

Tất cả IPC calls đều qua `preload.cjs` → `window.electronAPI`:

```javascript
// Thông tin app
electronAPI.getVersion()
electronAPI.getDownloadsPath()

// Workflows
electronAPI.listWorkflows()
electronAPI.startWorkflow({workflowId, values, venvDir, useGlobalPython})  → sessionId
electronAPI.stopWorkflow(sessionId)
// Events stream qua: ipcRenderer.on("workflow:event", handler)
// Event: {sessionId, type: "started"|"stdout"|"stderr"|"closed", data: str}

// Result Viewer
electronAPI.getResultViewerDefaults()
electronAPI.listResultViewerRuns({featureDbPath})
electronAPI.listResultViewerClusters({featureDbPath, runId, label})
electronAPI.listResultViewerAssignments({featureDbPath, runId, clusterId})
electronAPI.clearResultViewerClusterFlags({featureDbPath, clusterId})

// Dialog
electronAPI.browsePath(mode)  // mode: "file" | "directory"
```

---

## Workflow execution

Mỗi workflow được định nghĩa trong `workflows/` folder (tại root DamageDetector) dưới dạng JSON metadata. Khi user click "Run":

```javascript
// main.js spawns:
spawn("python", ["-m", "workflows", "run", workflowId, "--values-json", valuesPath], {
  cwd: repoRoot,
  env: { ...process.env, PYTHONPATH: repoRoot }
})
// stdout/stderr được stream realtime qua IPC → WorkflowTerminal.jsx
```

---

## Data flow: Result Viewer

```
User nhập paths → click "Load"
    ↓
fetchResultViewerRuns()  →  electronAPI.listResultViewerRuns()
                         →  main.js query SQLite
    ↓
Screen chuyển sang "results"
User chọn run + label
    ↓
fetchResultViewerClusters()  →  listResultViewerClusters()
                             →  clustersByLabel[label] = [...]
    ↓
User click cluster
    ↓
openResultViewerCluster()   →  listResultViewerAssignments()
                            →  assignments = [{image_path, label, ...}]
Screen chuyển sang "detail"
    ↓
ClusterDetailView render ImageGrid với thumbnails từ imageRootPath
```

---

## Styling conventions

- Dùng Tailwind utility classes trực tiếp trong JSX
- Theme variables qua CSS custom properties (xem `styles.css`)
- Icon: `@tabler/icons-react` — import `{ IconName } from "@tabler/icons-react"`
- Reusable components trong `components/ui/` — dùng `cn()` từ `cn.js` để merge classnames

---

## Lưu ý khi sửa code

- `preload.cjs` dùng CommonJS (`.cjs` extension) vì Electron preload không support ESM. Tất cả file khác trong `electron/` dùng ESM.
- `better-sqlite3` (Node.js SQLite) được dùng trong main process — không phải Python SQLite. Schema phải match với schema từ `data_split/export.py` và `object_detection/damage_scan/sqlite_store.py`.
- Electron `contextIsolation: true` — renderer process không có `require`. Mọi Node.js API phải expose qua `preload.cjs`.
- Vite base `'./'` trong `vite.config.js` là cần thiết cho Electron file:// protocol.
- Sessions trong `workflowsSlice` được store trong memory — khi reload app, history mất. Đây là by-design.
