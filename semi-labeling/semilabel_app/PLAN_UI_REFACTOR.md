# Plan: Tối ưu giao diện Semi-Labeling App

## Mục tiêu

1. Ít nút nhất có thể
2. UX ngon nhất có thể
3. Layout bố trí phù hợp và đều

---

## Phân tích hiện trạng

App hiện có 4 tab + 1 dialog kết nối:

| Vấn đề | Cụ thể |
|--------|--------|
| **Quá nhiều nút** | Mỗi trang Review có 8 controls, Prototype có 6 nút. Hàm `_button()` ignore param `primary`/`danger` |
| **UX rời rạc** | Quyết định label nằm ở panel phải (260px), xa khu vực nhìn ảnh. Phải move chuột qua lại |
| **Layout cứng** | Splitter `[320, 760, 260]` fix cứng, panel phải hẹp, text metadata thường bị cắt |
| **ui_kit.py bỏ không** | Có sẵn `Card`, `Toolbar`, `InfoPanel`, `Chip`, `PercentBar` nhưng không trang nào dùng |
| **Không có phím tắt** | Người dùng phải click chuột cho mọi thao tác |
| **Thiếu progress** | Không biết đã review được bao nhiêu / tổng bao nhiêu |

---

## 1. Giảm nút xuống tối thiểu

### Review page & QA page (chung class `ReviewPage`)

**Hiện tại:** `Filter combo + Limit spin + Write JSON + 4 label buttons + pending_label = 8 controls`

**Sau sửa:**
- **Top bar:** `Filter combo | Limit spin | Progress bar` (3 controls)
- **Dưới ảnh:** 4 label buttons có màu + hint phím tắt (`1` `2` `3` `4`) → 1 row ngang
- **Panel phải:** PickedList (pending list clickable) + nút "Write JSON (N)" cuối panel
- **Xóa:** `pending_label` riêng lẻ, `save_btn` trên top bar cũ

**Tổng:** 8 controls → 4 controls

**⚠️ Fix đi kèm:** `_button()` hiện tại ignore param `primary`/`danger` → xóa hàm này,  
xài `primary_button()` / `danger_button()` từ `ui_kit.py`, hoặc viết helper mới trong page.

### Prototype page

**Hiện tại:** `Label combo + Pick + Pick reject + Write JSON + Count label + Domain hint = 6 controls`

**Sau sửa:**
- **Top bar:** `Label combo | Domain hint | Picks count | Write JSON` (4 controls gọn)
- **Dưới ảnh:** 2 nút "Pick as [label]" + "Reject" (có hint phím tắt `Enter` / `R`)
- **Xóa:** `pick_btn`, `reject_btn` rời, `count`, `domain_hint` rải rác

**Tổng:** 6 controls → 5 controls (gộp đẹp hơn)

**⚠️ Performance:** Mỗi lần `pick()` gọi `refresh()` → chạy lại `_diverse_order()` toàn bộ O(N log N).  
Với 500+ candidates, mỗi lần pick/unpick tốn CPU không cần thiết. Giải pháp:

```python
# Cache _diverse_order result per label, chỉ invalidate khi load() lại từ DB
self._diverse_cache: dict[str, list[Any]] = {}
```

### Images page

**Hiện tại:** `Source combo + Limit spin + Reload + Summary = 4 controls`

- **Xóa nút "Reload"** — combo và spin đã tự reload khi đổi giá trị, nút này thừa

**Tổng:** 4 → 3 controls

### Connection bar

**Hiện tại:** `QLabel "Connected" + connection_label (dài) + Change connection button`

→ **Bỏ qua, không sửa.** Connection bar hiện tại 1 dòng gọn rồi, thay đổi không đáng kể,  
giữ nguyên để giảm scope.

### Tổng kết

**~22 controls → ~14 controls (giảm ~36%)**

---

## 2. UX cải tiến

### 2.1 Decision bar dưới ảnh (thay vì panel phải)

```
┌─────────────────────────────────────┐
│                                     │
│          ẢNH (BoxImage)             │
│                                     │
├─────────────────────────────────────┤
│  [1] Crack  [2] Mold  [3] Spall   [4] Reject  │  ← màu, kèm phím tắt
│  #1024  crack  0.8723              │  ← caption
└─────────────────────────────────────┘
```

- Bấm nút (hoặc gõ phím 1/2/3/4) → tự động nhảy sang item tiếp theo
- Phím `Z` để undo quyết định gần nhất
- Phím `Space` / `↓` để skip qua item kế

### 2.2 Phím tắt toàn cục

| Tab | Phím | Hành động |
|-----|------|-----------|
| Review/QA | `1` `2` `3` `4` | Chọn label crack/mold/spall/reject |
| Review/QA | `Space` `↓` | Nhảy item kế |
| Review/QA | `↑` | Nhảy item trước |
| Review/QA | `Z` | Undo quyết định gần nhất |
| Review/QA | `Ctrl+S` | Write JSON |
| Prototype | `Enter` | Pick as current label |
| Prototype | `R` | Reject |
| Prototype | `Space` `↓` | Nhảy item kế |
| Prototype | `U` | Unpick item hiện tại |
| Toàn cục | `Ctrl+1-4` | Chuyển tab |

### ⚠️ 2.2b Vấn đề keyboard focus

`QComboBox` (Filter) và `QSpinBox` (Limit) khi có focus sẽ **nuốt phím 1-4, Space, ↑↓**.  
Phải dùng `QShortcut` với context `Qt.ShortcutContext.WindowShortcut` hoặc `ApplicationShortcut`  
để shortcut hoạt động bất kể widget nào đang focus. Hoặc cài `eventFilter` trên window để chặn  
phím trước khi tới widget con.

```python
# Ví dụ: shortcut luôn hoạt động dù QComboBox đang focus
QtWidgets.QShortcut(QtGui.QKeySequence("1"), self, context=Qt.ShortcutContext.WindowShortcut)
```

### 2.3 Panel phải chuyển thành info-only

Dùng `InfoPanel` từ `ui_kit.py` hiển thị metadata dạng key-value gọn gàng. Dùng `Card` bọc các section:

```
┌─ Item Info ──────────┐
│ ID:       1024       │
│ Image:    DSC01.png  │
│ Label:    crack      │
│ Score:    0.8723     │
│ Reasons:  ...        │
├─ Decision ───────────┤
│ Selected: crack      │
├─ Progress ───────────┤
│ ████████░░  45/500   │
├─ Pending ────────────┤
│ #1021  mold→crack  ✕ │
│ #1018  spall→reject ✕ │
│ [Write JSON (2)]     │
└──────────────────────┘
```

### 2.4 Progress bar

Thêm `QProgressBar` trong panel phải hiển thị `đã quyết định / tổng`.

**⚠️ Vấn đề reset:** Sau `write_json()` → `self.pending.clear()` → progress về 0 nếu tính bằng `len(pending)`.

Hai hướng giải quyết:

| Hướng | Cách làm | Pros/Cons |
|-------|----------|-----------|
| **Session progress** | Giữ `self._decided_ids: set[int]` riêng, không clear khi write, hiển thị `len(_decided_ids) / total` | Chính xác tiến độ thật, nhưng cần thêm 1 biến nữa |
| **Batch progress** | Bỏ progress bar, chỉ hiển thị `"Pending: N / M"` dạng text đơn giản | Ít code hơn, đỡ phức tạp |

→ **Chọn session progress** vì biết đã review qua bao nhiêu item là thông tin quan trọng.

### 2.5 Undo / Unpick

- Click vào item trong PickedList để bỏ chọn
- Phím `Z` để undo decision gần nhất

**⚠️ Implementation detail:** `self.pending` là `dict[int, dict]` — key là `result_id`, không có thứ tự.  
Cần thêm `deque[int]` track thứ tự insert:

```python
self._pending_order: deque[int] = deque()  # result_id theo thứ tự quyết định
```

Khi undo: `pop()` cái cuối, xóa key trong `self.pending`, jump list về item đó.

---

## 3. Layout đều và hợp lý

### 3.1 Splitter tỉ lệ động

```python
# Thay vì fix cứng [320, 760, 260]
# Dùng setStretchFactor để responsive
split.setStretchFactor(0, 2)   # list panel:  ~22%
split.setStretchFactor(1, 5)   # image panel: ~56%
split.setStretchFactor(2, 2)   # info panel:  ~22%
```

### 3.2 Center panel bố cục dọc

```
┌─ BoxImage (stretch=1) ─────────────┐
├─ Decision bar (fixed height=40) ───┤
└────────────────────────────────────┘
```

**⚠️ Cần wrapper widget:** `BoxImage` là widget vẽ ảnh thuần (paintEvent), không chứa layout.  
Phải tạo 1 `QWidget` container bọc `BoxImage` + `QFrame` (decision bar) trong `QVBoxLayout`:

```python
center = QtWidgets.QWidget()
center_layout = QtWidgets.QVBoxLayout(center)
center_layout.setContentsMargins(0, 0, 0, 0)
center_layout.setSpacing(0)
center_layout.addWidget(self.image, 1)    # BoxImage — giãn hết không gian
center_layout.addWidget(decision_bar)      # QFrame — fixed height 40-44px
split.addWidget(center)                    # thay vì split.addWidget(self.image)
```

### 3.3 Sử dụng ui_kit components

- `Card` → bọc từng section trong panel phải
- `InfoPanel` → hiển thị metadata
- `Toolbar` → top bar mỗi page
- `Chip` → hiển thị count/pending badges
- `PercentBar` → hiển thị progress

### 3.4 Style nút label có màu đồng bộ

```python
LABEL_BUTTON_STYLE = {
    "crack":  ("#4a90d9", "Crack [1]"),
    "mold":   ("#27ae60", "Mold [2]"),
    "spall":  ("#f39c12", "Spall [3]"),
    "reject": ("#e74c3c", "Reject [4]"),
}
```

Mỗi nút có màu nền nhạt + viền đậm theo label color, text kèm phím tắt.

**⚠️ Cách implement:** Dùng QSS (`setStyleSheet`) cho nút — đơn giản, dễ maintain,  
khác với `BoxImage.paintEvent` dùng QPainter (vẽ ảnh + boxes phức tạp thì QPainter hợp lý hơn).  
Nguyên tắc: QSS cho widget đơn giản (nút, frame, label), QPainter cho widget vẽ đồ họa.

---

## 4. Các file cần sửa (theo thứ tự)

| File | Thay đổi |
|------|----------|
| `ui/widgets/ui_kit.py` | Sửa `primary_button`/`danger_button` cho dễ xài; thêm `DecisionBar` widget (row ngang chứa nút label có màu); `PickedList` đã có sẵn |
| `ui/main_window.py` | Refactor lớn nhất: layout lại toàn bộ 4 page + MainWindow; xóa `_button()`; thêm keyboard shortcuts; decision bar; panel phải info-only; progress |
| `ui/widgets/box_image.py` | Thêm method `set_decision_state(label)` để hiển thị indicator trên ảnh (optional improvement) |
| `config/defaults.py` | Không cần sửa (shortcut map hardcode trong page, không cần config) |

---

## 5. Thứ tự thực hiện

1. **Refactor `ReviewPage`** trước (quan trọng nhất, dùng nhiều nhất)
   - Gộp 4 label buttons thành decision bar dưới ảnh
   - Panel phải → info-only với InfoPanel + Card
   - Thêm keyboard shortcuts
   - Thêm progress bar
   - Thêm undo

2. **Refactor `QA page`** (chung class nên tự động được)

3. **Refactor `PrototypePage`**
   - Decision bar dưới ảnh (2 nút thay vì 4 nút rời)
   - Panel phải gọn hơn
   - Keyboard shortcuts

4. **Refactor `ImageOverviewPage`**
   - Xóa nút Reload thừa
   - Cải thiện panel phải

5. **Tinh chỉnh `MainWindow`**
   - Global keyboard shortcut dispatch (install `eventFilter` hoặc QShortcut)
   - Splitter responsive (stretch factor thay vì pixel cứng)
   - Connection bar giữ nguyên

6. **Style tổng thể** — đảm bảo consistent spacing, màu sắc, font

---

## 6. Trước / Sau (so sánh nhanh)

### Review page

| Thành phần | Trước | Sau |
|------------|-------|-----|
| Top controls | 6 widgets (Filter, Limit, spacer, Pending label, Write JSON) | 3 widgets (Filter, Limit, Progress bar) |
| Label buttons | 4 nút trong panel phải | 4 nút dưới ảnh, có màu |
| Write JSON | Nút trên top | Nút trong panel phải (gần pending list) |
| Metadata | QLabel text dump | InfoPanel dạng key-value |
| Pending | Chỉ hiện số | Hiển thị list có nút bỏ chọn |
| Phím tắt | Không có | 1-4, Space, Z, Ctrl+S |

### Prototype page

| Thành phần | Trước | Sau |
|------------|-------|-----|
| Top controls | 6 widgets | 4 widgets |
| Action buttons | 2 nút trong top bar | 2 nút dưới ảnh |
| Decision feedback | QLabel text | Inline caption dưới nút |
| Phím tắt | Không có | Enter, R, Space, U |

### Images page

| Thành phần | Trước | Sau |
|------------|-------|-----|
| Top controls | 4 widgets | 3 widgets (xóa Reload) |
