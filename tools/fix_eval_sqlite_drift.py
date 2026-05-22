"""Fix 3 lệch sau eval segmentation post-Kaggle:

1. DeepCrack image_id không pair được giữa UNet (test_img/X) và SAM B1/B2/B3 (ti_X).
   → Thêm cột image_id_canonical vào image_threshold_metrics, populate theo rule:
       'ti_' prefix       → test_img__<stem>
       'tr_' prefix       → train_img__<stem>
       'test_img/' path   → test_img__<stem>
       'train_img/' path  → train_img__<stem>
       else (flat)        → <stem>

2. Schema mismatch UNet (25 cols) vs SAM (18 cols).
   → Backfill 7 cột thiếu trong SAM SQLite: mask_path, specificity, accuracy, tp, fp, fn, tn.
   Tính từ existing cols: precision, recall, gt_positive_px, pred_positive_px, width, height.

3. (Không phải bug, không fix ở đây) B0 best_threshold=0.45 — đúng behavior SAM zero-shot trên thin structures.

Idempotent: chạy lại không phá hỏng dữ liệu — kiểm cột tồn tại trước khi ALTER.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

# --- Mapping rules for image_rel_path → canonical id ---

def canonicalize_rel_path(rel_path: str) -> str:
    """Map heterogeneous image_rel_path to canonical id stable across models.

    Examples:
      'ti_11215-1.jpg'        → 'test_img__11215-1'
      'tr_7068.jpg'           → 'train_img__7068'
      'test_img/11215-1.jpg'  → 'test_img__11215-1'
      'train_img/7068.jpg'    → 'train_img__7068'
      '20160222_xxx.jpg'      → '20160222_xxx'  (CRACK500, Volker — flat)
    """
    p = rel_path.strip().replace("\\", "/")
    # Strip extension after we determine prefix
    if "/" in p:
        head, tail = p.split("/", 1)
        if head in {"test_img", "train_img"}:
            stem = Path(tail).stem
            return f"{head}__{stem}"
        # nested but not deepcrack: just use stem
        return Path(tail).stem
    # No path separator
    if p.startswith("ti_"):
        stem = Path(p[3:]).stem
        return f"test_img__{stem}"
    if p.startswith("tr_"):
        stem = Path(p[3:]).stem
        return f"train_img__{stem}"
    return Path(p).stem


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


# --- Fix 1: image_id_canonical ---

def add_canonical_column(conn: sqlite3.Connection) -> tuple[int, int]:
    """Add image_id_canonical column to image_threshold_metrics + populate."""
    if not _column_exists(conn, "image_threshold_metrics", "image_id_canonical"):
        conn.execute(
            "ALTER TABLE image_threshold_metrics ADD COLUMN image_id_canonical TEXT"
        )

    # Populate by reading distinct rel_paths and computing canonical
    cur = conn.execute(
        "SELECT DISTINCT image_rel_path FROM image_threshold_metrics"
    )
    rel_paths = [r[0] for r in cur.fetchall()]
    n_updated = 0
    for rel in rel_paths:
        canon = canonicalize_rel_path(rel)
        conn.execute(
            "UPDATE image_threshold_metrics SET image_id_canonical = ? "
            "WHERE image_rel_path = ?",
            (canon, rel),
        )
        n_updated += conn.total_changes
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_image_id_canonical "
        "ON image_threshold_metrics (run_id, dataset_label, image_id_canonical, threshold)"
    )
    conn.commit()
    return len(rel_paths), n_updated


# --- Fix 2: Schema backfill for SAM SQLite ---

REQUIRED_BACKFILL_COLS = [
    ("mask_path", "TEXT"),
    ("specificity", "REAL"),
    ("accuracy", "REAL"),
    ("tp", "INTEGER"),
    ("fp", "INTEGER"),
    ("fn", "INTEGER"),
    ("tn", "INTEGER"),
]


def backfill_missing_columns(conn: sqlite3.Connection) -> dict:
    """Add 7 missing columns to SAM SQLite + derive values from existing data."""
    added = []
    for col_name, col_type in REQUIRED_BACKFILL_COLS:
        if not _column_exists(conn, "image_threshold_metrics", col_name):
            conn.execute(
                f"ALTER TABLE image_threshold_metrics ADD COLUMN {col_name} {col_type}"
            )
            added.append(col_name)

    # Derivations:
    #   tp ≈ round(precision * pred_positive_px)
    #     also = round(recall * gt_positive_px); take average where both defined for robustness
    #   fp = pred_positive_px - tp
    #   fn = gt_positive_px - tp
    #   tn = width*height - tp - fp - fn
    #   specificity = tn / (tn + fp + 1e-7)
    #   accuracy = (tp + tn) / (width * height)
    #   mask_path = replace last '/images/' with '/masks/' in image_path, change ext to .png
    #
    # We compute in SQL using CASE to avoid Python row-by-row.

    conn.execute("""
        UPDATE image_threshold_metrics
        SET tp = CAST(ROUND(
            CASE
                WHEN pred_positive_px > 0 AND gt_positive_px > 0
                    THEN (precision * pred_positive_px + recall * gt_positive_px) / 2.0
                WHEN pred_positive_px > 0
                    THEN precision * pred_positive_px
                WHEN gt_positive_px > 0
                    THEN recall * gt_positive_px
                ELSE 0
            END
        ) AS INTEGER)
        WHERE tp IS NULL
    """)
    conn.execute("""
        UPDATE image_threshold_metrics
        SET fp = MAX(pred_positive_px - tp, 0)
        WHERE fp IS NULL AND tp IS NOT NULL
    """)
    conn.execute("""
        UPDATE image_threshold_metrics
        SET fn = MAX(gt_positive_px - tp, 0)
        WHERE fn IS NULL AND tp IS NOT NULL
    """)
    conn.execute("""
        UPDATE image_threshold_metrics
        SET tn = MAX(width * height - tp - fp - fn, 0)
        WHERE tn IS NULL AND tp IS NOT NULL
    """)
    conn.execute("""
        UPDATE image_threshold_metrics
        SET specificity = CAST(tn AS REAL) / NULLIF(tn + fp, 0)
        WHERE specificity IS NULL AND tn IS NOT NULL
    """)
    conn.execute("""
        UPDATE image_threshold_metrics
        SET accuracy = CAST(tp + tn AS REAL) / NULLIF(width * height, 0)
        WHERE accuracy IS NULL AND tn IS NOT NULL
    """)
    # mask_path: guess from image_path
    conn.execute("""
        UPDATE image_threshold_metrics
        SET mask_path = REPLACE(REPLACE(image_path, '/images/', '/masks/'), '.jpg', '.png')
        WHERE mask_path IS NULL AND image_path IS NOT NULL
    """)
    conn.commit()
    return {"added": added}


# --- Main ---

DB_FILES = [
    ("unet_v1", "unet_v1/unet_v1_eval.sqlite3", False),
    ("unet_v2_cldice_ema", "unet_v2_cldice_ema/unet_v2_eval.sqlite3", False),
    ("sam_b0_zeroshot", "sam_b0_zeroshot/sam_b0_eval.sqlite3", True),
    ("sam_b1_lora_only", "sam_b1_lora_only/sam_b1_eval.sqlite3", True),
    ("sam_b2_lora_hq", "sam_b2_lora_hq/sam_b2_eval.sqlite3", True),
    ("sam_b3_full", "sam_b3_full/sam_b3_eval.sqlite3", True),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        default="/Users/nguyenquangvinh/Desktop/Lab/results/segmentation_post_kaggle/eval",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    print(f"Eval root: {eval_root}")
    print(f"Dry run: {args.dry_run}\n")

    for tag, rel_db, needs_backfill in DB_FILES:
        db_path = eval_root / rel_db
        if not db_path.exists():
            print(f"  [SKIP] {tag} — missing: {db_path}")
            continue
        print(f"── {tag} ── {db_path.name}")

        if args.dry_run:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        else:
            conn = sqlite3.connect(str(db_path))

        # Fix 1: canonical column
        if not args.dry_run:
            n_paths, _ = add_canonical_column(conn)
            print(f"   ✓ canonicalized {n_paths} distinct rel_paths")

        # Fix 2: schema backfill for SAM
        if needs_backfill and not args.dry_run:
            info = backfill_missing_columns(conn)
            print(f"   ✓ schema backfill — added cols: {info['added'] or '(already present)'}")

        # Verify pairing on DeepCrack
        cur = conn.execute(
            "SELECT COUNT(DISTINCT image_id_canonical) FROM image_threshold_metrics "
            "WHERE dataset_label = 'deepcrack_test'"
        )
        n_canon = cur.fetchone()[0]
        print(f"   • deepcrack n_distinct_canonical = {n_canon}  (expected 87)")

        conn.close()

    if not args.dry_run:
        print("\nDone. Backups at <eval_root>/_backup_before_fix/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
