"""
Combine two datasets (main + secondary) into a new folder.

Rules:
  - Train from secondary merges into the main's train split.
  - Test and query from secondary become the validation set (val_query, val_test).
  - Test (and query) from the main remain as the test (and query) set.
  - Secondary image IDs, objectIDs, and camIDs are shifted to avoid collisions.

Both datasets are expected to have:
  - train.csv, test.csv, query.csv  (columns: cameraID, imageName, objectID)
  - train_classes.csv, test_classes.csv, query_classes.csv  (columns: cameraID, imageName, objectID, Class)
  - image_train/, image_test/, image_query/  folders with numbered .jpg files
"""

import os
import csv
import shutil
import argparse
from pathlib import Path

# The main dataset may use 'Corresponding Indexes' while the secondary uses 'objectID'.
# We normalise everything to 'objectID'.
OBJ_COL_ALIASES = ["objectID", "Corresponding Indexes"]


def read_csv(path):
    """Read a CSV file and return header + rows."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    return header, rows


def write_csv(path, header, rows):
    """Write header + rows to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def get_max_image_id(image_dir):
    """Return the maximum numeric image ID found in a directory of NNNNNN.jpg files."""
    max_id = 0
    if not os.path.isdir(image_dir):
        return max_id
    for fname in os.listdir(image_dir):
        stem = os.path.splitext(fname)[0]
        try:
            max_id = max(max_id, int(stem))
        except ValueError:
            continue
    return max_id


def find_obj_col(header):
    """Find the objectID column index, trying known aliases. Returns (index, name) or (None, None)."""
    for alias in OBJ_COL_ALIASES:
        if alias in header:
            return header.index(alias), alias
    return None, None


def normalize_header(header):
    """Rename 'Corresponding Indexes' -> 'objectID' so all outputs are consistent."""
    return ["objectID" if h in OBJ_COL_ALIASES else h for h in header]


def get_max_object_id(csv_path):
    """Return the maximum objectID from a CSV (handles both column name variants)."""
    if not os.path.isfile(csv_path):
        return 0
    header, rows = read_csv(csv_path)
    idx, _ = find_obj_col(header)
    if idx is None:
        return 0  # no objectID column in this CSV
    max_id = 0
    for row in rows:
        try:
            max_id = max(max_id, int(row[idx]))
        except (ValueError, IndexError):
            continue
    return max_id


def get_max_cam_id(csv_path, cam_col="cameraID"):
    """Return the maximum camera ID number (e.g. 'c004' -> 4) from a CSV."""
    if not os.path.isfile(csv_path):
        return 0
    header, rows = read_csv(csv_path)
    idx = header.index(cam_col)
    max_id = 0
    for row in rows:
        try:
            cam_str = row[idx]
            # Expect format like 'c001', 'c012', etc.
            max_id = max(max_id, int(cam_str.lstrip("c")))
        except (ValueError, IndexError):
            continue
    return max_id


def shift_csv_rows(rows, header, img_shift, obj_shift, cam_shift,
                   img_col="imageName", cam_col="cameraID"):
    """
    Shift image names, objectIDs, and cameraIDs in the rows.
    Returns new rows with shifted values.
    """
    img_idx = header.index(img_col)
    cam_idx = header.index(cam_col)
    obj_idx, _ = find_obj_col(header)

    new_rows = []
    for row in rows:
        new_row = list(row)

        # Shift imageName: '000123.jpg' -> shift by img_shift
        old_name = new_row[img_idx]
        stem, ext = os.path.splitext(old_name)
        new_id = int(stem) + img_shift
        new_row[img_idx] = f"{new_id:06d}{ext}"

        # Shift objectID
        if obj_idx is not None:
            new_row[obj_idx] = str(int(new_row[obj_idx]) + obj_shift)

        # Shift cameraID: 'c001' -> 'c001+cam_shift'
        old_cam = new_row[cam_idx]
        cam_num = int(old_cam.lstrip("c")) + cam_shift
        new_row[cam_idx] = f"c{cam_num:03d}"

        new_rows.append(new_row)
    return new_rows


def copy_images_shifted(src_dir, dst_dir, img_shift):
    """Copy images from src_dir to dst_dir, shifting filenames by img_shift."""
    if not os.path.isdir(src_dir):
        return
    for fname in sorted(os.listdir(src_dir)):
        stem, ext = os.path.splitext(fname)
        try:
            new_id = int(stem) + img_shift
        except ValueError:
            # Non-numeric filename, copy as-is
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
            continue
        new_name = f"{new_id:06d}{ext}"
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, new_name))


def copy_images(src_dir, dst_dir):
    """Copy all images from src_dir to dst_dir without renaming."""
    if not os.path.isdir(src_dir):
        return
    for fname in sorted(os.listdir(src_dir)):
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))


def main():
    parser = argparse.ArgumentParser(description="Combine two ReID datasets.")
    parser.add_argument("--main", type=str,
                        default="/scratch/dr/Urban-ReID/Datasets/Urban2026",
                        help="Path to the main dataset")
    parser.add_argument("--secondary", type=str,
                        default="/scratch/dr/Urban-ReID/Datasets/UAM_Unified",
                        help="Path to the secondary dataset")
    parser.add_argument("--output", type=str,
                        default="/scratch/dr/Urban-ReID/Combined_dataset",
                        help="Path for the combined output dataset")
    args = parser.parse_args()

    main_dir = Path(args.main)
    sec_dir = Path(args.secondary)
    out_dir = Path(args.output)

    # Verify secondary dataset exists
    assert sec_dir.is_dir(), f"Secondary dataset not found: {sec_dir}"

    # ---- Determine shifts based on main dataset maximums ----
    # Collect max IDs across all main splits
    main_max_img = 0
    main_max_obj = 0
    main_max_cam = 0

    main_has_data = main_dir.is_dir() and any(
        (main_dir / f).is_file() for f in ["train.csv", "test.csv", "query.csv"]
    )

    if main_has_data:
        for split in ["train", "test", "query"]:
            csv_path = main_dir / f"{split}.csv"
            img_dir = main_dir / f"image_{split}"
            if csv_path.is_file():
                main_max_obj = max(main_max_obj, get_max_object_id(str(csv_path)))
                main_max_cam = max(main_max_cam, get_max_cam_id(str(csv_path)))
            if img_dir.is_dir():
                main_max_img = max(main_max_img, get_max_image_id(str(img_dir)))

    # Shifts: start secondary IDs right after the main's max
    img_shift = main_max_img   # secondary image 000001 -> main_max_img + 1
    obj_shift = main_max_obj   # secondary objectID 1     -> main_max_obj + 1
    cam_shift = main_max_cam   # secondary cam c001       -> c(main_max_cam + 1)

    print(f"Main dataset: {main_dir}")
    print(f"  Has data: {main_has_data}")
    print(f"  Max image ID: {main_max_img}")
    print(f"  Max objectID: {main_max_obj}")
    print(f"  Max cameraID: c{main_max_cam:03d}")
    print()
    print(f"Secondary dataset: {sec_dir}")
    print(f"  Shifts: img_shift={img_shift}, obj_shift={obj_shift}, cam_shift={cam_shift}")
    print()
    print(f"Output: {out_dir}")
    print()

    # ---- Create output directory structure ----
    os.makedirs(out_dir, exist_ok=True)
    for d in ["image_train", "image_test", "image_query", "image_val_test", "image_val_query"]:
        os.makedirs(out_dir / d, exist_ok=True)

    # ================================================================
    # 1. TRAIN: main train + shifted secondary train
    # ================================================================
    print("Processing TRAIN split...")

    # Copy main train images and CSV
    if main_has_data and (main_dir / "train.csv").is_file():
        main_train_header, main_train_rows = read_csv(str(main_dir / "train.csv"))
        main_train_header = normalize_header(main_train_header)
        copy_images(str(main_dir / "image_train"), str(out_dir / "image_train"))
        print(f"  Main train: {len(main_train_rows)} rows copied")
    else:
        main_train_header = ["cameraID", "imageName", "objectID"]
        main_train_rows = []
        print("  Main train: empty (no data)")

    # Handle _classes.csv for main train
    if main_has_data and (main_dir / "train_classes.csv").is_file():
        main_tcls_header, main_tcls_rows = read_csv(str(main_dir / "train_classes.csv"))
        main_tcls_header = normalize_header(main_tcls_header)
    else:
        main_tcls_header = ["cameraID", "imageName", "objectID", "Class"]
        main_tcls_rows = []

    # Read secondary train
    sec_train_header, sec_train_rows = read_csv(str(sec_dir / "train.csv"))
    sec_train_shifted = shift_csv_rows(
        sec_train_rows, sec_train_header, img_shift, obj_shift, cam_shift
    )

    # Per-split image shift for secondary train
    copy_images_shifted(
        str(sec_dir / "image_train"), str(out_dir / "image_train"), img_shift
    )
    print(f"  Secondary train: {len(sec_train_shifted)} rows shifted and merged")

    # Merge train CSVs (ensure same header)
    combined_train_rows = main_train_rows + sec_train_shifted
    write_csv(str(out_dir / "train.csv"), main_train_header, combined_train_rows)

    # Handle train_classes.csv
    if (sec_dir / "train_classes.csv").is_file():
        sec_tcls_header, sec_tcls_rows = read_csv(str(sec_dir / "train_classes.csv"))
        sec_tcls_shifted = shift_csv_rows(
            sec_tcls_rows, sec_tcls_header, img_shift, obj_shift, cam_shift
        )
        combined_tcls_rows = main_tcls_rows + sec_tcls_shifted
        write_csv(str(out_dir / "train_classes.csv"), main_tcls_header, combined_tcls_rows)
        print(f"  train_classes.csv: {len(combined_tcls_rows)} total rows")

    # ================================================================
    # 2. TEST: main test stays as test
    # ================================================================
    print("\nProcessing TEST split (main only)...")

    if main_has_data and (main_dir / "test.csv").is_file():
        main_test_header, main_test_rows = read_csv(str(main_dir / "test.csv"))
        write_csv(str(out_dir / "test.csv"), main_test_header, main_test_rows)
        copy_images(str(main_dir / "image_test"), str(out_dir / "image_test"))
        print(f"  Main test: {len(main_test_rows)} rows")
    else:
        print("  Main test: empty (no data)")

    if main_has_data and (main_dir / "test_classes.csv").is_file():
        main_testcls_header, main_testcls_rows = read_csv(str(main_dir / "test_classes.csv"))
        write_csv(str(out_dir / "test_classes.csv"), main_testcls_header, main_testcls_rows)

    # ================================================================
    # 3. QUERY: main query stays as query
    # ================================================================
    print("Processing QUERY split (main only)...")

    if main_has_data and (main_dir / "query.csv").is_file():
        main_query_header, main_query_rows = read_csv(str(main_dir / "query.csv"))
        write_csv(str(out_dir / "query.csv"), main_query_header, main_query_rows)
        copy_images(str(main_dir / "image_query"), str(out_dir / "image_query"))
        print(f"  Main query: {len(main_query_rows)} rows")
    else:
        print("  Main query: empty (no data)")

    if main_has_data and (main_dir / "query_classes.csv").is_file():
        main_qcls_header, main_qcls_rows = read_csv(str(main_dir / "query_classes.csv"))
        write_csv(str(out_dir / "query_classes.csv"), main_qcls_header, main_qcls_rows)

    # ================================================================
    # 4. VALIDATION: secondary test -> val_test, secondary query -> val_query
    # ================================================================
    print("\nProcessing VALIDATION splits (secondary test/query -> val)...")

    # Secondary test -> val_test
    sec_test_header, sec_test_rows = read_csv(str(sec_dir / "test.csv"))
    sec_test_shifted = shift_csv_rows(
        sec_test_rows, sec_test_header, img_shift, obj_shift, cam_shift
    )
    write_csv(str(out_dir / "val_test.csv"), sec_test_header, sec_test_shifted)
    copy_images_shifted(
        str(sec_dir / "image_test"), str(out_dir / "image_val_test"), img_shift
    )
    print(f"  val_test: {len(sec_test_shifted)} rows")

    if (sec_dir / "test_classes.csv").is_file():
        sec_testcls_header, sec_testcls_rows = read_csv(str(sec_dir / "test_classes.csv"))
        sec_testcls_shifted = shift_csv_rows(
            sec_testcls_rows, sec_testcls_header, img_shift, obj_shift, cam_shift
        )
        write_csv(str(out_dir / "val_test_classes.csv"), sec_testcls_header, sec_testcls_shifted)

    # Secondary query -> val_query
    sec_query_header, sec_query_rows = read_csv(str(sec_dir / "query.csv"))
    sec_query_shifted = shift_csv_rows(
        sec_query_rows, sec_query_header, img_shift, obj_shift, cam_shift
    )
    write_csv(str(out_dir / "val_query.csv"), sec_query_header, sec_query_shifted)
    copy_images_shifted(
        str(sec_dir / "image_query"), str(out_dir / "image_val_query"), img_shift
    )
    print(f"  val_query: {len(sec_query_shifted)} rows")

    if (sec_dir / "query_classes.csv").is_file():
        sec_qcls_header, sec_qcls_rows = read_csv(str(sec_dir / "query_classes.csv"))
        sec_qcls_shifted = shift_csv_rows(
            sec_qcls_rows, sec_qcls_header, img_shift, obj_shift, cam_shift
        )
        write_csv(str(out_dir / "val_query_classes.csv"), sec_qcls_header, sec_qcls_shifted)

    # ================================================================
    # 5. Copy any extra files (sample_submission, readme, etc.)
    # ================================================================
    print("\nCopying extra files...")
    for src in [main_dir, sec_dir]:
        if not src.is_dir():
            continue
        for f in src.iterdir():
            if f.is_file() and f.name not in {
                "train.csv", "test.csv", "query.csv",
                "train_classes.csv", "test_classes.csv", "query_classes.csv",
            }:
                dst = out_dir / f.name
                if not dst.exists():
                    shutil.copy2(str(f), str(dst))
                    print(f"  Copied: {f.name}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("COMBINATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {out_dir}")
    print()

    # Count output files
    for split_name in ["train", "test", "query", "val_test", "val_query"]:
        csv_path = out_dir / f"{split_name}.csv"
        img_dir = out_dir / f"image_{split_name}"
        if csv_path.is_file():
            _, rows = read_csv(str(csv_path))
            n_img = len(os.listdir(str(img_dir))) if img_dir.is_dir() else 0
            print(f"  {split_name:12s}: {len(rows):6d} CSV rows, {n_img:6d} images")

    print()
    print("ID Shifts applied to secondary:")
    print(f"  Image ID shift:  +{img_shift}")
    print(f"  ObjectID shift:  +{obj_shift}")
    print(f"  CameraID shift:  +{cam_shift}")


if __name__ == "__main__":
    main()
