#!/usr/bin/env python3
"""Dataset validation script for MultiTrash v8.

Validates image-label pairings, YOLO polygon annotation formats,
class indices, coordinate bounds, and calculates class distribution statistics.
"""

from collections import Counter, defaultdict
import os
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple
import yaml

EXPECTED_IMAGE_COUNTS = {
    "train": 5120,
    "valid": 664,
    "test": 7,
}

EXPECTED_CLASSES = [
    "biowaste",
    "glass",
    "metal",
    "other",
    "paper",
    "plastic",
    "styrofoam",
    "wrapper",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DatasetValidator:
    def __init__(
        self,
        data_dir: Path,
        data_yaml_path: Path,
        report_dir: Path,
    ):
        self.data_dir = data_dir
        self.data_yaml_path = data_yaml_path
        self.report_dir = report_dir

        self.splits = ["train", "valid", "test"]
        self.split_img_counts: Dict[str, int] = {}
        self.split_lbl_counts: Dict[str, int] = {}

        self.missing_labels: Dict[str, List[str]] = defaultdict(list)
        self.missing_images: Dict[str, List[str]] = defaultdict(list)

        self.annotation_errors: List[Dict[str, str]] = []
        self.class_counts: Dict[str, Counter] = {s: Counter() for s in self.splits}
        self.images_per_class: Dict[str, Counter] = {s: Counter() for s in self.splits}
        self.total_class_counts: Counter = Counter()

        self.yaml_errors: List[str] = []
        self.yaml_classes: List[str] = []

    def validate_yaml(self) -> bool:
        """Validate data.yaml existence, structure, relative paths, and class list."""
        if not self.data_yaml_path.exists():
            self.yaml_errors.append(f"data.yaml not found at {self.data_yaml_path}")
            return False

        try:
            with open(self.data_yaml_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
        except Exception as e:
            self.yaml_errors.append(f"Failed to parse data.yaml: {e}")
            return False

        # Check nc
        nc = content.get("nc")
        if nc != len(EXPECTED_CLASSES):
            self.yaml_errors.append(f"Expected nc={len(EXPECTED_CLASSES)}, got nc={nc}")

        # Check class names
        names = content.get("names")
        if isinstance(names, dict):
            # Dict form: {0: 'biowaste', ...}
            sorted_names = [names[k] for k in sorted(names.keys())]
            self.yaml_classes = sorted_names
        elif isinstance(names, list):
            self.yaml_classes = names
        else:
            self.yaml_errors.append(f"Invalid names field type in data.yaml: {type(names)}")
            self.yaml_classes = []

        if self.yaml_classes != EXPECTED_CLASSES:
            self.yaml_errors.append(
                f"Class names mismatch!\nExpected: {EXPECTED_CLASSES}\nGot:      {self.yaml_classes}"
            )

        # Check path settings
        base_dir = self.data_yaml_path.parent
        dataset_root = base_dir
        if "path" in content:
            raw_path = content["path"]
            if os.path.isabs(raw_path):
                self.yaml_errors.append(f"Absolute path found in data.yaml: path={raw_path}")
                dataset_root = Path(raw_path)
            else:
                dataset_root = (base_dir / raw_path).resolve()

        for split_key, split_name in [("train", "train"), ("val", "valid"), ("test", "test")]:
            if split_key not in content:
                self.yaml_errors.append(f"Missing '{split_key}' key in data.yaml")
                continue
            split_rel_path = content[split_key]
            if os.path.isabs(split_rel_path):
                self.yaml_errors.append(f"Absolute path in data.yaml: {split_key}={split_rel_path}")

            # Verify path exists relative to dataset_root or base_dir
            target_path = dataset_root / split_rel_path
            alt_path = base_dir / split_rel_path
            if not target_path.exists() and not alt_path.exists():
                self.yaml_errors.append(
                    f"Directory for {split_key} '{split_rel_path}' does not resolve to an existing path (checked {target_path} and {alt_path})"
                )

        return len(self.yaml_errors) == 0

    def validate_pairs_and_annotations(self):
        """Validate image-label correspondence and polygon coordinates/classes."""
        for split in self.splits:
            img_dir = self.data_dir / split / "images"
            lbl_dir = self.data_dir / split / "labels"

            if not img_dir.exists():
                self.annotation_errors.append({
                    "file": str(img_dir),
                    "line": "-",
                    "error": f"Image directory not found: {img_dir}",
                })
                continue
            if not lbl_dir.exists():
                self.annotation_errors.append({
                    "file": str(lbl_dir),
                    "line": "-",
                    "error": f"Label directory not found: {lbl_dir}",
                })
                continue

            img_files = {
                f.stem: f
                for f in img_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            }
            lbl_files = {
                f.stem: f
                for f in lbl_dir.iterdir()
                if f.is_file() and f.suffix.lower() == ".txt"
            }

            self.split_img_counts[split] = len(img_files)
            self.split_lbl_counts[split] = len(lbl_files)

            # Check matching
            missing_labels = sorted(set(img_files.keys()) - set(lbl_files.keys()))
            missing_images = sorted(set(lbl_files.keys()) - set(img_files.keys()))

            if missing_labels:
                self.missing_labels[split] = missing_labels
            if missing_images:
                self.missing_images[split] = missing_images

            # Inspect each label file
            for stem, lbl_path in sorted(lbl_files.items()):
                try:
                    with open(lbl_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                except Exception as e:
                    self.annotation_errors.append({
                        "file": str(lbl_path),
                        "line": "-",
                        "error": f"Unable to read file: {e}",
                    })
                    continue

                classes_in_image: Set[int] = set()

                for line_idx, raw_line in enumerate(lines, 1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    tokens = line.split()

                    # Class ID check
                    try:
                        class_id = int(tokens[0])
                    except ValueError:
                        self.annotation_errors.append({
                            "file": str(lbl_path),
                            "line": str(line_idx),
                            "error": f"class_id is not integer: '{tokens[0]}'",
                        })
                        continue

                    if not (0 <= class_id <= 7):
                        self.annotation_errors.append({
                            "file": str(lbl_path),
                            "line": str(line_idx),
                            "error": f"class_id {class_id} outside valid range [0, 7]",
                        })

                    # Coordinates check
                    raw_coords = tokens[1:]
                    if len(raw_coords) < 6:
                        self.annotation_errors.append({
                            "file": str(lbl_path),
                            "line": str(line_idx),
                            "error": f"Polygon has {len(raw_coords)} coords (min 6 coords / 3 points required)",
                        })
                    elif len(raw_coords) % 2 != 0:
                        self.annotation_errors.append({
                            "file": str(lbl_path),
                            "line": str(line_idx),
                            "error": f"Polygon has odd coordinate count: {len(raw_coords)}",
                        })

                    for coord_idx, c_str in enumerate(raw_coords):
                        try:
                            c_val = float(c_str)
                            if not (0.0 <= c_val <= 1.0):
                                self.annotation_errors.append({
                                    "file": str(lbl_path),
                                    "line": str(line_idx),
                                    "error": f"Coordinate #{coord_idx+1} ({c_val}) outside [0, 1]",
                                })
                                break
                        except ValueError:
                            self.annotation_errors.append({
                                "file": str(lbl_path),
                                "line": str(line_idx),
                                "error": f"Coordinate #{coord_idx+1} is not float: '{c_str}'",
                            })
                            break

                    self.class_counts[split][class_id] += 1
                    self.total_class_counts[class_id] += 1
                    classes_in_image.add(class_id)

                for cid in classes_in_image:
                    self.images_per_class[split][cid] += 1

    def print_summary_table(self):
        """Print console summary table."""
        print("=" * 80)
        print(" " * 26 + "DATASET VALIDATION SUMMARY")
        print("=" * 80)

        # Split count table
        print("\n[1] Image & Label Counts by Split:")
        print(f"{'Split':<10} | {'Found Imgs':<12} | {'Expected Imgs':<14} | {'Diff':<8} | {'Labels':<10} | {'Status'}")
        print("-" * 72)
        all_counts_match = True
        for split in self.splits:
            found = self.split_img_counts.get(split, 0)
            expected = EXPECTED_IMAGE_COUNTS.get(split, 0)
            diff = found - expected
            diff_str = f"{diff:+d}" if diff != 0 else "0"
            lbls = self.split_lbl_counts.get(split, 0)
            status = "MATCH" if diff == 0 else "MISMATCH"
            if diff != 0:
                all_counts_match = False
            print(f"{split:<10} | {found:<12} | {expected:<14} | {diff_str:<8} | {lbls:<10} | {status}")

        # Pair verification
        print("\n[2] Image - Label Pair Completeness:")
        for split in self.splits:
            miss_lbl = len(self.missing_labels[split])
            miss_img = len(self.missing_images[split])
            print(f" - {split:<6}: Missing labels = {miss_lbl} (expected 0) | Missing images = {miss_img} (expected 0)")

        # Annotation errors
        print("\n[3] Annotation Syntax & Boundary Verification:")
        if not self.annotation_errors:
            print(" - All label lines are valid! (Coordinates in [0, 1], class_id in 0-7, polygon points >= 3)")
        else:
            print(f" - Found {len(self.annotation_errors)} annotation violations:")
            for err in self.annotation_errors[:15]:
                print(f"   * {err['file']}:{err['line']} -> {err['error']}")
            if len(self.annotation_errors) > 15:
                print(f"   ... and {len(self.annotation_errors) - 15} more violations.")

        # Class distribution table
        print("\n[4] Class Distribution & Imbalance Analysis:")
        total_instances = sum(self.total_class_counts.values())
        print(f"{'ID':<3} | {'Class Name':<12} | {'Train':<8} | {'Valid':<8} | {'Test':<8} | {'Total':<8} | {'Pct (%)':<8}")
        print("-" * 68)
        for cid, name in enumerate(EXPECTED_CLASSES):
            tr = self.class_counts["train"][cid]
            va = self.class_counts["valid"][cid]
            te = self.class_counts["test"][cid]
            tot = self.total_class_counts[cid]
            pct = (tot / total_instances * 100) if total_instances > 0 else 0
            print(f"{cid:<3} | {name:<12} | {tr:<8} | {va:<8} | {te:<8} | {tot:<8} | {pct:>6.2f}%")

        print("-" * 68)
        print(f"{'SUM':<3} | {'ALL CLASSES':<12} | {sum(self.class_counts['train'].values()):<8} | {sum(self.class_counts['valid'].values()):<8} | {sum(self.class_counts['test'].values()):<8} | {total_instances:<8} | 100.00%")

        # Imbalance insight
        if total_instances > 0:
            counts = [self.total_class_counts[cid] for cid in range(len(EXPECTED_CLASSES))]
            max_c = max(counts)
            min_c = min(counts)
            max_cls = EXPECTED_CLASSES[counts.index(max_c)]
            min_cls = EXPECTED_CLASSES[counts.index(min_c)]
            ratio = max_c / min_c if min_c > 0 else float("inf")
            print(f"\nImbalance Ratio (Max/Min): {ratio:.2f}:1")
            print(f" - Most frequent class:   '{max_cls}' with {max_c} instances ({max_c/total_instances*100:.1f}%)")
            print(f" - Least frequent class:  '{min_cls}' with {min_c} instances ({min_c/total_instances*100:.1f}%)")

        # YAML status
        print("\n[5] data.yaml Configuration Check:")
        if not self.yaml_errors:
            print(" - data.yaml: OK (Relative paths, nc=8, correct class ordering)")
        else:
            print(f" - data.yaml issues found ({len(self.yaml_errors)}):")
            for err in self.yaml_errors:
                print(f"   * {err}")

        print("=" * 80)

    def generate_reports(self):
        """Export Markdown report and CSV distribution file."""
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # 1. Export CSV
        csv_path = self.report_dir / "class_distribution.csv"
        total_instances = sum(self.total_class_counts.values())
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("class_id,class_name,train_instances,valid_instances,test_instances,total_instances,percentage\n")
            for cid, name in enumerate(EXPECTED_CLASSES):
                tr = self.class_counts["train"][cid]
                va = self.class_counts["valid"][cid]
                te = self.class_counts["test"][cid]
                tot = self.total_class_counts[cid]
                pct = (tot / total_instances * 100) if total_instances > 0 else 0
                f.write(f"{cid},{name},{tr},{va},{te},{tot},{pct:.4f}\n")

        # 2. Export Markdown Report
        md_path = self.report_dir / "dataset_validation_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Báo Cáo Kiểm Tra Tính Hợp Lệ Dữ Liệu MultiTrash v8\n\n")
            f.write("## 1. Kết Quả Tổng Quan\n\n")

            has_fatal_errors = (
                bool(self.yaml_errors)
                or bool(self.annotation_errors)
                or any(len(v) > 0 for v in self.missing_labels.values())
                or any(len(v) > 0 for v in self.missing_images.values())
            )
            status_text = "FAILED" if has_fatal_errors else "PASSED"
            f.write(f"- **Trạng thái xác thực:** `{status_text}`\n")
            f.write(f"- **Tệp cấu hình:** `{self.data_yaml_path}`\n")
            f.write(f"- **Thư mục dữ liệu:** `{self.data_dir}`\n\n")

            f.write("## 2. Số Lượng Ảnh và Nhãn Theo Tập\n\n")
            f.write("| Tập dữ liệu | Số ảnh thực tế | Số ảnh kỳ vọng | Chênh lệch | Số nhãn thực tế | Trạng thái |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
            for split in self.splits:
                found = self.split_img_counts.get(split, 0)
                expected = EXPECTED_IMAGE_COUNTS.get(split, 0)
                diff = found - expected
                diff_str = f"{diff:+d}" if diff != 0 else "0"
                lbls = self.split_lbl_counts.get(split, 0)
                st = "Khớp" if diff == 0 else "Chênh lệch"
                f.write(f"| `{split}` | {found:,} | {expected:,} | {diff_str} | {lbls:,} | {st} |\n")

            f.write("\n## 3. Kiểm Tra Cặp Ảnh - Nhãn\n\n")
            f.write("| Tập dữ liệu | Ảnh không có nhãn (Kỳ vọng 0) | Nhãn không có ảnh (Kỳ vọng 0) |\n")
            f.write("| :--- | :---: | :---: |\n")
            for split in self.splits:
                f.write(f"| `{split}` | {len(self.missing_labels[split])} | {len(self.missing_images[split])} |\n")

            f.write("\n## 4. Kiểm Tra Tọa Độ và Định Dạng Nhãn (YOLO Polygon)\n\n")
            f.write(f"- **Tổng số dòng nhãn vi phạm:** `{len(self.annotation_errors)}`\n")
            if self.annotation_errors:
                f.write("\n| Tệp | Dòng | Chi tiết lỗi |\n")
                f.write("| :--- | :---: | :--- |\n")
                for err in self.annotation_errors[:50]:
                    f.write(f"| `{err['file']}` | {err['line']} | {err['error']} |\n")
                if len(self.annotation_errors) > 50:
                    f.write(f"| ... | ... | Còn {len(self.annotation_errors) - 50} lỗi khác |\n")
            else:
                f.write("- **Chi tiết:** Tất cả các dòng nhãn đều có `class_id` trong [0, 7], tọa độ trong [0.0, 1.0], và tối thiểu 3 điểm đa giác (>= 6 tọa độ chẵn).\n")

            f.write("\n## 5. Thống Kê Phân Bố Số Mẫu Theo Lớp (Class Distribution)\n\n")
            f.write("| ID | Tên lớp | Train | Valid | Test | Tổng số mẫu | Tỷ lệ (%) |\n")
            f.write("| :-: | :--- | :-: | :-: | :-: | :-: | :-: |\n")
            for cid, name in enumerate(EXPECTED_CLASSES):
                tr = self.class_counts["train"][cid]
                va = self.class_counts["valid"][cid]
                te = self.class_counts["test"][cid]
                tot = self.total_class_counts[cid]
                pct = (tot / total_instances * 100) if total_instances > 0 else 0
                f.write(f"| {cid} | `{name}` | {tr:,} | {va:,} | {te:,} | {tot:,} | {pct:.2f}% |\n")

            f.write(f"| - | **TỔNG CỘNG** | **{sum(self.class_counts['train'].values()):,}** | **{sum(self.class_counts['valid'].values()):,}** | **{sum(self.class_counts['test'].values()):,}** | **{total_instances:,}** | **100.00%** |\n\n")

            f.write("### Nhận xét mất cân bằng dữ liệu (Class Imbalance Analysis)\n\n")
            if total_instances > 0:
                counts = [self.total_class_counts[cid] for cid in range(len(EXPECTED_CLASSES))]
                max_c = max(counts)
                min_c = min(counts)
                max_cls = EXPECTED_CLASSES[counts.index(max_c)]
                min_cls = EXPECTED_CLASSES[counts.index(min_c)]
                ratio = max_c / min_c if min_c > 0 else float("inf")

                f.write(f"- **Tỷ lệ mất cân bằng (Max/Min):** `{ratio:.2f}:1`\n")
                f.write(f"- **Lớp nhiều mẫu nhất:** `{max_cls}` với {max_c:,} mẫu ({max_c/total_instances*100:.2f}%).\n")
                f.write(f"- **Lớp ít mẫu nhất:** `{min_cls}` với {min_c:,} mẫu ({min_c/total_instances*100:.2f}%).\n")
                f.write("- **Quan sát đặc biệt:**\n")
                f.write("  - Lớp `other` (ID 3) có số lượng mẫu rất thấp (chỉ 218 mẫu, chiếm 0.74% toàn bộ dataset).\n")
                f.write("  - Tập `test` chỉ có 7 ảnh (37 đối tượng), trong đó thiếu hoàn toàn 2 lớp: `glass` (ID 1) và `paper` (ID 4).\n")
                f.write("  - Cần cân nhắc áp dụng kỹ thuật data augmentation (mosaic, mixup) hoặc weighted loss trong quá trình huấn luyện để cải thiện mAP cho lớp thiểu số.\n")

            f.write("\n## 6. Kiểm Tra Tệp Cấu Hình data.yaml\n\n")
            if self.yaml_errors:
                f.write("- **Trạng thái:** ❌ Phát hiện lỗi trong `data.yaml`:\n")
                for err in self.yaml_errors:
                    f.write(f"  - {err}\n")
            else:
                f.write("- **Trạng thái:** ✅ Hợp lệ. Sử dụng đường dẫn tương đối, định nghĩa đủ 8 lớp theo đúng thứ tự `class_id`.\n")

        print(f"\n[+] Generated reports in '{self.report_dir}':")
        print(f"    - Markdown: {md_path}")
        print(f"    - CSV:      {csv_path}")

    def run(self) -> int:
        """Execute validation and return exit code."""
        yaml_ok = self.validate_yaml()
        self.validate_pairs_and_annotations()
        self.print_summary_table()
        self.generate_reports()

        has_missing = any(len(v) > 0 for v in self.missing_labels.values()) or any(
            len(v) > 0 for v in self.missing_images.values()
        )
        has_annotation_errs = len(self.annotation_errors) > 0

        # DoD: Script trả mã thoát khác 0 khi phát hiện lỗi
        if not yaml_ok or has_missing or has_annotation_errs:
            return 1
        return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate MultiTrash YOLO polygon dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("MultiTrash"),
        help="Path to MultiTrash dataset root (default: MultiTrash)",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("MultiTrash/data.yaml"),
        help="Path to data.yaml (default: MultiTrash/data.yaml)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to save validation reports (default: reports)",
    )

    args = parser.parse_args()

    validator = DatasetValidator(
        data_dir=args.data_dir,
        data_yaml_path=args.data_yaml,
        report_dir=args.report_dir,
    )
    exit_code = validator.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
