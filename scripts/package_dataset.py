#!/usr/bin/env python3
"""Dataset packaging script for MultiTrash v8.

Packages the validated dataset into a single archive (.zip or .tar.gz)
with zero-modification cloud paths and generates a SHA-256 checksum for integrity verification.
"""

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tarfile
import time
import zipfile


def calculate_sha256(filepath: Path) -> str:
    """Calculate SHA-256 checksum of a file in streaming chunks."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):  # 1MB buffer
            sha256.update(chunk)
    return sha256.hexdigest()


def package_dataset(
    data_dir: Path,
    output_dir: Path,
    archive_format: str = "zip",
) -> Path:
    """Package dataset directory into a clean archive."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory '{data_dir}' not found.")

    yaml_file = data_dir / "data.yaml"
    if not yaml_file.exists():
        raise FileNotFoundError(f"data.yaml not found at '{yaml_file}'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"{data_dir.name}_v8.{archive_format}"
    archive_path = output_dir / archive_name

    print(f"[*] Packaging dataset '{data_dir}' into '{archive_path}'...")
    start_time = time.time()

    # Collect files
    file_list = []
    total_raw_bytes = 0
    for root, _, files in os.walk(data_dir):
        for file in files:
            full_path = Path(root) / file
            rel_path = full_path.relative_to(data_dir.parent)
            file_list.append((full_path, rel_path))
            total_raw_bytes += full_path.stat().st_size

    print(f"[*] Total files to package: {len(file_list)} ({total_raw_bytes / (1024*1024):.2f} MB)")

    if archive_format == "zip":
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for idx, (full_path, rel_path) in enumerate(file_list, 1):
                zf.write(full_path, arcname=str(rel_path))
                if idx % 1000 == 0 or idx == len(file_list):
                    print(f"    Archived {idx}/{len(file_list)} files...", end="\r", flush=True)
    elif archive_format in ("tar.gz", "tgz"):
        with tarfile.open(archive_path, "w:gz") as tf:
            for idx, (full_path, rel_path) in enumerate(file_list, 1):
                tf.add(full_path, arcname=str(rel_path))
                if idx % 1000 == 0 or idx == len(file_list):
                    print(f"    Archived {idx}/{len(file_list)} files...", end="\r", flush=True)
    else:
        raise ValueError(f"Unsupported format: {archive_format}")

    print()
    elapsed = time.time() - start_time
    archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"[+] Archive created in {elapsed:.2f}s ({archive_size_mb:.2f} MB)")

    # Compute SHA-256
    print("[*] Computing SHA-256 checksum...")
    checksum = calculate_sha256(archive_path)
    checksum_file = output_dir / f"{archive_name}.sha256"
    with open(checksum_file, "w", encoding="utf-8") as f:
        f.write(f"{checksum}  {archive_name}\n")

    print(f"[+] SHA-256: {checksum}")
    print(f"[+] Checksum saved to: {checksum_file}")

    return archive_path


def main():
    parser = argparse.ArgumentParser(description="Package MultiTrash v8 dataset for cloud training.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("MultiTrash"),
        help="Path to dataset root folder (default: MultiTrash)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist"),
        help="Path to output directory (default: dist)",
    )
    parser.add_argument(
        "--format",
        choices=["zip", "tar.gz"],
        default="zip",
        help="Archive format (default: zip)",
    )

    args = parser.parse_args()

    try:
        package_dataset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            archive_format=args.format,
        )
    except Exception as e:
        print(f"[!] Error packaging dataset: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
