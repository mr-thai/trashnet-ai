#!/usr/bin/env python3
"""Inference Benchmark Script for Trash-Net (US-2.4).

Measures inference latency, 95th percentile (p95), and FPS across 100 frames
for WasteDetector at multiple image resolutions (e.g. 640x640, 480x480).
Saves benchmark metrics to reports/benchmark_report.md.
"""

import argparse
import os
from pathlib import Path
import platform
import sys
import time
from typing import Dict, List
import cv2
import numpy as np
import psutil

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.detector import WasteDetector


def run_benchmark(
    weights_path: Path,
    image_path: Path,
    imgsz_list: List[int],
    num_frames: int = 100,
    device: str = "cpu",
    report_dir: Path = Path("reports"),
) -> Dict[int, dict]:
    """Run benchmark loop across specified resolutions."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Benchmark test image not found at {image_path}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Could not read image from {image_path}")

    results = {}
    print("=" * 70)
    print(f"TRASH-NET INFERENCE BENCHMARK ({num_frames} frames per resolution)")
    print(f"Device: {device.upper()} | Model: {weights_path.name}")
    print("=" * 70)

    for imgsz in imgsz_list:
        print(f"\n[*] Benchmarking resolution: {imgsz}x{imgsz} ...")
        detector = WasteDetector(
            model_path=weights_path,
            imgsz=imgsz,
            device=device,
            warmup=True,
        )

        latencies_ms = []
        for i in range(num_frames):
            t0 = time.perf_counter()
            _ = detector.detect(frame)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        latencies_arr = np.array(latencies_ms)
        avg_latency = float(np.mean(latencies_arr))
        p50 = float(np.percentile(latencies_arr, 50))
        p95 = float(np.percentile(latencies_arr, 95))
        min_lat = float(np.min(latencies_arr))
        max_lat = float(np.max(latencies_arr))
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

        results[imgsz] = {
            "avg_ms": avg_latency,
            "p50_ms": p50,
            "p95_ms": p95,
            "min_ms": min_lat,
            "max_ms": max_lat,
            "fps": fps,
        }

        print(f"    Avg Latency: {avg_latency:.2f} ms | p95: {p95:.2f} ms | FPS: {fps:.2f}")

    # Generate Markdown Report
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / "benchmark_report.md"

    cpu_name = platform.processor() or "Unknown CPU"
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Đo Hiệu Năng Suy Luận Trash-Net (US-2.4)\n\n")
        f.write("## 1. Môi Trường Phần Cứng & Cấu Hình\n\n")
        f.write(f"- **Hệ điều hành:** `{platform.system()} {platform.release()}`\n")
        f.write(f"- **CPU:** `{cpu_name}` ({os.cpu_count()} vCPUs)\n")
        f.write(f"- **Dung lượng RAM:** `{ram_gb:.1f} GB`\n")
        f.write(f"- **Thiết bị suy luận:** `{device.upper()}`\n")
        f.write(f"- **Trọng số mô hình:** `{weights_path.name}`\n")
        f.write(f"- **Số khung hình thử nghiệm:** `{num_frames}`\n\n")

        f.write("## 2. Kết Quả Đo Độ Trễ & FPS\n\n")
        f.write("| Kích thước (imgsz) | Độ trễ TB (ms) | p50 (ms) | p95 (ms) | Min (ms) | Max (ms) | FPS đạt được |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for imgsz, metrics in results.items():
            f.write(
                f"| `{imgsz}x{imgsz}` | {metrics['avg_ms']:.2f} | {metrics['p50_ms']:.2f} | "
                f"{metrics['p95_ms']:.2f} | {metrics['min_ms']:.2f} | {metrics['max_ms']:.2f} | "
                f"**{metrics['fps']:.1f} FPS** |\n"
            )

        f.write("\n## 3. Nhận Xét & Khuyến Nghị Triển Khai\n\n")
        fps_640 = results.get(640, {}).get("fps", 0)
        fps_480 = results.get(480, {}).get("fps", 0)
        f.write(f"- Ở độ phân giải chuẩn `640x640`, mô hình đạt **{fps_640:.1f} FPS** trên {device.upper()}.\n")
        if 480 in results:
            f.write(f"- Khi hạ độ phân giải xuống `480x480`, tốc độ tăng lên **{fps_480:.1f} FPS** (cải thiện ~{((fps_480/fps_640)-1)*100:.1f}%).\n")
        f.write("- **Khuyến nghị cho Desktop GUI:** Với webcam thời gian thực trên CPU, kích thước `480x480` hoặc `640x640` kết hợp kỹ thuật đa luồng (QThread) sẽ đảm bảo giao diện không bị giật lag.\n")

    print(f"\n[+] Benchmark report saved to: {report_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Trash-Net WasteDetector performance.")
    parser.add_argument("--weights", type=Path, default=Path("yolo11n-seg.pt"), help="Path to model weights")
    parser.add_argument("--image", type=Path, default=Path("tests/data/sample_waste.jpg"), help="Path to benchmark image")
    parser.add_argument("--frames", type=int, default=100, help="Number of benchmark iterations (default: 100)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to benchmark on ('cpu' or 'cuda')")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 480], help="Resolutions to benchmark")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"), help="Report output folder")

    args = parser.parse_args()
    run_benchmark(
        weights_path=args.weights,
        image_path=args.image,
        imgsz_list=args.imgsz,
        num_frames=args.frames,
        device=args.device,
        report_dir=args.report_dir,
    )


if __name__ == "__main__":
    main()
