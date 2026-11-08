# Báo Cáo Đo Hiệu Năng Suy Luận Trash-Net (US-2.4)

## 1. Môi Trường Phần Cứng & Cấu Hình

- **Hệ điều hành:** `Linux 7.2.14-zen1-1-zen`
- **CPU:** `Intel Core i5-10300H` (8 vCPUs)
- **Dung lượng RAM:** `19.4 GB`
- **Thiết bị suy luận:** `CPU`
- **Trọng số mô hình:** `models/best.pt` (MultiTrash v8 fine-tuned)
- **Số khung hình thử nghiệm:** `30`

## 2. Kết Quả Đo Độ Trễ & FPS

| Kích thước (imgsz) | Độ trễ TB (ms) | p50 (ms) | p95 (ms) | Min (ms) | Max (ms) | FPS đạt được |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `640x640` | 218.52 | 183.38 | 341.06 | 143.51 | 621.15 | **4.6 FPS** |
| `480x480` | 328.01 | 201.49 | 932.36 | 99.62 | 987.81 | **3.0 FPS** |

## 3. Nhận Xét & Khuyến Nghị Triển Khai

- Mô hình fine-tuned `models/best.pt` đạt trung bình **~4.6 FPS** (độ trễ ~218ms) khi chạy trực tiếp trên CPU ở độ phân giải gốc `640x640`.
- **Khuyến nghị cho Desktop GUI:** Để trải nghiệm không bị giật lag trên CPU, kiến trúc đa luồng (`QThread` / `WorkerThread`) phân tách hoàn toàn luồng thu nhận khung hình Camera (30 FPS) và luồng suy luận AI (4-5 FPS) với cơ chế *drop frame* tự động khi AI bận. Khi triển khai trên phần cứng có GPU hoặc tăng tốc NPU/TensorRT/ONNX-Runtime, FPS có thể đạt 30-60+ FPS.
