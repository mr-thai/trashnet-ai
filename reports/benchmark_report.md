# Báo Cáo Đo Hiệu Năng Suy Luận Trash-Net (US-2.4)

## 1. Môi Trường Phần Cứng & Cấu Hình

- **Hệ điều hành:** `Linux 7.2.11-zen1-1-zen`
- **CPU:** `Unknown CPU` (8 vCPUs)
- **Dung lượng RAM:** `19.4 GB`
- **Thiết bị suy luận:** `CPU`
- **Trọng số mô hình:** `yolo11n-seg.pt`
- **Số khung hình thử nghiệm:** `20`

## 2. Kết Quả Đo Độ Trễ & FPS

| Kích thước (imgsz) | Độ trễ TB (ms) | p50 (ms) | p95 (ms) | Min (ms) | Max (ms) | FPS đạt được |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `640x640` | 66.83 | 64.39 | 77.29 | 58.40 | 82.25 | **15.0 FPS** |
| `480x480` | 43.97 | 41.62 | 61.27 | 37.83 | 78.53 | **22.7 FPS** |

## 3. Nhận Xét & Khuyến Nghị Triển Khai

- Ở độ phân giải chuẩn `640x640`, mô hình đạt **15.0 FPS** trên CPU.
- Khi hạ độ phân giải xuống `480x480`, tốc độ tăng lên **22.7 FPS** (cải thiện ~52.0%).
- **Khuyến nghị cho Desktop GUI:** Với webcam thời gian thực trên CPU, kích thước `480x480` hoặc `640x640` kết hợp kỹ thuật đa luồng (QThread) sẽ đảm bảo giao diện không bị giật lag.
