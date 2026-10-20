# Model Card: MultiTrash-YOLO11n-seg (`best.pt`)

Bản đóng gói trọng số mô hình trí tuệ nhân tạo nhận diện và phân đoạn thực thể rác thải phục vụ hệ thống Trash-Net (US-1.5).

---

## 1. Thông Tin Tổng Quan (Model Details)

- **Tên mô hình:** MultiTrash YOLO11 Nano Segmentation
- **Tên tệp đóng gói:** `models/best.pt`
- **Phiên bản kiến trúc:** YOLO11n-seg (`ultralytics 8.3.142`)
- **Tác vụ (Task):** Instance Segmentation (Phát hiện và phân đoạn đường biên đa giác)
- **Ngày hoàn tất huấn luyện:** 2026-10-20T10:48:02 UTC
- **Kích thước tệp:** 6,007,460 bytes (~6.01 MB)
- **Mã băm toàn vẹn (SHA-256):**
  ```text
  f096a1ce1d5f742c6c751dc81e64b407158469ba8480cf358e45d21adb865730
  ```

---

## 2. Thông Số Huấn Luyện (Training Parameters)

- **Số lượng epochs:** 50
- **Kích thước ảnh đầu vào (`imgsz`):** 640 × 640
- **Batch size:** Dynamic (phù hợp GPU đám mây)
- **Tối ưu hóa (Optimizer):** Auto (AdamW / SGD tích hợp Ultralytics)
- **Tệp trọng số gốc ban đầu:** `yolo11n-seg.pt` (transfer learning)

---

## 3. Chỉ Số Hiệu Năng Đạt Được (Validation Metrics)

Đo lường trên tập kiểm định MultiTrash v8:

| Chỉ số đánh giá | Hộp bao quanh (Bounding Box) | Mặt nạ phân đoạn (Mask Polygon) |
| :--- | :--- | :--- |
| **Độ chính xác (Precision)** | 75.51% | 75.24% |
| **Độ nhạy (Recall)** | 62.71% | 62.48% |
| **mAP@50** | **69.68%** | **69.09%** |
| **mAP@50-95** | **52.55%** | **48.27%** |
| **Tổn thất nghiệm thu (Loss)** | `box_loss: 0.8295` | `seg_loss: 1.8726` |

---

## 4. Danh Sách Lớp Nhận Diện & Ánh Xạ Thùng Rác (Classes & Mapping)

Mô hình được huấn luyện để phân biệt 8 lớp đối tượng rác thải và tự động phân loại vào 3 thùng đích:

| Class ID | Tên Lớp Kỹ Thuật | Tên Tiếng Việt | Nhóm Thùng Rác Ánh Xạ |
| :---: | :--- | :--- | :--- |
| `0` | `biowaste` | Rác hữu cơ / sinh học | 🟢 **Hữu cơ (ORGANIC)** |
| `1` | `glass` | Thủy tinh | 🔵 **Tái chế (RECYCLABLE)** |
| `2` | `metal` | Lon, kim loại | 🔵 **Tái chế (RECYCLABLE)** |
| `3` | `other` | Rác vô cơ / tạp chất | ⚪ **Vô cơ / Khác (RESIDUAL)** |
| `4` | `paper` | Giấy, bìa carton | 🔵 **Tái chế (RECYCLABLE)** |
| `5` | `plastic` | Chai lọ nhựa | 🔵 **Tái chế (RECYCLABLE)** |
| `6` | `styrofoam` | Xốp hộp cơm | ⚪ **Vô cơ / Khác (RESIDUAL)** |
| `7` | `wrapper` | Màng bọc nilon / bao bì | ⚪ **Vô cơ / Khác (RESIDUAL)** |

---

## 5. Hướng Dẫn Nạp Trọng Số & Suy Luận Cục Bộ (Inference Usage)

### Nạp trực tiếp qua Ultralytics YOLO:
```python
from ultralytics import YOLO
import cv2

# Nạp mô hình
model = YOLO("models/best.pt")

# Thực hiện suy luận phân đoạn thực thể
results = model.predict("tests/data/sample_waste.jpg", conf=0.5, imgsz=640)

# Duyệt kết quả
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        print(f"Detected: {model.names[cls_id]} with confidence {float(box.conf[0]):.2f}")
```

### Sử dụng qua lớp bọc Trash-Net (`WasteDetector`):
```python
from core.detector import WasteDetector

detector = WasteDetector(model_path="models/best.pt", conf=0.5, imgsz=640)
detections = detector.detect(frame)
```

---

## 6. Lưu Ý Khi Xuất ONNX (Export Considerations)
- Để xuất sang định dạng ONNX phục vụ tăng tốc OpenVINO / TensorRT:
  ```bash
  pip install onnx onnxruntime
  yolo export model=models/best.pt format=onnx imgsz=640 opset=12
  ```
