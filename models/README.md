# Models Directory (US-1.5 & US-5.2)

Thư mục lưu trữ các tệp trọng số phân đoạn thực thể phục vụ hệ thống Trash-Net.

## Các tệp chính:
- `best.pt`: Tệp trọng số mô hình YOLO11n-seg đã huấn luyện hoàn tất qua 50 epochs trên bộ dữ liệu MultiTrash v8 (mAP50 đạt ~69.7%).
- `MODEL_CARD.md`: Thẻ mô hình ghi nhận chi tiết siêu dữ liệu (ngày huấn luyện, phiên bản ultralytics, SHA-256 hash, danh sách lớp, chỉ số mAP và hướng dẫn nạp).
- `yolo11n-seg.pt`: Trọng số dự phòng ban đầu.

## Cơ chế tự động tìm nạp (Model Path Priority):
1. `models/best.pt` (Khuyến nghị chính thức)
2. `best.pt` (Thư mục gốc)
3. `models/yolo11n-seg.pt`
4. `yolo11n-seg.pt`
