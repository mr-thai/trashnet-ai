# Báo Cáo Kiểm Tra Tính Hợp Lệ Dữ Liệu MultiTrash v8

## 1. Kết Quả Tổng Quan

- **Trạng thái xác thực:** `PASSED`
- **Tệp cấu hình:** `MultiTrash/data.yaml`
- **Thư mục dữ liệu:** `MultiTrash`

## 2. Số Lượng Ảnh và Nhãn Theo Tập

| Tập dữ liệu | Số ảnh thực tế | Số ảnh kỳ vọng | Chênh lệch | Số nhãn thực tế | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `train` | 5,120 | 5,120 | 0 | 5,120 | Khớp |
| `valid` | 664 | 664 | 0 | 664 | Khớp |
| `test` | 7 | 7 | 0 | 7 | Khớp |

## 3. Kiểm Tra Cặp Ảnh - Nhãn

| Tập dữ liệu | Ảnh không có nhãn (Kỳ vọng 0) | Nhãn không có ảnh (Kỳ vọng 0) |
| :--- | :---: | :---: |
| `train` | 0 | 0 |
| `valid` | 0 | 0 |
| `test` | 0 | 0 |

## 4. Kiểm Tra Tọa Độ và Định Dạng Nhãn (YOLO Polygon)

- **Tổng số dòng nhãn vi phạm:** `0`
- **Chi tiết:** Tất cả các dòng nhãn đều có `class_id` trong [0, 7], tọa độ trong [0.0, 1.0], và tối thiểu 3 điểm đa giác (>= 6 tọa độ chẵn).

## 5. Thống Kê Phân Bố Số Mẫu Theo Lớp (Class Distribution)

| ID | Tên lớp | Train | Valid | Test | Tổng số mẫu | Tỷ lệ (%) |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: |
| 0 | `biowaste` | 4,413 | 494 | 1 | 4,908 | 14.81% |
| 1 | `glass` | 3,065 | 239 | 0 | 3,304 | 9.97% |
| 2 | `metal` | 3,241 | 453 | 4 | 3,698 | 11.16% |
| 3 | `other` | 159 | 38 | 21 | 218 | 0.66% |
| 4 | `paper` | 4,833 | 716 | 0 | 5,549 | 16.74% |
| 5 | `plastic` | 6,995 | 878 | 5 | 7,878 | 23.77% |
| 6 | `styrofoam` | 2,072 | 178 | 5 | 2,255 | 6.80% |
| 7 | `wrapper` | 4,808 | 530 | 1 | 5,339 | 16.11% |
| - | **TỔNG CỘNG** | **29,586** | **3,526** | **37** | **33,149** | **100.00%** |

### Nhận xét mất cân bằng dữ liệu (Class Imbalance Analysis)

- **Tỷ lệ mất cân bằng (Max/Min):** `36.14:1`
- **Lớp nhiều mẫu nhất:** `plastic` với 7,878 mẫu (23.77%).
- **Lớp ít mẫu nhất:** `other` với 218 mẫu (0.66%).
- **Quan sát đặc biệt:**
  - Lớp `other` (ID 3) có số lượng mẫu rất thấp (chỉ 218 mẫu, chiếm 0.74% toàn bộ dataset).
  - Tập `test` chỉ có 7 ảnh (37 đối tượng), trong đó thiếu hoàn toàn 2 lớp: `glass` (ID 1) và `paper` (ID 4).
  - Cần cân nhắc áp dụng kỹ thuật data augmentation (mosaic, mixup) hoặc weighted loss trong quá trình huấn luyện để cải thiện mAP cho lớp thiểu số.

## 6. Kiểm Tra Tệp Cấu Hình data.yaml

- **Trạng thái:** ✅ Hợp lệ. Sử dụng đường dẫn tương đối, định nghĩa đủ 8 lớp theo đúng thứ tự `class_id`.
