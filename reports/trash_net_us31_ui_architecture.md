# Kiến Trúc Cửa Sổ Chính PyQt6/PySide6 Cho Ứng Dụng Phân Loại Rác (US-3.1)

Ghi chép kỹ thuật về việc triển khai giao diện phân vùng không méo khung hình, thích ứng đa thư viện và hiển thị trạng thái phân loại rác trực quan.

## 1. Cấu Trúc Thành Phần Giao Diện (UI Architecture)

Bố cục phân tầng dạng module hoá thông qua `QSplitter` tỷ lệ 65:35, phân cách rõ rệt giữa vùng thị giác và thông tin kết quả.

| Thành phần | File triển khai | Vai trò chính |
| :--- | :--- | :--- |
| **Lớp tương thích** | `ui/qt_compat.py` | Trừu tượng hoá PyQt6 và PySide6, tránh khóa cứng vendor (vendor lock-in) |
| **Vùng xem ảnh/video** | `ui/widgets/video_view.py` | Giữ nguyên tỷ lệ khung hình (Aspect Ratio), vẽ qua `paintEvent` chống giật lag |
| **Biểu ngữ thùng rác** | `ui/widgets/bin_banner.py` | Chỉ thị thùng đích màu tương phản cao (WCAG AAA), trạng thái chờ "Chưa phát hiện rác" |
| **Thẻ kết quả** | `ui/widgets/result_card.py` | Thẻ phân loại hiển thị tên tiếng Việt, độ tin cậy %, huy hiệu thùng đích |
| **Bảng điều khiển** | `ui/widgets/control_panel.py` | Các nút chức năng nguồn dữ liệu và thanh trượt ngưỡng tin cậy |
| **Cửa sổ chính** | `ui/main_window.py` | Điểm kết nối trung tâm, quản lý kích thước tối thiểu 1280×720 (DoD) |

## 2. Bảng Màu 3 Thùng Rác Đích (Color Palette)

Thiết kế màu sắc đảm bảo độ tương phản cao, trực quan và dễ nhận biết ngay cả từ khoảng cách xa:

```
- THÙNG HỮU CƠ (ORGANIC):        #2E7D32 (Xanh lá đậm, chữ trắng #FFFFFF)
- THÙNG TÁI CHẾ (RECYCLABLE):     #1565C0 (Xanh dương cobalt, chữ trắng #FFFFFF)
- THÙNG VÔ CƠ / KHÁC (RESIDUAL):    #C62828 (Đỏ cam / Crimson, chữ trắng #FFFFFF)
- TRẠNG THÁI CHỜ (IDLE):          #374151 (Xám than slate, chữ #E5E7EB)
```

## 3. Kỹ Thuật Giữ Tỷ Lệ Khung Hình Không Méo (Aspect Ratio Preservation)

Thay vì dùng `QLabel.setScaledContents(True)` gây biến dạng tỷ lệ khung hình, widget `VideoView` nạp ảnh thành `QPixmap` và tính toán trong `paintEvent`:

```python
scaled = self._pixmap.scaled(
    self.size(),
    Qt.AspectRatioMode.KeepAspectRatio,
    Qt.TransformationMode.SmoothTransformation,
)
x = (self.width() - scaled.width()) // 2
y = (self.height() - scaled.height()) // 2
painter.drawPixmap(x, y, scaled)
```

---
[[trash-net-task - Lộ trình triển khai ứng dụng phân loại rác hoàn chỉnh]]
