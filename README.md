# Trash-Net — Hệ Thống Phân Loại & Nhận Diện Rác Thông Minh Thời Gian Thực (v1.0.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyQt6 / PySide6](https://img.shields.io/badge/GUI-PyQt6%20%7C%20PySide6-green.svg)](https://riverbankcomputing.com/software/pyqt/)
[![YOLO11n-seg](https://img.shields.io/badge/YOLO11-Segmentation-yellowgreen.svg)](https://github.com/ultralytics/ultralytics)
[![Tests Passing](https://img.shields.io/badge/tests-108%2F108%20passed-brightgreen.svg)](tests/)
[![Ruff Linter](https://img.shields.io/badge/linter-Ruff%20clean-blueviolet.svg)](https://github.com/astral-sh/ruff)
[![Mypy Type-Checked](https://img.shields.io/badge/types-Mypy%20verified-informational.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Trash-Net là ứng dụng nhận diện và phân đoạn rác thải thời gian thực dựa trên thị giác máy tính và trí tuệ nhân tạo biên (Edge AI), được xây dựng trên nền tảng **YOLO11n-seg** và **PyQt6/PySide6**. Ứng dụng tự động phân đoạn thực thể (Instance Segmentation), nhận diện 8 lớp rác thải phổ biến từ tập dữ liệu MultiTrash, hỗ trợ cả ảnh tĩnh, video và camera trực tiếp với khả năng chống rung, khử trùng lặp và vận hành bền bỉ nhiều giờ liên tục.

---

## 🌟 Tính Năng Nổi Bật (v1.0.0)

- **Phân đoạn thực thể tức thời (Real-time Instance Segmentation)**:
  - Ứng dụng mô hình YOLO11n-seg gọn nhẹ (~6.2 MB), trích xuất chính xác mặt nạ đa giác (polygon contours) và bounding box không cắt mép màn hình.
  - Tự động nhận diện thiết bị tính toán: **CUDA** (NVIDIA) ➔ **MPS** (Apple Silicon) ➔ **CPU**, tự động fallback về CPU an toàn nếu gặp sự cố driver hoặc tràn VRAM (CUDA Out-of-Memory).
- **Giao diện tối giản, tập trung vào nhận diện thực tế**:
  - Tối ưu hóa cho các bãi rác đa vật thể (Multi-Object Trash Piles), loại bỏ các nhãn nhóm thừa thãi gây rối mắt.
  - Nhãn hiển thị trực quan, tinh gọn: Tên lớp tiếng Việt chuẩn hóa kèm phần trăm tin cậy (ví dụ: `Nhựa (85%)`, `Kim loại (72%)`).
- **Khử trùng lặp & Điều khiển động thời gian thực (Interactive Sliders)**:
  - **Thanh trượt Ngưỡng tin cậy (Confidence)**: Kéo chỉnh từ `0.10` đến `0.95`.
  - **Thanh trượt Ngưỡng IoU (NMS Suppression)**: Kéo chỉnh từ `0.10` đến `0.95` giúp triệt tiêu tức thì các hộp bao đè lấn lên nhau trên cùng một đống rác.
  - **Bộ lọc Top-K (Giới hạn hiển thị)**: Tùy chọn lọc hiển thị `Tất cả`, `Top 5`, `Top 10`, `Top 15`, `Top 20`, `Top 30` vật thể có điểm tin cậy cao nhất.
  - **Bật/Tắt từng lớp đồ họa**: Tự do bật tắt độc lập `Mặt nạ`, `Hộp bao`, và `Nhãn`.
- **Co giãn thích ứng theo độ phân giải (Resolution-Adaptive Rendering)**:
  - Nét vẽ và kích thước chữ tự động co giãn theo kích thước ảnh thực tế (`0.45` trên ảnh nhỏ đến `1.8` trên ảnh 4K/Full HD), không bị che khuất rác và không bị vỡ nét.
- **Vận hành liên tục không đơ lag (Zero-Freeze Concurrency Engine)**:
  - Bộ đệm đơn khung (`size-1 buffer`) đảm bảo frame mới nhất luôn được xử lý, loại bỏ độ trễ tích lũy.
  - Cấu hình nhân V4L2 `BUFFERSIZE = 1` triệt tiêu độ trễ đệm của phần cứng camera trên Linux.
  - Cơ chế điều tiết luồng giao diện (UI Throttling 500ms) loại bỏ hoàn toàn hiện tượng tạo/hủy hàng trăm widget mỗi giây gây đơ máy khi stream webcam lâu dài.
- **Hệ thống cấu hình động tập trung (Dynamic Configuration)**:
  - Tệp cấu hình trung tâm `config/app.yaml` cho phép thay đổi toàn bộ thông số model, camera, smoother, và UI mà **không cần sửa một dòng code**.
  - Hỗ trợ đầy đủ cờ dòng lệnh CLI (`python main.py --help`) và biến môi trường (`TRASHNET_*`).
  - Màu sắc thùng rác và phân nhóm đồng bộ 100% từ `config/bin_mapping.yaml`.

---

## 💻 Yêu Cầu Hệ Thống

- **Hệ điều hành**: Linux (Ubuntu 20.04+, Arch Linux, Fedora,...), macOS 12+, hoặc Windows 10/11.
- **Python**: Phiên bản `>= 3.10` (Khuyến nghị Python 3.11 hoặc 3.12).
- **Thiết bị ghi hình**: Camera USB hoặc webcam tích hợp (tương thích V4L2 trên Linux).
- **Cấu hình phần cứng tối thiểu**:
  - CPU: Tối thiểu 4 nhân (x86_64 hoặc ARM64).
  - RAM: Tối thiểu 4 GB (Khuyến nghị 8 GB).
  - GPU (Tùy chọn): GPU NVIDIA hỗ trợ CUDA để tăng tốc suy luận tối đa.

---

## 🚀 Khởi Chạy Nhanh (Quick Start)

### Cách 1: Sử dụng script tự động (Khuyên dùng)

Dự án cung cấp kịch bản khởi chạy tự động chuẩn hóa môi trường:

```bash
./run.sh
```

> **`run.sh` sẽ tự động**:
> 1. Kiểm tra và khởi tạo môi trường ảo Python (`.venv`) nếu chưa có.
> 2. Cài đặt các thư viện cần thiết từ `requirements.txt`.
> 3. Kiểm tra tệp trọng số `models/best.pt` hoặc `best.pt`.
> 4. Khởi chạy ứng dụng Desktop.

---

### Cách 2: Khởi chạy thủ công

```bash
# 1. Kích hoạt môi trường ảo
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Chạy ứng dụng mặc định
python main.py
```

---

## ⚙️ Tùy Biến Qua Dòng Lệnh (CLI Arguments)

Ứng dụng hỗ trợ ghi đè linh hoạt mọi thông số từ terminal khi khởi chạy:

```bash
python main.py --help
```

```text
options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Đường dẫn tới tệp trọng số YOLO (.pt)
  -c CONF, --conf CONF  Ngưỡng tin cậy phát hiện mặc định (0.0, 1.0]
  --iou IOU             Ngưỡng IoU khử trùng lặp NMS (0.0, 1.0]
  --imgsz IMGSZ         Độ phân giải vuông đầu vào suy luận (mặc định: 640)
  -d DEVICE, --device DEVICE
                        Thiết bị tính toán: 'cuda', 'mps', hoặc 'cpu'
  --half                Bật cờ suy luận bán chính xác FP16 trên CUDA
  --config CONFIG       Đường dẫn tới tệp cấu hình YAML tùy biến
  --camera CAMERA       Chỉ mục thiết bị camera mặc định (ví dụ: 0, 1, 2)
```

**Ví dụ sử dụng:**
```bash
# Chạy với model tùy biến, ngưỡng tin cậy 0.5, ngưỡng IoU 0.4 trên GPU CUDA:
python main.py -m models/best.pt -c 0.50 --iou 0.40 -d cuda

# Chạy với camera phụ (index 1) và file cấu hình riêng:
python main.py --camera 1 --config config/app.yaml
```

---

## 📂 Cấu Trúc Dự Án (Clean Architecture)

```text
trash-Net/
├── ui/                     # TẦNG GIAO DIỆN (PyQt6 / PySide6)
│   ├── main_window.py      # Cửa sổ chính, tích hợp QSplitter & điều phối luồng
│   ├── worker.py           # Luồng nền suy luận AI bất đồng bộ (InferenceWorker)
│   ├── webcam_thread.py    # Luồng thu hình camera V4L2 không trễ đệm
│   ├── constants.py        # Bảng màu WCAG AAA, giao diện tối & từ điển nhãn tiếng Việt
│   ├── qt_compat.py        # Lớp tương thích lai PyQt6 / PySide6
│   └── widgets/            # Các thành phần Widget chuyên biệt
│       ├── video_view.py   # Khung hiển thị ảnh/video, giữ tỉ lệ & vẽ overlay thích ứng
│       ├── control_panel.py# Bảng điều khiển camera, slider Conf, IoU, Top-K & overlay toggles
│       ├── result_card.py  # Danh sách thẻ kết quả hiển thị rác phát hiện
│       ├── bin_banner.py   # Banner chỉ dẫn thùng rác quy chuẩn
│       ├── guideline_card.py # Thẻ hướng dẫn sơ chế và xử lý rác
│       └── audit_dialog.py # Hộp thoại xem lịch sử kiểm toán & xuất CSV
├── core/                   # TẦNG NGHIỆP VỤ LÕI (Decoupled, 100% không phụ thuộc GUI)
│   ├── config.py           # Trình nạp cấu hình tập trung AppConfig & biến môi trường
│   ├── detector.py         # YOLO11n-seg Wrapper, fallback CPU/GPU OOM & vector polygon
│   ├── mapper.py           # Quản lý ánh xạ nhãn sang thùng & nạp mã màu từ YAML
│   ├── smoother.py         # Thuật toán khử nhấp nháy chuỗi thời gian & IoU Tracker
│   ├── ood.py              # Bộ lọc rác bất thường ngoài phân phối (Out-of-Distribution)
│   ├── flagging.py         # Module gắn cờ rác nguy hại & xuất JSON metadata
│   ├── audit_log.py        # Quản lý nhật ký phiên kiểm toán & xuất CSV UTF-8 with BOM
│   ├── guidelines.py       # Trình tra cứu chỉ dẫn sơ chế rác tiếng Việt
│   └── types.py            # Kiểu dữ liệu bất biến (Bin enum, Detection dataclass)
├── config/                 # CẤU HÌNH TẬP TRUNG
│   ├── app.yaml            # Cấu hình vận hành hệ thống (Model, Camera, Smoother, UI)
│   ├── bin_mapping.yaml    # Ánh xạ nhãn MultiTrash sang nhóm thùng rác & mã màu
│   └── guidelines.yaml     # Nội dung hướng dẫn phân loại và sơ chế rác tiếng Việt
├── models/                 # TRỌNG SỐ MÔ HÌNH
│   ├── best.pt             # Trọng số mô hình chính thức (MultiTrash v8 Segmentation)
│   └── yolo11n-seg.pt      # Trọng số nền tảng Ultralytics YOLO11n
├── tests/                  # BỘ KIỂM THỬ TỰ ĐỘNG HÓA TOÀN DIỆN (108 Test Cases)
│   ├── test_config.py      # Kiểm thử nạp YAML, biến môi trường & fallback cấu hình
│   ├── test_detector.py    # Kiểm thử suy luận, kiểm tra biên ảnh rỗng, IoU, device fallback
│   ├── test_evaluation_and_accuracy.py # Kiểm định độ chính xác trên Ground Truth thực tế
│   ├── test_mapper.py      # Kiểm thử phân nhóm thùng và nạp mã màu từ YAML
│   ├── test_smoother.py    # Kiểm thử bộ lọc làm mượt chuỗi thời gian
│   ├── test_ood_and_flagging.py # Kiểm định phát hiện OOD và gắn cờ rác nguy hại
│   ├── test_audit_log.py   # Kiểm thử nhật ký phiên và xuất báo cáo CSV
│   ├── test_guidelines.py  # Kiểm thử nội dung hướng dẫn sơ chế
│   ├── test_structure.py   # Kiểm thử tính độc lập của tầng core (Zero GUI import)
│   └── test_ui.py          # Kiểm thử toàn diện 37 ca kiểm thử giao diện & đa luồng
├── scripts/                # Kịch bản phụ trợ và kiểm chuẩn
├── run.sh                  # Kịch bản khởi chạy tự động chuẩn hóa
├── pyproject.toml          # Khai báo gói dự án PEP 517/621
├── requirements.txt        # Thư viện vận hành chính thức
└── requirements-dev.txt    # Thư viện phục vụ kiểm thử và phát triển
```

---

## 🧪 Kiểm Chuẩn Chất Lượng (Quality Assurance & CI/CD)

Mọi thay đổi trên codebase đều được xác thực nghiêm ngặt qua bộ kiểm thử headless:

```bash
# 1. Kiểm tra cú pháp và quy chuẩn mã nguồn với Ruff
ruff check --no-cache core/ ui/ main.py

# 2. Kiểm tra an toàn kiểu dữ liệu tĩnh với Mypy
mypy core/

# 3. Chạy toàn bộ 108 bài kiểm thử tự động (Headless offscreen)
QT_QPA_PLATFORM=offscreen pytest tests/ -v
```

**Kết quả kiểm chuẩn v1.0.0:**
- **Ruff Linter:** `All checks passed!` (0 cảnh báo, 0 lỗi).
- **Mypy Type-Checker:** `Success: no issues found in 10 source files` (100% Type-safe).
- **Pytest Suite:** **108 / 108 tests PASSED** (Thời gian chạy: ~25-35s).

---

## 🔧 Xử Lý Sự Cố Thường Gặp (Troubleshooting)

### 1. Khuôn mặt hoặc vật dụng văn phòng bị nhận diện thành rác
- **Nguyên nhân:** Mô hình được huấn luyện trên 8 nhóm rác thải (`plastic`, `glass`, `metal`, `paper`,...). Khuôn mặt người hoặc đồ dùng công nghệ là dữ liệu ngoài phân phối (**OOD**). Khi để thanh trượt Confidence quá thấp (`<= 0.25`), mô hình có xu hướng đoán bừa vào nhãn gần nhất.
- **Khắc phục:** Kéo thanh trượt **Ngưỡng tin cậy (Confidence) lên mức `>= 0.45 – 0.60`** và hướng camera vào khu vực chứa rác.

### 2. Các hộp bao nhận diện bị đè lấn nhau trên đống rác lớn
- **Khắc phục:** 
  1. Kéo thanh trượt **Ngưỡng IoU (Khử đè lấn)** về mức thấp hơn (`0.30 – 0.45`) để thuật toán NMS tự động loại bỏ các hộp đè lên nhau.
  2. Chọn chế độ **Top 10** hoặc **Top 15** ở mục *Giới hạn số lượng hiển thị* để màn hình luôn thoáng đãng.

### 3. Lỗi quyền truy cập Camera trên Linux (`/dev/video*`)
- **Khắc phục:** Thêm tài khoản người dùng vào nhóm `video` và đăng nhập lại:
  ```bash
  sudo usermod -aG video $USER
  ```

### 4. Lỗi thiếu thư viện đồ họa Qt trên Linux (`Could not load Qt platform plugin "xcb"`)
- **Ubuntu/Debian:**
  ```bash
  sudo apt update && sudo apt install -y libegl1 libgl1 libxcb-cursor0 libxkbcommon-x11-0
  ```
- **Arch Linux / Manjaro:**
  ```bash
  sudo pacman -S --needed libxcb xcb-util-cursor libxkbcommon-x11
  ```

---

## 📜 Giấy Phép & Đóng Góp

Dự án được phân phối dưới giấy phép mã nguồn mở **MIT License**. Mọi ý kiến đóng góp, báo cáo lỗi hoặc đề xuất cải tiến thuật toán đều được hoan nghênh thông qua GitHub Issues và Pull Requests!
