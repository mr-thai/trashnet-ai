# Trash-Net — Hệ Thống Phân Loại & Nhận Diện Rác Thông Minh Thời Gian Thực (v1.0.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyQt6 / PySide6](https://img.shields.io/badge/GUI-PyQt6%20%7C%20PySide6-green.svg)](https://riverbankcomputing.com/software/pyqt/)
[![YOLO11n-seg](https://img.shields.io/badge/YOLO11-Segmentation-yellowgreen.svg)](https://github.com/ultralytics/ultralytics)
[![Tests Passing](https://img.shields.io/badge/tests-113%2F113%20passed-brightgreen.svg)](tests/)
[![Ruff Linter](https://img.shields.io/badge/linter-Ruff%20clean-blueviolet.svg)](https://github.com/astral-sh/ruff)
[![Mypy Type-Checked](https://img.shields.io/badge/types-Mypy%20verified-informational.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Trash-Net là ứng dụng nhận diện và phân đoạn rác thải thời gian thực dựa trên thị giác máy tính và trí tuệ nhân tạo biên (Edge AI), được xây dựng trên nền tảng **YOLO11n-seg** và **PyQt6/PySide6**. Hệ thống hỗ trợ tự động phân đoạn thực thể (Instance Segmentation), nhận diện 8 lớp rác phổ biến từ tập dữ liệu MultiTrash, phân loại vào 3 thùng quy chuẩn, hỗ trợ 3 chế độ vận hành (Ảnh tĩnh, Video tệp tin và Webcam trực tiếp) với khả năng chống rung nhấp nháy, khử trùng lặp và vận hành bền bỉ.

---

## 🌟 Tính Năng Nổi Bật

### 1. Ba Chế Độ Nhận Diện Linh Hoạt (Multi-Input Modes)
- **Ảnh tĩnh (Static Image)**: Kéo thả trực tiếp (Drag & Drop) hoặc mở tệp ảnh hỗ trợ đường dẫn tiếng Việt Unicode (`.jpg`, `.png`, `.webp`, `.bmp`, `.tiff`).
- **Phát video quét theo nhịp AI (Inference-Paced Video Playback)**: Phân tích video từng khung hình đồng bộ theo tốc độ suy luận của phần cứng (`Zero-Drop`), tự động điều tiết tốc độ phát để không bỏ sót bất kỳ vật thể rác nào.
- **Webcam thời gian thực (Zero-Latency Streaming)**: Kết nối camera USB hoặc camera tích hợp, đọc qua backend V4L2 chuẩn Linux với bộ đệm đơn khung (`size-1 buffer`) triệt tiêu hoàn toàn độ trễ tích lũy.

### 2. Phân Đoạn Thực Thể & Nhận Diện Đa Vật Thể (Instance Segmentation)
- **Mô hình YOLO11n-seg tối ưu**: Trọng lượng nhẹ (~6.2 MB), phân đoạn chính xác đa giác viền thực thể (polygon contours) và bounding box không cắt mép màn hình.
- **Tự động nhận diện thiết bị tính toán**: Ưu tiên **CUDA** (NVIDIA GPU) ➔ **MPS** (Apple Silicon) ➔ **CPU**, tự động fallback về CPU an toàn nếu gặp sự cố driver hoặc tràn bộ nhớ VRAM (CUDA Out-of-Memory).
- **Hỗ trợ độ chính xác nửa (FP16)**: Tự động kích hoạt trên GPU CUDA để tăng tốc độ xử lý khung hình.

### 3. Khử Rung Nhấp Nháy & Chống Trùng Lặp (Temporal Smoothing & Filtering)
- **Bộ làm mượt chuỗi thời gian (TemporalSmoother)**: Theo dõi vật thể liên tục qua các khung hình dựa trên chỉ số IoU; yêu cầu nhãn rác duy trì ổn định liên tục tối thiểu 1.5 giây trước khi cập nhật nhãn, loại bỏ hiện tượng nhấp nháy do nhiễu camera.
- **Thanh trượt điều khiển động thời gian thực (Interactive Sliders)**:
  - **Ngưỡng tin cậy (Confidence)**: Kéo chỉnh từ `0.10` đến `0.95`.
  - **Ngưỡng IoU (NMS Suppression)**: Kéo chỉnh từ `0.10` đến `0.95` giúp lọc bỏ các khung bao đè lấn nhau trên cùng một đống rác.
  - **Giới hạn số lượng (Top-K Filter)**: Tùy chọn lọc hiển thị `Tất cả`, `Top 5`, `Top 10`, `Top 15`, `Top 20`, `Top 30` vật thể có độ tin cậy cao nhất.
  - **Bật/Tắt từng lớp đồ họa**: Bật tắt độc lập hiển thị `Mặt nạ`, `Khung bao` và `Nhãn`.

### 4. An Toàn & Kiểm Toán (Safety & Compliance)
- **Phát hiện dữ liệu bất thường (OOD Detection)**: Tự động cảnh báo và chuyển sang thùng *Không xác định* đối với các vật thể lạ ngoài phân phối dữ liệu huấn luyện.
- **Gắn cờ rác nguy hại (Hazardous Waste Flagging)**: Cho phép người dùng báo cáo thủ công các loại rác nguy hại (pin, đồ điện tử...), lưu ảnh JPEG kèm file siêu dữ liệu JSON vào thư mục `flagged/`.
- **Nhật ký kiểm toán & Xuất báo cáo (Audit Trail & CSV Export)**: Ghi nhận tự động các sự kiện phân loại có cơ chế chống ghi trùng lặp (Cooldown); xuất báo cáo CSV định dạng chuẩn UTF-8 BOM hiển thị chuẩn tiếng Việt trên Microsoft Excel.

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

### Cách 1: Khởi chạy tự động bằng script (Khuyên dùng)
```bash
./run.sh
```
*Script sẽ tự động kiểm tra môi trường ảo `.venv`, cài đặt thư viện thiếu và khởi động ứng dụng.*

---

### Cách 2: Khởi chạy thủ công
```bash
# 1. Kích hoạt môi trường ảo
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate

# 2. Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# 3. Khởi chạy ứng dụng
python main.py
```

---

## ⚙️ Tùy Biến Qua Dòng Lệnh (CLI Options)

Ứng dụng hỗ trợ ghi đè linh hoạt mọi thông số từ terminal khi khởi chạy:

```bash
python main.py --help
```

| Tham số | Tác dụng | Mặc định |
| :--- | :--- | :--- |
| `-m`, `--model` | Đường dẫn tới tệp trọng số YOLO (`.pt`) | `models/best.pt` |
| `-c`, `--conf` | Ngưỡng tin cậy phát hiện mặc định `(0.0, 1.0]` | `0.25` |
| `--iou` | Ngưỡng IoU khử trùng lặp NMS `(0.0, 1.0]` | `0.45` |
| `--imgsz` | Độ phân giải vuông đầu vào suy luận | `640` |
| `-d`, `--device` | Thiết bị tính toán: `'cuda'`, `'mps'`, hoặc `'cpu'` | Tự động nhận diện |
| `--half` | Bật cờ suy luận bán chính xác FP16 trên CUDA | Tự động trên CUDA |
| `--config` | Đường dẫn tới tệp cấu hình YAML tùy biến | `config/app.yaml` |
| `--camera` | Chỉ mục thiết bị camera mặc định (ví dụ: `0`, `1`) | `0` |

**Ví dụ sử dụng:**
```bash
# Chạy với model tùy biến, ngưỡng tin cậy 0.50, ngưỡng IoU 0.40 trên GPU CUDA:
python main.py -m models/best.pt -c 0.50 --iou 0.40 -d cuda

# Chạy với camera ngoài (index 1) và file cấu hình riêng:
python main.py --camera 1 --config config/app.yaml
```

---

## 📂 Cấu Trúc Dự Án (Clean Modular Architecture)

```text
trash-Net/
├── ui/                     # TẦNG GIAO DIỆN (PyQt6 / PySide6)
│   ├── main_window.py      # Cửa sổ chính, tích hợp QSplitter & điều phối luồng
│   ├── worker.py           # Luồng nền suy luận AI bất đồng bộ (InferenceWorker)
│   ├── webcam_thread.py    # Luồng thu hình camera V4L2 không trễ đệm (WebcamThread)
│   ├── video_thread.py     # Luồng phân tích tệp video quét theo nhịp AI (VideoThread)
│   ├── constants.py        # Bảng màu WCAG AAA, giao diện tối & từ điển nhãn tiếng Việt
│   ├── qt_compat.py        # Lớp trừu tượng tương thích lai PyQt6 / PySide6
│   └── widgets/            # Các thành phần giao diện chuyên biệt
│       ├── video_view.py   # Khung hiển thị ảnh/video, giữ tỉ lệ & vẽ overlay thích ứng
│       ├── control_panel.py# Bảng điều khiển camera, slider Conf, IoU, Top-K & overlay toggles
│       ├── result_card.py  # Danh sách thẻ kết quả hiển thị rác phát hiện
│       ├── bin_banner.py   # Banner chỉ dẫn thùng rác quy chuẩn
│       └── audit_dialog.py # Hộp thoại xem lịch sử kiểm toán & xuất báo cáo CSV
├── core/                   # TẦNG NGHIỆP VỤ LÕI (Decoupled, 100% không phụ thuộc GUI)
│   ├── config.py           # Trình nạp cấu hình tập trung AppConfig & biến môi trường
│   ├── detector.py         # YOLO11n-seg Wrapper, fallback CPU/GPU OOM & trích xuất polygon
│   ├── mapper.py           # Quản lý ánh xạ nhãn sang thùng & nạp mã màu từ YAML
│   ├── smoother.py         # Thuật toán khử nhấp nháy chuỗi thời gian & IoU Tracker
│   ├── ood.py              # Bộ lọc rác bất thường ngoài phân phối (Out-of-Distribution)
│   ├── flagging.py         # Module gắn cờ rác nguy hại & xuất JSON metadata
│   ├── audit_log.py        # Quản lý nhật ký phiên kiểm toán & xuất CSV UTF-8 with BOM
│   └── types.py            # Kiểu dữ liệu bất biến (Bin enum, Detection dataclass)
├── config/                 # CẤU HÌNH TẬP TRUNG (YAML)
│   ├── app.yaml            # Cấu hình vận hành hệ thống (Model, Camera, Smoother, UI)
│   └── bin_mapping.yaml    # Ánh xạ nhãn MultiTrash sang nhóm thùng rác & mã màu
├── models/                 # TRỌNG SỐ MÔ HÌNH
│   ├── best.pt             # Trọng số mô hình chính thức (MultiTrash v8 Segmentation)
│   └── yolo11n-seg.pt      # Trọng số nền tảng Ultralytics YOLO11n
├── tests/                  # BỘ KIỂM THỬ TỰ ĐỘNG HÓA TOÀN DIỆN (113 Test Cases)
│   ├── test_config.py      # Kiểm thử nạp YAML, biến môi trường & fallback cấu hình
│   ├── test_detector.py    # Kiểm thử suy luận, kiểm tra biên ảnh rỗng, IoU, device fallback
│   ├── test_evaluation_and_accuracy.py # Kiểm định độ chính xác trên Ground Truth thực tế
│   ├── test_mapper.py      # Kiểm thử phân nhóm thùng và nạp mã màu từ YAML
│   ├── test_smoother.py    # Kiểm thử bộ lọc làm mượt chuỗi thời gian
│   ├── test_ood_and_flagging.py # Kiểm định phát hiện OOD và gắn cờ rác nguy hại
│   ├── test_audit_log.py   # Kiểm thử nhật ký phiên và xuất báo cáo CSV
│   ├── test_structure.py   # Kiểm thử tính độc lập của tầng core (Zero GUI import)
│   ├── test_ui.py          # Kiểm thử 37 ca kiểm thử giao diện & đa luồng
│   ├── test_video_thread.py# Kiểm thử luồng phân tích video quét theo nhịp AI
│   └── test_vietnamese_aliases.py # Kiểm thử tính toàn vẹn của giao diện hàm mở rộng
├── scripts/                # Kịch bản phụ trợ và kiểm chuẩn benchmark
├── run.sh                  # Kịch bản khởi chạy tự động chuẩn hóa
├── pyproject.toml          # Khai báo gói dự án PEP 517/621
├── requirements.txt        # Thư viện vận hành chính thức
└── requirements-dev.txt    # Thư viện phục vụ kiểm thử và phát triển
```

---

## 🧪 Kiểm Chuẩn Chất Lượng (Quality Assurance & CI/CD)

Mọi thay đổi trên codebase đều được xác thực nghiêm ngặt qua bộ công cụ kiểm thử:

```bash
# 1. Kiểm tra cú pháp và quy chuẩn mã nguồn với Ruff
ruff check --no-cache core/ ui/ main.py

# 2. Kiểm tra an toàn kiểu dữ liệu tĩnh với Mypy
mypy core/

# 3. Chạy toàn bộ 113 bài kiểm thử tự động (Headless offscreen)
QT_QPA_PLATFORM=offscreen pytest tests/ -v
```

**Kết quả kiểm chuẩn v1.0.0:**
- **Ruff Linter:** `All checks passed!` (0 cảnh báo, 0 lỗi cú pháp).
- **Mypy Type-Checker:** `Success: no issues found in 10 source files` (100% Type-safe).
- **Pytest Suite:** **113 / 113 tests PASSED** (Thời gian chạy: ~25-35s).

---

## 🔧 Xử Lý Sự Cố Thường Gặp (Troubleshooting)

### 1. Khuôn mặt hoặc vật dụng văn phòng bị nhận diện thành rác
- **Nguyên nhân:** Mô hình được huấn luyện trên 8 nhóm rác thải (`plastic`, `glass`, `metal`, `paper`,...). Khuôn mặt người hoặc đồ dùng công nghệ là dữ liệu ngoài phân phối (**OOD**). Khi để thanh trượt Confidence quá thấp (`<= 0.25`), mô hình có xu hướng gán vào nhãn gần nhất.
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

