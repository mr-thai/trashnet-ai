"""Điểm khởi chạy chính cho ứng dụng Trash-Net Desktop."""

# Nạp trước torchvision để đăng ký các toán tử C++ (torchvision::nms) khi chạy qua PyInstaller
try:
    import torchvision  # noqa: F401
    import torchvision.ops  # noqa: F401
except Exception:
    pass

# Điểm vào ứng dụng: Hỗ trợ cả tên tiếng Anh (run_app) và bí danh tiếng Việt (chay_ung_dung)
from ui.main_window import run_app, chay_ung_dung  # noqa: F401

if __name__ == "__main__":
    # Khởi chạy giao diện chính Trash-Net Desktop (chay_ung_dung tương đương run_app)
    chay_ung_dung()


