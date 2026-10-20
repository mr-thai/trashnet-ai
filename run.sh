#!/usr/bin/env bash
# ==============================================================================
# Trash-Net: One-click launcher script (US-5.3)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "  🚀 Đang khởi động Trash-Net — Phân Loại Rác Thông Minh"
echo "======================================================================"

# 1. Check or initialize Python virtual environment
if [ ! -d ".venv" ]; then
    echo "[*] Chưa tìm thấy môi trường ảo .venv. Đang tự động khởi tạo..."
    python3 -m venv .venv
    echo "[*] Đang cài đặt các thư viện phụ thuộc từ requirements.txt..."
    ./.venv/bin/pip install --upgrade pip
    ./.venv/bin/pip install -r requirements.txt
    echo "[✓] Thiết lập môi trường ảo hoàn tất!"
fi

# 2. Check model weights
if [ ! -f "models/best.pt" ] && [ ! -f "best.pt" ] && [ ! -f "models/yolo11n-seg.pt" ] && [ ! -f "yolo11n-seg.pt" ]; then
    echo "[!] Cảnh báo: Chưa tìm thấy tệp trọng số (models/best.pt hoặc yolo11n-seg.pt)!"
    echo "    Ứng dụng vẫn sẽ khởi chạy ở chế độ giao diện chờ."
fi

# 3. Launch application via standard package entry point
echo "[*] Đang mở ứng dụng Trash-Net..."
export PYTHONPATH=.
./.venv/bin/python -m trashnet "$@"
