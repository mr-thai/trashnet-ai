"""UI utility functions: Unicode image loader and validation (US-3.3).
"""

from pathlib import Path
from typing import Union
import cv2
import numpy as np

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FILE_DIALOG_IMAGE_FILTER = "Hình ảnh (*.jpg *.jpeg *.png *.bmp *.webp)"


def load_image_unicode(file_path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk using OpenCV with full Unicode path support.

    Standard cv2.imread fails on paths containing non-ASCII / Vietnamese characters.
    Using numpy.fromfile + cv2.imdecode handles any Unicode file path robustly.

    Args:
        file_path: Absolute or relative path to the image file.

    Returns:
        Numpy array containing BGR image data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file extension is unsupported or content is corrupted/empty.
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Tệp không tồn tại: '{path}'")

    if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
        exts_str = ", ".join(sorted(VALID_IMAGE_EXTENSIONS))
        raise ValueError(
            f"Định dạng tệp '{path.suffix}' không được hỗ trợ. "
            f"Vui lòng chọn ảnh định dạng: {exts_str}"
        )

    try:
        raw_bytes = np.fromfile(str(path), dtype=np.uint8)
        if raw_bytes.size == 0:
            raise ValueError(f"Tệp '{path.name}' rỗng (0 bytes).")

        image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Không thể giải mã dữ liệu ảnh từ '{path.name}'. Tệp có thể đã bị hỏng.")

        return image
    except Exception as exc:
        if isinstance(exc, (ValueError, FileNotFoundError)):
            raise
        raise ValueError(f"Lỗi khi đọc tệp ảnh '{path.name}': {exc}") from exc
