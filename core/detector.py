"""Lớp bao đóng (wrapper) mô hình phân đoạn YOLO phục vụ nhận diện rác thải Trash-Net.

[KIẾN TRÚC & LUỒNG HOẠT ĐỘNG]:
1. Tự động nhận diện phần cứng (CUDA GPU -> Apple Silicon MPS -> CPU).
2. Tải trọng số mô hình (.pt), thực hiện khởi động ấm (warm-up) để nạp trước kernel tính toán.
3. Chạy suy luận với cơ chế phòng vệ chống tràn bộ nhớ GPU (OOM Fallback).
4. Phân tích kết quả: Bounding box, polygon mask và tự động ánh xạ loại rác vào thùng chứa.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import torch
from ultralytics import YOLO

from core.mapper import BinMapper
from core.types import Detection


class ModelLoadError(Exception):
    """Raised when loading model weights fails or weights file is corrupted."""
    pass


class InvalidFrameError(ValueError):
    """Raised when an invalid image frame is passed for inference."""
    pass


class WasteDetector:
    """Wrapper class around YOLO segmentation model for waste detection.

    [GIẢI THÍCH NGHIỆP VỤ - BỘ NHẬN DIỆN RÁC THẢI]:
    Lớp bao đóng (wrapper) mô hình phân đoạn YOLO phục vụ nhận diện rác thải.
    Tách biệt hoàn toàn với giao diện PyQt và chi tiết bên trong Ultralytics.
    Tự động chọn phần cứng (CUDA GPU -> Apple Silicon MPS -> CPU), nạp trọng số (.pt),
    chạy khởi động ấm (warmup), và chuyển đổi kết quả thành danh sách đối tượng Detection.

    Attributes:
        model_path (Path): Path to model weights file (.pt).
        conf (float): Default confidence threshold for detections.
        iou (float): Default NMS IoU threshold for detections.
        imgsz (int): Image dimension used for model inference.
        device (str): Computation device in use ('cuda', 'mps', or 'cpu').
        half (bool): Whether FP16 half-precision inference is enabled.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: Optional[str] = None,
        half: Optional[bool] = None,
        warmup: bool = True,
        bin_mapper: Optional[BinMapper] = None,
    ) -> None:
        """Initialize and warm up WasteDetector.

        Args:
            model_path: Path to YOLO weights (.pt).
            conf: Default detection confidence threshold in range (0.0, 1.0].
            iou: Default NMS IoU threshold in range (0.0, 1.0].
            imgsz: Input image size for inference (default: 640).
            device: Target hardware device ('cuda', 'mps', 'cpu', or None for auto-detect).
            half: Whether to enable FP16 half-precision (None = auto-enable on CUDA).
            warmup: Whether to execute one dummy forward pass on init.
            bin_mapper: Optional BinMapper instance. If None, default mapping is used.

        Raises:
            FileNotFoundError: If model_path does not exist.
            ModelLoadError: If model file cannot be loaded or is corrupted.
            ValueError: If conf, iou, or imgsz are out of valid bounds.
        """
        if not (0.0 < conf <= 1.0):
            raise ValueError(f"Confidence threshold 'conf' must be in (0.0, 1.0], got {conf}")
        if not (0.0 < iou <= 1.0):
            raise ValueError(f"IoU threshold 'iou' must be in (0.0, 1.0], got {iou}")
        if imgsz <= 0:
            raise ValueError(f"Image size 'imgsz' must be positive, got {imgsz}")

        self._model_path = Path(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.mapper = bin_mapper or BinMapper()

        # Tự động chọn thiết bị tính toán tối ưu: CUDA -> MPS -> CPU
        if device is None:
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            dev_str = str(device).lower()
            if dev_str.startswith("cuda") and not torch.cuda.is_available():
                self._device = "cpu"
            else:
                self._device = dev_str

        # Tự động bật FP16 trên GPU CUDA để tăng tốc độ suy luận
        if half is None:
            self.half = bool(self._device.startswith("cuda") and torch.cuda.is_available())
        else:
            self.half = bool(half)

        self._model: Optional[YOLO] = None
        self.load(self._model_path)

        if warmup:
            self._warmup()

    @property
    def device(self) -> str:
        """Tên thiết bị tính toán đang hoạt động ('cuda', 'mps', hoặc 'cpu')."""
        return self._device

    @property
    def half_precision(self) -> bool:
        """Trạng thái kích hoạt độ chính xác nửa FP16."""
        return self.half

    @property
    def iou_threshold(self) -> float:
        """Ngưỡng IoU khử trùng lặp NMS mặc định."""
        return self.iou

    @property
    def model_path(self) -> Path:
        """Đường dẫn đến file trọng số mô hình hiện tại."""
        return self._model_path

    @property
    def names(self) -> dict:
        """Bảng ánh xạ class_id sang tên lớp rác từ mô hình."""
        if self._model is not None and hasattr(self._model, "names"):
            return self._model.names
        return {}

    def load(self, model_path: Union[str, Path]) -> None:
        """Load model weights from disk.

        [GIẢI THÍCH NGHIỆP VỤ - TẢI MÔ HÌNH]:
        Nạp file trọng số YOLO (.pt) từ ổ đĩa và khởi tạo instance YOLO.
        Kiểm tra file tồn tại và bắt các ngoại lệ hỏng file để ném ra ModelLoadError.

        Args:
            model_path: Path to .pt model weights file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ModelLoadError: If the weights file is corrupted or cannot be parsed.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model weights file not found: '{path.resolve()}'. "
                f"Please provide a valid weights file (.pt)."
            )

        if not path.is_file():
            raise ModelLoadError(f"Specified path is not a file: '{path.resolve()}'")

        try:
            self._model = YOLO(str(path))
            self._model_path = path
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load YOLO model from '{path}': {exc}"
            ) from exc

    def _warmup(self) -> None:
        """Thực hiện một lượt suy luận giả định để nạp sẵn kernel tính toán vào GPU/CPU."""
        if self._model is None:
            return
        try:
            dummy_frame = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self._model.predict(
                source=dummy_frame,
                device=self._device,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )
        except Exception:
            # Nếu khởi động GPU thất bại (tràn VRAM hoặc lỗi driver), tự động lùi về CPU
            if self._device.startswith("cuda"):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._device = "cpu"
                self.half = False
                try:
                    dummy_frame = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                    self._model.predict(
                        source=dummy_frame,
                        device="cpu",
                        imgsz=self.imgsz,
                        conf=self.conf,
                        iou=self.iou,
                        verbose=False,
                    )
                except Exception:
                    pass

    def detect(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> List[Detection]:
        """Perform instance segmentation inference on an input frame.

        [GIẢI THÍCH NGHIỆP VỤ - NHẬN DIỆN RÁC THẢI]:
        1. Tiếp nhận khung hình mảng numpy ảnh BGR từ Webcam hoặc File ảnh.
        2. Chạy dự đoán phân đoạn đối tượng (Instance Segmentation) qua YOLO.
        3. Phân tích kết quả: Tọa độ bounding box, đa giác polygon mask, nhãn lớp.
        4. Ánh xạ từng loại rác vào thùng tương ứng (Tái chế / Hữu cơ / Vô cơ).
        5. Trả về danh sách đối tượng Detection sắp xếp giảm dần theo độ tin cậy.

        Args:
            frame: Numpy ndarray of shape (H, W, 3) representing an image in BGR format.
            conf: Optional override confidence threshold for this detection call.
            iou: Optional override NMS IoU threshold for this detection call.

        Returns:
            List[Detection]: List of structured detections, sorted descending by confidence.
                             Returns empty list [] if no objects are detected.

        Raises:
            InvalidFrameError: If frame is None, not an ndarray, or has empty/invalid dimensions.
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            raise RuntimeError("Model is not loaded. Call load() before detect().")

        if frame is None or not isinstance(frame, np.ndarray):
            raise InvalidFrameError(f"Input frame must be a valid numpy.ndarray, got {type(frame)}")

        if frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
            raise InvalidFrameError(
                f"Input frame must have 3 non-empty dimensions (H, W, 3), got shape {getattr(frame, 'shape', None)}"
            )

        threshold = conf if conf is not None else self.conf
        if not (0.0 < threshold <= 1.0):
            raise ValueError(f"Confidence threshold must be in (0.0, 1.0], got {threshold}")

        target_iou = iou if iou is not None else self.iou
        if not (0.0 < target_iou <= 1.0):
            raise ValueError(f"IoU threshold must be in (0.0, 1.0], got {target_iou}")

        orig_h, orig_w = frame.shape[:2]

        if self._model is None:
            raise ModelLoadError("Model is not initialized.")

        # Chạy suy luận với cơ chế bảo vệ chống tràn VRAM GPU (OOM Fallback sang CPU)
        try:
            raw_results: Any = self._model.predict(
                source=frame,
                conf=threshold,
                iou=target_iou,
                imgsz=self.imgsz,
                device=self._device,
                verbose=False,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if self._device.startswith("cuda"):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._device = "cpu"
                self.half = False
                raw_results = self._model.predict(
                    source=frame,
                    conf=threshold,
                    iou=target_iou,
                    imgsz=self.imgsz,
                    device="cpu",
                    verbose=False,
                )
            else:
                raise exc

        detections: List[Detection] = []
        if not raw_results:
            return detections

        result = raw_results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes_data = result.boxes.xyxy.cpu().numpy()
        conf_data = result.boxes.conf.cpu().numpy()
        cls_data = result.boxes.cls.cpu().numpy()

        masks_xy = None
        if result.masks is not None and hasattr(result.masks, "xy"):
            masks_xy = result.masks.xy

        for idx in range(len(boxes_data)):
            x1, y1, x2, y2 = boxes_data[idx]
            box_int = (
                max(0, int(round(x1))),
                max(0, int(round(y1))),
                min(orig_w, int(round(x2))),
                min(orig_h, int(round(y2))),
            )
            score = float(conf_data[idx])
            class_id = int(cls_data[idx])
            class_name = self.names.get(class_id, f"class_{class_id}")
            bin_type = self.mapper.map_class_to_bin(class_id)

            polygon_pts: List[Tuple[int, int]] = []
            if masks_xy is not None and idx < len(masks_xy) and len(masks_xy[idx]) > 0:
                pts = np.asarray(masks_xy[idx], dtype=np.float32)
                if pts.size > 0 and pts.ndim == 2 and pts.shape[1] >= 2:
                    pts_clipped = np.rint(pts).astype(int)
                    pts_clipped[:, 0] = np.clip(pts_clipped[:, 0], 0, orig_w)
                    pts_clipped[:, 1] = np.clip(pts_clipped[:, 1], 0, orig_h)
                    polygon_pts = [(int(p[0]), int(p[1])) for p in pts_clipped]
                else:
                    polygon_pts = [
                        (box_int[0], box_int[1]),
                        (box_int[2], box_int[1]),
                        (box_int[2], box_int[3]),
                        (box_int[0], box_int[3]),
                    ]
            else:
                polygon_pts = [
                    (box_int[0], box_int[1]),
                    (box_int[2], box_int[1]),
                    (box_int[2], box_int[3]),
                    (box_int[0], box_int[3]),
                ]

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=score,
                    box=box_int,
                    polygon=polygon_pts,
                    bin_type=bin_type,
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # =========================================================================
    # BÍ DANH TIẾNG VIỆT (VIETNAMESE METHOD ALIASES)
    # Giúp người mới đọc hiểu luồng xử lý mà không làm gãy chuẩn quốc tế
    # =========================================================================
    tai_mo_hinh = load
    nhan_dien = detect
    khoi_dong_am = _warmup


# =============================================================================
# BÍ DANH LỚP TIẾNG VIỆT (VIETNAMESE CLASS ALIASES)
# =============================================================================
BoNhanDienRac = WasteDetector
LoiTaiMoHinh = ModelLoadError
LoiKhungHinhKhongHopLe = InvalidFrameError

