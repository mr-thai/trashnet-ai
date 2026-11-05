"""Model wrapper for YOLO instance segmentation in Trash-Net (US-2.1 & US-2.3).

Decoupled from GUI frameworks (PyQt/PySide) and Ultralytics internals.
Handles device detection, model loading, warm-up, FP16/half-precision optimization,
and returns structured Detection objects with bin mappings sorted by confidence.
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

        # Auto-detect hardware device: CUDA -> MPS -> CPU with graceful fallback
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

        # Configure FP16 half precision: auto-enable on CUDA unless explicitly set
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
        """Get the active computation device name ('cuda', 'mps', or 'cpu')."""
        return self._device

    @property
    def half_precision(self) -> bool:
        """Get whether FP16 half-precision is enabled."""
        return self.half

    @property
    def iou_threshold(self) -> float:
        """Get the default NMS IoU threshold."""
        return self.iou

    @property
    def model_path(self) -> Path:
        """Get path to currently loaded model weights."""
        return self._model_path

    @property
    def names(self) -> dict:
        """Get mapping of class_id to class name from loaded model."""
        if self._model is not None and hasattr(self._model, "names"):
            return self._model.names
        return {}

    def load(self, model_path: Union[str, Path]) -> None:
        """Load model weights from disk.

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
        """Execute one forward pass with a dummy blank image to prime GPU/CPU kernels."""
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
            # If warmup fails on GPU (e.g. CUDA OOM or driver issue), fallback to CPU
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

        Passes BGR frame directly to YOLO (Ultralytics handles internal color formatting),
        parses raw predictions, converts polygon masks to original image pixel coordinates,
        maps classes to sorting bins, and returns structured Detection objects sorted descending by confidence.

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

        # Execute prediction with GPU OOM protection and automatic CPU fallback
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

        boxes_data = result.boxes.xyxy.cpu().numpy()  # [[x1, y1, x2, y2], ...]
        conf_data = result.boxes.conf.cpu().numpy()    # [conf, ...]
        cls_data = result.boxes.cls.cpu().numpy()      # [cls_id, ...]

        # Extract polygon contours if segmentation masks exist
        masks_xy = None
        if result.masks is not None and hasattr(result.masks, "xy"):
            masks_xy = result.masks.xy  # List of (N, 2) arrays in original image pixel coords

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

            # Extract polygon pixel coordinates using vectorized NumPy operations
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
                # Fallback: 4 corners of bounding box if mask is missing
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

        # Sort descending by confidence score (AC requirement)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
