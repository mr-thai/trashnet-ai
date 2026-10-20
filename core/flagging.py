"""Manual reporting and hazardous waste flagging system (US-4.4).

Persists flagged camera frames and structured JSON metadata to the `flagged/`
directory for safety logging and future retraining dataset collection.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import cv2
import numpy as np

from core.types import Detection


def save_flagged_frame(
    frame: np.ndarray,
    detections: List[Detection],
    is_hazardous: bool = True,
    note: str = "",
    flagged_dir: Union[str, Path] = "flagged",
) -> Tuple[Path, Path]:
    """Save the current frame image and metadata JSON into the flagged directory.

    Args:
        frame: OpenCV BGR image array.
        detections: List of current active Detection objects.
        is_hazardous: Whether the user marked the object as hazardous (batteries, e-waste).
        note: Optional user comment or explanation.
        flagged_dir: Target destination directory path.

    Returns:
        Tuple of (saved_image_path, saved_json_path).
    """
    out_dir = Path(flagged_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    timestamp_str = now.isoformat()
    base_name = f"flagged_{now.strftime('%Y%m%d_%H%M%S_%f')}"

    img_path = out_dir / f"{base_name}.jpg"
    json_path = out_dir / f"{base_name}.json"

    # Save frame as JPEG
    cv2.imwrite(str(img_path), frame)

    # Format structured metadata
    metadata: Dict[str, Any] = {
        "timestamp": timestamp_str,
        "image_filename": img_path.name,
        "hazardous": bool(is_hazardous),
        "user_note": str(note),
        "detection_count": len(detections),
        "detections": [
            {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": round(float(d.confidence), 4),
                "bin_type": d.bin_type.value,
                "box": list(d.box),
            }
            for d in detections
        ],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return img_path, json_path
