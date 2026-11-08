"""Core inference and domain logic package for Trash-Net.

Independent of GUI (PyQt/PySide) and desktop presentation frameworks.
"""

from core.audit_log import AuditEntry, AuditLogger, MucKiemToan, NhatKyHeThong
from core.config import (
    AppConfig,
    ModelConfig,
    SmootherConfig,
    CameraConfig,
    UIConfig,
    CauHinhHeThong,
    CauHinhMoHinh,
    CauHinhBoLamMuot,
    CauHinhCamera,
    CauHinhGiaoDien,
)
from core.detector import (
    WasteDetector,
    ModelLoadError,
    InvalidFrameError,
    BoNhanDienRac,
    LoiTaiMoHinh,
    LoiKhungHinhKhongHopLe,
)
from core.flagging import save_flagged_frame, luu_khung_hinh_nghi_ngo
from core.guidelines import (
    GuidelineManager,
    get_waste_guideline,
    QuanLyHuongDan,
    lay_huong_dan_phan_loai,
)
from core.mapper import BinMapper, map_class_to_bin, AnhXaThungRac, anh_xa_thung_rac
from core.ood import OODDetector, BoKiemTraVatTheLa
from core.smoother import (
    TemporalSmoother,
    TrackedTarget,
    TrackState,
    compute_iou,
    BoLamMuotTheoThoiGian,
    DoiTuongTheoDoi,
    TrangThaiTheoDoi,
    tinh_chi_so_iou,
)
from core.types import Bin, Detection, ThungRac, KetQuaNhanDien

__all__ = [
    # Standard English exports
    "AuditEntry",
    "AuditLogger",
    "AppConfig",
    "ModelConfig",
    "SmootherConfig",
    "CameraConfig",
    "UIConfig",
    "WasteDetector",
    "ModelLoadError",
    "InvalidFrameError",
    "Bin",
    "Detection",
    "BinMapper",
    "map_class_to_bin",
    "TemporalSmoother",
    "TrackedTarget",
    "TrackState",
    "compute_iou",
    "OODDetector",
    "save_flagged_frame",
    "GuidelineManager",
    "get_waste_guideline",
    # Vietnamese aliases (Bí danh tiếng Việt cho người mới dễ tiếp cận)
    "MucKiemToan",
    "NhatKyHeThong",
    "CauHinhHeThong",
    "CauHinhMoHinh",
    "CauHinhBoLamMuot",
    "CauHinhCamera",
    "CauHinhGiaoDien",
    "BoNhanDienRac",
    "LoiTaiMoHinh",
    "LoiKhungHinhKhongHopLe",
    "ThungRac",
    "KetQuaNhanDien",
    "AnhXaThungRac",
    "anh_xa_thung_rac",
    "BoLamMuotTheoThoiGian",
    "DoiTuongTheoDoi",
    "TrangThaiTheoDoi",
    "tinh_chi_so_iou",
    "BoKiemTraVatTheLa",
    "luu_khung_hinh_nghi_ngo",
    "QuanLyHuongDan",
    "lay_huong_dan_phan_loai",
]
