"""Unit tests verifying Vietnamese function and class aliases (Giải pháp 3).

Ensures that 100% of Vietnamese aliases point to the correct underlying
English implementations, work identically, and maintain complete backward compatibility.
"""

import pytest
from core import (
    # Classes
    BoNhanDienRac,
    WasteDetector,
    LoiTaiMoHinh,
    ModelLoadError,
    LoiKhungHinhKhongHopLe,
    InvalidFrameError,
    AnhXaThungRac,
    BinMapper,
    QuanLyHuongDan,
    GuidelineManager,
    BoLamMuotTheoThoiGian,
    TemporalSmoother,
    DoiTuongTheoDoi,
    TrackedTarget,
    TrangThaiTheoDoi,
    TrackState,
    NhatKyHeThong,
    AuditLogger,
    MucKiemToan,
    AuditEntry,
    BoKiemTraVatTheLa,
    OODDetector,
    CauHinhHeThong,
    AppConfig,
    ThungRac,
    Bin,
    KetQuaNhanDien,
    Detection,
    # Functions
    anh_xa_thung_rac,
    map_class_to_bin,
    lay_huong_dan_phan_loai,
    get_waste_guideline,
    tinh_chi_so_iou,
    compute_iou,
    luu_khung_hinh_nghi_ngo,
    save_flagged_frame,
)
from ui.main_window import run_app, chay_ung_dung


class TestVietnameseAliasesIdentity:
    """Verify identity and reference equality of Vietnamese aliases."""

    def test_class_aliases_identity(self):
        """Classes should be identical references."""
        assert BoNhanDienRac is WasteDetector
        assert LoiTaiMoHinh is ModelLoadError
        assert LoiKhungHinhKhongHopLe is InvalidFrameError
        assert AnhXaThungRac is BinMapper
        assert QuanLyHuongDan is GuidelineManager
        assert BoLamMuotTheoThoiGian is TemporalSmoother
        assert DoiTuongTheoDoi is TrackedTarget
        assert TrangThaiTheoDoi is TrackState
        assert NhatKyHeThong is AuditLogger
        assert MucKiemToan is AuditEntry
        assert BoKiemTraVatTheLa is OODDetector
        assert CauHinhHeThong is AppConfig
        assert ThungRac is Bin
        assert KetQuaNhanDien is Detection

    def test_function_aliases_identity(self):
        """Standalone functions should be identical references."""
        assert anh_xa_thung_rac is map_class_to_bin
        assert lay_huong_dan_phan_loai is get_waste_guideline
        assert tinh_chi_so_iou is compute_iou
        assert luu_khung_hinh_nghi_ngo is save_flagged_frame
        assert chay_ung_dung is run_app

    def test_method_aliases_identity(self):
        """Class methods should be identical references."""
        assert WasteDetector.nhan_dien is WasteDetector.detect
        assert WasteDetector.tai_mo_hinh is WasteDetector.load
        assert BinMapper.anh_xa is BinMapper.map_class_to_bin
        assert GuidelineManager.lay_huong_dan is GuidelineManager.get_guideline
        assert TemporalSmoother.cap_nhat is TemporalSmoother.update
        assert TemporalSmoother.lam_muot is TemporalSmoother.update
        assert TemporalSmoother.dat_lai is TemporalSmoother.reset
        assert AuditLogger.ghi_nhat_ky is AuditLogger.log_detection
        assert AuditLogger.thong_ke_theo_thung is AuditLogger.get_summary_by_bin
        assert AuditLogger.xoa_trang is AuditLogger.clear
        assert OODDetector.kiem_tra_ngoai_phan_phoi is OODDetector.is_ood
        assert OODDetector.kiem_tra_nhan_dien is OODDetector.check_detection


class TestVietnameseAliasesFunctionality:
    """Verify that calling functions via Vietnamese aliases works as expected."""

    def test_anh_xa_thung_rac_execution(self):
        """Mapping class ID 0 should return Bin.ORGANIC."""
        result = anh_xa_thung_rac(0)
        assert result == Bin.ORGANIC
        assert result == ThungRac.ORGANIC

    def test_tinh_chi_so_iou_execution(self):
        """IoU between identical bounding boxes should be 1.0."""
        box = (10, 10, 50, 50)
        iou = tinh_chi_so_iou(box, box)
        assert iou == pytest.approx(1.0)

    def test_lay_huong_dan_phan_loai_execution(self):
        """Guideline lookup for plastic should return actionable instructions."""
        guideline = lay_huong_dan_phan_loai("plastic")
        assert "title" in guideline
        assert "Nhựa" in guideline["title"]
        assert len(guideline.get("steps", [])) > 0

    def test_docstrings_contain_vietnamese_explanations(self):
        """Core modules should provide rich Vietnamese docstrings for beginners."""
        assert "[GIẢI THÍCH NGHIỆP VỤ" in WasteDetector.__doc__
        assert "[GIẢI THÍCH NGHIỆP VỤ" in WasteDetector.detect.__doc__
        assert "[GIẢI THÍCH NGHIỆP VỤ" in BinMapper.__doc__
        assert "[GIẢI THÍCH NGHIỆP VỤ" in GuidelineManager.__doc__
        assert "[GIẢI THÍCH NGHIỆP VỤ" in TemporalSmoother.__doc__
        assert "[GIẢI THÍCH NGHIỆP VỤ" in AuditLogger.__doc__
        assert "[GIẢI THÍCH NGHIỆP VỤ" in compute_iou.__doc__
