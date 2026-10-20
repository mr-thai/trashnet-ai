"""Architecture and modular project structure tests (US-5.2).

Verifies decoupling of core from GUI frameworks, existence of mandatory
directories, entry point validity, and pyproject.toml package configuration.
"""

from pathlib import Path
import re
import tomllib


MANDATORY_DIRS = [
    "ui",
    "core",
    "models",
    "config",
    "scripts",
    "tests",
    "reports",
]


class TestModularProjectStructure:
    """Verify acceptance criteria for US-5.2."""

    def test_mandatory_directories_exist(self):
        """DoD AC: Cấu trúc đề xuất: ui/, core/, models/, config/, scripts/, tests/, reports/."""
        root = Path(".")
        for d in MANDATORY_DIRS:
            dir_path = root / d
            assert dir_path.exists() and dir_path.is_dir(), f"Missing directory: {d}"

    def test_core_package_is_decoupled_from_gui(self):
        """DoD AC: Gói core không phụ thuộc PyQt/PySide; kiểm thử core chạy được không cần màn hình."""
        core_dir = Path("core")
        py_files = list(core_dir.glob("*.py"))
        assert len(py_files) > 0

        forbidden_patterns = [
            re.compile(r"^\s*from\s+(PyQt|PySide|tkinter)", re.MULTILINE),
            re.compile(r"^\s*import\s+(PyQt|PySide|tkinter)", re.MULTILINE),
        ]

        for pf in py_files:
            content = pf.read_text(encoding="utf-8")
            for pat in forbidden_patterns:
                match = pat.search(content)
                assert match is None, f"Found forbidden GUI import in {pf.name}: {match.group(0)}"

    def test_trashnet_package_entry_point(self):
        """DoD AC: Có điểm vào duy nhất: python -m trashnet."""
        import trashnet
        assert hasattr(trashnet, "__version__")
        main_py = Path("trashnet/__main__.py")
        assert main_py.exists()
        content = main_py.read_text(encoding="utf-8")
        assert "run_app" in content

    def test_pyproject_toml_configuration(self):
        """Tác vụ kỹ thuật: Dùng pyproject.toml để khai báo gói."""
        pyproject_file = Path("pyproject.toml")
        assert pyproject_file.exists()
        with open(pyproject_file, "rb") as f:
            data = tomllib.load(f)

        assert "project" in data
        assert data["project"]["name"] == "trashnet"
        assert "dependencies" in data["project"]
        assert "scripts" in data["project"]
        assert data["project"]["scripts"]["trashnet"] == "ui.main_window:run_app"

    def test_ci_workflow_configuration(self):
        """DoD AC (US-5.4): CI configuration exists, runs ruff, mypy, and offscreen pytest."""
        import yaml

        ci_yaml = Path(".github/workflows/ci.yml")
        assert ci_yaml.exists(), "Missing .github/workflows/ci.yml"

        content = ci_yaml.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        assert "jobs" in data

        # Check for ruff, mypy, and offscreen pytest
        workflow_str = str(data)
        assert "ruff" in workflow_str
        assert "mypy" in workflow_str
        assert "pytest" in workflow_str
        assert "QT_QPA_PLATFORM" in workflow_str
        assert "offscreen" in workflow_str

    def test_tests_data_sample_image_exists(self):
        """DoD AC (US-5.4): tests/data/ contains sample image for inference testing."""
        sample_img = Path("tests/data/sample_waste.jpg")
        assert sample_img.exists()
        assert sample_img.stat().st_size > 0
