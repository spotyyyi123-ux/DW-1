import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def _no_fusion_calibration(monkeypatch, tmp_path):
    """Тесты не должны зависеть от локального файла калибровки на машине."""
    monkeypatch.setenv("DW1_FUSION_CALIBRATION", str(tmp_path / "no-calibration.json"))
    import dw1.models.fusion as fusion

    fusion._CALIBRATION = None
    fusion._CALIBRATION_MTIME = 0.0
