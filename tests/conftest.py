import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def _isolate_local_artifacts(monkeypatch, tmp_path):
    """Тесты не должны зависеть от локальных файлов калибровки/модели на машине."""
    monkeypatch.setenv("DW1_FUSION_CALIBRATION", str(tmp_path / "no-calibration.json"))
    monkeypatch.setenv("DW1_PROV_MODEL", str(tmp_path / "no-model.joblib"))

    import dw1.extractors.body_text as body_text
    import dw1.models.fusion as fusion

    fusion._CALIBRATION = None
    fusion._CALIBRATION_MTIME = 0.0
    body_text._MODEL = None
    body_text._MODEL_TRIED = False
