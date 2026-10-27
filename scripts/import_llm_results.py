"""Проверяет архив Colab и переносит только артефакты детектора, без текстов."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

from dw1.models.llm_detectors import LLMScoreStore

ALLOWED = {
    "llm_scores.parquet",
    "llm_scores.manifest.json",
    "llm_calibration.json",
    "llm_calibration_metrics.json",
    "llm_calibrated_predictions.parquet",
    "llm_calibration_report.md",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/runs/sprint2"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.archive) as archive, tempfile.TemporaryDirectory(dir=args.out.parent) as temp:
        stage = Path(temp)
        if set(archive.namelist()) != ALLOWED:
            raise ValueError("Неожиданное содержимое архива; нужны только артефакты LLM-детектора")
        for name in ALLOWED:
            info = archive.getinfo(name)
            if info.file_size > 50 * 1024 * 1024:
                raise ValueError("Артефакт слишком большой для этого пилотного запуска")
            (stage / name).write_bytes(archive.read(name))
        store = LLMScoreStore(stage / "llm_scores.parquet", stage / "llm_calibration.json")
        manifest = json.loads((stage / "llm_scores.manifest.json").read_text())
        if manifest.get("status") != "completed" or manifest.get("detector") != store.model_version:
            raise ValueError("Прогон не завершён или версия в манифесте не совпадает")
        args.out.mkdir(parents=True, exist_ok=True)
        for name in sorted(ALLOWED):
            shutil.copyfile(stage / name, args.out / name)
    print(f"Проверенные LLM-результаты сохранены: {args.out}")


if __name__ == "__main__":
    main()
