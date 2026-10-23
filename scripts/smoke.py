"""Smoke-проверка M0 (только stdlib): сквозной прогон пайплайна на фикстуре."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dw1.pipeline import analyze  # noqa: E402


def main() -> int:
    fixture = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample.eml"
    ev = analyze(fixture)
    d = ev.to_dict()
    assert d["schema_version"] == "1.0"
    assert len(d["signals"]) == 9, f"ожидалось 9 экстракторов, получено {len(d['signals'])}"
    assert d["parts"]["links"], "ссылка из HTML должна быть извлечена"
    assert d["fusion"]["rejection"] is True, "M0: фузия должна отказываться от ответа"

    print("SMOKE OK: 9 экстракторов, Evidence v1 валиден, ссылки извлечены")
    print(json.dumps({k: d[k] for k in ("message_id", "fusion", "campaign")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
