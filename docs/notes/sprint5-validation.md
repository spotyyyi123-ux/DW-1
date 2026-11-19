# Валидация RU и визуальных сигналов (Sprint 5, задача 4)

## RU-детекция: текущий baseline

Из Sprint 3 (`reports/sprint3-llm/metrics.json`, HC3/CoAT, Platt на train + isotonic на val):

| Детектор | Язык | n test | ECE до → после | F1 | ROC-AUC |
|---|---|---|---|---|---|
| GPT-Neo 2.7B | en | 90 | 0.236 → 0.024 | 0.990 | 0.989 |
| GPT-Neo 2.7B | ru | 71 | 0.230 → 0.169 | 0.381 | 0.382 |
| Qwen2.5-0.5B | en | 90 | 0.219 → 0.106 | 0.821 | 0.917 |
| Qwen2.5-0.5B | ru | 71 | 0.269 → 0.313 | 0.400 | 0.361 |

**Цель Sprint 5 (Алина):** RU-модель v3 (Qwen2.5-1.5B / mGPT) → сравнить с этим baseline.
Команда сравнения (когда появятся скоры):

```powershell
python scripts/compare_detectors.py --scores "qwen_v3=path/to/new_scores.parquet" `
  --scores "qwen_v1=data/runs/sprint3/qwen_scores.parquet"
```

## Визуальные сигналы: устойчивость QR/изображений

Скрипт: `scripts/validate_visual.py`; эталон — QR-картинка из `docs/examples/task3/images/`.
OpenCV 5.1.0 (`opencv-python-headless`), декодер `cv2.QRCodeDetector`.

| Искажение | QR-статус | Декодирован | pHash dist |
|---|---|---|---:|
| original | decoded | да | 0 |
| jpeg_q90 / q70 / q50 / q30 | decoded | да | 0 |
| resize_50 / resize_25 | decoded | да | 0 |
| **resize_15** | detected_undecoded | **нет** | 0 |
| rotate_5 / 15 / 45 | decoded | да | 8 / 20 / 26 |
| grayscale, blur_1/2/3, darken_0.7 | decoded | да | 0 |
| contrast_0.5, noise | decoded | да | 2 |
| **perspective** | detected_undecoded | **нет** | 22 |
| occlusion, downup | decoded | да | 0 |

**Decode rate: 19/21 = 90%**; провалы: сильный даунскейл (resize_15) и перспективный сдвиг
(perspective) — цели для задачи 3 (препроцессинг: upscale, adaptive threshold, поворот/перспектива).

pHash ведёт себя ожидаемо: 0 при сжатии/ресайзе/блюре, 8–26 при поворотах и перспективе —
это хороший индикатор «изображение изменено/повёрнуто».

## Как перезапустить

```powershell
python scripts/validate_visual.py
python scripts/external_validation.py
```

Артефакт: `data/runs/sprint5/visual_validation.json`.
