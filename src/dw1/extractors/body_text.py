# -*- coding: utf-8 -*-
"""Текст тела (SPEC §6 W2) — основная текстовая ветка.

Будет реализовано:
- стилометрия: burstiness, MTLD/TTR, профиль пунктуации, частота опечаток;
- TF-IDF (word+char n-grams) + HistGB — базовый детектор (W1 отдаёт скоры);
- внешний контур: Fast-DetectGPT (основной), Binoculars/DetectGPT (эталоны);
- калибровка по языкам (EN/RU, замер ECE), класс ai_polished (SemEval-8 subtask C)."""
from dw1.extractors import BaseExtractor, register


@register
class BodyTextExtractor(BaseExtractor):
    name = "body_text"
    version = "stub-0.1"
