# -*- coding: utf-8 -*-
"""Тема письма (SPEC §6 W2, отдельная короткая ветка).

Будет реализовано: стиль, эмодзи, urgency-лексика, длина, персональное обращение,
несоответствие Re:/Fwd: истории треда, несогласованность темы и тела
(подпись гибридного происхождения ai_polished)."""
from dw1.extractors import BaseExtractor, register


@register
class SubjectExtractor(BaseExtractor):
    name = "subject"
    version = "stub-0.1"
