# -*- coding: utf-8 -*-
"""Шаблон кампании (SPEC §6 W4).

Будет реализовано: SimHash/MinHash + LSH по нормализованному тексту, шинглы DOM,
pHash картинок, набор ссылок; извлечение скелета шаблона (общий каркас +
переменные слоты); агрегация на кластер — разброс слотов как признак машинной
генерации."""
from dw1.extractors import BaseExtractor, register


@register
class TemplateExtractor(BaseExtractor):
    name = "template"
    version = "stub-0.1"
