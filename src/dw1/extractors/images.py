"""Изображения (SPEC §6 W3).

Будет реализовано: спектральная форензика (FFT-пики, шумовые остатки, JPEG-артефакты,
EXIF), pHash для склейки кампаний, OCR -> текстовая ветка, CLIP-эмбеддинги
(внешний контур) -> обучаемый детектор real/AI (бенчмарк CIFAKE/GenImage)."""
from dw1.extractors import BaseExtractor, register


@register
class ImagesExtractor(BaseExtractor):
    name = "images"
    version = "stub-0.1"
