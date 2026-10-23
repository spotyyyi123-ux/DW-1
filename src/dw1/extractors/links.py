"""Ссылки (SPEC §6 W3).

Будет реализовано: длина/энтропия URL, глубина поддоменов, punycode/homoglyph,
TLD, цепочки редиректов, несоответствие anchor<->href, репутация домена,
детектор легитимных трекеров ESP."""
from dw1.extractors import BaseExtractor, register


@register
class LinksExtractor(BaseExtractor):
    name = "links"
    version = "stub-0.1"
