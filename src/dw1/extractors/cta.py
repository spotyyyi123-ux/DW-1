# -*- coding: utf-8 -*-
"""Call-to-action (SPEC §6 W2/W3).

Будет реализовано: извлечение императивов и целевых действий, интент-лексика
(«подтвердите», «обновите доступ», «в течение 24 часов»), дедлайны,
несоответствие текста ссылки и href, лексическое разнообразие CTA по кластеру."""
from dw1.extractors import BaseExtractor, register


@register
class CtaExtractor(BaseExtractor):
    name = "cta"
    version = "stub-0.1"
