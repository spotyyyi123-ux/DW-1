# -*- coding: utf-8 -*-
"""HTML письма (SPEC §6 W3) — три независимые группы признаков:

A. признаки генератора (LLM писал HTML): семантические теги, ровная вложенность,
   CSS-переменные, flex/grid, отсутствие Outlook-хаков (<!--[if mso]-->, таблицы);
B. признаки вредоносности: zero-width, плотность сущностей, разбиение комментариями,
   base64-блобы, скрытые блоки, формы POST на внешний домен;
C. гасители FP: трекеры/joiner ESP, List-Unsubscribe, Precedence: bulk."""
from dw1.extractors import BaseExtractor, register


@register
class HtmlExtractor(BaseExtractor):
    name = "html"
    version = "stub-0.1"
