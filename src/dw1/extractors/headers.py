# -*- coding: utf-8 -*-
"""Заголовки письма (SPEC §6 W3).

Будет реализовано: SPF/DKIM/DMARC, выравнивание From/Return-Path/Reply-To,
длина и аномалии Received-цепочки, X-Mailer/User-Agent (скриптовая отправка),
несогласованность Date/таймзоны, формат Message-ID -> категориальные фичи -> HistGB."""
from dw1.extractors import BaseExtractor, register


@register
class HeadersExtractor(BaseExtractor):
    name = "headers"
    version = "stub-0.1"
