"""QR-коды / quishing (SPEC §6 W3).

Будет реализовано: декодирование (Node jsqr/@zxing или cv2), URL -> ветка ссылок,
факт QR как интент-признак (обход почтовых сканеров), артефакты скриншота QR."""
from dw1.extractors import BaseExtractor, register


@register
class QrExtractor(BaseExtractor):
    name = "qr"
    version = "stub-0.1"
