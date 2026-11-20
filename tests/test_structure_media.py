import importlib.util
import io
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from dw1.extractors.headers import HeadersExtractor
from dw1.extractors.html import HtmlExtractor
from dw1.extractors.image_features import decode_qr, inspect_image
from dw1.extractors.links import LinksExtractor
from dw1.pipeline import analyze


def flags(extractor, parts):
    return {e["signal"]: e["value"] for e in extractor.extract(parts)["intent"]["evidence"]}


def test_headers_received_and_dates():
    evidence = flags(
        HeadersExtractor(),
        {
            "headers": {
                "from": "A <a@example.org>",
                "return_path": "b@other.org",
                "date_tz": "Wed, 28 Oct 2026 10:00:00 +0700",
                "received": [
                    "from b; Wed, 28 Oct 2026 10:00:00 +0700",
                    "from a; Wed, 28 Oct 2026 11:00:00 +0700",
                    "broken",
                ],
            }
        },
    )
    assert evidence["received_count"] == 3
    assert evidence["received_malformed"] == 1
    assert evidence["received_chronology_anomaly"]
    assert evidence["date_timezone_minutes"] == 420
    assert evidence["from_return_path_mismatch"]
    assert not flags(HeadersExtractor(), {"headers": {"date_tz": "garbage"}})[
        "date_valid_with_timezone"
    ]


def test_html_real_css_and_post_method():
    html = (
        '<main style="display:grid;opacity:0.5;font-size:0.5em">MailChimp'
        '<span style="display:none">hidden</span>'
        '<form action="https://example.org" method="get"></form>'
        '<form method="post" action="https://example.org"></form></main>'
    )
    evidence = flags(HtmlExtractor(), {"body_html": {"text": html}})
    assert evidence["html_obfuscation_hidden_blocks"] == 1
    assert evidence["html_obfuscation_external_post_forms"] == 1
    assert evidence["html_generator_has_grid"]
    assert evidence["html_mailing_esp_markers"] == ["mailchimp"]


def test_link_idna_and_mismatch():
    assert not LinksExtractor._anchor_mismatch("https://пример.рф", "xn--e1afmkfd.xn--p1ai")
    assert LinksExtractor._anchor_mismatch("https://bank.example", "evil.example")
    assert not LinksExtractor._anchor_mismatch("Подтвердить доступ", "example.org")


def test_phash_resize_and_invalid_image():
    image = Image.fromarray(np.random.default_rng(42).integers(0, 255, (128, 128), dtype=np.uint8))
    a, b = io.BytesIO(), io.BytesIO()
    image.save(a, format="PNG")
    image.resize((256, 256)).save(b, format="PNG")
    first, second = inspect_image(a.getvalue()), inspect_image(b.getvalue())
    assert first["status"] == "ok"
    assert first["width"] == 128
    assert len(first["phash"]) == 16
    assert (int(first["phash"], 16) ^ int(second["phash"], 16)).bit_count() <= 8
    assert all(np.isfinite(p["power_fraction"]) for p in first["fft_peaks"])
    assert inspect_image(b"broken")["status"] == "invalid"


def test_qr_attachment_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    qrcode = pytest.importorskip("qrcode")
    url = "https://example.invalid/qr-check"
    image = qrcode.make(url)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    msg = EmailMessage()
    msg["Subject"] = "QR test"
    msg.set_content("Fixture only")
    msg.add_attachment(buf.getvalue(), maintype="image", subtype="png", filename="code.png")
    eml = tmp_path / "qr.eml"
    eml.write_bytes(msg.as_bytes())
    monkeypatch.setenv("DW1_IMAGE_ARTIFACTS", str(tmp_path / "artifacts"))
    result = analyze(eml)
    assert result.parts["qr"][0]["text"] == url
    assert any(link["source"] == "qr" for link in result.parts["links"])
    assert flags(LinksExtractor(), result.parts)["link_0_domain"] == "example.invalid"
    assert Path(result.parts["images"][0]["preview_path"]).is_file()
    assert result.parts["attachments"][0]["name"] == "code.png"
    result.validate()


def test_qr_inverted_and_multiple_codes_share_base_coordinates():
    pytest.importorskip("cv2")
    qrcode = pytest.importorskip("qrcode")
    first = qrcode.make("https://example.invalid/one", box_size=5).convert("L")
    second = ImageOps.invert(
        qrcode.make("https://example.invalid/two", box_size=5).convert("L")
    )
    image = Image.new("L", (first.width + second.width + 20, first.height), "white")
    image.paste(first, (0, 0))
    image.paste(second, (first.width + 20, 0))
    result = decode_qr(image)
    assert result["status"] == "decoded"
    assert result["coordinate_size"] == list(image.size)
    decoded = [code for code in result["codes"] if code["decoded"]]
    assert sorted(code["text"] for code in decoded) == [
        "https://example.invalid/one",
        "https://example.invalid/two",
    ]
    for code in decoded:
        if code["text"].endswith("/one"):
            assert all(0 <= x < first.width for x, _ in code["points"])
        else:
            assert all(first.width + 20 <= x < image.width for x, _ in code["points"])


def test_qr_decoder_statuses(monkeypatch):
    import sys
    from types import SimpleNamespace

    image = Image.new("L", (80, 80), "white")
    monkeypatch.setitem(sys.modules, "cv2", None)
    assert decode_qr(image)["status"] == "decoder_unavailable"

    class DecoderError(Exception):
        pass

    def fail():
        raise DecoderError

    monkeypatch.setitem(
        sys.modules, "cv2", SimpleNamespace(QRCodeDetector=fail, error=DecoderError)
    )
    assert decode_qr(image)["status"] == "decode_error"

    points = np.array([[[10, 10], [60, 10], [60, 60], [10, 60]]])
    detector = SimpleNamespace(
        detectAndDecodeMulti=lambda _: (False, (), points, ()),
        detectAndDecode=lambda _: ("", None, None),
    )
    monkeypatch.setitem(
        sys.modules, "cv2", SimpleNamespace(QRCodeDetector=lambda: detector, error=DecoderError)
    )
    assert decode_qr(image, preprocess=False)["status"] == "detected_undecoded"
    detector.detectAndDecodeMulti = lambda _: (False, (), None, ())
    assert decode_qr(image, preprocess=False)["status"] == "not_found"


def test_qr_link_ingest_ignores_invalid_urls(tmp_path, monkeypatch):
    import dw1.ingest.mime as mime

    msg = EmailMessage()
    msg.set_content("https://example.invalid/already")
    msg.add_attachment(b"fixture", maintype="image", subtype="png")
    eml = tmp_path / "qr-urls.eml"
    eml.write_bytes(msg.as_bytes())
    urls = [
        "https://example.invalid/already",
        "HTTPS://EXAMPLE.INVALID/add",
        "https://[broken",
        "https:///missing-host",
        "https://example.invalid/a\nb",
        "javascript:alert(1)",
    ]
    monkeypatch.setattr(
        mime,
        "inspect_image",
        lambda _: {
            "sha256": "image",
            "qr": {"codes": [{"text": url, "decoded": True} for url in urls]},
        },
    )
    parts = mime.parse_eml(eml)["parts"]
    assert len(parts["qr"]) == len(urls)
    assert [link["url"] for link in parts["links"]] == urls[:2]
    assert parts["links"][1]["domain"] == "example.invalid"
    assert parts["links"][1]["source"] == "qr"


def test_card_preserves_rejection_and_escapes():
    spec = importlib.util.spec_from_file_location("render_card", "scripts/render_card.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    page = module.render(
        {
            "fusion": {"intent": {"label": "unknown", "confidence": 0}},
            "signals": {"body_text": {"intent": {"label": "phishing", "confidence": 0.99}}},
            "campaign": {"template_skeleton": {"text": "<script>bad</script>"}},
        }
    )
    assert "Фишинг" not in page
    assert "&lt;script&gt;" in page
