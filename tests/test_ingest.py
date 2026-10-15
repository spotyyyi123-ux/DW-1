# -*- coding: utf-8 -*-
from pathlib import Path

from dw1.ingest.mime import parse_eml

FIXTURE = Path(__file__).parent / "fixtures" / "sample.eml"


def test_parses_headers_and_auth():
    res = parse_eml(FIXTURE)
    headers = res["parts"]["headers"]
    assert "ivan.petrov@example.com" in headers["from"]
    assert headers["spf"] == "pass"
    assert headers["message_id_domain"] == "example.com"


def test_extracts_links_from_html():
    links = parse_eml(FIXTURE)["parts"]["links"]
    assert any("mail-confirm.example.net" in l["url"] for l in links)
    assert links[0]["domain"] == "mail-confirm.example.net"


def test_subject_and_body():
    parts = parse_eml(FIXTURE)["parts"]
    assert "подтвердите" in parts["subject"]["text"].lower()
    assert "24 часа" in parts["body_text"]["text"]


def test_sha256_present():
    assert len(parse_eml(FIXTURE)["raw_sha256"]) == 64
