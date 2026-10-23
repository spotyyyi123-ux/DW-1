import pytest

from dw1.evidence import SCHEMA_VERSION, Evidence, EvidenceError


def test_roundtrip_preserves_fields():
    ev = Evidence(message_id="abc", parts={"subject": {"text": "hi"}})
    d = ev.to_dict()
    assert d["message_id"] == "abc"
    assert Evidence.from_dict(d).message_id == "abc"


def test_defaults_validate():
    Evidence().validate()


def test_unknown_field_raises():
    with pytest.raises(EvidenceError):
        Evidence.from_dict({"schema_version": SCHEMA_VERSION, "nope": 1})


def test_bad_provenance_label_raises():
    ev = Evidence(signals={"body_text": {"ai_provenance": {"label": "alien"}}})
    with pytest.raises(EvidenceError):
        ev.validate()


def test_bad_confidence_raises():
    ev = Evidence(signals={"body_text": {"intent": {"label": "phishing", "confidence": 1.5}}})
    with pytest.raises(EvidenceError):
        ev.validate()


def test_unknown_signal_kind_raises():
    ev = Evidence(signals={"body_text": {"mood": {"label": "happy"}}})
    with pytest.raises(EvidenceError):
        ev.validate()


def test_wrong_schema_version_raises():
    with pytest.raises(EvidenceError):
        Evidence(schema_version="0.9").validate()
