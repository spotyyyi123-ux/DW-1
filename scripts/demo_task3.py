"""Generate inspectable synthetic email/image artifacts, NOT a quality benchmark."""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
from email.message import EmailMessage
from pathlib import Path

import qrcode

from dw1.campaigns.template import skeleton
from dw1.pipeline import analyze


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("docs/examples/task3"))
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    text_a = "Здравствуйте, Иван! Оплатите 100 RUB для бренда Alpha https://example.invalid/pay"
    text_b = "Здравствуйте, Анна! Оплатите 200 RUB для бренда Beta https://example.invalid/other"
    template = skeleton([text_a, text_b])
    index = [{"cluster_id": "synthetic-demo", "size": 2, "representative_text": text_a,
              "template_skeleton": template, "model_version": "synthetic-template-demo"}]
    (out / "campaign_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    image = qrcode.make("https://example.invalid/qr-payment")
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    msg = EmailMessage()
    msg["From"] = "Demo <demo@example.invalid>"
    msg["To"] = "analyst@example.invalid"
    msg["Subject"] = ""
    msg["Date"] = "Wed, 28 Oct 2026 10:00:00 +0700"
    msg["Received"] = "from demo.example.invalid; Wed, 28 Oct 2026 10:01:00 +0700"
    msg.set_content(text_a)
    msg.add_attachment(payload.getvalue(), maintype="image", subtype="png", filename="qr.png")
    eml = out / "synthetic.eml"
    eml.write_bytes(msg.as_bytes())
    os.environ["DW1_CAMPAIGN_INDEX"] = str(out / "campaign_index.json")
    os.environ["DW1_IMAGE_ARTIFACTS"] = str(out / "images")
    os.environ["DW1_PROV_MODEL"] = str(out / "no-model.joblib")
    os.environ["DW1_INTENT_MODEL"] = str(out / "no-intent.joblib")
    os.environ["DW1_FUSION_CALIBRATION"] = str(out / "no-calibration.json")
    evidence = analyze(eml).to_dict()
    assert evidence["parts"]["qr"][0]["text"] == "https://example.invalid/qr-payment"
    assert evidence["campaign"]["template_skeleton"] == template
    # Portable artifact paths, relative to the HTML file.
    for im in evidence["parts"]["images"]:
        if im.get("preview_path"):
            im["preview_path"] = "images/" + Path(im["preview_path"]).name
    evidence["ingest"]["source"] = "synthetic.eml"
    (out / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    spec = importlib.util.spec_from_file_location("render_card", "scripts/render_card.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (out / "card.html").write_text(module.render(evidence), encoding="utf-8")
    (out / "status.json").write_text(json.dumps({
        "scope": "Synthetic authored fixture, no third-party dataset",
        "bcubed_f1_top50": None,
        "reason": "Real corpus, external embeddings and human top-50 annotation not supplied",
        "qr_decoded": True, "skeleton": template}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Synthetic card and measured QR evidence: {out}")


if __name__ == "__main__":
    main()
