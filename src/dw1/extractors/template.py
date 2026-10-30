"""Expose a matched campaign skeleton without attributing AI authorship."""
from dw1.campaigns.cluster import cluster
from dw1.extractors import BaseExtractor, register
from dw1.extractors.images import measured_signal


@register
class TemplateExtractor(BaseExtractor):
    name = "template"
    version = "campaign-template-1.0"

    def extract(self, parts: dict) -> dict:
        campaign = cluster(parts)
        evidence = [{"signal": "campaign_template_skeleton",
                     "value": campaign.get("template_skeleton"), "weight": 0.0},
                    {"signal": "campaign_member_count",
                     "value": campaign.get("member_count", 1), "weight": 0.0}]
        return measured_signal(evidence, self.version)
