from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

class ContentSafetyService:
    def __init__(self, endpoint, key):
        self.client = ContentSafetyClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )

    def is_text_safe(self, text, categories=None):
        if categories is None:
            categories = ["Hate", "Sexual", "Violence", "SelfHarm"]

        try:
            options = AnalyzeTextOptions(text=text, categories=categories)
            response = self.client.analyze_text(options)
            categories_analysis = response.categories_analysis

            details = {cat.category: cat.severity for cat in categories_analysis}
            safe = all(cat.severity == 0 for cat in categories_analysis)

            return {"safe": safe, "details": details}

        except HttpResponseError as e:
            return {"safe": False, "details": str(e)}
