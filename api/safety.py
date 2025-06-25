from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from django.conf import settings

def is_text_safe(text):
    client = ContentSafetyClient(
        endpoint=settings.AZURE_CONTENT_SAFETY_ENDPOINT,
        credential=AzureKeyCredential(settings.AZURE_CONTENT_SAFETY_KEY)
    )

    try:
        options = AnalyzeTextOptions(
            text=text,
            categories=["Hate", "Sexual", "Violence", "SelfHarm"]
        )
        response = client.analyze_text(options)
        categories = response.categories_analysis

        details = {cat.category: cat.severity for cat in categories}
        safe = all(cat.severity == 0 for cat in categories)
        return {"safe": safe, "details": details}

    except HttpResponseError as e:
        return {"safe": False, "details": str(e)}
