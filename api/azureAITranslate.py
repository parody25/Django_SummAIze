import requests, uuid, json, os
from django.http import JsonResponse
from dotenv import load_dotenv
load_dotenv()

# Add your key and endpoint
key = os.getenv('API_KEY')
endpoint = os.getenv('AZUREAI_ARABIC_TRANSLATN_ENDPT')
location = "eastus"

class AzureTranslation:
    def __init__(self, textToTranslate):
        self.textToTranslate = textToTranslate
    
    def toEnglish(self):
        path = '/translate'
        constructed_url = endpoint + path
        params = {
            'api-version': '3.0',
            # 'from': 'ar',
            'to': ['en']
        }

        headers = {
            'Ocp-Apim-Subscription-Key': key,
            'Ocp-Apim-Subscription-Region': location,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

        body = [{
            'text': self.textToTranslate
        }]
        try:
            request = requests.post(constructed_url, params=params, headers=headers, json=body)
            response = request.json()
            finalText = response[0]['translations'][0]['text']
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
        return finalText
    
    def __del__(self):
        print("AzureTranslation object is destroyed.")