from django.db import models
import uuid

class Application(models.Model):
    application_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)

class PDFDocument(models.Model):
    application = models.ForeignKey(Application, on_delete=models.CASCADE, related_name='pdfs')
    pdf_name = models.CharField(max_length=255)
    time_uploaded = models.DateTimeField(auto_now_add=True)

class Embedding(models.Model):
    application = models.OneToOneField(Application, on_delete=models.CASCADE, related_name='embeddings')
    embeddings_file = models.CharField(max_length=255)

class ExtractedData(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    extracted_json = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Extracted Data {self.id}"
