
from django.db import models

class UploadedDataset(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='uploads/')
    text_column = models.CharField(max_length=128, default='review_text')
    label_column = models.CharField(max_length=128, default='label')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class TrainedModel(models.Model):
    dataset = models.ForeignKey(UploadedDataset, on_delete=models.CASCADE, related_name='models')
    algorithm = models.CharField(max_length=128)
    vectorizer_path = models.CharField(max_length=512)
    model_path = models.CharField(max_length=512)
    metrics_json = models.JSONField()
    train_seconds = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.algorithm} on {self.dataset.name}"
