
from django import forms
from .models import UploadedDataset

class UploadForm(forms.ModelForm):
    class Meta:
        model = UploadedDataset
        fields = ['name', 'file', 'text_column', 'label_column']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Dataset name'}),
            'file': forms.FileInput(attrs={'class': 'form-control'}),
            'text_column': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'review_text'}),
            'label_column': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'label'}),
        }

class TrainForm(forms.Form):
    ALGO_CHOICES = [
        ('lr', 'Logistic Regression'),
        ('rf', 'Random Forest'),
        ('xgb', 'XGBoost'),
        ('cat', 'CatBoost'),
    ]
    algorithms = forms.MultipleChoiceField(
        choices=ALGO_CHOICES,
        widget=forms.CheckboxSelectMultiple,
        required=True
    )
    test_size = forms.FloatField(initial=0.2, min_value=0.05, max_value=0.5)
    random_state = forms.IntegerField(initial=42)
