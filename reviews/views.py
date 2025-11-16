
import os, json, time, joblib, pandas as pd, numpy as np
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.contrib import messages
from .forms import UploadForm, TrainForm
from .models import UploadedDataset, TrainedModel
from .ml_utils import detect_columns, train_algorithms, apply_best
from sklearn.feature_extraction.text import CountVectorizer

def _read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, encoding='latin-1')
        except Exception as e:
            raise e

def home(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            ds = form.save()
            messages.success(request, 'Dataset uploaded successfully.')
            return redirect('eda', dataset_id=ds.id)
    else:
        form = UploadForm()
    datasets = UploadedDataset.objects.order_by('-uploaded_at')[:5]
    return render(request, 'home.html', {'form': form, 'datasets': datasets})

def eda(request, dataset_id):
    ds = get_object_or_404(UploadedDataset, id=dataset_id)
    df = _read_csv(ds.file.path)
    # Auto-detect columns if defaults aren't present
    text_col = ds.text_column if ds.text_column in df.columns else detect_columns(df)[0]
    label_col = ds.label_column if ds.label_column in df.columns else detect_columns(df)[1]

    # Simple EDA
    total = len(df)
    nulls = df.isna().sum().to_dict()
    preview = df.head(12).to_html(classes='table table-sm table-striped', index=False)
    class_counts = None
    if label_col in df.columns:
        class_counts = df[label_col].value_counts(dropna=False).to_dict()

    # top tokens (quick)
    top_terms = []
    try:
        cv = CountVectorizer(max_features=20, stop_words='english')
        tokens = cv.fit_transform(df[text_col].astype(str).fillna(''))
        vocab = cv.get_feature_names_out()
        counts = tokens.toarray().sum(axis=0)
        top_terms = sorted(zip(vocab, counts), key=lambda x: -x[1])[:20]
    except Exception:
        pass

    return render(request, 'eda.html', {
        'ds': ds,
        'total': total,
        'nulls': nulls,
        'preview': preview,
        'label_col': label_col,
        'text_col': text_col,
        'class_counts': class_counts,
        'top_terms': top_terms,
    })

def train(request, dataset_id):
    ds = get_object_or_404(UploadedDataset, id=dataset_id)
    df = _read_csv(ds.file.path)
    text_col = ds.text_column if ds.text_column in df.columns else detect_columns(df)[0]
    label_col = ds.label_column if ds.label_column in df.columns else detect_columns(df)[1]

    if request.method == 'POST':
        form = TrainForm(request.POST)
        if form.is_valid():
            algos = form.cleaned_data['algorithms']
            test_size = form.cleaned_data['test_size']
            random_state = form.cleaned_data['random_state']

            runs, best, split = train_algorithms(df, text_col, label_col, algos, test_size, random_state, out_dir=os.path.join(settings.MEDIA_ROOT, 'models'))
            if not runs:
                messages.error(request, 'No models ran. Ensure required libraries are installed (xgboost/catboost optional).')
                return redirect('train', dataset_id=ds.id)

            # Save each model metadata
            for r in runs:
                TrainedModel.objects.create(
                    dataset=ds,
                    algorithm=r['algorithm'],
                    vectorizer_path=r['vectorizer_path'],
                    model_path=r['model_path'],
                    metrics_json=r['metrics'],
                    train_seconds=r['seconds']
                )

            # Attach best marker to session for results page
            request.session['last_best_algo'] = best['algorithm']
            request.session['last_metrics'] = best['metrics']

            messages.success(request, 'Training complete. Showing results.')
            return redirect('results', dataset_id=ds.id)
    else:
        form = TrainForm(initial={'algorithms': ['lr','rf','xgb','cat']})

    return render(request, 'train.html', {
        'ds': ds,
        'form': form,
        'text_col': text_col,
        'label_col': label_col,
    })

def results(request, dataset_id):
    ds = get_object_or_404(UploadedDataset, id=dataset_id)
    models = TrainedModel.objects.filter(dataset=ds).order_by('-created_at')
    df = _read_csv(ds.file.path)
    # pick most recent best (by f1)
    best_tm = None
    if models.exists():
        best_tm = max(models, key=lambda m: m.metrics_json.get('f1', 0.0))

    preds_preview = None
    best_name = None
    metrics = None
    if best_tm:
        # Load vectorizer + model
        vec = joblib.load(best_tm.vectorizer_path)
        clf = joblib.load(best_tm.model_path)
        pred, prob = apply_best(df, best_tm.dataset.text_column if best_tm.dataset.text_column in df.columns else df.columns[0], vec, clf)
        out = df.copy()
        out['pred_label'] = np.where(pred==1, 'FAKE', 'GENUINE')
        if prob is not None:
            out['prob_fake'] = prob.round(4)
        preds_preview = out.head(20).to_html(classes='table table-striped table-sm', index=False)
        best_name = best_tm.algorithm
        metrics = best_tm.metrics_json

    return render(request, 'results.html', {
        'ds': ds,
        'models': models,
        'preds_preview': preds_preview,
        'best_name': best_name,
        'metrics': metrics
    })
