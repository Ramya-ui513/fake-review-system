
import time, json, os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Optional imports:
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except Exception:
    HAS_CAT = False

def detect_columns(df):
    # Heuristic to find text/label columns if not set
    text_candidates = ['review_text','text','review','content','comment','body']
    label_candidates = ['label','target','is_fake','fake','y','class']
    text_col = next((c for c in text_candidates if c in df.columns), df.columns[0])
    label_col = next((c for c in label_candidates if c in df.columns), df.columns[-1])
    return text_col, label_col

def clean_labels(y):
    # Convert to 0/1; consider 'fake'/'genuine' strings, True/False, etc.
    if y.dtype.kind in 'ifu':
        # Numeric -> map to 0/1
        # Normalize any nonzero to 1
        return (y.astype(float) > 0).astype(int).values
    # String/object
    y_lower = y.astype(str).str.lower()
    positives = set(['1','true','fake','fraud','spam','yes'])
    return y_lower.apply(lambda v: 1 if v.strip() in positives else 0).values

def train_algorithms(df, text_column, label_column, algorithms, test_size=0.2, random_state=42, out_dir='models'):
    os.makedirs(out_dir, exist_ok=True)
    X = df[text_column].astype(str).fillna('')
    y_raw = df[label_column]
    y = clean_labels(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    results = []
    runs = []

    def evaluate(pipe, algo_name):
        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_seconds = time.time() - t0
        y_pred = pipe.predict(X_test)
        # Proba optional
        try:
            y_prob = pipe.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = None
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        metrics = {
            'accuracy': round(float(acc),4),
            'precision': round(float(prec),4),
            'recall': round(float(rec),4),
            'f1': round(float(f1),4),
        }
        if auc is not None:
            metrics['roc_auc'] = round(float(auc),4)

        # Persist vectorizer & model if possible
        model_tag = f"{algo_name}_{int(time.time())}"
        vec_path = os.path.join(out_dir, f"{model_tag}_vectorizer.joblib")
        model_path = os.path.join(out_dir, f"{model_tag}_model.joblib")

        if hasattr(pipe, 'named_steps') and 'tfidf' in pipe.named_steps:
            joblib.dump(pipe.named_steps['tfidf'], vec_path)
        else:
            # Not expected, but fallback: dump full pipeline vectorizer
            for step_name, step in getattr(pipe, 'named_steps', {}).items():
                if hasattr(step, 'transform'):
                    joblib.dump(step, vec_path); break

        # Save classifier
        clf = pipe.named_steps.get('clf', pipe)
        joblib.dump(clf, model_path)

        return { 'algorithm': algo_name, 'metrics': metrics, 'seconds': round(train_seconds,3),
                 'vectorizer_path': vec_path, 'model_path': model_path, }

    # Build pipelines
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=2)
    pipes = []

    if 'lr' in algorithms:
        pipes.append(('Logistic Regression',
                      Pipeline([('tfidf', vec), ('clf', LogisticRegression(max_iter=1000))])))
    if 'rf' in algorithms:
        pipes.append(('Random Forest',
                      Pipeline([('tfidf', vec), ('clf', RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=random_state))])))
    if 'xgb' in algorithms and HAS_XGB:
        pipes.append(('XGBoost',
                      Pipeline([('tfidf', vec), ('clf', XGBClassifier(
                          n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                          random_state=random_state, n_jobs=-1, eval_metric='logloss'))])))
    if 'cat' in algorithms and HAS_CAT:
        pipes.append(('CatBoost',
                      Pipeline([('tfidf', vec), ('clf', CatBoostClassifier(
                          depth=6, iterations=400, learning_rate=0.1, verbose=False, random_seed=random_state))])))

    for name, pipe in pipes:
        runs.append(evaluate(pipe, name))

    # Best by F1
    best = max(runs, key=lambda r: r['metrics'].get('f1', 0.0)) if runs else None
    return runs, best, (X_test, y_test)

def apply_best(df, text_column, vectorizer, model):
    X = df[text_column].astype(str).fillna('')
    X_vec = vectorizer.transform(X)
    try:
        prob = model.predict_proba(X_vec)[:,1]
    except Exception:
        prob = None
    pred = model.predict(X_vec)
    return pred, prob
