import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
import config

__all__ = ['train_all_models', 'evaluate_models', 'get_classification_report', 'get_confusion_matrix', 'predict_text']

def train_all_models(_X_train, y_train):
    """Melatih SVM, Naive Bayes, KNN, dan XGBoost (jika tersedia)."""
    # Mapping label untuk XGBoost (membutuhkan target numerik)
    y_train_num = y_train.map(config.LABEL_MAP)

    models = {
        "SVM": SVC(kernel='linear', probability=True, random_state=config.RANDOM_STATE),
        "Naive Bayes": MultinomialNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }
    
    # Add XGBoost only if available
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=config.RANDOM_STATE)
    
    trained_models = {}
    for name, model in models.items():
        if name == "XGBoost":
            model.fit(_X_train, y_train_num)
        else:
            model.fit(_X_train, y_train)
        trained_models[name] = model
        
    return trained_models

def evaluate_models(_models, _X_test, y_test):
    """Menghitung akurasi untuk semua model."""
    y_test_num = y_test.map(config.LABEL_MAP)
    
    results = {}
    for name, model in _models.items():
        if name == "XGBoost":
            y_pred = model.predict(_X_test)
            acc = accuracy_score(y_test_num, y_pred)
        else:
            y_pred = model.predict(_X_test)
            acc = accuracy_score(y_test, y_pred)
        results[name] = acc
    return results

def get_classification_report(model, model_name, _X_test, y_test):
    """Menghasilkan DataFrame Classification Report."""
    if model_name == "XGBoost":
        y_pred = model.predict(_X_test)
        y_true = y_test.map(config.LABEL_MAP)
        target_names = ['Negative', 'Neutral', 'Positive']
    else:
        y_pred = model.predict(_X_test)
        y_true = y_test
        target_names = sorted(y_test.unique())
        
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()
    
def get_confusion_matrix(model, model_name, _X_test, y_test):
    """Menghasilkan DataFrame Confusion Matrix."""
    if model_name == "XGBoost":
        y_pred = model.predict(_X_test)
        y_true = y_test.map(config.LABEL_MAP)
        labels = [0, 1, 2]
    else:
        y_pred = model.predict(_X_test)
        y_true = y_test
        labels = config.LABEL_NAMES
        
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=config.LABEL_NAMES, columns=config.LABEL_NAMES)

def predict_text(text, model, model_name, vectorizer):
    """Memprediksi teks baru dari input pengguna."""
    X_new = vectorizer.transform([text])
    pred = model.predict(X_new)[0]
    
    if model_name == "XGBoost":
        reverse_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        return reverse_map[pred]
    return pred

# Force reload
