# evaluation.py

import time
from collections import Counter
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_models(simple: bool, random_state: int = 1, n_jobs: int = 1) -> Dict[str, Any]:
    """
    Initialize classification models based on the 'simple' flag.

    Args:
        simple (bool): If True, initialize a subset of models.
        random_state (int, optional): Random state for reproducibility. Defaults to 1.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 1.

    Returns:
        Dict[str, Any]: Dictionary of initialized models.
    """
    if simple:
        models = {
            'RF': RandomForestClassifier(random_state=random_state, n_jobs=n_jobs),
        }
    else:
        models = {
            'LR': LogisticRegression(random_state=random_state, n_jobs=n_jobs, max_iter=1000),
            'RF': RandomForestClassifier(random_state=random_state, n_jobs=n_jobs),
            'KNN': KNeighborsClassifier(n_jobs=n_jobs, n_neighbors=5),
            'XGB': XGBClassifier(random_state=random_state, n_jobs=n_jobs, use_label_encoder=False, eval_metric='logloss'),
            'NB': GaussianNB(),
            # 'MLP': MLPClassifier(random_state=random_state, max_iter=1000),
        }
    return models

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray = None, classes: np.ndarray = None) -> Dict[str, Any]:
    """
    Calculate various classification metrics.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        y_score (np.ndarray, optional): Predicted probabilities or scores. Required for ROC AUC. Defaults to None.
        classes (np.ndarray, optional): Array of unique classes. Required for multi-class ROC AUC. Defaults to None.

    Returns:
        Dict[str, Any]: Dictionary of calculated metrics.
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0, average='macro'),
        'Recall': recall_score(y_true, y_pred, average='macro'),
        'F1 Score': f1_score(y_true, y_pred, average='macro'),
    }

    # ROC AUC
    if y_score is not None and classes is not None:
        if len(classes) > 2:
            y_binarized = label_binarize(y_true, classes=classes)
            metrics['ROC AUC'] = roc_auc_score(y_binarized, y_score, average='macro', multi_class='ovr')
        else:
            metrics['ROC AUC'] = roc_auc_score(y_true, y_score[:, 1])

    return metrics

def get_class_distributions(y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> Dict[str, Dict[Any, int]]:
    """
    Calculate class distributions for training, validation, and testing datasets.

    Args:
        y_train (pd.Series): Training labels.
        y_val (pd.Series): Validation labels.
        y_test (pd.Series): Testing labels.

    Returns:
        Dict[str, Dict[Any, int]]: Dictionary containing class distributions.
    """
    distributions = {
        'Total Class Distribution': Counter(y_train.tolist() + y_val.tolist() + y_test.tolist()),
        'Train Class Distribution': Counter(y_train),
        'Validation Class Distribution': Counter(y_val),
        'Test Class Distribution': Counter(y_test),
    }
    return distributions

def test_classification_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    RANDOM_STATE: int = 1,
    simple: bool = False,
    n_jobs: int = 1
) -> pd.DataFrame:
    """
    Train and evaluate multiple classification models on the provided datasets.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing labels.
        RANDOM_STATE (int, optional): Random state for reproducibility. Defaults to 1.
        simple (bool, optional): If True, train a subset of models. Defaults to False.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 1.

    Returns:
        pd.DataFrame: DataFrame containing evaluation metrics for each model.
    """
    models = initialize_models(simple, random_state=RANDOM_STATE, n_jobs=n_jobs)
    results = []

    # Calculate class distributions
    distributions = get_class_distributions(y_train, y_val, y_test)
    classes = np.unique(np.concatenate([y_train, y_val, y_test]))

    for name, model in models.items():
        # logger.info(f"Training model: {name}")
        start_time = time.perf_counter()
        try:
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - start_time
        except Exception as e:
            logger.error(f"Model {name} failed to fit: {e}")
            continue

        # Predicting test data
        start_time = time.perf_counter()
        try:
            y_pred_test = model.predict(X_test)
            test_time = time.perf_counter() - start_time
        except Exception as e:
            logger.error(f"Model {name} failed to predict on test data: {e}")
            continue

        # Predicting validation data
        try:
            y_pred_val = model.predict(X_val)
        except Exception as e:
            logger.error(f"Model {name} failed to predict on validation data: {e}")
            continue

        # Predicting training data
        try:
            y_pred_train = model.predict(X_train)
        except Exception as e:
            logger.error(f"Model {name} failed to predict on training data: {e}")
            continue

        # Get predicted probabilities for ROC AUC
        try:
            if hasattr(model, "predict_proba"):
                y_score_test = model.predict_proba(X_test)
                y_score_val = model.predict_proba(X_val)
            else:
                y_score_test = model.decision_function(X_test)
                y_score_val = model.decision_function(X_val)
        except Exception as e:
            logger.warning(f"Model {name} does not support probability predictions: {e}")
            y_score_test = None
            y_score_val = None

        # Calculate metrics
        metrics_test = calculate_metrics(y_test.to_numpy(), y_pred_test, y_score_test, classes)
        metrics_val = calculate_metrics(y_val.to_numpy(), y_pred_val, y_score_val, classes)
        metrics_train = calculate_metrics(y_train.to_numpy(), y_pred_train)

        # Calculate Specificity and G-Mean for binary classification
        if len(classes) == 2:
            # Specificity is the true negative rate
            cm = confusion_matrix(y_test, y_pred_test)
            tn, fp, fn, tp = cm.ravel()
            specificity_test = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            g_mean_test = np.sqrt(specificity_test * metrics_test['Recall'])

            cm_val = confusion_matrix(y_val, y_pred_val)
            tn_val, fp_val, fn_val, tp_val = cm_val.ravel()
            specificity_val = tn_val / (tn_val + fp_val) if (tn_val + fp_val) > 0 else 0.0
            g_mean_val = np.sqrt(specificity_val * metrics_val['Recall'])

            cm_train = confusion_matrix(y_train, y_pred_train)
            tn_train, fp_train, fn_train, tp_train = cm_train.ravel()
            specificity_train = tn_train / (tn_train + fp_train) if (tn_train + fp_train) > 0 else 0.0
            g_mean_train = np.sqrt(specificity_train * metrics_train['Recall'])

            metrics_test.update({
                'Specificity': specificity_test,
                'G-Mean': g_mean_test
            })
            metrics_val.update({
                'Specificity': specificity_val,
                'G-Mean': g_mean_val
            })
            metrics_train.update({
                'Specificity': specificity_train,
                'G-Mean': g_mean_train
            })
        else:
            # For multi-class, Specificity and G-Mean are not directly applicable
            metrics_test['Specificity'] = np.nan
            metrics_test['G-Mean'] = np.nan
            metrics_val['Specificity'] = np.nan
            metrics_val['G-Mean'] = np.nan
            metrics_train['Specificity'] = np.nan
            metrics_train['G-Mean'] = np.nan

        # Compile results
        result = {
            'Model': name,
            'Test Accuracy': metrics_test['Accuracy'],
            'Test Balanced Accuracy': metrics_test['Balanced Accuracy'],
            'Test Precision': metrics_test['Precision'],
            'Test Recall': metrics_test['Recall'],
            'Test F1 Score': metrics_test['F1 Score'],
            'Test ROC AUC': metrics_test['ROC AUC'],
            'Test Specificity': metrics_test['Specificity'],
            'Test G-Mean': metrics_test['G-Mean'],
            'Training Instances': len(X_train),
            'Validation Instances': len(X_val),
            'Testing Instances': len(X_test),
            'Training Time (s)': round(train_time, 4),
            'Testing Time (s)': round(test_time, 4),
            'Train Accuracy': metrics_train['Accuracy'],
            'Train Precision': metrics_train['Precision'],
            'Train Recall': metrics_train['Recall'],
            'Train F1 Score': metrics_train['F1 Score'],
            'Train Specificity': metrics_train['Specificity'],
            'Train G-Mean': metrics_train['G-Mean'],
            'Validation Accuracy': metrics_val['Accuracy'],
            'Validation Precision': metrics_val['Precision'],
            'Validation Recall': metrics_val['Recall'],
            'Validation F1 Score': metrics_val['F1 Score'],
            'Validation ROC AUC': metrics_val.get('ROC AUC', np.nan),
            'Validation Specificity': metrics_val.get('Specificity', np.nan),
            'Validation G-Mean': metrics_val.get('G-Mean', np.nan),
            'Random State': RANDOM_STATE,
            'Total Class Distribution': distributions['Total Class Distribution'],
            'Train Class Distribution': distributions['Train Class Distribution'],
            'Validation Class Distribution': distributions['Validation Class Distribution'],
            'Test Class Distribution': distributions['Test Class Distribution'],
            'Model Parameters': model.get_params()
        }

        results.append(result)
        # logger.info(f"Completed evaluation for model: {name}")

    return pd.DataFrame(results)