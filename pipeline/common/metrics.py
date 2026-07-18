"""Shared evaluation metrics for probability models: Brier score, log loss, calibration curve."""
import numpy as np


def brier_score(y_true, p_pred):
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    return float(np.mean((p_pred - y_true) ** 2))


def log_loss(y_true, p_pred, eps=1e-15):
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.clip(np.asarray(p_pred, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred)))


def calibration_curve(y_true, p_pred, n_bins=10):
    """Return (bin_mid, predicted_mean, actual_mean, count) per bin, equal-width on [0,1]."""
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p_pred >= lo) & (p_pred < hi) if i < n_bins - 1 else (p_pred >= lo) & (p_pred <= hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_lo": lo,
            "bin_hi": hi,
            "predicted_mean": float(p_pred[mask].mean()),
            "actual_mean": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return rows


def accuracy(y_true, p_pred, threshold=0.5):
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    preds = (p_pred >= threshold).astype(float)
    return float(np.mean(preds == y_true))
