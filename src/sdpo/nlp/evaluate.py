"""Classification metrics for the 3-class sentiment comparison."""

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

LABELS = [0, 1, 2]  # negative, neutral, positive


def classification_report_dict(y_true, y_pred) -> dict:
    """accuracy, macro_f1 and a 3x3 confusion matrix (rows = true label)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro",
                                   zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }
