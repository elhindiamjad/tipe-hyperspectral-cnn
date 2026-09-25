"""Standard hyperspectral classification metrics."""
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def compute_metrics(y_true, y_pred, num_classes):
    """Return overall accuracy (OA), average accuracy (AA), Cohen's kappa,
    per-class accuracy and the confusion matrix.

    AA is the mean of per-class accuracies: unlike OA, it is not dominated by the
    large classes, which matters on Indian Pines (20 to 2455 samples per class).
    """
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    support = cm.sum(axis=1)
    per_class = np.divide(np.diag(cm), support, out=np.zeros(num_classes), where=support > 0)
    return {
        "overall_accuracy": float(np.trace(cm) / cm.sum()),
        "average_accuracy": float(per_class[support > 0].mean()),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class_accuracy": per_class.tolist(),
        "confusion_matrix": cm.tolist(),
    }
