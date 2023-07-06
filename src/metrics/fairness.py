from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.metrics import recall_score, f1_score
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import count, selection_rate, mean_prediction
from fairlearn.metrics import false_negative_rate, false_positive_rate
from fairlearn.metrics import true_negative_rate, true_positive_rate

def eval_clf_fairness(y_true, y_pred, z_true, binary=True):
    metrics = {
        "accuracy": accuracy_score,
        "confusion matrix": confusion_matrix,
        "count": count
    }
    if binary:
        metrics.update({
            "recall": recall_score,
            "precision": precision_score,
            "f1 score": f1_score,
            "mean prediction": mean_prediction,
            "selection rate": selection_rate,
            "false negative rate": false_negative_rate,
            "false positive rate": false_positive_rate,
            "true negative rate": true_negative_rate,
            "true positive rate": true_positive_rate
        })
    metric_frame = MetricFrame(metrics=metrics,
                               y_true=y_true,
                               y_pred=y_pred,
                               sensitive_features=z_true)
    return metric_frame