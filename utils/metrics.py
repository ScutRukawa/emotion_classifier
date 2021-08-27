from sklearn import metrics

val_y_pred = 0
val_y_true = 0


def calculate_metrics(y_pred, y_true):

    all_metrics = {
        'f1_score': metrics.f1_score(y_pred, y_true),
        'precision_score': metrics.precision_score(y_pred, y_true),
        'recall_score': metrics.recall_score(y_pred, y_true)
    }

    return all_metrics
