from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import numpy as np

from model_results import Metrics


def eval_rocauc(y_true, y_pred):
    rocauc_list = []

    for i in range(y_true.shape[1]):

        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)



def metrics(y_true, y_pred):
    # Concatenate all batches and compute metric
    f1_macro = f1_score(y_true, y_pred, average='macro')  # macro means compute per label and then average
    f1_micro = f1_score(y_true, y_pred, average='micro')
    ap_macro = average_precision_score(y_true=y_true, y_score=y_pred, average='macro')
    roc_auc_score = eval_rocauc(y_true, y_pred)


    return Metrics(f1_macro=f1_macro, f1_micro=f1_micro, auc_roc=roc_auc_score, ap_macro=ap_macro)