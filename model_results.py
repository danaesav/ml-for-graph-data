import numpy as np

class Metrics:
    def __init__(self, loss = None, f1_macro= None, f1_micro=None, ap_macro=None, auc_roc=None):
        self.loss = loss if loss else np.empty(0)
        self.f1_macro = f1_macro if f1_macro else np.empty(0)
        self.f1_micro = f1_micro if f1_micro else np.empty(0)
        self.ap_macro = ap_macro if ap_macro else np.empty(0)
        self.auc_roc = auc_roc if auc_roc else np.empty(0)

    def append_metrics(self, metrics):
        self.loss = np.append(self.loss, metrics.loss)
        self.f1_macro = np.append(self.f1_macro, metrics.f1_macro)
        self.f1_micro = np.append(self.f1_micro, metrics.f1_micro)
        self.ap_macro = np.append(self.ap_macro, metrics.ap_macro)
        self.auc_roc = np.append(self.auc_roc, metrics.auc_roc)

    def metric_list(self):
        return [self.loss, self.f1_macro, self.f1_micro, self.ap_macro, self.auc_roc]

    @staticmethod
    def aggregate_metrics(metrics):
        loss = []
        f1_macro = []
        f1_micro = []
        ap_macro = []
        auc_roc = []

        for i, metric in enumerate(metrics):
            loss.append(metric.loss)
            f1_macro.append(metric.f1_macro)
            f1_micro.append(metric.f1_micro)
            ap_macro.append(metric.ap_macro)
            auc_roc.append(metric.auc_roc)

        return Metrics(loss, f1_macro, f1_micro, ap_macro, auc_roc)

    def __str__(self):
        return (f"test-loss:{np.mean(self.loss):.4f}+-{np.std(self.loss):.4f}\n"
                + f"test-f1-macro:{np.mean(self.f1_macro):.4f}+-{np.std(self.f1_macro):.4f}\n"
                + f"test-f1-micro:{np.mean(self.f1_micro):.4f}+-{np.std(self.f1_micro):.4f}\n"
                + f"test-AP-macro:{np.mean(self.ap_macro):.4f}+-{np.std(self.ap_macro):.4f}\n"
                + f"test-AUC-ROC:{np.mean(self.auc_roc):.4f}+-{np.std(self.auc_roc):.4f}\n")


class ModelResults:
    def __init__(self, train_metrics = None, val_metrics = None, test_metrics = None):
        self.train_metrics = train_metrics if train_metrics else Metrics()
        self.val_metrics = val_metrics if val_metrics else Metrics()
        self.test_metrics = test_metrics if test_metrics else Metrics()

    def extend_metrics(self, metrics):
        self.train_metrics.extend_metrics(metrics.train_metrics)
        self.val_metrics.extend_metrics(metrics.val_metrics)
        self.test_metrics.extend_metrics(metrics.test_metrics)


    @staticmethod
    def aggregate_results(results):
        train_metrics = []
        val_metrics = []
        test_metrics = []
        for result in results:
            train_metrics.append(result.train_metrics)
            val_metrics.append(result.val_metrics)
            test_metrics.append(result.test_metrics)

        train_metrics = Metrics.aggregate_metrics(train_metrics)
        val_metrics = Metrics.aggregate_metrics(val_metrics)
        test_metrics = Metrics.aggregate_metrics(test_metrics)

        return ModelResults(train_metrics, val_metrics, test_metrics)
