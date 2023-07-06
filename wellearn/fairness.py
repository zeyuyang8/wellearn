from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.metrics import recall_score, f1_score
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import count, selection_rate, mean_prediction
from fairlearn.metrics import false_negative_rate, false_positive_rate
from fairlearn.metrics import true_negative_rate, true_positive_rate
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.reductions import GridSearch
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio
import pandas as pd

def dpr_and_eor(y_true, y_pred, z_true):
    dpr = demographic_parity_ratio(y_true, y_pred, sensitive_features=z_true)
    eor = equalized_odds_ratio(y_true, y_pred, sensitive_features=z_true)
    return dpr, eor

def dpr_and_acc(y_true, y_pred, z_true):
    dpr = demographic_parity_ratio(y_true, y_pred, sensitive_features=z_true)
    acc = accuracy_score(y_true, y_pred)
    return dpr, acc

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

def fairness_versus_performance(metrics, fairness_options, performance_options):
    """Return a tuple of fairness and performance metrics frames"""
    fairness_dict = {}
    performance_dict = {}
    for key in metrics:
        fairness_dict[key] = list(metrics[key].difference()[fairness_options].values)
        performance_dict[key] = list(metrics[key].overall[performance_options].values)
    fairness_frame = pd.DataFrame(fairness_dict, index=fairness_options)
    performance_frame = pd.DataFrame(performance_dict, index=performance_options)
    return fairness_frame, performance_frame

class FairSklearnModel:
    """Fair Logistic Regression model"""
    def __init__(self, model=LogisticRegression()):
        self.model = model
    
    def get_model(self):
        return self.model
        
    def predict(self, X_test, z_test=None):
        if z_test:
            return self.model.predict(X_test, sensitive_features=z_test)
        return self.model.predict(X_test)
    
    def fit_raw(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def fit_exp_grad(self, 
                     X_train, 
                     y_train, 
                     z_train, 
                     constraints=DemographicParity()):
        mitigator = ExponentiatedGradient(self.model, 
                                          constraints=constraints)
        mitigator.fit(X_train, y_train, sensitive_features=z_train)
        self.model = mitigator
        
    def fit_grid_search(self, 
                        X_train, 
                        y_train, 
                        z_train, 
                        grid_size=36,
                        constraints=DemographicParity()):
        sweep = GridSearch(
            estimator=self.model,
            constraints=constraints,
            grid_size=grid_size,
        )
        sweep.fit(X_train, y_train, sensitive_features=z_train)
        mitigator = sweep.predictors_[sweep.best_idx_]
        self.model = mitigator
    
    def fit_threshold_optimizer(self, 
                                X_train, 
                                y_train, 
                                z_train, 
                                constraints='demographic_parity', 
                                objective='accuracy_score'):
        mitigator = ThresholdOptimizer(estimator=self.model,
                                       constraints=constraints, 
                                       objective=objective, 
                                       predict_method='auto')
        mitigator.fit(X_train, y_train, sensitive_features=z_train)
        self.model = mitigator
