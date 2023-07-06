from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.reductions import GridSearch
from fairlearn.postprocessing import ThresholdOptimizer

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
