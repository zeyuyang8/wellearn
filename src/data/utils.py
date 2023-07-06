from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from fairlearn.preprocessing import CorrelationRemover


def one_hot_encoder_and_move_to_back(X):
    """One-hot encode the categorical features and move them to the back of the dataframe."""
    # One-hot encode the categorical features
    X_cat = pd.get_dummies(X.select_dtypes(include="category"), dtype=float)
    X_cat.index = X.index
    
    # Move the categorical features to the back of the dataframe
    X = X.drop(X.select_dtypes(include="category").columns, axis=1)
    X = pd.concat([X, X_cat], axis=1)
    return X


def feature_normalizer(X, normalizer=None):
    """Transform the columns of the features. Convert the categorical features to numerical features.
    Normalize the numerical features.
    """
    if not normalizer:
        normalizer = make_column_transformer(
            (
                Pipeline(steps=[("encoder", OrdinalEncoder())]),
                make_column_selector(dtype_include="category")
            ),
            (
                Pipeline(steps=[("normalizer", StandardScaler())]),
                make_column_selector(dtype_include="number")
            ),
            remainder="passthrough"
        )
        X_norm = normalizer.fit_transform(X)
    else:
        X_norm = normalizer.transform(X)
    X_norm = pd.DataFrame(X_norm, columns=X.columns)
    return X_norm, normalizer

def corr_remover(X, sensitive_id, ignore_ids=[], remover=None):
    """Remove the correlation between the sensitive feature and the other features."""
    # Select the columns to be removed from the correlation remover
    ignore_ids_copy = ignore_ids.copy()
    ignore_ids_copy.remove(sensitive_id)
    X_select = X.drop(ignore_ids_copy, axis=1)
    
    # Remove the correlation between the sensitive feature and the other features
    if not remover:
        remover = CorrelationRemover(sensitive_feature_ids=[sensitive_id])
        X_select_cr = remover.fit_transform(X_select)
    else:
        X_select_cr = remover.transform(X_select)
    
    # Convert the numpy array to a pandas dataframe
    cr_columns = list(X_select.columns)
    cr_columns.remove(sensitive_id)
    X_select_cr = pd.DataFrame(X_select_cr, columns=cr_columns)
    
    # Add the ignored columns back to the data
    X_select_cr[sensitive_id] = X[sensitive_id].values
    X_select_cr[ignore_ids] = X[ignore_ids].values
    X_select_cr = X_select_cr[X.columns]
    X_select_cr.index = X.index
    return X_select_cr, remover
