from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from .utils import feature_normalizer, corr_remover, one_hot_encoder_and_move_to_back

ACS_INCOME_ID = 43141
IBA_DEPRESSION_ID = 45040

def get_openml_data(data_id):
    data = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
    X = data.data
    y_true = data.target
    data_dict = {"features": X, "labels": y_true}
    return data_dict

def prepare_iba_depression(sen_attr='Sex', test_size=0.1, random_state=2023):
    data_dict = get_openml_data(data_id=IBA_DEPRESSION_ID)
    X, y = data_dict["features"], data_dict["labels"]  
    X = X.drop(['dataset'], axis=1)  # drop the column 'dataset' because it is not a feature
    z = X[sen_attr]
    categorical_ids = ['Sex', 'Race', 'Housing', 'Delay']
    X[categorical_ids] = X[categorical_ids].astype('category')
    X_c = one_hot_encoder_and_move_to_back(X)
    
    X_c_train, X_c_test, X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
        X_c, X, y, z, test_size=test_size, random_state=random_state
    )

    # Convert the categorical features to numerical features
    X_train_norm, normalizer = feature_normalizer(X_train)
    X_test_norm, dummy = feature_normalizer(X_test, normalizer=normalizer)
    
    # Remove the correlation between the sensitive feature and the other features
    X_train_norm_cr, remover = corr_remover(X_train_norm, sen_attr, ignore_ids=categorical_ids)
    X_test_norm_cr, dummy = corr_remover(X_test_norm, sen_attr, ignore_ids=categorical_ids, remover=remover)
    
    # Return the data
    data_dict = {
        "X_c": (X_c_train, X_c_test),
        "X": (X_train, X_test),
        "y": (y_train, y_test),
        "z": (z_train, z_test),
        "X_norm": (X_train_norm, X_test_norm),
        "X_norm_cr": (X_train_norm_cr, X_test_norm_cr),
    }
    return data_dict
    
