# import lightgbm
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

# functions to test are imported from train.py
from train import split_data, train_model, get_model_metrics

"""A set of simple unit tests for protecting against regressions in train.py"""


def test_split_data():
    test_data = {
        'Shell weight': [0, 0.15, 0.07, 0.21, 0.155],
        'Age': [2.5, 16.5, 8.5, 10.5, 11.5],
        'Length': [0.455, 0.35, 0.53, 0.44, 0.33],
        'Diameter': [0.365, 0.265, 0.42, 0.365, 0.255]
        }

    data_df = pd.DataFrame(data=test_data)
    data = split_data(data_df)

    # # verify that columns were removed correctly
    # assert "Age" not in data[0].data.columns
    # assert "Shell weight" not in data[0].data.columns
    # assert "Length" in data[0].data.columns

    # # verify that data was split as desired
    # assert data[0].data.shape == (4177, 9)
    # assert data[1].data.shape == (1, 2)


def test_train_model():
    data = __get_test_datasets()

    params = {
        "learning_rate": 0.05,
        "metric": "auc",
        "min_data": 1
    }

    model = train_model(data, params)

    # verify that parameters are passed in to the model correctly
    for param_name in params.keys():
        assert param_name in model.params
        assert params[param_name] == model.params[param_name]


def test_get_model_metrics():
    class MockModel:

        @staticmethod
        def predict(data):
            return np.array([0, 0])

    data = __get_test_datasets()

    metrics = get_model_metrics(MockModel(), data)

    # verify that metrics is a dictionary containing the auc value.
    assert "auc" in metrics
    auc = metrics["auc"]
    np.testing.assert_almost_equal(auc, 0.5)


def __get_test_datasets():
    """This is a helper function to set up some test data"""
    X_train = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    y_train = np.array([1, 1, 0, 1, 0, 1])
    X_test = np.array([7, 8]).reshape(-1, 1)
    y_test = np.array([0, 1])

    train_data = RandomForestRegressor.Dataset(X_train, y_train)
    valid_data = RandomForestRegressor.Dataset(X_test, y_test)
    data = (train_data, valid_data)
    return data
