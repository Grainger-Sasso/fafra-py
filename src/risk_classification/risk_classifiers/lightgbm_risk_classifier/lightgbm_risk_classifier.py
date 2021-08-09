from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import lightgbm as lgb
import optuna


class LightGBMRiskClassifier:
    # Actual preprocessing (e.g., scaling) will be done using the metrics that Grainger and Dr. Hernandez
    # have developed.

    def __init__(self, params: dict = None):
        # params is a dict of parameters, including hyperparameters, for the LightGBM risk classifier
        if params is None:
            params = {}
        self.model = lgb.LGBMClassifier(params)
        self.scaler = preprocessing.StandardScaler()

    def get_model(self) -> lgb.LGBMClassifier():
        return self.model

    def set_model(self, params: dict = None):
        if params is None:
            params = {}
        self.model = lgb.LGBMClassifier(params)

    def get_scaler(self) -> preprocessing.StandardScaler():
        return self.scaler

    def set_scaler(self, scaler=preprocessing.StandardScaler()):
        self.scaler = scaler

    def scale_input_data(self, x):
        self.scaler.fit(x)
        return self.scaler.transform(x)

    def scale_train_test_data(self, x_train, x_test):
        # Fit the scaler to the training data
        self.scaler.fit(x_train)
        # Transform the training data
        x_train_t = self.scaler.transform(x_train)
        # Transform the test data
        x_test_t = self.scaler.transform(x_test)
        return x_train_t, x_test_t

    def split_input_metrics(self, x, y, test_size=0.3, random_state=42):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test

    # objective function for optuna
    def opt_objective(self, trial):
        # use numpy to read data CSVs into numpy arrays
        x = np.genfromtxt('C:\\Users\\fancy\\Downloads\\x_data_metrics.csv', delimiter=',')
        y = np.genfromtxt('C:\\Users\\fancy\\Downloads\\y_data_metrics.csv', delimiter=',')
        train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.25)

        # create lgb dataset for lightgbm training
        lgbdata = lgb.Dataset(train_x, label=train_y)

        # set parameter search spaces for optuna to conduct hyperparameter optimization for max validation accuracy
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-15, 60.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-15, 60.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.01, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.01, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 20),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 100),
        }

        my_lgbm = lgb.train(params, lgbdata)
        raw_predictions = my_lgbm.predict(valid_x)
        predictions = np.rint(raw_predictions)
        accuracy = sklearn.metrics.accuracy_score(valid_y, predictions)
        return accuracy

    def fit_model(self, x: np.ndarray, y: np.ndarray):
        """
        Fits LightGBM model to input training vectors, x, and target values, y (notation is canonically used)
        :param x: Training vectors
        :type: np.ndarray
        :param y: Target values
        :type: np.ndarray
        :return: Trained model
        :rtype: lgb.LGBMClassifier()
        """
        self.model.fit(x, y)

    def make_prediction(self, samples):
        return self.model.predict(samples)

    def score_model(self, x_test, y_test):
        return self.model.score(x_test, y_test)

    def create_classification_report(self, y_test, y_pred):
        return classification_report(y_test, y_pred)


lgbm_risk_classifier = LightGBMRiskClassifier()

if __name__ == "__main__":
    print("Starting training")
    print("\n")
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # try lightgbm with LOOCV
    study = optuna.create_study(direction="maximize")
    study.optimize(lgbm_risk_classifier.opt_objective, n_trials=1000)
    trial = study.best_trial
    lgbm_risk_classifier.set_model(trial.params)

    # now try lightgbm with k-fold CV (k = 5 or k = 10)
    x = np.genfromtxt('x_data_metrics.csv', delimiter=',')
    y = np.genfromtxt('y_data_metrics.csv', delimiter=',')
    lgb_dataset_for_kfold_cv = optuna.integration.lightgbm.Dataset(x, label=y)
    k = 5
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
    }
    tuner = optuna.integration.lightgbm.LightGBMTunerCV(
        params, lgb_dataset_for_kfold_cv, early_stopping_rounds=100, folds=KFold(n_splits=k))
    tuner.run()

    print("Best LOOCV trial value was {}".format(trial.value))
    print("\n")
    print("Best k-fold CV score was {}".format(tuner.best_score))
    # lgbm_risk_classifier.set_model(tuner.best_params)
