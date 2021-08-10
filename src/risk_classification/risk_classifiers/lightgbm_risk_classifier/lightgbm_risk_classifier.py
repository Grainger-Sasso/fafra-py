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
    # Preprocessing (e.g., scaling) will be done using the metrics that Grainger and Dr. Hernandez
    # have developed.

    def __init__(self, params: dict = None):
        # params is a dict of parameters, including hyperparameters, for the LightGBM risk classifier
        if params is None:
            params = {}
        self.model = lgb.LGBMClassifier(params)

    def get_model(self) -> lgb.LGBMClassifier():
        return self.model

    def set_model(self, params: dict = None):
        if params is None:
            params = {}
        self.model = lgb.LGBMClassifier(params)

    # output the dataset that the user input; if user input none, then use numpy to generate default data
    def current_dataset(self, x=np.genfromtxt('x_data_metrics.csv'), y=np.genfromtxt('y_data_metrics.csv')):
        return x, y

    # objective function for optuna
    def opt_objective(self, trial):
        # get current dataset
        x, y = self.current_dataset()
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

    # train lightgbm risk classifier using 25% holdout cross-validation
    def train_loocv(self):
        print("Starting LOOCV training with test size of 0.25:\n")
        optuna.logging.set_verbosity(optuna.logging.ERROR)

        # train lightgbm
        study = optuna.create_study(direction="maximize")
        study.optimize(self.opt_objective, n_trials=1000)

        # get best trial's lightgbm (hyper)parameters and print best trial score
        trial = study.best_trial
        self.set_model(trial.params)
        print("Best LOOCV value was {}\n".format(trial.value))

    # train lightgbm risk classifier using k-fold cross-validation with default cross-validation being 5-fold
    def train_KFold(self, k: int = 5):
        print("Starting {}-fold CV training:".format(k))
        optuna.logging.set_verbosity(optuna.logging.ERROR)

        # get current dataset and create a lightgbm-compatible version of it
        x, y = self.current_dataset()
        lgb_dataset_for_kfold_cv = optuna.integration.lightgbm.Dataset(x, label=y)

        # set training parameters
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
        }

        # perform k-fold cross-validation (uses 1000 boosting rounds with 100 early stopping rounds
        tuner = optuna.integration.lightgbm.LightGBMTunerCV(
            params, lgb_dataset_for_kfold_cv, early_stopping_rounds=100, folds=KFold(n_splits=k))
        tuner.run()

        # get best trial's lightgbm (hyper)parameters and print best trial score
        self.set_model(tuner.best_params)
        print("Best {}-fold CV score was {}".format(k, tuner.best_score))

    def make_prediction(self, samples):
        return self.model.predict(samples)

    def score_model(self, x_test, y_test):
        return self.model.score(x_test, y_test)

    def create_classification_report(self, y_test, y_pred):
        return classification_report(y_test, y_pred)


if __name__ == "__main__":
    lgbm_risk_classifier = LightGBMRiskClassifier()

    # do 25% holdout CV on default dataset
    lgbm_risk_classifier.train_loocv()

    # do k-fold CV on default dataset (default k = 5)
    lgbm_risk_classifier.train_KFold()
