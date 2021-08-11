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


# LightGBM documentation: https://lightgbm.readthedocs.io/
# Optuna documentation: https://optuna.readthedocs.io/

# Optuna is a fast, extensive hyperparameter tuning library.

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

    # LOOCV objective function for optuna
    def opt_objective(self, trial):
        # get current dataset and then perform validation split
        x, y = self.current_dataset()
        train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.25)

        # create lgb dataset for lightgbm training
        lgbdata = lgb.Dataset(train_x, label=train_y)

        # Set parameter search spaces for optuna to conduct hyperparameter optimization for max validation accuracy.
        # See lightgbm_simple.py from https://github.com/optuna/optuna-examples/tree/main/lightgbm (a folder of LightGBM implementations using Optuna coded up by the Optuna 
        # authors) for how I implemented LOOCV training for LightGBMRiskClassifier.
     
        # For binary classification, use binary objective and binary_logloss metric. gbdt (gradient-boosted trees) should be the boosting_type.
        
        # lambda_l1 and lambda_l2 are the respective coefficients for L1 and L2 regularization.
        # In the lightgbm_simple.py file referenced above, the optuna authors used the same search space limits
        # for lambda_l1 and lambda l_2. However, Gabriel Tseng (source: https://medium.com/@gabrieltseng/gradient-boosting-and-xgboost-c306c1bcfaf5) says that 
        # a large value of lambda_l2 and a small (or 0) value of lambda_l1 should be used. This makes sense for our case since the dataset we input to the LightGBMRiskClassifier
        # has already been transformed by the metrics (i.e., feature engineering) devised by Grainger and Dr. Hernandez. Thus, lambda_l1 should be very small (in fact,
        # probably 0) and lambda_l2 should be large (or, at least, the upper bound of the search space of lambda_l2 should be large).
        
        # num_leaves is the number of leaves to use in the decision trees of gradient-boosted tree learning. I found that validation accuracy dropped when I replaced 256
        # with 512 as the upper bound of the search space of num_leaves for the original 340-sample dataset that Grainger gave. Maybe experiment with values larger than 
        # 256 for larger datasets? Values of num_leaves that are too large will lead to overfitting, of course.
        
        # feature_fraction and bagging_fraction are both floats <= 1.0 and should both be positive. I felt that [0.01, 1.0] was a pretty large (i.e., inclusive) search 
        # space for both feature_fraction and bagging_fraction. Future experiments should try changing (increasing) the lower bound of the search space of feature_fraction 
        # and/or bagging_fraction from 0.01 to some larger number <= 1.0.
        
        # bagging_freq should be a positive integer that denotes the number of decision trees after which bagging should be performed. Larger values of bagging_freq
        # will obviously lead to lower variance but may also lead to the LightGBM model underfitting. Smaller values of bagging_freq will lead to larger variance
        # and may not lower the bias by a significant amount.
        # Future experiments should change the upper bound for the search space of bagging_freq (I came up with the number 20 in my head randomly).
        
        # min_child_samples is a positive integer denoting the minimum number of data samples needed in a leaf of a decision tree. The LightGBM documentation says that 
        # this hyperparameter is very important for preventing overfitting. I do not know what a good search space is for min_child_samples, but I do 
        # know that excessively small values of min_child_samples will cause overfitting. The lightgbm_simple.py file from the Optuna authors referenced above 
        # uses [5, 100] as the search space for min_child_samples. Because the dataset that they use has more samples than the original 340-sample dataset 
        # that Grainger gave, I reduced the lower bound of the search space for min_child_samples from 5 to 3 and used the same upper bound of 100.
        
        # Looking at the LightGBM docs, there are a number of LightGBM hyperparameters that the params dict below does not include. However, the hyperparameters that the 
        # params dict does include seem to be by far the most important hyperparameters, and the hyperparameters that do not appear in the below params dict seem largely 
        # to be variations of the hyperparameters appearing in the below params dict or I/O parameters (which do not affect training), hyperparameters 
        # to aid distributed or GPU learning, or hyperparameters to control logging messages during training.
        
        # However, if one wants to use second-order optimization training instead of first-order, then include the hyperparameter min_sum_hessian_in_leaf 
        # in the below params dict, and make sure that the lower bound of the search space of min_sum_hessian_in_leaf is not too small (if it is too small,
        # then overfitting will happen).
        
        # This finishes the entire discussion on LightGBM hyperparameters.
        
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

    # Train lightgbm risk classifier using k-fold cross-validation with 5 being the default value of k.
    # See lightgbm_tuner_cv.py from https://github.com/optuna/optuna-examples/tree/main/lightgbm to see how I implemented k-fold CV training.
    def train_KFold(self, k: int = 5):
        print("Starting {}-fold CV training:\n".format(k))
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
