from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import random
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split

from src.risk_classification.risk_classifiers.classifier import Classifier
from src.risk_classification.input_metrics.input_metrics import InputMetrics
from src.risk_classification.input_metrics.input_metric import InputMetric

from src.risk_classification.input_metrics.metric_names import MetricNames

# LightGBM documentation: https://lightgbm.readthedocs.io/
# Optuna documentation: https://optuna.readthedocs.io/

# Optuna is a fast, extensive hyperparameter tuning library. Its hyperparameter tuning framework is exponentially faster than naive grid search. Optuna also supports 
# making plots for analysis of hyperparameters and features, such as a plot of feature importances.

class LightGBMRiskClassifier(Classifier):
    # Preprocessing (e.g., scaling) will be done using the metrics that Grainger and Dr. Hernandez
    # have developed.

    def __init__(self, params):
        """Constructor for LightGBMRiskClassifier.
           
           Args: 
            x, y (np.ndarray, np.ndarray): the current dataset (x is the matrix of examples, y is the vector of labels); if user did not input any dataset, set x, y to be the 
                                           first dataset that Grainger gave
            params (dict): is a dict of parameters, including hyperparameters, for the LightGBM risk classifier
        
        """
        model = lgb.LGBMClassifier(params)
        super().__init__('LightGBM', model)
        self.current_dataset = None

    # train lightgbm risk classifier using 33% holdout cross-validation
    def train_model_optuna(self,  x, y, **kwargs):
        self.current_dataset = [x, y]
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        # train lightgbm
        study = optuna.create_study(direction="minimize")
        study.optimize(self.opt_objective, n_trials=100)
        # optuna.visualization.plot_optimization_history(study)
        # get best trial's lightgbm (hyper)parameters and print best trial score
        trial = study.best_trial
        lgbdata = lgb.Dataset(x, label=y, feature_name=kwargs['names'])
        trial.params["objective"]="binary"
        trial.params['is_unbalanced'] = kwargs['is_unbalanced']
        self.params = trial.params
        model = lgb.train(trial.params, lgbdata)
        print("in LightGMB",trial.params,model.params)
        self.set_model(model)

    def train_model_optuna_multiclass(self, x, y, num_classes, **kwargs):
        self.current_dataset = [x, y]
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        # train lightgbm
        study = optuna.create_study(direction="minimize")
        study.optimize(self.opt_objective, n_trials=100)
        # optuna.visualization.plot_optimization_history(study)
        # get best trial's lightgbm (hyper)parameters and print best trial score
        trial = study.best_trial
        lgbdata = lgb.Dataset(x, label=y, feature_name=kwargs['names'])
        # TODO: need to specify the number of classes for classifier, see error on debug
        trial.params["objective"] = "multiclass"
        trial.params["num_classes"] = num_classes
        trial.params['is_unbalanced'] = kwargs['is_unbalanced']
        self.params = trial.params
        model = lgb.train(trial.params, lgbdata)
        print("in LightGMB", trial.params, model.params)
        self.set_model(model)
        # print("Best LOOCV value was {}\n".format(trial.value))

    def train_model(self, input_metrics, **kwargs):
        x_train, x_test, y_train, y_test = self.split_input_metrics(input_metrics)
        x_train, x_test = self.scale_train_test_data(x_train, x_test)
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'num_leaves': 40,
            'learning_rate': 0.1,
            'feature_fraction': 0.9
                  }
        eval_results = {}
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=200,
                        valid_sets=[lgb_train, lgb_eval],
                        valid_names=['train', 'valid'],
                        evals_result=eval_results
        )
        self.set_model(gbm)
        return eval_results

    def group_cv(self, x, y, groups, feature_names, multiclass, viz=False):
        n_splits = 5
        lw = 10
        # Shuffle the groups to create uniform distribution of classes in splits
        # x, y, groups = self.shuffle_groups(x, y, groups)
        # Create the test and train splits with group k-fold
        # cv = KFold(n_splits=5, shuffle=True)
        # cv = GroupShuffleSplit(n_splits=5)
        # cv = GroupKFold(n_splits=5)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True)
        # For every split, scale data, train model, and score model, append to results
        scores = []
        fig, ax = plt.subplots()
        for split_num, (train_ixs, test_ixs) in enumerate(cv.split(x, y, groups)):
            x_train = [x[ix] for ix in train_ixs]
            y_train = [y[ix] for ix in train_ixs]
            x_test = [x[ix] for ix in test_ixs]
            y_test = [y[ix] for ix in test_ixs]
            x_train, x_test = self.scale_train_test_data(x_train, x_test)
            num_classes = 3
            if multiclass:
                self.train_model_optuna_multiclass(x_train, y_train, num_classes, names=feature_names, is_unbalanced=True)
            else:
                self.train_model_optuna(x_train, y_train, names=feature_names, is_unbalanced=True)
            y_pred = self.make_prediction(x_test, multiclass)
            scores.append(self.score_model_pred(y_test, y_pred, multiclass))
            if viz:
                self.plot_cv_indices(cv, x, y, groups, ax,
                                     n_splits, train_ixs, test_ixs,
                                     split_num, lw)
        if viz:
            cmap_data = plt.cm.Paired
            cmap_cv = plt.cm.coolwarm
            # Plot the data classes and groups at the end
            ax.scatter(
                range(len(x)), [split_num + 1.5] * len(x), c=y, marker="_", lw=lw, cmap=cmap_data
            )

            ax.scatter(
                range(len(x)), [split_num + 2.5] * len(x), c=groups, marker="_", lw=lw, cmap=cmap_cv
            )
            # Formatting
            yticklabels = list(range(n_splits)) + ["class", "group"]
            ax.set(
                yticks=np.arange(n_splits + 2) + 0.5,
                yticklabels=yticklabels,
                xlabel="Sample index",
                ylabel="CV iteration",
                ylim=[n_splits + 2.2, -0.2],
            )
            ax.set_title("{}".format(type(cv).__name__), fontsize=15)
            plt.show()
        return scores

    def score_model_pred(self, y_true, y_pred, multiclass):
        perf_metrics = {}
        # Assess model accuracy
        perf_metrics['accuracy'] = self.assess_accuracy(y_true, y_pred)
        # Assess model F1 score
        perf_metrics['f1_score'] = self.assess_f1_score(y_true, y_pred)
        # TODO: Assess ROC AUC
        self.assess_roc_auc(y_true, y_pred, multiclass)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        cm_df = pd.DataFrame(cm, index=[0, 1, 2], columns=[0, 1, 2])
        # perf_metrics['confusion_matrix'] = self.assess_conf_matrix(y_true, y_pred, multiclass)
        perf_metrics['confusion_matrix'] = cm_df
        return perf_metrics

    def assess_conf_matrix(self, y_true, y_pred, multiclass):
        if multiclass:
            return self.multilabel_confusion_matrix(y_true, y_pred)
        else:
            return confusion_matrix(y_true, y_pred)

    def assess_accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def assess_f1_score(self, y_true, y_pred):
        # Assessment of the models precision and recall
        return f1_score(y_true, y_pred, average=None)

    def assess_roc_auc(self, y_true, y_pred, multiclass):
        # TODO: research and implement the binary and multiclass approach
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
        pass

    def viz_groups(self, classes, groups):
        cmap_data = plt.cm.Paired
        cmap_cv = plt.cm.coolwarm
        fig, ax = plt.subplots()
        ax.scatter(
            range(len(groups)),
            [0.5] * len(groups),
            c=groups,
            marker="_",
            lw=50,
            cmap=cmap_cv,
        )
        ax.scatter(
            range(len(groups)),
            [3.5] * len(groups),
            c=classes,
            marker="_",
            lw=50,
            cmap=cmap_data,
        )
        ax.set(
            ylim=[-1, 5],
            yticks=[0.5, 3.5],
            yticklabels=["Data\ngroup", "Data\nclass"],
            xlabel="Sample index",
        )
        plt.show()

    def plot_cv_indices(self, cv, x, y, group, ax, n_splits, train_ixs, test_ixs, split_num, lw):
        """Create a sample plot for indices of a cross-validation object."""
        cmap_data = plt.cm.Paired
        cmap_cv = plt.cm.coolwarm
        # Generate the training/testing visualizations for each CV split
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(x))
        indices[test_ixs] = 1
        indices[train_ixs] = 0
        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [split_num + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    def shuffle_groups(self, x, y, groups):
        z = list(zip(x, y, groups))
        random.shuffle(z)
        x, y, groups = zip(*z)
        return np.array(x), np.array(y), np.array(groups)

    # def train_model(self, x, y, **kwargs):
    #     lgb_train = lgb.Dataset(x, y)
    #     params = {
    #         'boosting_type': 'gbdt',
    #         'objective': 'binary',
    #         'num_leaves': 40,
    #         'learning_rate': 0.1,
    #         'feature_fraction': 0.9
    #               }
    #     gbm = lgb.train(params,
    #                     lgb_train,
    #                     num_boost_round=200,
    #     )
    #     self.set_model(gbm)

    def make_prediction(self, samples, multiclass=False, **kwargs):
        raw_predictions = self.model.predict(samples)
        if not multiclass:
            return np.rint(raw_predictions)
        else:
            return [np.argmax(line) for line in raw_predictions]

    def multilabel_confusion_matrix(self, y_test, y_pred):
        return multilabel_confusion_matrix(y_test, y_pred,labels=[0,1,2])

    def score_model(self, x_test, y_test, multiclass=False, **kwargs):
        raw_predictions = self.model.predict(x_test)
        if not multiclass:
            predictions = np.rint(raw_predictions)
        else:
            predictions = [np.argmax(line) for line in raw_predictions]
        return accuracy_score(y_test, predictions), predictions

    # Train lightgbm risk classifier using k-fold cross-validation with 5 being the default value of k.
    # See lightgbm_tuner_cv.py from https://github.com/optuna/optuna-examples/tree/main/lightgbm to see how I implemented k-fold CV training.
    def cross_validate(self, x, y, folds=5, **kwargs):
        # TODO: may be able to switch out the portion prior to CV with train model fxn, not sure difference
        #optuna.logging.set_verbosity(optuna.logging.ERROR)
        lgb_dataset_for_kfold_cv = optuna.integration.lightgbm.Dataset(x, label=y)

        # Set training parameters for Optuna's LightGBM k-fold CV algorithm.
        # These are the only parameters to input to Optuna's LightGBM k-fold CV algorithm that actually affect training (other than categorical_feature). All the other 
        # parameters to Optuna's LightGBM k-fold CV algorithm affect only logging messages or saving LightGBM models.
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
        }

        # perform optimal parameter search using k-fold cv
        # (uses 1000 boosting rounds with 100 early stopping rounds)
        tuner = optuna.integration.lightgbm.LightGBMTunerCV(
            params, lgb_dataset_for_kfold_cv, early_stopping_rounds=100, folds=KFold(n_splits=folds))
        tuner.run()
        lgbdata = lgb.Dataset(x, label=y)
        model = lgb.LGBMClassifier(objective=tuner.best_params['objective'],
                    eval_metric=tuner.best_params['metric'],
                    reg_alpha=tuner.best_params['lambda_l1'],
                    reg_lambda=tuner.best_params['lambda_l2'],
                    feature_fraction=tuner.best_params['feature_fraction'],
                    bagging_fraction=tuner.best_params['bagging_fraction'],
                    bagging_freq=tuner.best_params['bagging_freq'],
                    min_child_samples=tuner.best_params['min_child_samples'])
        # get best trial's lightgbm (hyper)parameters and print best trial score
        self.set_model(model)
        return self.cross_validator.cross_val_model(model, x, y, folds)
        # print("Best {}-fold CV score was {}".format(k, tuner.best_score))

    def cv_existing_model(self, x, y, folds=5):
        return cross_val_score(self.model, x, y, folds)

    # LOOCV objective function for optuna
    def opt_objective(self, trial):
        """
        # https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258
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
        :param trial:
        :return:
        """
        # get current dataset and then perform validation split
        x = self.current_dataset[0]
        y = self.current_dataset[1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
        # x_train_t, x_test_t = self.scale_train_test_data(x_train, x_test)
        # create lgb dataset for lightgbm training
        lgbdata = lgb.Dataset(x_train, label=y_train)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-15, 60.0,
                                             log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-15, 60.0,
                                             log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction",
                                                    0.01, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction",
                                                    0.01, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 20),
            "min_child_samples": trial.suggest_int("min_child_samples", 3,
                                                   100),
        }
        my_lgbm = lgb.train(params, lgbdata)
        raw_predictions = my_lgbm.predict(x_test)
        predictions = np.rint(raw_predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        return rmse


if __name__ == "__main__":
    lgbm_risk_classifier = LightGBMRiskClassifier()

    # do 25% holdout CV on default dataset
    lgbm_risk_classifier.train_loocv()

    # do k-fold CV on default dataset (default k = 5)
    lgbm_risk_classifier.train_KFold()
