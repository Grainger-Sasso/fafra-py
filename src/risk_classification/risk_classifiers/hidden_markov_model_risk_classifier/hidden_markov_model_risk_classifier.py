from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
from hmmlearn import hmm


# using Gaussian emission since Carapace Analytics will be using accelerometric, not gyroscopic, data;
# Yuwono et. al showed that the accelerometric data input to RCE tends to be Gaussian, so no mixture
# seems to be required (since FFT and DFT are both linear combinations of the signals);
# also, Yuwono et. al used only one mixture in their Hidden Markov Model (though they did not specify whether it
# was Gaussian, although it is reasonable to assume that the mixture they used was indeed Gaussian), and hence
# our Bayes filter needs to cover just Gaussian distributions (hence we'll use a Kalman filter as
# our Bayes filter since Kalman filter is the Bayes filter for Gaussian distributions)

class GaussianHMMRiskClassifier:
    def __init__(self, **kwargs):
        # Use covariance_type="full" since diagonal covariance matrix allows for accurate modelling iff
        # the number of mixtures is > 1 (according to a stackexchange answer).
        # n_components = length of input data vectors.
        # Set n_iter to some integer >= 100 and <= thousands.
        # Leave "params" and "init_params" to their default value (which is "stmc").
        # Leave all other GaussianHMM constructor parameters to their default values.
        # That covers all the GaussianHMM constructor parameters.
        if kwargs is None:
            # instead of kwargs = {}, will know n_components (and n_iter) when data dimensions will be known
            kwargs = {}
        self.model = hmm.GaussianHMM(kwargs)

        # Boolean variable of instance self to keep track of whether self.model has been trained using
        # k-fold cross-validation.
        self.isKFoldtrained = False

    def get_model(self) -> hmm.GaussianHMM():
        return self.model

    def set_model(self, **kwargs):
        self.model = hmm.GaussianHMM(kwargs)

    def get_hmm_params(self) -> dict:
        return dict(startprob_=self.model.startprob_,
                    transmat_=self.model.transmat_,
                    means_=self.model.means_,
                    covars_=self.model.covars_)

    def set_hmm_params(self, other):
        # other must be another instance of GaussianHMMRiskClassifier()

        if not isinstance(other, GaussianHMMRiskClassifier):
            raise TypeError("Input variable 'other' was not an instance of GaussianHMMRiskClassifier")

        self.model.startprob_ = other.model.startprob_
        self.model.transmat_ = other.model.transmat_,
        self.model.means_ = other.model.means_,
        self.model.covars_ = other.model.covars_

    # HMM is used in an unsupervised manner in Yuwono's paper. We will use LOOCV on the x data to ensure
    # that our HMM is maximally generalizable.
    def fit_model_loocv(self, x: np.ndarray, y: np.ndarray, length: int):
        # randomly split x data into train/validation (need labels y in order to do this
        x_train, y_1, x_valid, y_2 = train_test_split(x, y, test_size=0.25)

        # initially attempt to fit self.model to the train split (using EM algorithm)
        lengths = [length for i in range(0, 0.75 * x.shape[0])]
        self.model.fit(x_train, lengths)

        # get current model's validation score
        valid_lengths = [length for i in range(0, x_valid.shape[0])]
        current_score = self.model.score(x_valid, valid_lengths)

        # Try EM algorithm 500 times, choosing only the trial that has the
        # best HMM parameters.
        for n in range(1, 500):
            # create another random split of x data into train/validation
            x_train, y_1, x_valid, y_2 = train_test_split(x, y, test_size=0.25)

            # Create a temporary instance of GaussianHMMRiskClassifier w/ same constructor inputs as self, then
            # fit temporary model to the train split (using EM algorithm).
            temp_hmmgmm = GaussianHMMRiskClassifier(self.model.__dict__)
            temp_hmmgmm.model.fit(x_train, lengths)

            temp_score = temp_hmmgmm.model.score(x_valid, valid_lengths)
            if current_score < temp_score:
                self.set_hmm_params(temp_hmmgmm)
                current_score = temp_score

    # private function that returns predictor defined by the k models in k-fold cross-validation
    # only commit this function; do not push it
    def __kfold_function(self, *args):
        return lambda x: (1 / len(args)) * np.sum(gaussianhmm(x) for gaussianhmm in args)

# insert params into hmm_gmm_risk_classifier below (using names of hmm.GaussianHMM params; no need to input in a dict)
# gaussian_hmm_risk_classifier = GaussianHMMRiskClassifier(n_iter = ..., n_components = ...)
