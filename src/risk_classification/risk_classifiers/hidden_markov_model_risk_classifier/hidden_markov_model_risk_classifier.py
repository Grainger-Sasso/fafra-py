from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
from hmmlearn import hmm


# using Gaussian emission since Carapace Analytics will be using accelerometric, not gyroscopic, data;
# Yuwono et. al showed that the accelerometric data input to RCE tends to be Gaussian, so no mixture
# seems to be required (since FFT and DFT are both linear combinations of the signals)
# and hence our Bayes filter needs to cover just Gaussian distributions (hence we'll use a Kalman filter as
# our Bayes filter since Kalman filter is the Bayes filter for Gaussian distributions)


class GaussianHMMRiskClassifier:
    hmm_params = {}
    def __init__(self, **kwargs):
        # use covariance_type="full"
        if kwargs is None:
            kwargs = {}
        self.model = hmm.GaussianHMM(kwargs)
        hmm_params = kwargs

    def get_model(self) -> hmm.GaussianHMM():
        return self.model

    def set_model(self, **kwargs):
        self.model = hmm.GaussianHMM(kwargs)

    # def train_valid_split(self, x: np.ndarray, y: np.ndarray, test_size=0.25, random_state=42):
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    #     return x_train, x_test, y_train, y_test

    # HMM is used in an unsupervised manner in Yuwono's paper
    #def fit_model(self, x: np.ndarray, lengths: int):
        #for n in range(1, 500):
            #temp_hmmgmm = GaussianHMMRiskClassifier(self.get_model().__dict__)
            #temp_hmmgmm.fit(x, lengths)

            # if temp_hmmgmm's model is better than self.model:
                # self.model.startprob_ = temp_hmmgmm.startprob_
                # self.model.transmat_ = temp_hmmgmm.transmat_
                # self.model.means_ = temp_hmmgmm.means_
                # self.model.covars_ = temp_hmmgmm.covars_

# insert params into hmm_gmm_risk_classifier below (using names of hmm.GaussianHMM params; no need to input in a dict)
# gaussian_hmm_risk_classifier = GaussianHMMRiskClassifier(*insert params w/names here*)