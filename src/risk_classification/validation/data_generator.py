from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt


class DataGenerator:

    def generate_data(self, n_samples=1000, n_features=2,
                      centers=2, random_state=0):
        return make_blobs(n_samples=n_samples, n_features=n_features,
                          centers=centers, random_state=random_state)


