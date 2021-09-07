from sklearn.datasets import make_blobs


class DataGenerator:

    def generate_data(self, n_samples=100, n_features=8,
                      centers=2, random_state=0):
        return make_blobs(n_samples=n_samples, n_features=n_features,
                          centers=centers, random_state=random_state)


