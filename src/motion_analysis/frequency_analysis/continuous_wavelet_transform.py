import pywt
import matplotlib.pyplot as plt

class CWT:
    def __init__(self):
        pass

    def apply_cwt(self, data):
        pass


import pywt
import matplotlib.pyplot as plt

# Example taken from:
# https://towardsdatascience.com/multiple-time-series-classification-by-using-continuous-wavelet-transformation-d29df97c0442


def split_indices_per_label(y):
    indicies_per_label = [[] for x in range(0, 6)]
    # loop over the six labels
    for i in range(6):
        indicies_per_label[i] = np.where(y == i)[0]
    return indicies_per_label


def plot_cwt_coeffs_per_label(X, label_indicies, label_names, signal, sample,
                              scales, wavelet):
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True,
                            figsize=(12, 5))

    for ax, indices, name in zip(axs.flat, label_indicies, label_names):
        # apply  PyWavelets continuous wavelet transfromation function
        coeffs, freqs = pywt.cwt(X[indices[sample], :, signal], scales,
                                 wavelet=wavelet)
        # create scalogram
        ax.imshow(coeffs, cmap='coolwarm', aspect='auto')
        ax.set_title(name)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('Scale')
        ax.set_xlabel('Time')
    plt.tight_layout()


# list of list of sample indicies per activity
train_labels_indicies = split_indices_per_label(y_train)

# signal indicies: 0 = body acc x, 1 = body acc y, 2 = body acc z, 3 = body gyro x, 4 = body gyro y, 5 = body gyro z, 6 = total acc x, 7 = total acc y, 8 = total acc z
signal = 3  # signal index
sample = 1  # sample index of each label indicies list
scales = np.arange(1, 65)  # range of scales
wavelet = 'morl'  # mother wavelet

plot_cwt_coeffs_per_label(X_train, train_labels_indicies, LABEL_NAMES, signal,
                          sample, scales, wavelet)


