import numpy as np
from scipy.fft import fft, fftfreq
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class FastFourierTransform:
    def __init__(self):
        pass

    def perform_fft(self, data: np.array, sampling_rate: float, detrend=True):
        # y_fft = np.fft.fft(data)
        # x_fft = np.fft.fftfreq(n=data.size, d=1/sampling_rate)
        # return x_fft[:len(x_fft)//2], y_fft.real[:len(y_fft)//2]
        if detrend:
            data = data - np.mean(data)
            data = self._apply_polynomial_detrend(data, sampling_rate)
        y_fft = fft(data)
        x_fft = fftfreq(data.size, 1/sampling_rate)
        return x_fft[:len(x_fft) // 2], abs(y_fft[:len(y_fft) // 2])

    def _apply_polynomial_detrend(self, data, sampling_rate):
        """
        https://www.mathworks.com/matlabcentral/answers/124471-fft-significant-peak-in-0-hz-component
        https://www.investopedia.com/terms/d/detrend.asp
        https://towardsdatascience.com/removing-non-linear-trends-from-timeseries-data-b21f7567ed51
        :param data:
        :param sampling_rate:
        :return:
        """
        time = np.linspace(0, len(data)*(1/sampling_rate), len(data))
        time = np.reshape(time, (len(time), 1))
        pf = PolynomialFeatures(degree=2)
        Xp = pf.fit_transform(time)
        md2 = LinearRegression()
        md2.fit(Xp, data)
        trendp = md2.predict(Xp)
        detrpoly = [data[i] - trendp[i] for i in range(0, len(data))]
        return np.array(detrpoly)




# from scipy.fft import fft, fftfreq
# # Number of sample points
# N = 600
# # sample spacing
# T = 1.0 / 800.0
# x = np.linspace(0.0, N*T, N, endpoint=False)
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# yf = fft(y)
# xf = fftfreq(N, T)[:N//2]
# import matplotlib.pyplot as plt
# plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# plt.grid()
# plt.show()
