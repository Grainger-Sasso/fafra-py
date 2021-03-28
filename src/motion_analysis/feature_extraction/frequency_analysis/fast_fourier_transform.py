import numpy as np
from scipy.fft import fft, fftfreq


class FastFourierTransform:
    def __init__(self):
        pass

    def perform_fft(self, data: np.array, sampling_rate: float):
        # y_fft = np.fft.fft(data)
        # x_fft = np.fft.fftfreq(n=data.size, d=1/sampling_rate)
        # return x_fft[:len(x_fft)//2], y_fft.real[:len(y_fft)//2]
        y_fft = fft(data)
        x_fft = fftfreq(data.size, 1/sampling_rate)
        return x_fft[:len(x_fft) // 2], abs(y_fft[:len(y_fft) // 2])




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
