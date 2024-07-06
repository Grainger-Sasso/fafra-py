import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    path = '/home/grainger-sasso/Desktop/fafra_guest_shared/datasets/Upstate/Neville/test/20240621-123716_sensor_data.csv'
    df = pd.read_csv(path)
    z_acc = np.array(df['Unnamed: 3'][2:])
    z_acc = z_acc.astype(float)
    time = np.arange(len(z_acc))
    # print(type(z_acc), type(z_acc[0]))
    # print(type(time), type(time[0]))
    plt.plot(time, z_acc)
    plt.show()

if __name__ == '__main__':
    main()