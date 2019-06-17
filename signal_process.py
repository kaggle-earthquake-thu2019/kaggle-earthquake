import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.fftpack as fft

if __name__ == '__main__':
    file_path = "earthquakes/"
    filename = "earthquake_2.csv"
    signal_data = pd.read_csv(filepath_or_buffer=file_path + filename, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

    data_size = signal_data.shape[0]
    seg_size = 150000
    segments = int(np.ceil(data_size / seg_size))

    i = 0
    rows = np.floor(segments / 20) + 1
    plt.figure(figsize = (20, 5 * rows))
    for seg_id in tqdm(range(segments)):
        start = seg_id * seg_size
        end = seg_id * seg_size + seg_size
        if end >= data_size:
            end = data_size - 1
            start = data_size - seg_size
        time_to_failure = signal_data['time_to_failure'][start:end].values
        segment = signal_data['acoustic_data'][start:end].values

        
        if seg_id % 20 == 0:
            xc = pd.Series(segment)
            zc = fft.fftshift(fft.fft(xc))
            low = 60000
            high = 900000
            zc[0:low - 1] = 0
            zc[high:-1] = 0
            low_signal = fft.ifft(zc)
            '''mag = np.sqrt(np.real(zc) ** 2 + np.imag(zc) ** 2)
            freq_fft = fft.fftfreq(len(xc), d = 0.1)
            ax = plt.subplot(rows, 3, int(1 + i))
            ax.set_title(time_to_failure[-1])
            plt.plot(mag)
            ax.set_ylim([0, 20000])
            
            '''
            ax = plt.subplot(rows, 2, int(1 + i))
            ax.set_title([time_to_failure[-1],"low"])
            plt.plot(low_signal)
            ax = plt.subplot(rows, 2, int(2 + i))
            ax.set_title([time_to_failure[-1],"original"])
            plt.plot(xc)
            
            i += 2
