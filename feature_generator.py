import os
import numpy as np
import pandas as pd
import warnings
from scipy import stats
from scipy import signal
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

computer_feature = False


def load_data(file_path):
    signal_data = pd.read_csv(filepath_or_buffer=file_path,
                              dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
    return signal_data


def get_spectrum(input_signal, window=100):
    """
    returns the Fourier power spectrum for a given signal segment
    output is a pandas Series
    output.index is the frequencies
    output.values is the amplitudes for each frequencies
    default moving average window is 10
    """
    sampling_frequency = np.ceil(150000 / 0.0375)
    input_signal = np.asarray(input_signal, dtype='float32')

    # Remove the mean
    input_signal -= input_signal.mean()

    # Estimate power spectral density using a periodogram.
    frequencies, power_spectrum = signal.periodogram(
        input_signal, sampling_frequency, scaling='spectrum')

    # Run a running windows average of 10-points to smooth the signal (default).
    power_spectrum = pd.Series(power_spectrum, index=frequencies).rolling(window=window).mean()

    return pd.Series(power_spectrum)


def generate_features(seg_id, segment, X):
    xc = pd.Series(segment)
    zc = np.fft.fft(xc)

    # time domain
    X.loc[seg_id, 'mean'] = xc.mean()
    # X.loc[seg_id, 'std'] = xc.std()
    X.loc[seg_id, 'max'] = xc.max()
    X.loc[seg_id, 'min'] = xc.min()
    X.loc[seg_id, 'autocorr'] = xc.autocorr()

    X.loc[seg_id, 'mad'] = xc.mad()
    X.loc[seg_id, 'kurt'] = xc.kurtosis()
    X.loc[seg_id, 'skew'] = xc.skew()
    # X.loc[seg_id, 'med'] = xc.median()

    # X.loc[seg_id, 'sem'] = xc.sem()
    # X.loc[seg_id, 'var'] = xc.var()

    X.loc[seg_id, 'diff_ave'] = xc.diff().mean()
    X.loc[seg_id, 'diff_max'] = xc.diff().max()
    X.loc[seg_id, 'diff_mad'] = xc.diff().mad()
    X.loc[seg_id, 'diff_kurt'] = xc.diff().kurtosis()
    X.loc[seg_id, 'diff_skew'] = xc.diff().skew()

    # first last 50000
    for window in [50000, 10000, 5000]:
        X.loc[seg_id, 'mean_first' + str(window)] = xc[:window].mean()
        # X.loc[seg_id, 'std_first' + str(window)] = xc[:window].std()
        X.loc[seg_id, 'autocorr_first' + str(window)] = xc[:window].autocorr()
        X.loc[seg_id, 'max_first' + str(window)] = xc[:window].max()
        # X.loc[seg_id, 'med_first' + str(window)] = xc[:window].median()
        # X.loc[seg_id, 'min_first' + str(window)] = xc[:window].min()
        X.loc[seg_id, 'mad_first' + str(window)] = xc[:window].mad()
        X.loc[seg_id, 'kurt_first' + str(window)] = xc[:window].kurtosis()
        X.loc[seg_id, 'skew_first' + str(window)] = xc[:window].skew()

        X.loc[seg_id, 'mean_last' + str(window)] = xc[-window:].mean()
        # X.loc[seg_id, 'std_last' + str(window)] = xc[-window:].std()
        X.loc[seg_id, 'autocorr_last' + str(window)] = xc[-window:].autocorr()
        X.loc[seg_id, 'max_last' + str(window)] = xc[-window:].max()
        # X.loc[seg_id, 'med_last' + str(window)] = xc[-window:].median()
        # X.loc[seg_id, 'min_last' + str(window)] = xc[-window:].min()
        X.loc[seg_id, 'mad_last' + str(window)] = xc[-window:].mad()
        X.loc[seg_id, 'kurt_last' + str(window)] = xc[-window:].kurtosis()
        X.loc[seg_id, 'skew_last' + str(window)] = xc[-window:].skew()

    # absolute
    abs_xc = pd.Series(np.abs(xc))
    X.loc[seg_id, 'abs_mean'] = abs_xc.mean()

    # frequency domain
    realFFT = pd.Series(np.real(zc))
    imagFFT = pd.Series(np.imag(zc))
    X.loc[seg_id, 'r_mean'] = realFFT.mean()
    # X.loc[seg_id, 'r_std'] = realFFT.std()
    X.loc[seg_id, 'r_max'] = realFFT.max()
    # X.loc[seg_id, 'r_min'] = realFFT.min()

    X.loc[seg_id, 'i_mean'] = imagFFT.mean()
    # X.loc[seg_id, 'i_std'] = imagFFT.std()
    X.loc[seg_id, 'i_max'] = imagFFT.max()
    # X.loc[seg_id, 'i_min'] = imagFFT.min()

    X.loc[seg_id, 'r_mad'] = realFFT.mad()
    X.loc[seg_id, 'r_kurt'] = realFFT.kurtosis()
    X.loc[seg_id, 'r_skew'] = realFFT.skew()
    X.loc[seg_id, 'r_var'] = realFFT.var()
    X.loc[seg_id, 'r_sem'] = realFFT.sem()

    X.loc[seg_id, 'i_mad'] = imagFFT.mad()
    X.loc[seg_id, 'i_kurt'] = imagFFT.kurtosis()
    X.loc[seg_id, 'i_skew'] = imagFFT.skew()
    # X.loc[seg_id, 'i_var'] = imagFFT.var()
    # X.loc[seg_id, 'i_sem'] = imagFFT.sem()

    # quantile
    X.loc[seg_id, 'q95'] = np.quantile(xc, 0.95)
    X.loc[seg_id, 'q91'] = np.quantile(xc, 0.91)
    X.loc[seg_id, 'q99'] = np.quantile(xc, 0.99)

    X.loc[seg_id, 'q01'] = np.quantile(xc, 0.01)
    X.loc[seg_id, 'q05'] = np.quantile(xc, 0.05)
    X.loc[seg_id, 'q09'] = np.quantile(xc, 0.09)

    # # rolling windows
    for window in [3000, 5000, 10000]:
        roll_mean = pd.Series(xc.rolling(window=window).mean().dropna().values)
        zc_rolling = np.fft.fft(xc.rolling(window=window).mean().dropna().values)
        # roll_mean_realfft = pd.Series(np.real(zc_rolling))
        roll_mean_imagfft = pd.Series(np.imag(zc_rolling))

        X.loc[seg_id, 'roll_mean_mad_' + str(window)] = roll_mean.mad()
        # X.loc[seg_id, 'roll_mean_kurt_' + str(window)] = roll_mean.kurtosis()
        X.loc[seg_id, 'roll_mean_skew_' + str(window)] = roll_mean.skew()

        # X.loc[seg_id, 'roll_mean_r_mean' + str(window)] = roll_mean_realfft.mean()
        # X.loc[seg_id, 'roll_mean_r_max' + str(window)] = roll_mean_realfft.max()

        X.loc[seg_id, 'roll_mean_i_mean' + str(window)] = roll_mean_imagfft.mean()
        X.loc[seg_id, 'roll_mean_i_max' + str(window)] = roll_mean_imagfft.max()
        X.loc[seg_id, 'roll_mean_i_skew' + str(window)] = roll_mean_imagfft.skew()
        # X.loc[seg_id, 'roll_mean_i_kurt' + str(window)] = roll_mean_imagfft.kurtosis()

    # power spectrum
    max_frequency = 300000
    min_frequency = 100
    power_spectrum = get_spectrum(segment, window=10).dropna()
    power_spectrum = power_spectrum[power_spectrum.index < max_frequency]
    power_spectrum = power_spectrum[power_spectrum.index > min_frequency]

    X.loc[seg_id, 'spectrum_mean'] = power_spectrum.mean()
    X.loc[seg_id, 'spectrum_std'] = power_spectrum.std()
    X.loc[seg_id, 'spectrum_max'] = power_spectrum.max()
    X.loc[seg_id, 'spectrum_min'] = power_spectrum.min()
    X.loc[seg_id, 'spectrum_med'] = power_spectrum.median()
    X.loc[seg_id, 'spectrum_skew'] = power_spectrum.skew()
    X.loc[seg_id, 'spectrum_kurt'] = power_spectrum.kurtosis()
    X.loc[seg_id, 'spectrum_mad'] = power_spectrum.mad()



def data_process(file_path):
    signal_data = load_data(file_path)
    data_size = signal_data.shape[0]
    seg_size = 150000
    segments = int(np.ceil(data_size / seg_size))

    train_X = pd.DataFrame(index=range(segments), dtype=np.float64)
    train_Y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])

    for seg_id in tqdm(range(segments)):
        start = seg_id * seg_size
        end = seg_id * seg_size + seg_size
        if end >= data_size:
            end = data_size - 1
            start = data_size - seg_size
        time_to_failure = signal_data['time_to_failure'][start:end].values

        segment = signal_data['acoustic_data'][start:end].values
        generate_features(seg_id, segment, train_X)

        train_Y.loc[seg_id, 'time_to_failure'] = time_to_failure[-1]

    print("finish loading")
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)

    return train_X, train_Y


def make_dataset(file_dir, output_dir):
    files = os.listdir(file_dir)

    X, Y = data_process('../earthquakes/earthquake_1.csv')
    X.to_hdf(output_dir + "1_x.hdf", 'data')
    Y.to_hdf(output_dir + "1_y.hdf", 'data')

    for file in files:
        if not os.path.isdir(file) and file.split(".")[1] == "csv":  # 判断是否是文件夹，不是文件夹才打开
            print(f"loading {file}")
            if file != "earthquake_1.csv":
                file_name = file.split(".")[0].split("_")[1]
                x, y = data_process(file_dir + file)
                X.to_hdf(output_dir + f"{file_name}_x.hdf", 'data')
                Y.to_hdf(output_dir + f"{file_name}_y.hdf", 'data')
                X = X.append(x)
                Y = Y.append(y)

    print(X.shape, Y.shape)
    X.to_hdf(output_dir + "train_x.hdf", 'data')
    Y.to_hdf(output_dir + "train_y.hdf", 'data')


def make_submission(submission_dir, submission_file, output_dir):
    submission = pd.read_csv(submission_file, index_col='seg_id')
    submission_x = pd.DataFrame(dtype=np.float64, index=submission.index)
    for seg_id in tqdm(submission_x.index):
        test_df = pd.read_csv(submission_dir + seg_id + '.csv', dtype={'acoustic_data': np.int16})
        segment = test_df['acoustic_data'].values
        generate_features(seg_id, segment, submission_x)
    # scale
    scaler = StandardScaler()
    scaler.fit(submission_x)
    submission_x = pd.DataFrame(scaler.transform(submission_x), columns=submission_x.columns)
    # save
    submission_x.to_hdf(output_dir + "submission.hdf", 'data')


if __name__ == '__main__':
    file_dir = '../earthquakes/'
    output_dir = '../train/'
    submission_path = '../test/'
    submission_file = '../test/sample_submission.csv'
    if not computer_feature:
        make_dataset(file_dir, output_dir)
    make_submission(submission_path, submission_file, output_dir)
