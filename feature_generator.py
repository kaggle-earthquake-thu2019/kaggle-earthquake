import os
import numpy as np
import pandas as pd
import warnings
from scipy import stats
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold


warnings.filterwarnings("ignore")

computer_feature = False


def load_data(file_path):
    signal_data = pd.read_csv(filepath_or_buffer=file_path,
                              dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
    return signal_data


def generate_features(seg_id, segment, X):
    xc = pd.Series(segment)
    zc = np.fft.fft(xc)

    X.loc[seg_id, 'mean'] = xc.mean()
    X.loc[seg_id, 'std'] = xc.std()
    X.loc[seg_id, 'max'] = xc.max()
    X.loc[seg_id, 'min'] = xc.min()

    X.loc[seg_id, 'mad'] = xc.mad()
    X.loc[seg_id, 'kurt'] = xc.kurtosis()
    X.loc[seg_id, 'skew'] = xc.skew()
    X.loc[seg_id, 'med'] = xc.median()


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


if __name__ == '__main__':
    file_dir = '../earthquakes/'
    output_dir = '../train/'
    if not computer_feature:
        make_dataset(file_dir, output_dir)
