import numpy as np
import pandas as pd
import warnings
from scipy import stats
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")


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
    seg_size = 150000
    segments = int(np.floor(signal_data.shape[0] / seg_size))

    train_X = pd.DataFrame(index=range(segments), dtype=np.float64)
    train_Y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])

    for seg_id in tqdm(range(segments)):
        start = seg_id * seg_size
        end = seg_id * seg_size + seg_size
        time_to_failure = signal_data['time_to_failure'][start:end].values

        if np.abs(time_to_failure[0] - time_to_failure[-1]) > 1:
            continue

        segment = signal_data['acoustic_data'][start:end].values
        generate_features(seg_id, segment, train_X)

        train_Y.loc[seg_id, 'time_to_failure'] = time_to_failure[-1]

    print("finish loading")
    # scaler = StandardScaler()
    # scaler.fit(train_X)
    # train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)

    return train_X, train_Y


def train(train_data, train_label, test_data, test_label):
    X, y = np.array(train_data.values), np.array(train_label.values).reshape(-1, )
    X_test, y_test = np.array(train_data.values), np.array(train_label.values).reshape(-1, )
    # print(X, y)
    reg = CatBoostRegressor(iterations=5,
                            depth=2,
                            learning_rate=1,
                            loss_function='MAE')
    print(reg)
    reg.fit(X, y)
    print(reg.feature_importances_)
    pred = []
    for i in range(len(test_data.values)):
        output = reg.predict([test_data.values[i]])
        print(output, test_label.values[i])
        pred.append(output)
    print("mean absolute error: ", mean_absolute_error(test_label.values, pred))


if __name__ == '__main__':
    file_path = '../earthquakes/earthquake_3.csv'
    test_file = '../earthquakes/earthquake_1.csv'
    full_data = '../input/train.csv'
    train_data, train_label = data_process(file_path)
    test_data, test_label = data_process(test_file)
    train(train_data, train_label, test_data, test_label)
