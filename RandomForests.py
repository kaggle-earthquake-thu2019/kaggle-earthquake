from sklearn.ensemble.forest import RandomForestRegressor
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.datasets import make_regression

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    signal_data = pd.read_csv(filepath_or_buffer=file_path,
                              dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
    return signal_data


def generate_features(seg_id, segment, X):
    xc = pd.Series(segment)

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
        segment = signal_data['acoustic_data'][seg_id * seg_size:seg_id * seg_size + seg_size]
        generate_features(seg_id, segment, train_X)
        train_Y.loc[seg_id, 'time_to_failure'] = signal_data['time_to_failure'].values[-1]
    print("finish loading")
    scaler = StandardScaler()
    scaler.fit(train_X)
    scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)

    # print(train_X.shape)
    # for i in range(5):
    #     print(train_X.values[i])
    #     print(train_Y.values[i])
    # print(train_X.head(5))
    # print(train_Y.head(5))
    return scaled_train_X, train_Y


def train(train_data, train_label, test_data, test_label):
    X, y = np.array(train_data.values), np.array(train_label.values).reshape(-1,)
    print(X.shape, y.shape)
    rf_reg = RandomForestRegressor(criterion='mae')
    rf_reg.fit(X, y)
    print(rf_reg.feature_importances_)
    pred = []
    for i in range(len(test_data.values)):
        output = rf_reg.predict([test_data.values[i]])
        pred.append(output)
    print(mean_absolute_error(test_label.values, pred))


if __name__ == '__main__':
    file_path = '../earthquakes/earthquake_3.csv'
    train_path = '../input/train.csv'
    test_file = '../earthquakes/earthquake_1.csv'
    train_data, train_label = data_process(file_path)
    test_data, test_label = data_process(test_file)
    train(train_data, train_label, test_data, test_label)
