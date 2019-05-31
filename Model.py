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

from catboost import CatBoostRegressor
import lightgbm as lgb


warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_data(file_path, file_name):
    X = pd.read_hdf(file_path+file_name+"_x.hdf", 'data')
    Y = pd.read_hdf(file_path+file_name+"_y.hdf", 'data')
    print(X.shape, Y.shape)
    return X, Y


def train_cat(train_data, train_label, test_data, test_label):
    X, y = np.array(train_data.values), np.array(train_label.values).reshape(-1, )
    X_test, y_test = np.array(test_data.values), np.array(test_label.values).reshape(-1, )
    predict = np.zeros(len(X_test))
    # KFold
    kfold = KFold(n_splits=8, shuffle=True)
    for fold_, (train_id, valid_id) in enumerate(kfold.split(X, y)):
        print(f"fold {fold_}")
        train_x, train_y = X[train_id], y[train_id]
        valid_x, valid_y = X[valid_id], y[valid_id]
        reg = CatBoostRegressor(iterations=20000,
                                depth=2,
                                learning_rate=1,
                                loss_function='MAE')
        reg.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)])
        predict += reg.predict(X_test) / kfold.n_splits
    mae = mean_absolute_error(predict, y_test)
    return mae


def train_lgb(train_data, train_label, test_data, test_label):
    X, y = np.array(train_data.values), np.array(train_label.values).reshape(-1, )
    X_test, y_test = np.array(test_data.values), np.array(test_label.values).reshape(-1, )
    predict = np.zeros(len(X_test))
    params = {'num_leaves': 51,
              'min_data_in_leaf': 10,
              'objective': 'regression',
              'max_depth': -1,
              'learning_rate': 0.001,
              "boosting": "gbdt",
              "feature_fraction": 0.91,
              "bagging_freq": 1,
              "bagging_fraction": 0.91,
              "bagging_seed": 42,
              "metric": 'mae',
              "lambda_l1": 0.1,
              "verbosity": -1,
              "nthread": -1,
              "random_state": 42}
    # KFold
    kfold = KFold(n_splits=8, shuffle=True)
    for fold_, (train_id, valid_id) in enumerate(kfold.split(X, y)):
        print(f"fold {fold_}")
        train_x, train_y = X[train_id], y[train_id]
        valid_x, valid_y = X[valid_id], y[valid_id]
        reg = lgb.LGBMRegressor(**params, n_estimators=20000, n_jobs=-1)
        reg.fit(train_x, train_y,
                eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='mae')
        predict += reg.predict(X_test) / kfold.n_splits
    mae = mean_absolute_error(predict, y_test)
    return mae


if __name__ == '__main__':
    file_dir = '../train/'
    train_data, train_label = load_data(file_dir, "train")
    test_data, test_label = load_data(file_dir, "1")
    cat_mae = train_cat(train_data, train_label, test_data, test_label)
    lgb_mae = train_lgb(train_data, train_label, test_data, test_label)
    print("catboost mae: ", cat_mae)
    print("lightGBM mae: ", lgb_mae)