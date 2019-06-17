import os
import sys
import numpy as np
import pandas as pd
import warnings
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import time
from sklearn import linear_model


warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
n_iter_cat = 2000
n_iter_lgb = 10000
n_fold = 16


def load_data(file_path, file_name):
    X = pd.read_hdf(file_path+file_name+"_x.hdf", 'data')
    Y = pd.read_hdf(file_path+file_name+"_y.hdf", 'data')
    features = X.columns.values
    X, Y = np.array(X.values), np.array(Y.values).reshape(-1, )
    print(X.shape, Y.shape)
    return X, Y, features


def load_submission(submission_path, submission_data):
    submission_x = pd.read_hdf(submission_data)
    submission = pd.read_csv(submission_path, index_col='seg_id')
    return submission, np.array(submission_x.values)


def train_lasso(x_train, y_train, submission_x, features, a):
    predict_submission = np.zeros(len(submission_x))

    # reg = linear_model.Lasso(alpha=0.1)
    # reg.fit(x_train, y_train)
    # predict_submission = reg.predict(submission_x)

    # KFold
    kfold = KFold(n_splits=n_fold, shuffle=True)
    mae = 0
    for fold_, (train_id, valid_id) in enumerate(kfold.split(x_train, y_train)):
        print(f"fold {fold_}")
        train_x, train_y = x_train[train_id], y_train[train_id]
        valid_x, valid_y = x_train[valid_id], y_train[valid_id]
        reg = linear_model.Lasso(alpha=a)
        reg.fit(train_x, train_y)
        reg.coef_

        predict_submission += reg.predict(submission_x) / kfold.n_splits
        predict_valid = reg.predict(valid_x)
        mae += mean_absolute_error(predict_valid, valid_y) / kfold.n_splits

    return predict_submission, mae


if __name__ == '__main__':
    file_dir = 'train/'
    output_dir = 'output_add/'
    train_data, train_label, features = load_data(file_dir, "train")

    # submission
    submission_df, submission_x = load_submission(file_dir + "sample_submission.csv", file_dir + "submission.hdf")

    # train and predict
    for a in [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]:
        lasso_start = time.time()
        submission_ttf, lasso_mae = train_lasso(train_data, train_label, submission_x, features, a)
        lasso_end = time.time()

        # save submission
        timestamp = str(time.time()).split(".")[0]
        output_txt = output_dir + timestamp + '_result.txt'
        submission_df['time_to_failure'] = submission_ttf
        submission_df.to_csv(output_dir + timestamp + 'a=' + str(a) + '_submission.csv', index=True)

        # save result
        with open(output_txt, 'w+') as output:
            print(f"LASSO mae: {lasso_mae}, use {lasso_end - lasso_start}s", file=output)
        print(f"LASSO mae,alpha=, {a}, :, {lasso_mae}, use {lasso_end - lasso_start}s")
