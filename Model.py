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
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import time

from catboost import CatBoostRegressor
import lightgbm as lgb


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


def train_lgb(x_train, y_train, submission_x, features):
    predict_submission = np.zeros(len(submission_x))
    feature_importance_df = pd.DataFrame()
    params = {'num_leaves': 63,
              'min_data_in_leaf': 10,
              'objective': 'regression',
              'max_depth': -1,
              'learning_rate': 0.01,
              "boosting": "gbdt",
              "feature_fraction": 0.9,
              "bagging_freq": 1,
              "bagging_fraction": 0.9,
              "bagging_seed": 3,
              "metric": 'mae',
              "lambda_l1": 0.1,
              "verbosity": -1,
              "nthread": -1,
              "random_state": 30
              }
    # KFold
    kfold = KFold(n_splits=n_fold, shuffle=True)
    mae = 0
    for fold_, (train_id, valid_id) in enumerate(kfold.split(x_train, y_train)):
        print(f"fold {fold_}")
        train_x, train_y = x_train[train_id], y_train[train_id]
        valid_x, valid_y = x_train[valid_id], y_train[valid_id]
        reg = lgb.LGBMRegressor(**params, n_estimators=n_iter_lgb)
        reg.fit(train_x, train_y,
                eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='mae',
                early_stopping_rounds=1500)

        predict_submission += reg.predict(submission_x) / kfold.n_splits
        predict_valid = reg.predict(valid_x, num_iteration=reg.best_iteration_)
        mae += mean_absolute_error(predict_valid, valid_y) / kfold.n_splits
        # count features importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = features
        fold_importance_df["importance"] = reg.feature_importances_[:len(features)]
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    return mae, predict_submission, feature_importance_df


def features_importance(feature_importance_df, timestamp):
    cols = (feature_importance_df[["Feature", "importance"]]
            .groupby("Feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:-1].index)
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

    plt.figure(figsize=(14, 26))
    sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (averaged over folds)')
    plt.tight_layout()
    plt.savefig('../output/' + timestamp + '_lgbm_importances.png')


if __name__ == '__main__':
    file_dir = '../train/'
    output_dir = '../output/'
    train_data, train_label, features = load_data(file_dir, "train")

    # submission
    submission_df, submission_x = load_submission(file_dir + "sample_submission.csv", file_dir + "submission.hdf")

    # train and predict
    lgb_start = time.time()
    lgb_mae, submission_ttf, feature_df = train_lgb(train_data, train_label, submission_x, features)
    lgb_end = time.time()

    # save submission
    timestamp = str(time.time()).split(".")[0]
    output_txt = output_dir + timestamp + '_result.txt'
    submission_df['time_to_failure'] = submission_ttf
    submission_df.to_csv(output_dir + timestamp + '_submission.csv', index=True)

    # save result
    with open(output_txt, 'w+') as output:
        print(f"LightGBM mae: {lgb_mae}, use {lgb_end - lgb_start}s", file=output)
        print(f"LightGBM mae: {lgb_mae}, use {lgb_end - lgb_start}s")

    # show feature importance
    features_importance(feature_df, timestamp)
