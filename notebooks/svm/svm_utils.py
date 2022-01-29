import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import (
    confusion_matrix, roc_curve, 
    precision_recall_fscore_support, plot_roc_curve, roc_auc_score
)

import plotly.express as px


# reading and basic preprocessing
def read_saving(fname, ids):

    col_date = 'D_TRANSACTION_DATE'
    col_amt = 'D_TRAN_AMOUNT'
    col_mult = 'D_TRAN_TYPE'

    df_trans = pd.read_csv(fname)
    df_trans = df_trans[df_trans['id'].isin(ids)]
    df_trans = tidy_basic(df_trans, col_date, col_amt, col_mult)
    return df_trans


def read_credit(fname, ids):

    col_date = 'TRANSACTION_DATE'
    col_amt = 'TRANSACTION_AMT'
    col_mult = 'D_TRAN_TYPE'

    df_trans = pd.read_csv(fname)
    df_trans = df_trans[df_trans['Product'] == 'Credit Card']
    df_trans = df_trans[df_trans['id'].isin(ids)]
    df_trans = tidy_basic(df_trans, col_date, col_amt, col_mult)
    return df_trans


def read_current(fname, ids):

    col_date = 'D_TRANSACTION_DATE'
    col_amt = 'D_TRAN_AMOUNT'
    col_mult = 'D_TRAN_TYPE'    

    df_trans = pd.read_csv(fname)
    df_trans = df_trans[df_trans['id'].isin(ids)]
    df_trans = tidy_basic(df_trans, col_date, col_amt, col_mult)
    return df_trans


def tidy_basic(df_trans, col_date, col_amt, col_mult):

    df_trans['date'] = pd.to_datetime(df_trans[col_date])
    df_trans['transact_d'] = df_trans['transact_c'] = df_trans[col_amt]
    df_trans.loc[df_trans[col_mult] == 'C', 'transact_d'] = 0
    df_trans.loc[df_trans[col_mult] == 'D', 'transact_c'] = 0
    df_trans = df_trans[['id', 'date', 'transact_c', 'transact_d']]

    return df_trans  
    
    
# summarise sum by id and day
def summarise_id_day(df_trans, ids, start_date, end_date):
    df_trans = df_trans[df_trans['date'] < pd.to_datetime(end_date)]
    dates = pd.date_range(start=start_date, end=end_date)
    df_fillers = pd.DataFrame({
        'id': np.tile(ids, len(dates)),
        'date': np.repeat(dates, len(ids)),
        'transact_c': 0, 
        'transact_d': 0
    })
    df_trans = df_trans.append(df_fillers)

    df_trans['day'] = (pd.to_datetime(df_trans['date']) \
                        .apply(lambda x: x.value) - pd.to_datetime('1/1/2021').value) // 86400000000000

    df_trans = df_trans.sort_values(['id', 'day'])
    df_trans = df_trans.groupby(['id', 'day']).sum().reset_index(drop=False)
    df_trans = df_trans.set_index('id')
    
    return df_trans


# roll df by cid, return dict
def roll_df(df_trans, ids, rolling_window):
    dict_rolled = {}
    for cid in tqdm(ids):
        df = df_trans.loc[cid, ][['transact_c', 'transact_d']]
        df = df.rolling(window=rolling_window).apply(lambda x : sum(x) / rolling_window)
        df = df.dropna().reset_index(drop=True)
        dict_rolled[cid] = df
    return dict_rolled


def roll_df_cache(df_trans, ids, rolling_window, cache_dir):
    path = os.path.join(cache_dir, str(rolling_window))
    if not os.path.isdir(path):
        dict_rolled = roll_df(df_trans, ids, rolling_window)
        os.mkdir(path)
        for cid in dict_rolled:
            dict_rolled[cid].to_csv(os.path.join(path, '{}.csv').format(cid), index=False)
    else:
        dict_rolled = {}
        for cid in tqdm(ids):
            dict_rolled[cid] = pd.read_csv(os.path.join(path, '{}.csv').format(cid))
            
    return dict_rolled


def split_by_idxs(X, y, idxs):
    return X[idxs, :], np.array(y)[idxs]


def log_transform(array):
    return np.log(array.replace({0: 1}))


def transacts_to_bins(transacts, bins):
    feat = np.zeros((len(bins)))
    tmp = np.digitize(transacts, bins) - 1
    tmp = pd.DataFrame(tmp).reset_index(drop=False).groupby(0).count()['index']
    feat[tmp.index] = tmp
    return feat


def plot_features(X, meta, y):

    colnames = []
    for col, n in meta:
        for i in range(n):
            colnames.append(col + str(i))

    df_X = pd.DataFrame(X)
    df_X.columns = colnames
    df_X['y'] = y
    df_X = df_X.melt('y', colnames, var_name='transact')
    df_X['type'] = df_X['transact'].str.extract('([a-zA-Z_]+)')
    df_X['transact'] = df_X['transact'].str.extract('([0-9]+)').astype(int)

    df_X_avg = df_X.groupby(['type', 'y', 'transact']).mean()
    df_X_avg = df_X_avg.reset_index(drop=False)
    df_X_avg['amount'] = df_X_avg['transact']

    fig = px.line(df_X_avg, x="amount", y="value", color='y', line_shape='spline', facet_col='type')
    fig.show()


def get_svm_classifier():
    return make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))