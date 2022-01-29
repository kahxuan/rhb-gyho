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
from svm_utils import *


def get_features_amount(df, bins):
    feat_c = log_transform(df['transact_c'])
    feat_d = log_transform(df['transact_d'])
    feat_c = transacts_to_bins(feat_c, bins['transact_c'])
    feat_d = transacts_to_bins(feat_d, bins['transact_d'])
    meta = [('transact_c', len(feat_c)), ('transact_d', len(feat_d))]
    return np.concatenate([feat_c, feat_d]), meta


def get_X_amount(dict_rolled, ids, bins):

    X = np.zeros((len(ids), sum([len(bins[x]) for x in bins])))
    for i, cid in enumerate(ids):
        feat, meta = get_features_amount(dict_rolled[cid], bins)
        X[i, :] = feat
        
    return X, meta


def get_diff(df):
    df['diff_c'] = df['transact_c'] - df['transact_d']
    df['diff_d'] = df['transact_d'] - df['transact_c']
    df.loc[df['diff_c'] < 0, 'diff_c'] = 0
    df.loc[df['diff_d'] < 0, 'diff_d'] = 0 
    df.loc[df['diff_c'] < 1, 'diff_c'] = 1
    df.loc[df['diff_d'] < 1, 'diff_d'] = 1
    return df.drop(['transact_c', 'transact_d'], axis=1)


def get_features_diff(df, bins):
    df = get_diff(df)
    feat_c = log_transform(df['diff_c'])
    feat_d = log_transform(df['diff_d'])
    feat_c = transacts_to_bins(feat_c, bins['diff_c'])
    feat_d = transacts_to_bins(feat_d, bins['diff_d'])
    meta = [('diff_c', len(feat_c)), ('diff_d', len(feat_d))]
    return np.concatenate([feat_c, feat_d]), meta


def get_X_diff(dict_rolled, ids, bins):

    X = np.zeros((len(ids), sum([len(bins[x]) for x in bins])))
    for i, cid in enumerate(ids):
        feat, meta = get_features_diff(dict_rolled[cid], bins)
        X[i, :] = feat
        
    return X, meta

