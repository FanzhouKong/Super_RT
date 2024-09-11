import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
from toolsets.search import quick_search_values, string_search

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
pd.options.mode.chained_assignment = None  # default='warn'
from rdkit.Chem import rdFMCS
from rdkit import Chem
from tqdm import tqdm
def serrf_single(data_working, qc_idx, compound_name):
    rf_temp = build_qc_model(data_working, qc_idx, compound_name)
    predicted = make_predictions(rf_temp, x= data_working.drop(columns = compound_name), y_raw=data_working[compound_name])
    return(compound_name,predicted)
def serrf_norm(data_working, compound_list, qc_idx):
    data_normalized = data_working.copy()
    for compound in tqdm(compound_list):
        rf_temp = build_qc_model(data_working, qc_idx, col_name=compound)
        predicted = make_predictions(rf_temp, x= data_working.drop(columns = compound), y_raw=data_working[compound])
    data_normalized[compound]=predicted
    data_normalized[data_normalized<0]=0
    return(data_normalized)
def cv_plot(raw_data, compound_name, qc_idx= None, savepath = None):
    # print(qc_idx is None)
    if 'sampleType' in raw_data.columns:
        qc_data = string_search(raw_data, 'sampleType', 'qc')
    elif qc_idx is not None:
        # print(qc_idx is None == False)
        qc_data = raw_data[qc_idx]
    else:
        print('no qc info provided')
        return()
    fig, ax = plt.subplots()
    sns.scatterplot( x=raw_data['time'], y=raw_data[compound_name])
    sns.scatterplot(x=qc_data['time'], y=qc_data[compound_name], color='red')
    plt.xticks([])
    plt.legend(['raw', 'qc'])
    plt.title(compound_name)
    # plt.show()
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
    # return(calc_cv(raw_data[compound_name]), calc_cv(qc_data[compound_name]))
    # print('the cv overall is: ', calc_cv(raw_data[compound_name]))
    # print('the cv on qc is: ', calc_cv(qc_data[compound_name]))
def autoscale(col):
    scaled = []
    for item in col:
        scaled.append((item-col.mean())/col.std())
    return  scaled
def find_col_idx(raw_data, anchor):
    for col in raw_data.columns:
        if anchor in col:
            # print(raw_data.columns.get_loc(col))
            break
    return raw_data.columns.get_loc(col)
def calc_cv(list):
    return(np.round(pd.Series(list).std()/pd.Series(list).mean()*100, 2))
def readin_data(data_path, return_raw = False):

    raw_data = pd.read_csv(data_path)
    # return(raw_data)
    compound_list = raw_data.columns[4:]
    qc_idx = raw_data['sampleType']=='qc'
    if 'sampleType' in raw_data.columns:
        data_working = raw_data.drop(columns=['sampleType', 'label'])
    else:
        data_working = raw_data.copy()
    data_working = make_dummy_features(data_working)
    if return_raw == True:
        return(data_working, qc_idx, compound_list, raw_data)
    else:
        return(data_working, qc_idx,compound_list)
def build_qc_model(data_working, qc_idx, col_name):
    qc_data = data_working[qc_idx]
    y = qc_data[col_name].values
    n_trees = 200
    x = qc_data.drop(columns = col_name, axis = 1)
    rf = RandomForestRegressor(n_estimators = n_trees, random_state =0)
    rf.fit(x, y)
    return(rf)
def make_predictions(rf, x, y_raw):
    y_pred = rf.predict(x)
    
    predicted = y_raw/y_pred*pd.Series(y_raw).median()
    return(predicted)
def make_dummy_features(x, dummy_col = 'batch'):
    x_working = x.copy()
    dummy_feature = pd.get_dummies(x[dummy_col])

    x_working.drop(columns = dummy_col, axis = 1, inplace = True)
    x_working = pd.concat([ dummy_feature,x_working], axis = 1)
    return x_working