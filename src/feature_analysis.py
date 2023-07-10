#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Al Shahriar"
__copyright__ = "Copyright Danfoss Turbocor 2023, The Meanline ML Project"
__credits__ = ["Al Shahriar"]
__license__ = "Private"
__version__ = "1.0.0"
__maintainer__ = "Al Shahriar"
__email__ = "al.shahriar@danfoss.com"
__status__ = "Pre-production"

"""Detail description of TT_post_process.py

@Usage:
    Histogram to check if there are any biases in the features
@Date: 
    July 7 2023
@Files
    Required:
        TT Input files
        TT Output files
    Optional:
        compressor.py
@Output
    Files:
        Batch data (both training and testing)
        Accumulated data (both training and testing)
"""
# %%
import os
import shutil
from datetime import datetime
import re
# Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import imageio
# %% Loading

n_features = 18 #Number of input columns
n_labels = 3 #Number of output columns
if 0:
    train_data_dir = r"../training_data"
    test_data_dir = r"../testing_data"    
    train_data_fname = "train_parameters.pkl"
    test_data_fname = "test_parameters.pkl"
    train_full_dir = os.path.join(train_data_dir, train_data_fname)
    test_full_dir = os.path.join(test_data_dir, test_data_fname)
else:
    train_full_dir = r"C:\Users\U423018\asr\machine_learning\training_data\training_parameters.pkl"
    test_full_dir = r"C:\Users\U423018\asr\machine_learning\testing_data\testing_parameters.pkl"  

train_data_df = pd.read_pickle(train_full_dir)
test_data_df = pd.read_pickle(test_full_dir)
# train_data = train_data_df.to_numpy()
# test_data = test_data_df.to_numpy()
list_of_final_columns = train_data_df.columns.to_list()
list_of_final_columns = [lines.replace(' ','_') for lines in list_of_final_columns] # removing spaces
list_of_final_columns = [re.sub('[!,*)@#%(&$?.^-]', '', lines) for lines in list_of_final_columns] # removing special char except underscore
train_data_df.columns = list_of_final_columns;
test_data_df.columns = list_of_final_columns;
# %% Plotting histograms
n_groups = 3;
max_eta = train_data_df.Isentropic_machine_efficiency_TS.max()
min_eta = train_data_df.Isentropic_machine_efficiency_TS.min()
delta_eta = np.linspace(min_eta, max_eta, n_groups+1)
eta_values = np.array(train_data_df.Isentropic_machine_efficiency_TS.values)

range1_txt = str(round(delta_eta[0],2))+"-"+str(round(delta_eta[1],2))
range2_txt = str(round(delta_eta[1],2))+"-"+str(round(delta_eta[2],2))
range3_txt = str(round(delta_eta[2],2))+"-"+str(round(delta_eta[3],2))

bool_val = (eta_values < delta_eta[1])
eta_group = np.multiply(bool_val, 1)

cond1 = ((eta_values >= delta_eta[1]))
cond2 = (eta_values < delta_eta[2])
bool_val = np.all([cond1,cond2],  axis=0)
eta_group = eta_group + np.multiply(bool_val, 2)

cond1 = ((eta_values >= delta_eta[2]))
cond2 = (eta_values <= delta_eta[3])
bool_val = np.all([cond1,cond2],  axis=0)
eta_group = eta_group + np.multiply(bool_val, 3)

zero_values = np.argwhere(eta_group == 0)

train_data_df["eta_group"] = eta_group
# fig = train_data_df.pivot(columns='eta_group').Inlet_mass_flow.plot(kind = 'hist', stacked=True, bins=10, edgecolor="black")
new_df = pd.DataFrame()

for (columnName, columnData) in train_data_df.iteritems():
    print('Column Name : ', columnName)
    new_df["eta_group"] = eta_group
    new_df["data"] = columnData
    fig = new_df.pivot(columns='eta_group').data.plot(kind = 'hist', stacked=True, bins=10, edgecolor="black")
    # print('Column Contents : ', columnData.values)
    # plt.hist(columnData.values,edgecolor="black",bins=10)
    xlbl_txt = columnName.replace("_"," ")
    plt.xlabel(xlbl_txt)
    plt.ylabel('count')
    plt.legend([r"$\eta$ = "+range1_txt,r"$\eta$ = "+range2_txt,r"$\eta$ = "+range3_txt])
    plt.savefig(os.path.join(r"hist_figures", columnName+r".png"))
    plt.close()
    
# %%

n_groups = 3;
max_eta = train_data_df.Isentropic_machine_efficiency_TS.max()
min_eta = train_data_df.Isentropic_machine_efficiency_TS.min()
delta_eta = np.linspace(min_eta, max_eta, n_groups+1)
eta_values = np.array(train_data_df.Isentropic_machine_efficiency_TS.values)

range1_txt = str(round(delta_eta[0],2))+"-"+str(round(delta_eta[1],2))
range2_txt = str(round(delta_eta[1],2))+"-"+str(round(delta_eta[2],2))
range3_txt = str(round(delta_eta[2],2))+"-"+str(round(delta_eta[3],2))

bool_val = (eta_values < delta_eta[1])
eta_group = np.multiply(bool_val, 1)

cond1 = ((eta_values >= delta_eta[1]))
cond2 = (eta_values < delta_eta[2])
bool_val = np.all([cond1,cond2],  axis=0)
eta_group = eta_group + np.multiply(bool_val, 2)

cond1 = ((eta_values >= delta_eta[2]))
cond2 = (eta_values <= delta_eta[3])
bool_val = np.all([cond1,cond2],  axis=0)
eta_group = eta_group + np.multiply(bool_val, 3)

zero_values = np.argwhere(eta_group == 0)

train_data_df["eta_group"] = eta_group
# fig = train_data_df.pivot(columns='eta_group').Inlet_mass_flow.plot(kind = 'hist', stacked=True, bins=10, edgecolor="black")
new_df = pd.DataFrame()

for (columnName, columnData) in train_data_df.iteritems():
    print('Column Name : ', columnName)
    new_df["eta_group"] = eta_group
    new_df["data"] = columnData
    fig = new_df.pivot(columns='eta_group').data.plot(kind = 'hist', stacked=True, bins=10, edgecolor="black")
    # print('Column Contents : ', columnData.values)
    # plt.hist(columnData.values,edgecolor="black",bins=10)
    xlbl_txt = columnName.replace("_"," ")
    plt.xlabel(xlbl_txt)
    plt.ylabel('count')
    plt.legend([r"$\eta$ = "+range1_txt,r"$\eta$ = "+range2_txt,r"$\eta$ = "+range3_txt])
    plt.savefig(os.path.join(r"hist_figures", columnName+r".png"))
    plt.close()    