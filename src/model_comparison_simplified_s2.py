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
    Comparing the accuracy of  different trained model
@Date: 
    July 11 2023
@Files
    saved_weights_[case_ID].h5
    model_[case_ID].h5
@Output
    Files:
        Output png images
"""
# %% Load libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# System or built-in
import os
import warnings
import datetime
from time import gmtime
from time import strftime
import shutil
import re

from get_input_parameter_range import get_input_parameter_range
from param_bounds_s2 import get_param_bounds_np
from param_bounds_s2 import get_col_list

import matplotlib.pyplot as plt

from pandas import read_csv
from pandas.plotting import scatter_matrix

from matplotlib import pyplot
import matplotlib.cm as cm

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow import py_function
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.utils import plot_model
# %%
def normalize_data(data):
    # Normalize the data
    data_transformed = StandardScaler()
    data_transformed.fit(data)
    # Normalize inputs/outputs to zero mean and unit variance
    data = data_transformed.transform(data)
    return data

def inverse_normalize_data(data):
    # Normalize the data
    data_transformed = StandardScaler()
    data_transformed.fit(data)
    # Normalize inputs/outputs to zero mean and unit variance
    data = data_transformed.transform(data)
    return data
    
# %%
in_col = 44
output_dir = "model_comparison_images"
# Three cases
# model_ID = ["2023_07_13_17_5832","2023_07_13_19_0159","2023_07_13_20_0415"]
# optimized model
model_ID = ["2023_08_25_15_0713"]
test_data_dir = r"../testing_data_s2"
test_data_fname = r"testing_parameters_s2.pkl" 

inlet_mf_txt = "Inlet_mass_flow";

test_full_dir = os.path.join(test_data_dir, test_data_fname)
test_data_df = pd.read_pickle(test_full_dir)
test_data = test_data_df.to_numpy()

x_test = test_data[:,:in_col] #input values
y_test = test_data[:,in_col:] #output values


# %% Load model and calclate accuracy
tfv = str(tf.__version__)
epoch = []
accuracy = []
losses = []

for i in range(len(model_ID)):
    case_ID = model_ID[i]
    print(case_ID)

    file = open("saved_variables_"+case_ID+".pickle", 'rb')
    [train_data_dir,test_data_dir,train_data_fname,test_data_fname] = pickle.load(file)
    [x_trans,y_trans] = pickle.load(file)
    [case_ID] = pickle.load(file)
    [saved_weights_dir,saved_model_dir] = pickle.load(file)
    [execution_time] = pickle.load(file)
    if tfv<='2.10.1': # for tf 2.10
        [learning_rate_user,beta_1_user,activation,layers,initial_weights,reg,epochs] = pickle.load(file)
    file.close()
    
    print(train_data_fname)
    execution_time = strftime("%H:%M", gmtime(execution_time))
    print("execution time:", (execution_time))
    x_test_transformed = x_trans.transform(x_test)
    y_test_transformed = y_trans.transform(y_test)
    
    if tfv>'2.10.1': # for tf 2.12
        model = tf.keras.models.load_model(saved_model_dir)
    else: # tf v = 2.10
        model = tf.keras.models.load_model(saved_model_dir,compile=False)
        opt = Adam(learning_rate=learning_rate_user, beta_1=beta_1_user) #Optimizer 
        model.compile(optimizer=opt, loss='mse', metrics='accuracy')
        
    loss, acc = model.evaluate(x_test_transformed, y_test_transformed, verbose=0)
    print("Trained model, accuracy: {:5.2f}%".format(100 * acc))
    
    y_pred_transformed = model.predict(x_test_transformed,verbose=0) #uses NN weights saved at last epoch
    
    model.load_weights(saved_weights_dir) #NN weights saved from epoch with lowest loss value
    # model.compile(optimizer=opt, loss='mse', metrics='accuracy')
    y_pred_best_transformed = model.predict(x_test_transformed,verbose=0) #prediction when loss value is lowest during training

    y_pred = y_trans.inverse_transform(y_pred_transformed)
    y_pred_best = y_trans.inverse_transform(y_pred_best_transformed)
    
    y_test = np.asarray(y_test).astype('float32')
    y_pred_best = np.asarray(y_pred_best).astype('float32')
    y_pred = np.asarray(y_pred).astype('float32')
    
    #mse = tf.keras.losses.MeanSquaredError()
    mape = tf.keras.losses.MeanAbsolutePercentageError()
    #mre = tf.keras.metrics.MeanRelativeError(y_trans)
    #print('MSE accuracy: %f' % (1-mse(y_test, y_pred).numpy()))
    mape_error = mape(y_test, y_pred_best).numpy()
    print('MAPE accuracy: {:.2f}%'.format(100- mape_error))
    #print('MRE accuracy: %f' % (1- mre(y_test, y_pred).numpy()))
    
    l2_error = np.linalg.norm(y_test - y_pred, 2)/np.linalg.norm(y_test, 2)
    print('Relative L2 accuracy: {:.2f}%'.format(100*(1-l2_error)))
    l2_error_best = np.linalg.norm(y_test - y_pred_best, 2)/np.linalg.norm(y_test, 2)
    print('Relative L2 accuracy (best): {:.2f}%'.format(100*(1-l2_error_best)))
    
    log_file_name = "training_"+case_ID+".log"
    df = pd.read_csv(log_file_name)
    epoch.append(df.epoch.values)
    accuracy.append(df.accuracy.values)
    losses.append(df.loss.values)

    # Plot the true vs predicted
    fig, ax2 = plt.subplots()
    #ax2.plot(x_train[:,1],y_train[:,0],'o',label='Training points')
    ax2.plot(x_test[:,0],y_test[:,2],".", markersize=1,label='True')
    ax2.plot(x_test[:,0],y_pred_best[:,2],".", markersize=1,label='Predicted')
    ax2.legend()
    plt.xlabel('mass flow rate [kg/s]')
    plt.ylabel('$\eta$ (T-S)')
    fig.savefig(r'ml_images/etaTS.png')

    error_txt = ["PR","Power","efficiency"]
    for error_index in range(0,3):
        errtxt = error_txt[error_index]
        # Colored error: MF vs eta
        fig, ax2 = plt.subplots()
        error_rel = (abs(y_test - y_pred)/y_test)
        # error is eta
        error_rel_eta = np.array(error_rel[:,error_index])
        error_rel_eta = np.log((error_rel_eta))
        plt.scatter(x_test[:,0],y_test[:,2], s=1, c=error_rel_eta, cmap=cm.jet, edgecolors=None)
        plt.colorbar(label="error: "+errtxt, orientation="vertical")
        plt.xlabel('mass flow rate [kg/s]')
        plt.ylabel('$\eta$ (T-S)')
        fig.savefig(r'ml_images/error'+errtxt+r'_etaTS_colored.png')
    
        # Colored error: MF vs PR
        fig, ax2 = plt.subplots()
        error_rel = (abs(y_test - y_pred)/y_test)
        error_rel_eta = np.array(error_rel[:,error_index])
        error_rel_eta = np.log((error_rel_eta))
        plt.scatter(x_test[:,0],y_test[:,0], s=1, c=error_rel_eta, cmap=cm.jet, edgecolors=None)
        plt.colorbar(label="error: "+errtxt, orientation="vertical")
        plt.xlabel('mass flow rate [kg/s]')
        plt.ylabel('PR (T-S)')
        fig.savefig(r'ml_images/error'+errtxt+r'_PRTS_colored.png')
        
        
        # Colored error: MF vs Power
        fig, ax2 = plt.subplots()
        error_rel = (abs(y_test - y_pred)/y_test)
        error_rel_eta = np.array(error_rel[:,error_index])
        error_rel_eta = np.log((error_rel_eta))
        plt.scatter(x_test[:,0],y_test[:,1], s=1, c=error_rel_eta, cmap=cm.jet, edgecolors=None)
        plt.colorbar(label="error: "+errtxt, orientation="vertical")
        plt.xlabel('mass flow rate [kg/s]')
        plt.ylabel('Power')
        fig.savefig(r'ml_images/error'+errtxt+r'_Power_colored.png')


    error_rel = (abs(y_test - y_pred)/y_test)
    error_rel_eta = np.array(error_rel[:,2])
    error_rel_eta = np.log((error_rel_eta))
    
    # input_bound_np,input_bound_df = get_input_parameter_range_s2()
    #input_bound_np,units = get_param_bounds_np()
    #col_list = get_col_list()
    #input_bound_df = pd.DataFrame(input_bound_np,index=col_list)

    nCol = len(test_data_df.columns)
    for iCol in range(0,nCol-3):
        colName = test_data_df.columns[iCol]
        plt.figure()
        plt.scatter(test_data_df[inlet_mf_txt].values,test_data_df[colName].values, s=1, c=error_rel_eta, cmap=cm.jet, edgecolors=None)
        #plt.plot(test_data_df["Inlet mass flow"].values,test_data_df[colName].values,".", markersize=1)
        plt.xlabel("Inlet mass flow rate")
        #plt.ylabel(input_bound_df.index[iCol])
        #plt.xlim(input_bound_df.loc["Mass flow rate"].values)
        #plt.ylim(input_bound_df.iloc[iCol].values)
        
        ftxt_simple = test_data_df.columns[iCol]
        ftxt_simple = ftxt_simple.replace('>','_') # removing spaces
        ftxt_simple = ftxt_simple.replace(' ','_') # removing spaces
        ftxt_simple = re.sub('[!,*)@#%(&$?.^-]', '', ftxt_simple) # removing special char except underscore   
        fname = "zone_"+ftxt_simple+"flowrate.png"
        plt.savefig(r"ml_images/"+fname)

    # Plot the true vs predicted
    fig, ax2 = plt.subplots()
    #ax2.plot(x_train[:,1],y_train[:,0],'o',label='Training points')
    ax2.plot(x_test[:,0],y_test[:,0],".", markersize=1,label='True')
    ax2.plot(x_test[:,0],y_pred_best[:,0],".", markersize=1,label='Predicted')
    ax2.legend()
    plt.xlabel('mass flow rate [kg/s]')
    plt.ylabel('PR (T-S)')
    fig.savefig(r'ml_images/PRTS.png')

    # Plot the true vs predicted
    fig, ax2 = plt.subplots()
    #ax2.plot(x_train[:,1],y_train[:,0],'o',label='Training points')
    ax2.plot(x_test[:,0],y_test[:,1],".", markersize=1,label='True')
    ax2.plot(x_test[:,0],y_pred_best[:,1],".", markersize=1,label='Predicted')
    ax2.legend()
    plt.xlabel('mass flow rate [kg/s]')
    plt.ylabel('Power [W]')
    fig.savefig(r'ml_images/Power.png')    
    
    #plt.figure(1)
    #df.plot.line(x = "epoch",y="accuracy")
    #plt.hold(True)
    #plt.figure(2)
    #df.plot.line(x = "epoch",y="loss")
    #plt.hold(True)
# %%
fig, ax = plt.subplots()
for i in range(len(model_ID)):
    plt.plot(epoch[i],accuracy[i], label=model_ID[i])
plt.ylim([0.9, 0.1])
ax.legend()
# %%
fig, ax = plt.subplots()
for i in range(len(model_ID)):
    plt.plot(epoch[i],losses[i],label=model_ID[i])
plt.ylim([0, 0.1])
ax.legend()
