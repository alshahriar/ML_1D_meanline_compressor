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
    Comparing different trained model
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
from datetime import datetime
import shutil

import matplotlib.pyplot as plt

from pandas import read_csv
from pandas.plotting import scatter_matrix

from matplotlib import pyplot

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
in_col = 18
output_dir = "model_comparison_images"
model_ID = ["2023_07_11_08_4940","2023_07_11_13_0411","2023_07_11_10_5056"]
read_method  = 1

test_data_dir = r"../testing_data"    
test_data_fname = r"testing_parameters_92.pkl" 
test_full_dir = os.path.join(test_data_dir, test_data_fname)
test_data_df = pd.read_pickle(test_full_dir)
test_data = test_data_df.to_numpy()    

x_test = test_data[:,:in_col] #input values
y_test = test_data[:,in_col:] #output values


# %% Load model and calclate accuracy

nEpochMax = 25000

epoch = np.zeros([25000,len(model_ID)])
accuracy = np.zeros([25000,len(model_ID)])
losses = np.zeros([25000,len(model_ID)])

for i in range(len(model_ID)):
    case_ID = model_ID[i]
    #print(case_ID)

    file = open("saved_variables_"+case_ID+".pickle", 'rb')
    [train_data_dir,test_data_dir,train_data_fname,test_data_fname] = pickle.load(file)
    [x_trans,y_trans] = pickle.load(file)
    [case_ID] = pickle.load(file)
    [saved_weights_dir,saved_model_dir] = pickle.load(file)
    file.close()
    
    print(train_data_fname)
    
    x_test_transformed = x_trans.transform(x_test)
    y_test_transformed = y_trans.transform(y_test)

    model = tf.keras.models.load_model(saved_model_dir)
    loss, acc = model.evaluate(x_test_transformed, y_test_transformed, verbose=0)
    print("Trained model, accuracy: {:5.4f}%".format(100 * acc))
    
    y_pred_transformed = model.predict(x_test_transformed,verbose=0) #uses NN weights saved at last epoch
    
    model.load_weights(saved_weights_dir) #NN weights saved from epoch with lowest loss value
    # model.compile(optimizer=opt, loss='mse', metrics='accuracy')
    y_pred_best_transformed = model.predict(x_test_transformed,verbose=0) #prediction when loss value is lowest during training

    y_pred = y_trans.inverse_transform(y_pred_transformed)
    y_pred_best = y_trans.inverse_transform(y_pred_best_transformed)
    
    #mse = tf.keras.losses.MeanSquaredError()
    mape = tf.keras.losses.MeanAbsolutePercentageError()
    #mre = tf.keras.metrics.MeanRelativeError(y_trans)
    #print('MSE accuracy: %f' % (1-mse(y_test, y_pred).numpy()))
    print('MAPE accuracy: {:.4f}%'.format(100- mape(y_test, y_pred).numpy()))
    #print('MRE accuracy: %f' % (1- mre(y_test, y_pred).numpy()))
    
    l2_error = np.linalg.norm(y_test - y_pred, 2)/np.linalg.norm(y_test, 2)
    print('Relative L2 accuracy: {:.4f}%'.format(100*(1-l2_error)))
    l2_error_best = np.linalg.norm(y_test - y_pred_best, 2)/np.linalg.norm(y_test, 2)
    print('Relative L2 accuracy (best): {:.4f}%'.format(100*(1-l2_error_best)))
    
    log_file_name = "training_"+case_ID+".log"
    df = pd.read_csv(log_file_name)
    
    epoch[:,i] = df.epoch.values
    accuracy[:,i] = df.accuracy.values
    losses[:,i] = df.loss.values
    
    #plt.figure(1)
    #df.plot.line(x = "epoch",y="accuracy")
    #plt.hold(True)
    #plt.figure(2)
    #df.plot.line(x = "epoch",y="loss")
    #plt.hold(True)
# %%
for i in range(len(model_ID)):
    plt.plot(epoch[:,i],accuracy[:,i])
plt.ylim([0.9, 1])
# %%
for i in range(len(model_ID)):
    plt.plot(epoch[:,i],losses[:,i])
plt.ylim([0, 0.1])
