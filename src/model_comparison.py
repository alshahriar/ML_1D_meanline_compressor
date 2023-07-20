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

"""Detail description

@Usage:
    Comparing the accuracy of different trained models using the raw data
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
train_data_fname_list = [r"training_parameters.pkl",r"training_parameters_92.pkl",r"training_parameters_92.pkl"]
read_method  = 1

# %% Load model and calclate accuracy

for i in range(len(model_ID)):
    train_data_dir = r"../training_data"
    test_data_dir = r"../testing_data"    
    # need to load training data for scaling parameters
    train_data_fname = train_data_fname_list[i] 
    # fixed testing data for all model
    test_data_fname = r"testing_parameters_92.pkl" 
    train_full_dir = os.path.join(train_data_dir, train_data_fname)
    test_full_dir = os.path.join(test_data_dir, test_data_fname)
    train_data_df = pd.read_pickle(train_full_dir)
    test_data_df = pd.read_pickle(test_full_dir)
    train_data = train_data_df.to_numpy()
    test_data = test_data_df.to_numpy()    
    
    # Training data
    x_train = train_data[:,:in_col] #input values
    y_train = train_data[:,in_col:] #output values
    x_test = test_data[:,:in_col] #input values
    y_test = test_data[:,in_col:] #output values
    
    x_trans = StandardScaler()
    x_trans.fit(x_train)
    y_trans = StandardScaler()
    y_trans.fit(y_train)
    x_test_transformed = x_trans.transform(x_test)
    y_test_transformed = y_trans.transform(y_test)    
    
    case_ID = model_ID[i]
    print(case_ID)
    # Recreate the exact same model, including its weights and the optimizer
    model_name = "model_"+case_ID+".h5"
    model = tf.keras.models.load_model(r"complete_model/"+model_name)
    
    # Show the model architecture
    # model.summary()
    # y_pred = model.predict(x_test_transformed) #uses NN weights saved at last epoch
    
    loss, acc = model.evaluate(x_test_transformed, y_test_transformed, verbose=0)
    print("Trained model, accuracy: {:5.2f}%".format(100 * acc))
    
    