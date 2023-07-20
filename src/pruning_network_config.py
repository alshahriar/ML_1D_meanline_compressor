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
    Finding the optimized number of hidden layers and neurons
@Date: 
    July 13 2023
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
in_col = 18
output_dir = "model_comparison_images"
# Three cases
# model_ID = ["2023_07_13_17_5832","2023_07_13_19_0159","2023_07_13_20_0415"]
# optimized model
model_ID = ["2023_07_14_09_2440"]
nEpochMax = 25000

epoch = np.zeros([nEpochMax,len(model_ID)])
accuracy = np.zeros([nEpochMax,len(model_ID)])
losses = np.zeros([nEpochMax,len(model_ID)])

# %% Load model and calclate accuracy


for i in range(len(model_ID)):
    case_ID = model_ID[i]
    #print(case_ID)

    file = open("saved_variables_"+case_ID+".pickle", 'rb')
    [train_data_dir,test_data_dir,train_data_fname,test_data_fname] = pickle.load(file)
    [x_trans,y_trans] = pickle.load(file)
    [case_ID] = pickle.load(file)
    [saved_weights_dir,saved_model_dir] = pickle.load(file)
    file.close()   
    print("Case loaded: ", train_data_fname)
    
    model = tf.keras.models.load_model(saved_model_dir) 
    model.load_weights(saved_weights_dir) # best epoch   
    
    log_file_name = "training_"+case_ID+".log"
    for j in range(len(model.layers)):
        
        # Weights
        Z = abs(model.layers[j].get_weights()[0])
        x = np.arange(0,Z.shape[1]+1,1)
        y = np.arange(0,Z.shape[0]+1,1)
        fig, ax = plt.subplots()
        pc1 = ax.pcolormesh(x, y, Z)
        plt.colormaps['Reds']
        plt.set_cmap('Reds')
        plt.colorbar(pc1)
        plt.savefig(os.path.join(r"pruning","model_"+(case_ID)+"_weights_"+str(j)+r".png"))
        plt.close()
        
        # Biases
        bb = abs(model.layers[j].get_weights()[1])
        plt.plot(bb)
        plt.savefig(os.path.join(r"pruning","model_"+(case_ID)+"_biases_"+str(j)+r".png"))
        plt.close()
        
        