#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Krista Walters, Al Shahriar"
__copyright__ = "Copyright Danfoss Turbocor 2023, The Meanline ML Project"
__credits__ = ["Al Shahriar"]
__license__ = "Private"
__version__ = "1.0.0"
__maintainer__ = "Al Shahriar"
__email__ = "al.shahriar@danfoss.com"
__status__ = "Pre-production"

"""Detail description

@Usage:
    Build a model based on optimized hyperparameters
@Date: 
    July 10 2023
@Files
    training data
    testing data
@Output
    Files:
        weights.h5
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
import time

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

# %% Custom functions
class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch%5000==0:
            print("epoch = ", epoch,", Loss = ", "%.4f"%logs["loss"],", Acc = ","%.4f"%logs["accuracy"])

class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        self.epochs = []
        # use this value as reference to calculate cummulative time taken
        self.timetaken = tf.timestamp()
    def on_epoch_end(self,epoch,logs = {}):
        self.times.append(tf.timestamp() - self.timetaken)
        self.epochs.append(epoch)
        if epoch>5:
            #print("a",epoch)
            print("epoch = ", epoch," time: %.3f" % (self.times[epoch]-self.times[epoch-1]))
        
# %% User inputs

# Locations
train_data_dir = r"../training_data_s2"
train_data_fname = r"training_parameters_s2.pkl"
train_full_dir = os.path.join(train_data_dir, train_data_fname)

clear_current_weights = 1 # will create a backup of the existing weights
# Inputs
in_col = 41 #Number of input columns
out_col = 9 #Number of output columns

# Number of hidden layers
n_layers = 4;
# layers in the NN, first and last are input and output layers, respectively
# In between are the hidden layers
layers = [in_col,125,125,125,125,out_col];
# Specify NN hyperparameters
activation = 'tanh' #Activation function
initial_weights = None #'he_uniform' #Initial guess of the model parameters
reg = None #l1(0.0001) #regularizer
learning_rate_user = 0.001;
beta_1_user = 0.9
epochs = 100000 #Number of epochs
# batch_size = full batch - added later in the code

if len(layers)!=n_layers+2:
    import sys
    sys.exit("Mismatch of n_layers and neuron list")

# If want to run a model for smaller size
trim_flag = 0
if trim_flag==1:
    nExample = 5000
    wrn_txt = "\n \n Data will trimmed for first "+str(nExample)+" examples \n \n"
    warnings.warn(wrn_txt)

# Save model once the job is done
save_flag = 1

now = datetime.now()
case_ID = now.strftime("%Y_%m_%d_%H_%M%S")
print("Running training for case: ",case_ID)

# %% GPU
use_gpu_flag = 1

if use_gpu_flag==1:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"
# %% Import data
train_data_df = pd.read_pickle(train_full_dir)
train_data = train_data_df.to_numpy()
    
# Removing duplicates
# define the location of the dataset
# calculate duplicates
dups = train_data_df.duplicated()
if dups.any():
    print("Duplicates found on train data")
    train_data_df = train_data_df.drop_duplicates()

# Checking dimensions
if len(train_data_df.columns)!= (in_col+out_col):
    import sys
    sys.exit("Imported data dimension mismatch")
    print("expecting cols: ",in_col+out_col, ", but: loaded cols: ",len(train_data_df.columns))
    
# For faster analysis doing for nExample examples only
if trim_flag==1:
    old_train_data = train_data
    train_data = train_data[:nExample,:]

# Training data
x_train = train_data[:,:in_col] #input values
y_train = train_data[:,in_col:] #output values

# Get the parameters for normalization
# based on training data
x_trans = StandardScaler()
x_trans.fit(x_train)
y_trans = StandardScaler()
y_trans.fit(y_train)

# Normalize inputs/outputs to zero mean and unit variance
x_train = x_trans.transform(x_train)
y_train = y_trans.transform(y_train)
# %% Initialize the NN model and hyperparameters

opt = Adam(learning_rate=learning_rate_user, beta_1=beta_1_user) #Optimizer 
batch_size = x_train.shape[0] #Batch size
num_inputs = x_train.shape[1] #Number of inputs
num_outputs = y_train.shape[1] #Number of outputs
# layers = [num_inputs] + 4 * [100] + [num_outputs] #NN architecture

# Making backup of the current weights
dir_name_weight = "weights"
file_name_weight = "weights.h5"
fullpath_weights = os.path.join(dir_name_weight, file_name_weight)
if clear_current_weights==1:
    target = os.path.join(dir_name_weight, "backup_"+case_ID+"_"+file_name_weight)
    if(os.path.isfile(fullpath_weights)):
        shutil.move(fullpath_weights, target)

# Build the NN structure 
model = Sequential()
model.add(Dense(layers[1], input_shape=(layers[0],), kernel_initializer=initial_weights,
                bias_initializer=initial_weights, activation=activation))
for i in range(len(layers)-3):
    model.add(Dense(layers[i+2], kernel_initializer=initial_weights,
                    bias_initializer=initial_weights, activation=activation))
model.add(Dense(layers[-1], activity_regularizer=reg))
#plot_model(model, to_file="model_plot.png", show_shapes=True)

# loss=mse is mean squared error
model.compile(optimizer=opt, loss='mse', metrics='accuracy') #Complile the model

# model checkpointing: save best model
checkpoint = ModelCheckpoint(fullpath_weights,monitor='loss',
                            verbose=0, save_best_only=True, mode='min',
                            save_freq='epoch')

# Saves the training loss values
csv_logger = CSVLogger("training_"+case_ID+".log")    

# Earlystopping is needed to prevent the model from overfitting if the
# number of epochs is very high
earlystopping = EarlyStopping(monitor='loss', min_delta=0, patience=2000,
                              verbose=0, mode='auto', baseline=None,
                              restore_best_weights=False)  

timetaken = timecallback()

# %% Train the model and save the results
start_time = time.time()
with tf.device(device_name):
    r = model.fit(x_train,y_train, epochs=epochs,  
              batch_size=batch_size,verbose=0,
              callbacks=[checkpoint,csv_logger,earlystopping,LossAndErrorPrintingCallback()]) 

execution_time = (time.time() - start_time)
print("Execution time: %.2f seconds" % execution_time)

# Saving the entire model
if save_flag==1:
    model_name = "model_"+case_ID+".h5"
    model_dir = r"complete_model"
    saved_model_dir = os.path.join(model_dir, model_name)
    model.save(saved_model_dir)

# Saving weigths
fullpath_weights = os.path.join(dir_name_weight, file_name_weight)
saved_weights_dir = os.path.join(dir_name_weight, "saved_"+case_ID+"_"+file_name_weight)
if(os.path.isfile(fullpath_weights)):
    shutil.copyfile(fullpath_weights, saved_weights_dir)

# Saving important variables
print("Case number: "+case_ID)
file = open("saved_variables_"+case_ID+".pickle", 'wb')
test_data_dir = "NaN"
test_data_fname = "NaN"
pickle.dump([train_data_dir,test_data_dir,train_data_fname,test_data_fname],file)
pickle.dump([x_trans,y_trans],file)
pickle.dump([case_ID],file)
pickle.dump([saved_weights_dir,saved_model_dir],file)
pickle.dump([execution_time],file)
pickle.dump([learning_rate_user,beta_1_user,activation,layers,initial_weights,reg,epochs],file)
file.close()