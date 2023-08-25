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

"""Detail description of TT_post_process.py

@Usage:
    Build a model based on the data generated by TurboTides
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
        if epoch%1000==0:
            print("epoch = ", epoch,", Loss = ", "%.4f"%logs["loss"],", Acc = ","%.4f"%logs["accuracy"])

# %% User inputs

# Directory locations
read_method = 1
if read_method==0:
    train_data_dir = r"../training_data/training_batch_173.csv"
    test_data_dir = r"../testing_data/testing_batch_173.csv"
    read_method = 0
else:
    train_data_dir = r"../training_data"
    test_data_dir = r"../testing_data"    
    train_data_fname = r"training_parameters.pkl"
    test_data_fname = r"testing_parameters.pkl"
    train_full_dir = os.path.join(train_data_dir, train_data_fname)
    test_full_dir = os.path.join(test_data_dir, test_data_fname)
    read_method  = 1

clear_current_weights = 1 # will create a backup of the existing weights
in_col = 18 #Number of input columns
out_col = 3 #Number of output columns

# Number of hidden layers
n_layers = 4;
layers = [in_col,100,100,100,100,out_col];
# Specify NN hyperparameters
activation = 'tanh' #Activation function
initial_weights = None #'he_uniform' #Initial guess of the model parameters
reg = None #l1(0.0001) #regularizer
learning_rate_user = 0.001;
beta_1_user = 0.9
epochs = 50000 #Number of epochs
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
# %% Import data
if read_method==0:
    train_data = np.loadtxt(train_data_dir, delimiter=",")
    test_data = np.loadtxt(test_data_dir, delimiter=",")
    train_data_df = pd.DataFrame(train_data)
    test_data_df = pd.DataFrame(test_data)
else:
    train_data_df = pd.read_pickle(train_full_dir)
    test_data_df = pd.read_pickle(test_full_dir)
    train_data = train_data_df.to_numpy()
    test_data = test_data_df.to_numpy()    
    
# Removing duplicates
# define the location of the dataset
# calculate duplicates
dups = train_data_df.duplicated()
if dups.any():
    print("Duplicates found on train data")
    train_data_df = train_data_df.drop_duplicates()
dups = test_data_df.duplicated()
if dups.any():
    print("Duplicates found on test data")
    test_data_df = test_data_df.drop_duplicates()

# Checking dimensions
if len(train_data_df.columns)!= (in_col+out_col):
    import sys
    sys.exit("Imported data dimension mismatch")
    print("expecting cols: ",in_col+out_col, ", but: loaded cols: ",len(train_data_df.columns))
    
# For faster analysis doing for nExample examples only
if trim_flag==1:
    old_train_data = train_data
    old_test_data = test_data
    train_data = train_data[:nExample,:]
    test_data = test_data[:int(nExample*0.2),:]

# Training data
x_train = train_data[:,:in_col] #input values
y_train = train_data[:,in_col:] #output values

# Testing data
x_test = test_data[:,:in_col] #input values
y_test = test_data[:,in_col:] #output values

# Get the parameters for normalization
# based on training data
x_trans = StandardScaler()
x_trans.fit(x_train)
y_trans = StandardScaler()
y_trans.fit(y_train)

# Normalize inputs/outputs to zero mean and unit variance
x_train = x_trans.transform(x_train)
y_train = y_trans.transform(y_train)
x_test = x_trans.transform(x_test)
y_test = y_trans.transform(y_test)


# %% Hyperparameters

opt = Adam(learning_rate=learning_rate_user, beta_1=beta_1_user) #Optimizer 
batch_size = x_train.shape[0] #Batch size
#layers in the NN, first and last are input and output layers, respectively
#Inbetween are the hidden layers
num_inputs = x_train.shape[1] #Number of inputs
num_outputs = y_train.shape[1] #Number of outputs
# layers = [num_inputs] + 4 * [100] + [num_outputs] #NN architecture

# %% NN model

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
earlystopping = EarlyStopping(monitor='loss', min_delta=0, patience=1000,
                              verbose=0, mode='auto', baseline=None,
                              restore_best_weights=False)  

# Create a callback that saves the model's weights every 10 epochs
# vb_callback = tf.keras.callbacks.Callback(verbose = 0 if epochs % 10 else 0)

start_time = time.time()
#Train the model
r = model.fit(x_train,y_train, epochs=epochs,  
              batch_size=batch_size,verbose=0,
              callbacks=[checkpoint,csv_logger,earlystopping,LossAndErrorPrintingCallback()]) 

execution_time = (time.time() - start_time)
print("Execution time: %.2f seconds" % execution_time)
# %% Saving results
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
# %% Prediction
y_pred = model.predict(x_test) #uses NN weights saved at last epoch

# load weights
model.load_weights(fullpath_weights) #NN weights saved from epoch with lowest loss value

model.compile(optimizer=opt, loss='mse', metrics='accuracy')
y_pred_best = model.predict(x_test) #prediction when loss value is lowest during training

# Evaluate the model
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("Trained model, accuracy: {:5.2f}%".format(100 * acc))

#for layer in model.layers :
#    print(layer.name+" : input ("+str(layer.input_shape)+") output ("+str(layer.output_shape)+")")

# #Plot the Mass flow vs  Efficiency (true and predicted)
# fig, ax2 = plt.subplots(figsize=(15,15))
# #ax2.plot(x_train[:,0],y_train[:,2],'o',label='Training points')
# ax2.plot(x_test[:,0],y_test[:,2],'o',label='True')
# ax2.plot(x_test[:,0],y_pred_best[:,2],'*',label='Predicted')
# ax2.legend()
# fig.savefig('Comparison_plot.png')

# %% Scale data back to original

# Scale to original values 
x_train = x_trans.inverse_transform(x_train)
y_train = y_trans.inverse_transform(y_train)
x_test = x_trans.inverse_transform(x_test)
y_test = y_trans.inverse_transform(y_test)
y_pred = y_trans.inverse_transform(y_pred)
y_pred_best = y_trans.inverse_transform(y_pred_best)

# %% Evaluating errors

# Calculate the testing errors
# Relative L2 norm error
l2_error = np.linalg.norm(y_test - y_pred, 2)/np.linalg.norm(y_test, 2)
print('Relative L2 error_u: %e' % (l2_error))
# best weight 
l2_error_best = np.linalg.norm(y_test - y_pred_best, 2)/np.linalg.norm(y_test, 2)
print('Relative L2 error_u (best): %e' % (l2_error_best))

# Root Mean Square Error (RMSE )
num_testing = test_data.shape[0]
rmse = np.linalg.norm(y_test - y_pred, 2)/np.sqrt(num_testing)
print('RMSE: %e' % (rmse))
# best weight
rmse_best = np.linalg.norm(y_test - y_pred_best, 2)/np.sqrt(num_testing)
print('RMSE (best): %e' % (rmse_best))

# %% Plotting

# Plot the loss function values
fig, ax1 = plt.subplots()
ax1.plot(r.history['loss'], label='loss')
ax1.legend()

# Plot the true vs predicted Efficiency
fig, ax2 = plt.subplots(figsize=(15,15))
#ax2.plot(x_train[:,1],y_train[:,0],'o',label='Training points')
ax2.plot(x_test[:,0],y_test[:,2],'o',label='True')
ax2.plot(x_test[:,0],y_pred_best[:,2],'*',label='Predicted')
ax2.legend()
fig.savefig(r'ml_images/plot1.png')

# Summay of the NN structure
print(model.summary())
# Loss function value
print("minimum loss: " + str(min(r.history['loss'])))

# %% Saving important variables
file = open("saved_variables_"+case_ID+".pickle", 'wb')
pickle.dump([train_data_dir,test_data_dir,train_data_fname,test_data_fname],file)
pickle.dump([x_trans,y_trans],file)
pickle.dump([case_ID],file)
pickle.dump([saved_weights_dir,saved_model_dir],file)
pickle.dump([execution_time],file)
pickle.dump([opt,learning_rate_user,beta_1_user],file)
file.close()
