# Author: Krista Walters. Al Shahriar
# Jun 21 2023

# %% Load libraries
import numpy as np
import pandas as pd
import tensorflow as tf

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


# %% inputs necessary from the user

# Locations
read_method = 1
clear_current_weights = 1 # will create a backup
# Inputs
in_col = 18 #Number of input columns
out_col = 3 #Number of output columns

# Number of hidden layers
n_layers = 4;
layers = [in_col,100,100,100,100,out_col];
#Specify NN hyperparameters
activation = 'tanh' #Activation function
initial_weights = None #'he_uniform' #Initial guess of the model parameters
reg = None #l1(0.0001) #regularizer
learning_rate_user = 0.001;
beta_1_user = 0.9
epochs = 5 #Number of epochs

if len(layers)!=n_layers+2:
    import sys
    sys.exit("Mismatch of n_layers and neuron list")

# If want to run a model for smaller size
trim_flag = 1
if trim_flag==1:
    nExample = 25000
    warnings.warn("Data will trimmed for first "+str(nExample)+" examples")

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

# For faster analysis doing for 5000 examples only
if trim_flag==1:
    old_train_data = train_data
    old_test_data = test_data
    train_data = train_data[:nExample,:]
    test_data = test_data[:int(nExample*0.2),:]
    
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

# Training data
x_train = train_data[:,:in_col] #input values
y_train = train_data[:,in_col:] #output values

# Testing data
x_test = test_data[:,:in_col] #input values
y_test = test_data[:,in_col:] #output values

# Normalize the data
x_trans = StandardScaler()
x_trans.fit(x_train)
y_trans = StandardScaler()
y_trans.fit(y_train)

# Normalize inputs/outputs to zero mean and unit variance
x_train = x_trans.transform(x_train)
y_train = y_trans.transform(y_train)
y_test = y_trans.transform(y_test)
x_test = x_trans.transform(x_test)

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
    now = datetime.now()
    time_str = now.strftime("%Y%m%d%H%M%S")
    target = os.path.join(dir_name_weight, "backup_"+time_str+"_"+file_name_weight)
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
model.compile(optimizer=opt, loss='mse') #Complile the model

#save best model
checkpoint = ModelCheckpoint(fullpath_weights,monitor='loss',      #model checkpointing
                            verbose=0, save_best_only=True, mode='min',
                            save_freq='epoch')
csv_logger = CSVLogger('training.log')    #Saves the training loss values
earlystopping = EarlyStopping(monitor='loss', min_delta=0, patience=2000,  #earlystopping
                              verbose=0, mode='auto', baseline=None,
                              restore_best_weights=False)  

#Train the model
r = model.fit(x_train,y_train, epochs=epochs,  
              batch_size=batch_size,
              callbacks=[checkpoint,csv_logger,earlystopping]) 

# %%Prediction
y_pred = model.predict(x_test) #uses NN weights saved at last epoch

# load weights
model.load_weights(fullpath_weights) #NN weights saved from epoch with lowest loss value

model.compile(optimizer=opt, loss='mse', metrics='accuracy')
y_pred_best = model.predict(x_test) #prediction when loss value is lowest during training

# Evaluate the model
loss, acc = model.evaluate(x_train, y_train, verbose=0)
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

###################################################    Scale data back to original    ###########################################################

#Scale to original values 
x_train = x_trans.inverse_transform(x_train)
y_train = y_trans.inverse_transform(y_train)
x_test = x_trans.inverse_transform(x_test)
y_test = y_trans.inverse_transform(y_test)
y_pred    = y_trans.inverse_transform(y_pred)
y_pred_best = y_trans.inverse_transform(y_pred_best)

###########################################################    Error metrics    ##################################################################

#Calculate the testing errors
# Relative L2 norm error
l2_error = np.linalg.norm(y_test - y_pred, 2)/np.linalg.norm(y_test, 2)
print('Relative L2 error_u: %e' % (l2_error))
#best weight 
l2_error_best = np.linalg.norm(y_test - y_pred_best, 2)/np.linalg.norm(y_test, 2)
print('Relative L2 error_u (best): %e' % (l2_error_best))

#Root Mean Square Error (RMSE )
num_testing = test_data.shape[0]
rmse = np.linalg.norm(y_test - y_pred, 2)/np.sqrt(num_testing)
print('RMSE: %e' % (rmse))
#best weight
rmse_best = np.linalg.norm(y_test - y_pred_best, 2)/np.sqrt(num_testing)
print('RMSE (best): %e' % (rmse_best))

# We can also hand calucuate error by using this formula in excel: [abs(y_test-y_pred)/y_test] * 100
###################################################    Plots    #####################################################################################

#Plot the loss function values
fig, ax1 = plt.subplots()
ax1.plot(r.history['loss'], label='loss')
ax1.legend()

#Plot the true vs predicted Efficiency
fig, ax2 = plt.subplots(figsize=(15,15))
#ax2.plot(x_train[:,1],y_train[:,0],'o',label='Training points')
ax2.plot(x_test[:,0],y_test[:,2],'o',label='True')
ax2.plot(x_test[:,0],y_pred_best[:,2],'*',label='Predicted')
ax2.legend()
fig.savefig('//images/plot1.png')

#Summay of the NN structure
print(model.summary())
#Loss function value
print(min(r.history['loss']))
#The End
# %%
np.savez('saved_variables',y_pred=y_pred,y_pred_best=y_pred_best)
loaded_variables = np.load('saved_variables.npz')
print(loaded_variables.files)
# %%

