# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 08:52:14 2023

@Modified_author: Krista Walters
"""

####################################    Python Libraries    ############################################################
# Import required libraries

from scipy.io import savemat
import numpy as np
import pandas as pd
from numpy import append

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l1, l2

##################################    Data & Scaling    #################################################################

################################################
# Note: Once the NN is trained, it is used to make predictions. You need to provide in the inputs (x_pred) in order to
# get the outputs (y_pred). Training data is used just to normalize the data to be predicted, as was done during the
# NN training.
################################################

in_col = 18  # Number of input columns
out_col = 3  # Number of output columns


# Import data
# Training data
train_data = np.loadtxt(
    r"X:\CC\USCC-TLH\Common\Everyone\aero-thermal\Krista\Machine_Learning\ML_Codes_MiniTT\Training_Data\Mass_flow_all_RPM_all\Filtered_all\Training1-173_Filtered_all_Conditions_all.csv", delimiter=",")
# prediction data
pred_data = np.loadtxt(
    r"X:\CC\USCC-TLH\Common\Everyone\aero-thermal\Krista\Machine_Learning\ML_Codes_MiniTT\Testing_Data\Mass_flow_all_RPM_all\Filtered_all\Testing1-173_Filtered_all_Conditions_all.csv", delimiter=",")

# Training data
x_train = train_data[:, :in_col]  # input values
y_train = train_data[:, in_col:]  # output values
# Prediction data inputs
x_pred = pred_data[:, :in_col]  # input values

# Normalize the data
x_trans = StandardScaler()
x_trans.fit(x_train)
y_trans = StandardScaler()
y_trans.fit(y_train)

# Normalize inputs to zero mean and unit variance
x_pred = x_trans.transform(x_pred)


###############################################
# Note: These hyperparamters need to be the same as those used during NN training. They are specified here to recreate the
# NN architecture for prediction
##############################################

# Specify NN hyperparameters
activation = 'tanh'  # Activation function
initial_weights = None  # 'he_uniform' #Initial guess of the model parameters
reg = None  # l1(0.0001) #regularizer
opt = Adam(learning_rate=0.001, beta_1=0.9)  # Optimizer
# layers in the NN, first and last are input and output layers, respectively
# Inbetween are the hidden layers
num_inputs = x_train.shape[1]
num_outputs = y_train.shape[1]
layers = [num_inputs] + 4 * [100] + [num_outputs]


# Build the NN structure
model = Sequential()
model.add(Dense(layers[1], input_shape=(layers[0],), kernel_initializer=initial_weights,
                bias_initializer=initial_weights, activation=activation))
for i in range(len(layers)-3):
    model.add(Dense(layers[i+2], kernel_initializer=initial_weights,
                    bias_initializer=initial_weights, activation=activation))
model.add(Dense(layers[-1], activity_regularizer=reg))

# load weights from trained NN
model.load_weights("weights/1-173_&32000epochs_weights.h5")
model.compile(optimizer=opt, loss='mse')
y_pred = model.predict(x_pred)  # output prediction

# Scale to original values
x_pred = x_trans.inverse_transform(x_pred)
y_pred = y_trans.inverse_transform(y_pred)

# Exporting Data

# Creating array that combines x inputs and y prediction values
results = append(x_pred, y_pred, axis=1)

# Transport into MATLAB
pred = {'results': results}
savemat("Matlab_Scripts/results.mat", pred, appendmat=True, format='5',
        long_field_names=False, do_compression=False, oned_as='row')

# Transport into Excel Spreadsheet
titles = ["Inlet Mass Flow [kg/s]", "RPM", "Number of Main Blades", "TE Hub Blade angle [deg(m)]", "TE Tip Blade angle [deg(m)]", "LE Clearance [m]",
          "TE Clearance [m]", "Inclination angle [deg(delta)]", "Imp. Hub Radius [m]", "Imp. Shroud Radius [m]",
          "Imp. Outlet Radius [m]", "Imp. Outlet Width [m]", "Pinch radius (Rpin) [m]", "Pinch width (Bpin) [m]", "Diff. Outlet Radius [m]", "Diff. Outlet Width [m]",
          "Volute Throat Area [m^2]", "Volute Exit Diameter [m]", "PR-Predicted", "Power-Predicted", "Efficiency-Predicted"]
df = pd.DataFrame(results, columns=titles)
df.to_excel(excel_writer="Excel_Spreadsheets/results.xlsx")


