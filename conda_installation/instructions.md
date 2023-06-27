## Author: Al Shahriar
## Instuctions to install anaconda, tensorflow, other necessary tools in Windows
## Extra useful information: 
# Create your own environment in conda, so that you can try different versions of Python without uninstalling another version.
# Watch a youtube tutorial first (for Windows) then come back here.
# Install Anaconda (check to add the path to the environment).
# Open anaconda-navigator.
# Open PowerShell.
# ==========================================================================
## For windows: working version (new_env):
#Best practices: anaconda.com/blog/using-pip-in-a-conda-environment

#On the base environment: 
	conda update -n base -c defaults conda
	conda create --name ml_meanline
	conda activate ml_meanline

#On ml_meanline environment: 
	conda install -c anaconda python
	conda install -c conda-forge spyder
	conda install -c anaconda numpy     
	conda install -c conda-forge matplotlib
	conda install -c anaconda scikit-learn
	conda install -c anaconda git
	conda install -c conda-forge scikit-learn-intelex
	pip install tensorflow
	conda install -c anaconda pandas
#Open spyder from anaconda-navigator (or type spyder in powershell).
#Test the following code:

## % Tensorflow test
# ==========================================================================
import tensorflow as tf;
print(tf.__version__)
print(tf.version.VERSION)
print(tf.reduce_sum(tf.random.normal([1000, 1000])))
#Output should look something like:
# 2.12.0
# 2.12.0
# tf.Tensor(284.4109, shape=(), dtype=float32)

## % Numpy test
# ==========================================================================
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(arr)
print("result: ",type(arr))
#Output: 
#[1 2 3 4 5]
#<class 'numpy.ndarray'>

## % scikit-learn test
# ==========================================================================
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
X = [[ 1,  2,  3],  # 2 samples, 3 features
	 [11, 12, 13]]
y = [0, 1]  # classes of each sample
z = clf.fit(X, y)
print("result: " , type(z))
# Output: 
# <class 'sklearn.ensemble._forest.RandomForestClassifier'>