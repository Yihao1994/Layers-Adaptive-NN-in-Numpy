import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io
import sklearn
import sklearn.datasets

np.random.seed(3)
train_X, train_Y = sklearn.datasets.make_moons(n_samples=2800, noise=.2)

# Visualize the data
#plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);

# Output the data
train_set_x = train_X.T
train_set_y = train_Y
m_train = train_set_x.shape[1]