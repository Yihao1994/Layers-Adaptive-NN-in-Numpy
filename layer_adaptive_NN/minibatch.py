import numpy as np
import math

def minibatch(X, Y, mini_batch_size):
    global mini_batches
    m = X.shape[1]
    mini_batches = []
    
    #Shuffle the dataset
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[permutation]
    
    num_howmany_minibatches = math.floor(m/mini_batch_size)
    for i in range(0, num_howmany_minibatches):
        mini_batch_X = shuffled_X[:, i*mini_batch_size:(i+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[i*mini_batch_size:(i+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    if m % mini_batch_size !=0:
        mini_batch_X = shuffled_X[:, (i+1)*mini_batch_size:m]
        mini_batch_Y = shuffled_Y[(i+1)*mini_batch_size:m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
        
    return mini_batches