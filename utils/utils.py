import numpy as np
import random

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def sigmoid_grad(s):
    ds = s*(1-s)
    return ds

def softmax(x):
    orig_shape = x.shape

    if len(x.shape)>1:
        # nd array
        row_max = np.max(x, axis=1)[:,np.newaxis]
        x -= row_max #for numerical stability
        num = np.exp(x)
        den = np.sum(num, axis=1)[:,np.newaxis]
        x = num/den

    else: 
        #vector
        x -= np.max(x)
        x = np.exp(x)/np.sum(np.exp(x))

    assert x.shape == orig_shape
    return x

def normalizeRows(x):
    x /= np.sqrt(np.sum(x**2, axis=1, keepdims=True))
    return x
