import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def linear(x, b=0):
    return x

@jit(nopython=True, cache=True)
def sigmoid(x, b):
    return 1 / (1 + np.exp(-2*b*x))

@jit(nopython=True, cache=True)
def tan_h(x, b):
    return np.tanh(b*x)

@jit(nopython=True, cache=True)
def relu(x, b):
    return np.log(np.exp(x) + 1)

@jit(nopython=True, cache=True)
def derivative_linear(x, b=0):
    return np.ones_like(x)

@jit(nopython=True, cache=True)
def derivative_sigmoid(x, b):
    return 2*b*np.exp(-2*b*x) / (1 + np.exp(-2*b*x))**2

@jit(nopython=True, cache=True)
def derivative_tan_h(x, b):
    return b*(1 - np.tanh(b*x)**2)

@jit(nopython=True, cache=True)
def derivative_relu(x, b):
    return np.exp(x) / (np.exp(x) + 1)

@jit(nopython=True, cache=True)
def compute_activation(activation_function, excitement, beta):
    if activation_function == 'linear':
        return linear(excitement, beta)
    elif activation_function == 'sigmoid':
        return sigmoid(excitement, beta)
    elif activation_function == 'tan_h':
        return tan_h(excitement, beta)
    elif activation_function == 'relu':
        return relu(excitement, beta)
    else:
        raise ValueError("Invalid activation function")

@jit(nopython=True, cache=True)
def compute_activation_prime(activation_function, excitement, beta):
    if activation_function == 'linear':
        return derivative_linear(excitement, beta)
    elif activation_function == 'sigmoid':
        return derivative_sigmoid(excitement, beta)
    elif activation_function == 'tan_h':
        return derivative_tan_h(excitement, beta)
    elif activation_function == 'relu':
        return derivative_relu(excitement, beta)
    else:
        raise ValueError("Invalid activation function")