import numpy as np

class ActivationFunction:
    def __init__(self):
        pass

    def __call__(self, x, beta=1):
        pass

    def derivative(self, x, beta=1):
        pass

class Sigmoid(ActivationFunction):
    def __init__(self):
        pass

    def __call__(self, x, beta=1):
        return 1 / (1 + np.exp(-beta * x))
    
    def derivative(self, x, beta=1):
        return beta * self.__call__(x, beta) * (1 - self.__call__(x, beta))
    
class ReLU(ActivationFunction):
    def __init__(self):
        pass

    def __call__(self, x, beta=1):
        return np.maximum(0, x)
    
    def derivative(self, x, beta=1):
        return np.where(x > 0, 1, 0)
    
class Linear(ActivationFunction):
    def __init__(self):
        pass

    def __call__(self, x, beta=1):
        return x
    
    def derivative(self, x, beta=1):
        return 1
    
class Tanh(ActivationFunction):
    def __init__(self):
        pass

    def __call__(self, x, beta=1):
        return np.tanh(beta * x)
    
    def derivative(self, x, beta=1):
        return beta * (1 - np.tanh(beta * x)**2)
    
class Softmax(ActivationFunction):
    def __init__(self):
        pass

    def __call__(self, x, beta=1):
        return np.exp(x) / np.sum(np.exp(x))
    
    def derivative(self, x, beta=1):
        return self.__call__(x, beta) * (1 - self.__call__(x, beta))