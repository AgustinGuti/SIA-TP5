import numpy as np


activation_functions = {
    'linear': lambda x, b=0: x,
    'sigmoid': lambda x, b: 1 / (1 + np.exp(-2*b*x)),
    'tan_h': lambda x, b: np.tanh(b*x),
    'relu': lambda x, b: np.log(np.exp(x) + 1),
}

derivative_activation_functions = {
    'linear': lambda x, b=0: 1,
    'sigmoid': lambda x, b: 2*b*np.exp(-2*b*x) / (1 + np.exp(-2*b*x))**2,
    'tan_h': lambda x, b: b*(1 - np.tanh(b*x)**2),
    'relu': lambda x, b: np.exp(x) / (np.exp(x) + 1),
}  


class Layer:
    def __init__(self, dimensions, id, activation_function, beta, learning_rate, optimizer):
        self.id = id
        self.weights = np.random.randn(dimensions[0], dimensions[1])
        self.biases = np.random.randn(dimensions[0], 1)

        self.output_dimensions = dimensions[1]
        self.optimizer = optimizer

        self.activation_function = activation_function
        self.beta = beta

        self.learning_rate = learning_rate

        self.adam_params = AdamParams()

        self.optimizer = optimizer

        self.weight_acum = 0
        self.bias_acum = 0
        self.count = 0

        self.last_input = None
    
    def compute_activation(self, excitement):
        return activation_functions[self.activation_function](excitement, self.beta)
    
    def compute_activation_prime(self, excitement):
        return derivative_activation_functions[self.activation_function](excitement, self.beta)

    def forward(self, data):
        self.last_input = data
        self.last_excitement = np.dot(data, self.weights) # + self.biases
        self.last_output = self.compute_activation(self.last_excitement)
        return self.last_output
    
    def backward(self, gradient, iteration):
        gradient = np.multiply(self.compute_activation_prime(self.last_excitement), gradient)
        
        weights_error = np.dot(self.last_input.T, gradient)
        new_gradient = np.dot(gradient, self.weights.T)

        if self.optimizer == 'adam':
            weights_error = self.adam_params.get_delta(iteration, weights_error)

        self.weight_acum += weights_error * self.learning_rate / (1 + 0.001 * iteration)
        # if iteration % 1000 == 0 and self.id == 0:
        #     print(f'{self.learning_rate / (1 + 0.001 * iteration)}')

        return new_gradient
    
    def update(self):
        self.weights -= self.weight_acum
        self.weight_acum = 0
        self.count = 0



class AdamParams:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None
        self.v = None

        self.one_minus_beta1 = 1 - beta1
        self.one_minus_beta2 = 1 - beta2
        
    def get_delta(self, iteration, gradient):
        if self.m is None:
            self.m = np.zeros_like(gradient)
            self.v = np.zeros_like(gradient)

        self.m = self.beta1 * self.m + (self.one_minus_beta1) * gradient
        self.v = self.beta2 * self.v + (self.one_minus_beta2) * np.power(gradient, 2)

        m = np.divide(self.m, 1 - self.beta1 ** (iteration + 1))
        v = np.divide(self.v, 1 - self.beta2 ** (iteration + 1))

        # m = self.m
        # v = self.v
        return np.divide(m, (np.sqrt(v) + self.epsilon))