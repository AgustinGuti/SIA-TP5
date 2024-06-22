import numpy as np
import json
from ActivationFunctions import compute_activation, compute_activation_prime

class Layer:
    def __init__(self, dimensions, id, activation_function, beta, learning_rate, optimizer, use_backtracking_as_final_gradient=False):
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

        self.weights_error = np.empty_like(self.weights)
        self.use_backtracking_as_final_gradient = use_backtracking_as_final_gradient

        print(f'Layer {self.id} - weights shape: {self.weights.shape}')
    
    def forward(self, data):
        self.last_input = data
        self.last_excitement = np.dot(data, self.weights) # + self.biases
        self.last_output = compute_activation(self.activation_function, self.last_excitement, self.beta)
        return self.last_output
    
    def backward(self, gradient, iteration):        
        if not self.use_backtracking_as_final_gradient:
            np.multiply(compute_activation_prime(self.activation_function, self.last_excitement, self.beta), gradient, out=gradient)
        
        np.dot(self.last_input.T, gradient, out=self.weights_error)
        new_gradient = np.dot(gradient, self.weights.T)

        if self.optimizer == 'adam':
            self.adam_params.get_delta(iteration, self.weights_error, location=self.weights_error)

        self.weight_acum += self.weights_error * self.learning_rate # / (1 + 0.001 * iteration)
        # if iteration % 1000 == 0 and self.id == 0:
        #     print(f'{self.learning_rate / (1 + 0.001 * iteration)}')

        return new_gradient
    
    def update(self):
        np.subtract(self.weights, self.weight_acum, out=self.weights)
        self.weight_acum = 0
        self.count = 0

    def dump_weights_to_file(self, filename):
        weights = self.weights.tolist()
        bias = self.biases.tolist()
        data = {
            'weights': weights,
            'biases': bias
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_weights_from_file(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.weights = np.array(data['weights'])
        self.biases = np.array(data['biases'])


class VariationalLayer(Layer):
    def __init__(self, dimensions, id, activation_function, beta, learning_rate, optimizer):
        super().__init__(dimensions, id, activation_function, beta, learning_rate, optimizer)
        print(f'^^^ Variational ^^^')
    
    def forward(self, data):
        self.mean = data[:, : data.shape[1] // 2]
        self.std = data[:, data.shape[1] // 2:]
        z, eps = reparametrization_trick(self.mean, self.std)
        self.last_eps = eps
        self.last_output = z
        return np.array(z)
    
    def backward(self, gradient, iteration):
        
        # print(f'Layer {self.id} - gradient shape: {gradient.shape} - VAE')

        dE_mu = gradient
        dE_sigma = self.last_eps * gradient

        # KL divergence
        dKL_dmu = self.mean
        dKL_dsigma = 0.5 * (np.exp(0.5 * self.std) - 1)

        total_error = np.concatenate((dE_mu + dKL_dmu * 0, dE_sigma + dKL_dsigma * 0), axis=1)

        # encoder_output_error = np.concatenate((dE_mu, dE_sigma), axis=1)

        return total_error
    
    def update(self):
        np.subtract(self.weights, self.weight_acum, out=self.weights)
        self.weight_acum = 0
        self.count = 0

def reparametrization_trick(mean, std):
    eps = np.random.normal(0, 1)
    z = mean + std * eps
    return z, eps

class AdamParams:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None
        self.v = None

        self.one_minus_beta1 = 1 - beta1
        self.one_minus_beta2 = 1 - beta2
        
    def get_delta(self, iteration, gradient, location=None):
        if self.m is None:
            self.m = np.zeros_like(gradient)
            self.v = np.zeros_like(gradient)

        self.m = self.beta1 * self.m + (self.one_minus_beta1) * gradient
        self.v = self.beta2 * self.v + (self.one_minus_beta2) * np.power(gradient, 2)

        m = np.divide(self.m, 1 - self.beta1 ** (iteration + 1))
        v = np.divide(self.v, 1 - self.beta2 ** (iteration + 1))

        return np.divide(m, (np.sqrt(v) + self.epsilon), out=location)