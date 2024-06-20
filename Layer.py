import numpy as np
from numba import jit
from ActivationFunctions import compute_activation, compute_activation_prime

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

        self.weights_error = np.empty_like(self.weights)

        print(f'Layer {self.id} - weights shape: {self.weights.shape}')
    
    def forward(self, data):
        self.last_input = data
        # print(f'Layer {self.id} - input shape: {data.shape} - weights shape: {self.weights.shape} ')
        self.last_excitement = np.dot(data, self.weights) # + self.biases
        self.last_output = compute_activation(self.activation_function, self.last_excitement, self.beta)
        return self.last_output
    
    def backward(self, gradient, iteration, final_gradient=None):
        # print(f'Layer {self.id} - gradient shape: {gradient.shape} - is None: {final_gradient is None}')
        if final_gradient is not None:
            self.weight_acum += final_gradient * self.learning_rate / (1 + 0.001 * iteration)
            np.dot(self.last_input.T, final_gradient, out=self.weights_error)
            
            return np.dot(final_gradient, self.weights.T)
        
        np.multiply(compute_activation_prime(self.activation_function, self.last_excitement, self.beta), gradient, out=gradient)
        
        np.dot(self.last_input.T, gradient, out=self.weights_error)
        new_gradient = np.dot(gradient, self.weights.T)

        if self.optimizer == 'adam':
            self.adam_params.get_delta(iteration, self.weights_error, location=self.weights_error)

        self.weight_acum += self.weights_error * self.learning_rate / (1 + 0.001 * iteration)
        # if iteration % 1000 == 0 and self.id == 0:
        #     print(f'{self.learning_rate / (1 + 0.001 * iteration)}')

        return new_gradient
    
    def update(self):
        np.subtract(self.weights, self.weight_acum, out=self.weights)
        self.weight_acum = 0
        self.count = 0

class VariationalLayer(Layer):
    def __init__(self, dimensions, id, activation_function, beta, learning_rate, optimizer):
        super().__init__(dimensions, id, activation_function, beta, learning_rate, optimizer)
        print(f'Layer {self.id} - weights shape: {self.weights.shape}')
    
    def forward(self, data):
        mean = data[:, : data.shape[1] // 2]
        std = data[:, data.shape[1] // 2:]

        self.last_output = data

        # print(f'mean shape: {mean.shape} - std shape: {std.shape}')

        z, eps = reparametrization_trick(mean, std)
        # print(f'z shape: {z.shape}')
        self.last_eps = eps
        return np.array(z)
    
    def backward(self, gradient, iteration, final_gradient=None):
        
        # print(f'Layer {self.id} - gradient shape: {gradient.shape} - VAE')

        # dz_dmean = np.ones([self.last_delta_size, self.latent_space_size])
        #     dz_dstd = eps * \
        #         np.ones([self.last_delta_size, self.latent_space_size])

        #     mean_error = np.dot(last_delta, dz_dmean)
        #     std_error = np.dot(last_delta, dz_dstd)

        #     encoder_output_error = np.concatenate((mean_error, std_error), axis=1)

        dz_dmean = np.ones([2, 2])
        dz_dstd = self.last_eps * np.ones([2, 2])

        mean_error = np.dot(gradient, dz_dmean)
        std_error = np.dot(gradient, dz_dstd)

        encoder_output_error = np.concatenate((mean_error, std_error), axis=1)

        return encoder_output_error

        dE_dz_a = gradient[0]
        dE_deps_a = gradient[0] * self.last_eps

        dE_dz_b = gradient[1]
        dE_deps_b = gradient[1] * self.last_eps

        return np.array([dE_dz_a, dE_deps_a, dE_dz_b, dE_deps_b])

    
    def update(self):
        np.subtract(self.weights, self.weight_acum, out=self.weights)
        self.weight_acum = 0
        self.count = 0

def reparametrization_trick(mean, std):
    eps = np.random.normal(0, 1)
    # print(f'mean shape: {mean.shape} - std shape: {std.shape}')
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