import numpy as np
import sys
import json
import copy

from multiprocessing import Pool

activation_functions = {
    'linear': lambda x, b=0: x,
    'sigmoid': lambda x, b: 1 / (1 + np.exp(-2*b*x)),
    'tan_h': lambda x, b: np.tanh(b*x),
}

derivative_activation_functions = {
    'linear': lambda x, b=0: 1,
    'sigmoid': lambda x, b: 2*b*np.exp(-2*b*x) / (1 + np.exp(-2*b*x))**2,
    'tan_h': lambda x, b: b*(1 - np.tanh(b*x)**2),
}    

class NeuralNetwork:
    def __init__(self, layers: list[int], input_dimensions: int, output_dimensions: int, activation_function='linear', beta=100, learning_rate=0.01, optimizer=''):
        
        self.latent_layer = len(layers) + 1
        all_layers = [input_dimensions] + layers + [output_dimensions] + [layer for layer in reversed(layers)] + [input_dimensions]
        self.layers: list[Layer] = []

        self.input_dimensions = input_dimensions

        i = 0   
        while i < (len(all_layers) - 1):
            dimensions = all_layers[i:i + 2]
            if i == self.latent_layer:
                # self.layers.append(Layer(dimensions, i, 'linear', beta, learning_rate, optimizer))
                self.layers.append(Layer(dimensions, i, activation_function, beta, learning_rate, optimizer))
            else:
                self.layers.append(Layer(dimensions, i, activation_function, beta, learning_rate, optimizer))
            i += 1

        print(f'Layer: {self.layers[-1].weights.shape}')
            
        self.min_error = sys.maxsize

    # TODO: Check if this is correct
    def predict_latent_space(self, data_input):
        next_layer_input = data_input
        for i, layer in enumerate(self.layers):
            next_layer_input = layer.forward(next_layer_input)
            if i == self.latent_layer:
                return next_layer_input
        return next_layer_input


    def predict(self, data_input):
        next_layer_input = data_input
        for layer in self.layers:
            next_layer_input = layer.forward(next_layer_input)
        return next_layer_input
    
    def dump_weights_to_file(self, filename):
        weights = [layer.weights.tolist() for layer in self.layers]
        bias = [layer.biases.tolist() for layer in self.layers]
        data = {
            'weights': weights,
            'biases': bias
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_weights_from_file(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        for i, layer in enumerate(self.layers):
            layer.weights = np.array(data['weights'][i])
            layer.biases = np.array(data['biases'][i])
    
    def train(self, data_input, expected_output, iters):
        iteration = 0
        error_history = []
        pixel_error_history = []

        data_input = np.reshape(data_input, (len(data_input), self.input_dimensions, 1))
        expected_output = np.reshape(expected_output, (len(expected_output), self.input_dimensions, 1))

        input_len = len(data_input)

        best_pixel_error = sys.maxsize
        while iteration < iters:
            err = 0
            # mu = np.random.randint(input_len)
            for mu in range(input_len):
                result = self.predict(data_input[mu])
                err += calculate_error(result, expected_output[mu])
                gradient = calculate_error_derivative(result, expected_output[mu])

                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, iteration)

            error = err / len(data_input)

            pixel_error = self.calculate_pixel_error(data_input, expected_output)

            if pixel_error < best_pixel_error:
                best_pixel_error = pixel_error
                self.dump_weights_to_file('best_weights.txt')
                        
            if iteration % 100 == 0:
                print(f'Iteration: {iteration} - Pixel Error: {pixel_error} - Error: {error} - Best Pixel Error: {best_pixel_error}')

                error_history.append(error)
                pixel_error_history.append(pixel_error)

            if error < self.min_error:
                self.min_error = error

            if pixel_error == 0:
                break

            for layer in reversed(self.layers):
                layer.update()


            # print(f'Iteration: {iteration} - Error: {error} - Min Error: {self.min_error} {improved}')

            iteration += 1
        return self.min_error, iteration, error_history, pixel_error_history
    
    def calculate_pixel_error(self, data_input, expected_output):
        results = [self.predict(data_input[i]) for i in range(len(data_input))]
        errors = [get_different_pixel_count(clean_results(results[i]), expected_output[i]) for i in range(len(results))]
        return np.max(errors)

def calculate_error(predictions, expected_output):
    predictions = clean_results(predictions)
    return np.mean(np.power(expected_output - predictions, 2))

def calculate_error_derivative(predictions, expected_output):
    predictions = clean_results(predictions)
    return 2 * (predictions - expected_output) / np.size(expected_output)

def clean_results(results):
    cleaned = np.array([1 if i > 0.5 else 0 for i in results])
    return cleaned.reshape(results.shape)

def get_different_pixel_count(data1, data2):
    clean_data1 = clean_results(data1)
    return sum([1 for i in range(len(clean_data1)) if clean_data1[i] != data2[i]])

class Layer:
    def __init__(self, dimensions, id, activation_function, beta, learning_rate, optimizer):
        self.id = id
        self.weights = np.random.randn(dimensions[1], dimensions[0])
        self.biases = np.random.randn(dimensions[1], 1)

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
        output = np.dot(self.weights, data) + self.biases
        return self.compute_activation(output)
    
    def backward(self, gradient, iteration):
        # print(f'Layer: {self.id} - Grad Shape: {gradient.shape} - Weights: {self.weights.shape} - Last Input: {self.last_input.shape}')
        weights_gradient = np.dot(gradient, self.last_input.T)
        new_gradient = np.dot(self.weights.T, gradient)

        bias_error = gradient
        weights_error = weights_gradient

        if self.optimizer == 'adam':
            weights_error = self.adam_params.get_delta(iteration, weights_gradient)
            bias_error = self.adam_params.get_delta(iteration, gradient)

        self.weight_acum += weights_error * self.learning_rate
        self.bias_acum += bias_error * self.learning_rate
        self.count += 1

        return np.multiply(new_gradient, self.compute_activation_prime(self.last_input))
    
    def update(self):
        self.weights -= self.weight_acum/self.count
        self.biases -= self.bias_acum/self.count
        self.weight_acum = 0
        self.bias_acum = 0
        self.count = 0

class AdamParams:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None
        self.v = None

        self.one_minus_beta1 = 1
        self.one_minus_beta2 = 1
        
    def get_delta(self, iteration, gradient):
        # if self.m is None:
        self.m = np.zeros_like(gradient)
        self.v = np.zeros_like(gradient)

        self.m = self.beta1 * self.m + (self.one_minus_beta1) * gradient
        self.v = self.beta2 * self.v + (self.one_minus_beta2) * np.power(gradient, 2)

        m = np.divide(self.m, 1 - self.beta1 ** (iteration + 1))
        v = np.divide(self.v, 1 - self.beta2 ** (iteration + 1))

        # m = self.m
        # v = self.v
        return np.divide(m, (np.sqrt(v) + self.epsilon))